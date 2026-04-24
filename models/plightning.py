import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import BatchSizeFinder, LearningRateFinder

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
from concurrent.futures import ThreadPoolExecutor

# Path Settings
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(project_dir)

# 모듈 임포트
from models.model import SequenceDecoderModel, RegressionModel, ViT_FeatureExtractor, CNN_FeatureExtractor
from models.model import RegressionModel_Transformer, PositionalEncoding, VAE, MLP, SeqEmbeddingModel
from models.model import create_organ_mask, get_tgt_mask, create_pad_mask, text_global_pool
from models.model import PlantArchitectureTransformer
from src.plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from src.plant_tokenizer import generate_noise_plant_tokens
from src.plant_dataset import PlantDataset
from src.plantstring2model import plantstring2model
from src.plant_tokenizer import token2vec, vec2token
from src.string_to_xml_to_vec import vec2string
from src.image_process import process_leaf_image
from src.plant_architecture_utils import coordinates_to_angle
import pickle
import copy


# from open_clip.transformer import text_global_pool

# Disable fastpath for TransformerEncoder and MultiHeadAttention
# torch.backends.mha.set_fastpath_enabled(False)

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

class GaussianWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, sigma=0.5):
        super(GaussianWeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma

    def forward(self, inputs, targets):
        probabilities = F.softmax(inputs, dim=1)
        batch_size = probabilities.size(0)
        device = inputs.device

        # Vectorized Gaussian profile creation
        gauss_range = torch.arange(self.num_classes, device=device).unsqueeze(0).float()
        gauss_range = gauss_range.expand(batch_size, -1)
        gauss_center = targets.unsqueeze(1).float()

        gaussian = torch.exp(-0.5 * ((gauss_range - gauss_center) / self.sigma) ** 2)
        gaussian /= gaussian.sum(dim=1, keepdim=True)

        loss = -torch.sum(gaussian * torch.log(probabilities + 1e-12), dim=1)
        return loss
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def make_negative_imgs(image):
    # Suffle the image along the batch dimension. make sure i != j
    # Ensure i != j by checking for identity permutation and reshuffling if necessary
    batch_size = image.size(0)
    # 무작위로 인덱스를 섞음
    idx = np.random.permutation(batch_size)
    if 0:
        # 인덱스가 동일한 경우 요소를 교환하여 섞인 인덱스를 생성
        while np.array_equal(idx, np.arange(batch_size)):
            for i in range(batch_size):
                if i == idx[i]:
                    j = np.random.randint(0, batch_size)
                    idx[i], idx[j] = idx[j], idx[i]
    
    image = image[idx]

    # Add noise to the plant images
    transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)])

    image = transform(image)

    return image

def make_negative_seqs(seqs, shuffle=True, noise_level=0.2):
    # Suffle the seqs along the batch dimension. make sure i != j
    # Ensure i != j by checking for identity permutation and reshuffling if necessary
    batch_size = seqs.size(0)
    # 무작위로 인덱스를 섞음
    if shuffle:
        idx = np.random.permutation(batch_size)
        if 0:
            # 인덱스가 동일한 경우 요소를 교환하여 섞인 인덱스를 생성
            while np.array_equal(idx, np.arange(batch_size)):
                for i in range(batch_size):
                    if i == idx[i]:
                        j = np.random.randint(0, batch_size)
                        idx[i], idx[j] = idx[j], idx[i]
        seqs = seqs[idx]

    # Add noise to seq
    if 0:
        noises = generate_noise_plant_tokens(seqs)
    else:
        noises = torch.randn_like(seqs, requires_grad=True) * noise_level
    seqs = seqs + noises

    return seqs


class MainModule(pl.LightningModule):
    def __init__(self, num_layers=6, num_heads=8, 
                 num_tokens=EOS_TOKEN,
                 dim_model=768,
                 image_size=224, alpha=1.0, lr=1e-5, 
                 dropout=0.10, 
                 max_len=1024,
                 use_depth=False,
                 decoder_only=True,
                 vit_model="facebook/dinov2-small",
                 **kwargs):
        super(MainModule, self).__init__()
        self.save_hyperparameters()  # 전달된 모든 인수를 저장

        # self.automatic_optimization = False

        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.dim_model = dim_model
        self.image_size = image_size
        self.alpha = alpha
        self.lr = lr
        self.dropout = dropout
        self.use_depth = use_depth
        self.max_len = max_len
        self.num_warmup_steps = 1000
        self.num_training_steps = 10000

        # Handle additional keyword arguments
        self.extra_args = kwargs

        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            # Fix the weights 
            for param in self.depth_est_model.parameters():
                param.requires_grad = False
            self.depth_background = cv2.resize(cv2.imread(os.path.join(self.current_script_dir, "../src/assets/dirt.jpg")), (self.image_size, self.image_size))
            # Conver to RGB
            self.depth_background = cv2.cvtColor(self.depth_background, cv2.COLOR_BGR2RGB)

        self.image_encoder = ViT_FeatureExtractor(output_size=dim_model, 
                                                  use_depth=self.use_depth, image_size=image_size, vit_model=vit_model)
        
        self.sequence_decoder = SequenceDecoderModel(
            dim_model=self.dim_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_tokens=self.num_tokens,
            decoder_only=decoder_only,
            use_depth=self.use_depth,
            image_size=self.image_size,
            dropout=self.dropout,
            max_seq_length=max_len,
        )
        
        self.multihead_attn_weights = None
        self.self_attn_weights = None


        self.helios_path = os.path.join(self.current_script_dir, "../src/PlantString2Model/build")
        self.helios = plantstring2model(program_path=self.helios_path,
                                                        program_name="PlantString2Model",
                                                        display=":11.0", 
                                                        height=1.0,background_path=os.path.join(self.current_script_dir,"../src/assets/black.png"))
        self.prev_epoch = -1
        self.current_train_step = 0
        self.current_val_step = 0

    def forward(self, image, plant_info, tgt):
        if self.use_depth:
            image = self.add_depth_to_image(image)
        features = self.image_encoder(image, plant_info)
        out = self.sequence_decoder(features, tgt)
        out = out.permute(1, 0, 2)
        return out

    def generate(self, image, plant_info, stage='val'):
        device = image.device
        y_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
        if self.use_depth:
            image = self.add_depth_to_image(image)
        feature = self.image_encoder(image, plant_info)
        for i in range(self.max_len):
            try:
                if stage == 'val':
                    with torch.no_grad():
                       label_p  = self.sequence_decoder(feature, y_input)
                else:
                    label_p = self.sequence_decoder(feature, y_input)
            except Exception as e:
                print(e)
                print(f"Error in {i} iteration")
                break
            label = label_p.topk(1)[1].view(-1)[-1].item()  # num with highest probability

            # Stop if model predicts end of sentence
            if label == EOS_TOKEN or label == PAD_TOKEN:
                break

            # Make next tensor using label and params
            next_item = torch.tensor([[label]], dtype=torch.long, device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

        return y_input.squeeze(0)                    
    
    def label_loss_fn(self, pred, label, ignore_index=None, label_smoothing=0.0):
        # Define the number of classes (0 to 26)
        if 0:
            loss = F.cross_entropy(pred, label, ignore_index=ignore_index, reduction='sum') / pred.size(0)   
        else:
            loss = F.cross_entropy(pred, label, ignore_index=ignore_index)
        return loss 
        
    def add_depth_to_image(self, image, add_background=True):
    
        if add_background:
            depth_input = torch.zeros_like(image)
            # Add black background the images
            for i in range(image.size(0)):
                # Convert to numpy
                img = image[i].permute(1, 2, 0).cpu().numpy()
                # Mask 0 values
                mask = img == 0
                img[mask] = self.depth_background[mask]
                # Convert to tensor
                depth_input[i] = torch.tensor(img).permute(2, 0, 1)
        else:
            depth_input = image

        with torch.no_grad():
            inputs = self.depth_est_img_proc(images=depth_input, return_tensors="pt").to(image.device)
            outputs = self.depth_est_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        
        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        self.predicted_depth = depth
        # Rescale to 0-255
        depth = depth*255
        # cat depth to image
        image = torch.cat((image, depth), dim=1)

        return image
    
    def compute_loss(self, batch, mode):

        # Load batch and preprocess
        image, plant_info, y, lengths = batch
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        label = y_expected.long()

        # Decoder loss
        y_out = self(image, plant_info, y_input)

        # Reshape y_out and label for CrossEntropyLoss
        y_out = y_out.reshape(-1, y_out.size(-1))  # (N * L, C)
        label = label.reshape(-1)  # (N * L,)

        # Apply CrossEntropyLoss
        loss = F.cross_entropy(y_out, label, ignore_index=PAD_TOKEN)
        self.log(f'{mode}/loss', loss, batch_size=image.size(0), sync_dist=True)

        # Add images to tensorboard
        if (self.current_train_step == 0 and mode == "train") or (self.current_val_step == 0 and mode == "val"):
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_images(f'{mode}/input_images', image, self.current_epoch)
            if self.use_depth:
                tensorboard_logger.add_images(f'{mode}/depth_images', self.predicted_depth, self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, 'train')
        self.current_train_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, 'val')
        self.current_val_step += 1
        return loss

    def log_grads(self):
         for name, param in self.named_parameters():
            # if "seq_embedding_layer" in name:
            #     print(f"Gradient of {name} is {param.grad}")
            #     print(f"{name} requires_grad: {param.requires_grad}")
            if param.grad is not None:
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.current_epoch) # or global_step
                self.logger.experiment.add_histogram(f"{name}", param, self.current_epoch) # or global_step

    def on_after_backward(self):
        if self.prev_epoch != self.current_epoch:
            self.prev_epoch = self.current_epoch
            # self.log_grads()
            self.current_train_step = 0
            self.current_val_step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.num_warmup_steps, 
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
    
class MainDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, train_batch_size=16, val_batch_size=None,
                        num_workers=4, image_size=448, 
                        load_depth=True,
                        side_view=False,
                        process_leaf=False,
                        preload=False,
                        growth_stages=None,
                        partial_data=1.0,
                        **kwargs):
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.preload = preload
        self.process_leaf = process_leaf
        self.load_depth = load_depth
        self.pin_memory = False
        self.side_view = side_view

        self.growth_stages = growth_stages
        self.partial_data = partial_data

        # Handle additional keyword arguments
        self.extra_args = kwargs

    def load_or_create_dataset(self, dataset_dir, dataset_name, plot, stages, load_depth, process_leaf, side_view, preload, image_size, mode='', color_jitter=False, random_crop=False, random_erase=False):
        saved_dataset_name = os.path.join(dataset_dir, f"{dataset_name}.pkl")
        if os.path.exists(saved_dataset_name) and preload:
            print(f"Loading {dataset_name} dataset from .pkl file")
            with open(saved_dataset_name, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = PlantDataset(
                dataset_dir, plot=plot, stages=stages,
                load_depth=load_depth,
                process_leaf=process_leaf, side_view=side_view,
                preload=preload, image_size=image_size,
                mode=mode, color_jitter=color_jitter, random_crop=random_crop, random_erase=random_erase
            )
            if preload:
                # Check if the dataset is already saved
                if not os.path.exists(saved_dataset_name):
                    print(f"Saving {dataset_name} dataset to .pkl file")
                    with open(saved_dataset_name, "wb") as f:
                        pickle.dump(dataset, f)
        return dataset

    def setup(self, stage=None):
        train_ratio = 0.7 * self.partial_data
        val_ratio = 0.15 * self.partial_data
        test_ratio = 0.15 * self.partial_data

        growth_stages = self.growth_stages

        # Get the num plots from the last xml file
        xml_files = os.listdir(os.path.join(self.dataset_dir, "xml"))
        xml_files.sort()
        self.num_plots = int(xml_files[-1].split("_")[1]) + 1

        train_end = int(self.num_plots * train_ratio)
        val_end = train_end + int(self.num_plots * val_ratio)
        test_end = min(self.num_plots, val_end + int(self.num_plots * test_ratio)) # Ensure total sums up to num_plots

        train_plots = [f"{plot:04d}" for plot in range(train_end)]
        val_plots = [f"{plot:04d}" for plot in range(train_end, val_end)]
        test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]

        self.train_dataset = self.load_or_create_dataset(
            self.dataset_dir, "train_dataset", train_plots, growth_stages,
            self.load_depth, self.process_leaf, self.side_view,
            self.preload, self.image_size, mode='train', color_jitter=True, random_crop=True, random_erase=True
        )
        self.val_dataset = self.load_or_create_dataset(
            self.dataset_dir, "val_dataset", val_plots, growth_stages,
            self.load_depth, self.process_leaf, self.side_view,
            self.preload, self.image_size, mode='val'
        )
        self.test_dataset = self.load_or_create_dataset(
            self.dataset_dir, "test_dataset", test_plots, growth_stages,
            self.load_depth, self.process_leaf, self.side_view,
            self.preload, self.image_size, mode='test'
        )

        
    def collate_fn(self, batch):
        images = [f['pixel_values'] for f in batch]
        plant_info = [f['plant_info'] for f in batch]
        out = [f['labels'] for f in batch]
        lens = [len(f['labels']) for f in batch]
        max_length = max(lens)
        out_padded = np.ones([len(out), max_length]) * PAD_TOKEN
        for i, seq in enumerate(out):
            out_padded[i,:len(seq)] = seq
        images = torch.stack(images)
        plant_info = np.array(plant_info)
        plant_info = torch.tensor(plant_info, dtype=torch.float32)
        out_tensor = torch.tensor(out_padded, dtype=torch.long)
        return images, plant_info, out_tensor

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self, shuffle=True):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=shuffle,
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )
    

import unittest
import torch
from models.plightning import MainModule

class TestMainModule(unittest.TestCase):
    def setUp(self):
        self.model = MainModule()
        self.model.eval()  # 모델을 평가 모드로 설정
        self.image = torch.randn(1, 3, 224, 224)  # 임의의 이미지 텐서
        self.plant_info = torch.randn(1, 10)  # 임의의 식물 정보 텐서

    def test_generate(self):
        with torch.no_grad():
            result = self.model.generate(self.image, self.plant_info)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.dim(), 2)  # 결과 텐서는 2차원이어야 함
            self.assertEqual(result.size(0), 1)  # 배치 크기는 1이어야 함

    def test_generate_with_beam_search(self):
        with torch.no_grad():
            result = self.model.generate(self.image, self.plant_info, beam_size=5)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.dim(), 2)  # 결과 텐서는 2차원이어야 함
            self.assertEqual(result.size(0), 1)  # 배치 크기는 1이어야 함

    def test_generate_with_no_repeat_ngram(self):
        with torch.no_grad():
            result = self.model.generate(self.image, self.plant_info, no_repeat_ngram_size=3)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.dim(), 2)  # 결과 텐서는 2차원이어야 함
            self.assertEqual(result.size(0), 1)  # 배치 크기는 1이어야 함

if __name__ == '__main__':

    import os
    import sys
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    import cv2
    import matplotlib.pyplot as plt
    import shutil
    import subprocess
    from PIL import Image
    from torchvision import transforms
    from tqdm.notebook import tqdm

    from models.plightning import MainModule, MainDataModule
    from models.model import get_tgt_mask
    from src.plant_tokenizer import SOS_TOKEN, EOS_TOKEN, token2vec
    from src.string_to_xml_to_vec import vec2xml, pretty_print_xml, recursive_to_linked
    from src.plant_dataset import load_sideview_images
    from src.image_process import process_leaf_image

    def re_render_xml(output_path, filename, program_path, rotation=True):
        image_name = filename.split("/")[-1].split(".")[0]
        os.environ["DISPLAY"] = ":12.0"
        command = f"cd {program_path} && ./main -h 1.0 -o {output_path} -name {image_name} -tile none -f {os.path.join(output_path, filename)}"
        if rotation:
            command += " -r"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        return result

    def process_and_display_images(model, dataloader, n_figures, temp_folder, program_path):
        fig, axes = plt.subplots(3, n_figures, figsize=(20, 8))
        image_size = model.image_size
        device = model.device

        for idx, (image, plant_info, out, lengths) in enumerate(dataloader):
            if idx >= n_figures:
                break

            if image.dim() == 3:
                image = image.unsqueeze(0)

            image = image.to(device)
            plant_info = plant_info.to(device)
            out = torch.tensor(out).to(device)
            ground_truth = out.squeeze(0).cpu().numpy()

            plant_vec = token2vec(ground_truth)
            plant_xml = vec2xml(plant_vec, debug=True)
            plant_xml_file_name = f"{temp_folder}/plant_{idx}_gt.xml"
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)
            plant_xml = recursive_to_linked(plant_xml)
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)

            with torch.no_grad():
                result = model.generate(image, plant_info)
                result = result.cpu().numpy()

            plant_vec = token2vec(result)
            plant_xml = vec2xml(plant_vec, debug=True)
            plant_xml_file_name = f"{temp_folder}/plant_{idx}.xml"
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)
            plant_xml = recursive_to_linked(plant_xml)
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)

            re_render_xml(os.path.abspath(temp_folder), os.path.abspath(plant_xml_file_name), program_path)
            img, _ = load_sideview_images(temp_folder, plant_xml_file_name.replace("xml", "jpeg"), model.image_size, True)

            image_vis = image[0].permute(1, 2, 0).cpu()
            image_vis = cv2.normalize(np.array(image_vis), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            row, col = divmod(idx, n_figures)
            axes[row, col].imshow(image_vis[:, :, 0:3])
            axes[row, col].set_title(f"Input Image {idx + 1}")
            axes[row, col].axis('off')

            depth = model.predicted_depth.squeeze().cpu()
            axes[row+1, col].imshow(depth)
            axes[row+1, col].set_title(f"Estimated Depth Image {idx + 1}")
            axes[row+1, col].axis('off')

            axes[row + 2, col].imshow(img)
            axes[row + 2, col].set_title(f"Output Model {idx + 1}")
            axes[row + 2, col].axis('off')

        plt.tight_layout()
        plt.show()

    def main():
        # Add ../ as a directory to import from
        sys.path.append('../')

        # Load model
        model = MainModule.load_from_checkpoint("log/20250114_SideView_224_QuantizedParams/version_0/checkpoints/last.ckpt")
        model.eval()

        # Setup data module
        dataset_dir = "data/Sideview_Dec23_2024"
        datamodule = MainDataModule(dataset_dir,
                                    image_size=model.image_size,
                                    load_depth=False,
                                    train_batch_size=1, num_workers=0, process_leaf=True, preload=False, side_view=True)
        growth_stages = [f"{day:02d}" for day in range(0, 2)]
        datamodule.setup(growth_stages=growth_stages)
        datamodule.setup()
        dataloader = datamodule.test_dataloader()

        # Create temp folder
        temp_folder = "temp"
        shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder, exist_ok=True)

        # Process and display images
        process_and_display_images(model, dataloader, n_figures=5, temp_folder=temp_folder, program_path="src/GenerateDataset/build")

    main()


    #####
    unittest.main()