import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForCausalLM

import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from PIL import Image, ImageFile
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Add . as a directory to import from
import sys
import re

# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from plant_dataset import load_sideview_images
from image_process import process_leaf_image
from plant_tokenizer import vec2token, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE
from string_to_xml_to_vec import xml2vec, linked_to_recursive

class PlantDataset(Dataset):
    def __init__(self, root_dir, plot=None, stages=None, transform=None, 
                 image_size=224, load_depth=False, preload=True, side_view=False,
                 process_leaf=False, image_processor=None):

        self.root_dir = root_dir
        self.load_depth = load_depth          
        # images_path
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(root_dir, 'images')
        self.depth_image_dir = os.path.join(root_dir, 'depth')
        # plant_string_path
        self.plant_xml_dir = os.path.join(root_dir, 'xml')
        self.preload = preload
        # Get list of plant strings
        self.plant_xml_files = os.listdir(self.plant_xml_dir)
        # Get list of images
        self.image_files = [x.replace('.xml', '.jpeg') for x in self.plant_xml_files]
        if load_depth:
            self.depth_images = os.listdir(self.depth_image_dir)
            self.depth_images.sort()

        # Sort the lists
        self.image_files.sort()
        self.plant_xml_files.sort()

        self.img_size = image_size
        self.image_processor = image_processor
        # Filter with statges
        # Regular expression to extract plot and day numbers
        pattern = r"cowpea_(\d+)_day_(\d+)"
        if stages:
            self.image_files = [x for x in self.image_files if re.match(pattern, x).group(2) in stages]
            self.plant_xml_files = [x for x in self.plant_xml_files if re.match(pattern, x).group(2) in stages]
            if self.load_depth:
                self.depth_images = [x for x in self.depth_images if re.match(pattern, x).group(2) in stages]

        if plot:
            self.image_files = [x for x in self.image_files if re.match(pattern, x).group(1) in plot]
            self.plant_xml_files = [x for x in self.plant_xml_files if re.match(pattern, x).group(1) in plot]
            if self.load_depth:
                self.depth_images = [x for x in self.depth_images if re.match(pattern, x).group(1) in plot]
                
        self.transform = transform

        self.process_leaf = process_leaf
        self.side_view = side_view
        self.plant_string_raw = ""
        
        print(f"Total {len(self.image_files)} images and plant strings loaded")
        
        # self.param_scaler = joblib.load(os.path.join(self.current_script_dir,'scaler.pkl'))

        if self.preload:
            # Pre-load data
            self.images = []
            self.vec = []
            self.plant_infos = []
            print("Pre-loading data")
            for i in tqdm(range(len(self.image_files))):
                image, plant_info, vec = self.getitem(i)
                if image is not None:
                    self.images.append(image)
                    self.vec.append(vec)
                    self.plant_infos.append(plant_info)

    def __len__(self):
        return len(self.image_files)
    
    def getitem(self, idx):
        # Load image
        if self.side_view:
            leaf_img, plant_info = load_sideview_images(self.image_dir, self.image_files[idx], self.img_size, process_leaf=self.process_leaf)
                    
        else:
            try:
                image = Image.open(os.path.join(self.image_dir, self.image_files[idx]))
                # Convert to numpy array
                image = np.array(image)
            except:
                    print(f"Error loading {self.image_files[idx]}")
                    return None, None, None
            
            leaf_area, plant_width, plant_height, processed_img, (x,y,w,h) = process_leaf_image(np.array(image), 
                                                                                normalize=True, debug=False, sqaure_crop=True)
            plant_info = [leaf_area, plant_width, plant_height]
            if self.process_leaf:
                # Preprocess image
                leaf_img = cv2.resize(processed_img, (self.img_size, self.img_size))
            else:
                leaf_img = cv2.resize(np.array(image), (self.img_size, self.img_size))

            if self.load_depth:
                # Convert depth to grayscale
                depth = Image.open(os.path.join(self.depth_image_dir, self.depth_images[idx]))
                depth = np.array(depth)
                # Convert to grayscale if not already
                if len(depth.shape) > 2:
                    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

                if self.process_leaf:
                    # Crop the depth image
                    depth = depth[y:y+h, x:x+w]
                    
                # Normalize depth image to 0-255
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth = depth.astype(np.uint8)

                # Resize the images
                depth = cv2.resize(depth, (self.img_size, self.img_size))

                # Add depth channel
                leaf_img = np.concatenate((leaf_img, depth[:, :, np.newaxis]), axis=2)


        # Load XML file
        # Load and parse the XML file
        try:
            xml_path = os.path.join(self.plant_xml_dir, self.plant_xml_files[idx])
            tree = ET.parse(xml_path)
            # Get the root element
            root = tree.getroot()

            root = linked_to_recursive(root)
            plant_array = []
            xml2vec(root[0], plant_array) # Assume single plant
        except Exception as e:
            print(e)
            print(xml_path)

        return leaf_img, plant_info, plant_array
    
    def __getitem__(self, idx):

        if self.preload:
            image = self.images[idx]
            vec = self.vec[idx]
            plant_info = self.plant_infos[idx]
        else:
            image, plant_info, vec = self.getitem(idx)
            

        if self.transform:
            # Check if the image is a PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = self.transform(image)

            if isinstance(image, Image.Image):
                image = np.array(image)

        if vec:
            # Tokenize the plant structure
            out = vec2token(vec)
            # Add SOS and EOS tokens
            #out = np.concatenate(([SOS_TOKEN], out, [EOS_TOKEN]))
            out = np.concatenate((out, [EOS_TOKEN])) # Trainer will add SOS_TOKEN
            out_len = len(out)
        else:
            out = None
            out_len = 0

        # Conver to tensor
        image = torch.tensor(image)
        # Permute the image tensor
        image = image.permute(2, 0, 1)     

        if self.image_processor:
            image = self.image_processor(image,return_tensors="pt").pixel_values[0]
        return {"pixel_values": image, "labels": out}


# 필요한 객체 불러오기
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
)
# model = VisionEncoderDecoderModel.from_pretrained("log/20250320_Quantized_dataset/results")

model.config.decoder_start_token_id = SOS_TOKEN
model.config.bos_token_id = SOS_TOKEN
model.config.pad_token_id = PAD_TOKEN
model.config.eos_token_id = EOS_TOKEN
model.config.vocab_size = VOCAB_SIZE

# Dataset 인스턴스 생성
growth_stages = ["01"]
dataset = PlantDataset(root_dir="data/2000_Plots_20241210_Quantized", stages=growth_stages, 
                       process_leaf=True,
                       preload=False, image_processor=image_processor)

# 훈련 인자 설정
# Generate today's date string in YYYYMMDD format
from datetime import datetime
today_date_str = datetime.now().strftime('%Y%m%d')
exp_name = f"{today_date_str}_Quantized_dataset_100epoch"
training_args = TrainingArguments(
    output_dir=f'./log/{exp_name}/results',          # 모델 출력 디렉토리
    num_train_epochs=100,                             # 훈련 에포크 수
    per_device_train_batch_size=4,                   # 훈련 배치 사이즈
    per_device_eval_batch_size=4,                    # 평가 배치 사이즈
    warmup_steps=500,                                # 학습률 스케줄러를 위한 웜업 스텝 수
    weight_decay=0.01,                               # 가중치 감쇠
    logging_dir=f'./log/{exp_name}',                 # 로그 디렉토리
    logging_steps=10,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,                       # 평가 데이터셋 (여기서는 동일한 데이터셋 사용)
)

trainer.train()                                 # 모델 학습
trainer.save_model(f"./log/{exp_name}/results") # 모델 저장