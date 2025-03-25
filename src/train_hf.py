import torch
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModel
from transformers import VisionEncoderDecoderModel, BertConfig, BertModel, ViTModel, AutoConfig
import os
import argparse
from datetime import datetime

# Add argument parsing
parser = argparse.ArgumentParser(description='Train the Image to Plant Architecture model')
parser.add_argument('--image_size', type=int, default=448, help='Size of input images')
parser.add_argument('--side_view', type=str, default='True', help='Use side view images')
parser.add_argument('--preload', type=str, default='True', help='Preload dataset into memory')
parser.add_argument('--encoder_checkpoint', type=str, default='facebook/dinov2-small', help='Encoder checkpoint to use')
parser.add_argument('--decoder_checkpoint', type=str, default='google-bert/bert-base-uncased', help='Decoder checkpoint to use')
parser.add_argument('--dataset_path', type=str, default='data/2000_Plots_20241210_BetterQuantized', help='Path to the dataset')
parser.add_argument('--today_date_str', type=str, default=datetime.now().strftime('%Y%m%d'), help='Date string for experiment naming')
parser.add_argument('--log_dir', type=str, help='Main log directory')
parser.add_argument('--exp_name', type=str, help='Experiment name')
args = parser.parse_args()

# Convert string arguments to boolean
args.side_view = args.side_view.lower() == 'true'
args.preload = args.preload.lower() == 'true'

# Add . as a directory to import from
import sys

# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, META_TOKEN
from plant_dataset import PlantDataset
from utils import model_summary

# 1. 디코더 설정 정의
decoder_checkpoint = args.decoder_checkpoint
if "google-bert/bert" in decoder_checkpoint:
    decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
    decoder_config.max_position_embeddings = 2500  # 최대 시퀀스 길이 설정
    decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
    decoder_config.add_cross_attention=True
    decoder_config.is_decoder=True
elif "gpt2" in decoder_checkpoint:
    decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
    decoder_config.max_position_embeddings = 4096*2  # 최대 시퀀스 길이 설정
    decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
    decoder_config.add_cross_attention=True
    decoder_config.is_decoder=True
elif "google/bigbird-roberta" in decoder_checkpoint:
    decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
    decoder_config.max_position_embeddings = 4096*2  # 최대 시퀀스 길이 설정
    decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
    decoder_config.add_cross_attention=True
    decoder_config.is_decoder=True
    decoder_config.attention_type='original_full'

encoder_checkpoint = args.encoder_checkpoint
if "facebook/dinov2" in encoder_checkpoint:
    image_size = args.image_size
    encoder_config = AutoConfig.from_pretrained(encoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size
elif 'google/vit' in encoder_checkpoint:
    image_size = args.image_size
    encoder_config = AutoConfig.from_pretrained(encoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.size['width'] = image_size
    image_processor.size['height'] = image_size

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, decoder_checkpoint, 
    decoder_config=decoder_config, 
    encoder_config=encoder_config,
    decoder_ignore_mismatched_sizes=True,
)
# model = VisionEncoderDecoderModel.from_pretrained(
# "log/20250324_DinoV2Small_448_Bert_BaseBetterQuantize/results/checkpoint-1000"
# )
# model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# Freeze the encoder parameters
model.encoder.eval()
for param in model.encoder.parameters():
    param.requires_grad = False

# 5. 모델 설정 업데이트
model.config.decoder_start_token_id = SOS_TOKEN  # 디코더 시작 토큰
model.config.bos_token_id = SOS_TOKEN  # Beginning of sequence 토큰
model.config.pad_token_id = PAD_TOKEN  # 패딩 토큰
model.config.eos_token_id = EOS_TOKEN  # End of sequence 토큰

def custom_data_collator(features):
    # features는 데이터셋에서 반환된 샘플들의 리스트
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    
    # 패딩 처리: labels의 길이를 가장 긴 시퀀스에 맞춤
    max_label_length = max(len(f["labels"]) for f in features)
    labels = torch.stack([
        torch.cat([torch.tensor(f["labels"], dtype=torch.long), torch.full((max_label_length - len(f["labels"]),), PAD_TOKEN, dtype=torch.long)])
        for f in features
    ])

    # Plant info is integrated in labels, so don't need to return
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Dataset 인스턴스 생성
growth_stages = None # ["01"]
dataset_path = args.dataset_path
print("Loading Dataset...")
plant_architecture_dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                       process_leaf=True, image_size=image_size,
                       side_view=args.side_view,
                       preload=args.preload, image_processor=image_processor, add_sos_token=False)

# Split the dataset into Train, Validation, and Test sets
train_size = int(0.8 * len(plant_architecture_dataset))  # 80% for training
val_size = int(0.1 * len(plant_architecture_dataset))    # 10% for validation
test_size = len(plant_architecture_dataset) - train_size - val_size  # Remaining 10% for testing

# Use random_split with the seed set above
train_dataset, val_dataset, test_dataset = random_split(plant_architecture_dataset, [train_size, val_size, test_size])

# 훈련 인자 설정
today_date_str = args.today_date_str
encoder_name = args.encoder_checkpoint.split('/')[-1]
decoder_name = args.decoder_checkpoint.split('/')[-1]
side_view_str = "Sideview" if args.side_view else "TopView"

# Use provided experiment name if available, otherwise construct one
if args.exp_name:
    exp_name = args.exp_name
else:
    exp_name = f"{today_date_str}_{encoder_name}_{args.image_size}_{side_view_str}_{decoder_name}"

# Determine output directory
if args.log_dir:
    output_base_dir = f"{args.log_dir}/{exp_name}"
else:
    output_base_dir = f"./log/{exp_name}"

# Create output directory
os.makedirs(output_base_dir, exist_ok=True)
results_dir = f"{output_base_dir}/results"

batch_size = 8
num_train_epochs = 10
gradient_accumulation_steps = 4
warmup_steps = int(train_size * 0.2 // batch_size // gradient_accumulation_steps * num_train_epochs)
print(f"warmup_steps:{warmup_steps}")
training_args = TrainingArguments(
    output_dir=f"{output_base_dir}/checkpoints",     # 모델 출력 디렉토리
    num_train_epochs=num_train_epochs,               # 훈련 에포크 수
    per_device_train_batch_size=batch_size,          # 훈련 배치 사이즈
    per_device_eval_batch_size=1,                    # 평가 배치 사이즈
    warmup_steps=warmup_steps,                       # 학습률 스케줄러를 위한 웜업 스텝 수
    weight_decay=0.01,                               # 가중치 감쇠
    logging_dir=f"{output_base_dir}/logs",           # 로그 디렉토리
    logging_steps=10,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    fp16=True,
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_data_collator,
)

model_summary(model=model, max_depth=1)

# Check if model is already trained
if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
    print(f"Model checkpoint already exists at {results_dir}. Skipping training.")
    # Load the trained model to calculate metrics
    model = VisionEncoderDecoderModel.from_pretrained(results_dir)
else:
    print("Model training...")
    if 1:
        trainer.train()
    else:
        # Resume training from the latest checkpoint
        trainer.train(resume_from_checkpoint=True)
    trainer.save_model(results_dir) # 모델 저장

print("Calculating metrics...")
from calc_metric import calc_metric
calc_metric(model, dataset_path)