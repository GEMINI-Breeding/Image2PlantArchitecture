import torch
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForCausalLM
import os


# Add . as a directory to import from
import sys

# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, META_TOKEN
from plant_dataset import PlantDataset

# 필요한 객체 불러오기
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
)

model.config.decoder_start_token_id = SOS_TOKEN
model.config.bos_token_id = SOS_TOKEN
model.config.pad_token_id = PAD_TOKEN
model.config.eos_token_id = EOS_TOKEN
model.config.max_position_embeddings = 2048 
model.decoder.config.max_position_embeddings = 2048

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
growth_stages = ["01"]
dataset = PlantDataset(root_dir="data/2000_Plots_20241210_Quantized", stages=growth_stages, 
                       process_leaf=True,
                       preload=False, image_processor=image_processor, add_sos_token=False)

# Split the dataset into Train, Validation, and Test sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = int(0.1 * len(dataset))    # 10% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing

# Use random_split with the seed set above
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 훈련 인자 설정
# Generate today's date string in YYYYMMDD format
from datetime import datetime
today_date_str = datetime.now().strftime('%Y%m%d')
exp_name = f"{today_date_str}_Quantized_dataset_PlantMeta_FullData"
training_args = TrainingArguments(
    output_dir=f'./log/{exp_name}/results',          # 모델 출력 디렉토리
    num_train_epochs=10,                            # 훈련 에포크 수
    per_device_train_batch_size=8,                   # 훈련 배치 사이즈
    per_device_eval_batch_size=8,                    # 평가 배치 사이즈
    warmup_steps=1250,                                # 학습률 스케줄러를 위한 웜업 스텝 수
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
    eval_dataset=dataset,
    data_collator=custom_data_collator,            
)

trainer.train()                                 # 모델 학습
trainer.save_model(f"./log/{exp_name}/results") # 모델 저장