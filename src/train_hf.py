import torch
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModel
from transformers import VisionEncoderDecoderModel, BertConfig, BertModel, ViTModel, AutoConfig
import os

# Add . as a directory to import from
import sys

# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, META_TOKEN
from plant_dataset import PlantDataset
from utils import model_summary

# 1. 디코더 설정 정의
if 1:
    decoder_checkpoint = "google-bert/bert-base-uncased"
    #decoder_checkpoint = "google-bert/bert-large-uncased"
    decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
    decoder_config.max_position_embeddings = 2500  # 최대 시퀀스 길이 설정
    decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
    decoder_config.add_cross_attention=True
    decoder_config.is_decoder=True
elif 0:
    decoder_checkpoint = "gpt2"
    decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
    decoder_config.max_position_embeddings = 4096*2  # 최대 시퀀스 길이 설정
    decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
    decoder_config.add_cross_attention=True
    decoder_config.is_decoder=True
else:
    decoder_checkpoint = "google/bigbird-roberta-base"
    decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
    decoder_config.max_position_embeddings = 4096*2  # 최대 시퀀스 길이 설정
    decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
    decoder_config.add_cross_attention=True
    decoder_config.is_decoder=True
    decoder_config.attention_type='original_full'

if 1:
    image_size = 448
    #encoder_checkpoint = "facebook/dinov2-base"
    encoder_checkpoint = "facebook/dinov2-small"
    encoder_config = AutoConfig.from_pretrained(encoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size
else:
    image_size = 448
    encoder_checkpoint = "google/vit-base-patch16-224-in21k"
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

import evaluate
import numpy as np
metric = evaluate.load('sacrebleu')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Ensure labels and predictions are numpy arrays
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Mask out padding tokens in labels
    non_pad_mask = labels != PAD_TOKEN

    # Align predictions and labels by masking out padding tokens
    filtered_predictions = [pred[mask] for pred, mask in zip(predictions, non_pad_mask)]
    filtered_labels = [label[mask] for label, mask in zip(labels, non_pad_mask)]

    # Compute the metric
    return metric.compute(predictions=filtered_predictions, references=filtered_labels)


# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Dataset 인스턴스 생성
growth_stages = None # ["01"]
plant_architecture_dataset = PlantDataset(root_dir="data/2000_Plots_20241210_BetterQuantized", stages=growth_stages, 
#plant_architecture_dataset = PlantDataset(root_dir="data/20250311_Sideview_40Days", stages=growth_stages, 
                       process_leaf=True, image_size=image_size,
                       side_view=True,
                       preload=True, image_processor=image_processor, add_sos_token=False)

# Split the dataset into Train, Validation, and Test sets
train_size = int(0.8 * len(plant_architecture_dataset))  # 80% for training
val_size = int(0.1 * len(plant_architecture_dataset))    # 10% for validation
test_size = len(plant_architecture_dataset) - train_size - val_size  # Remaining 10% for testing

# Use random_split with the seed set above
train_dataset, val_dataset, test_dataset = random_split(plant_architecture_dataset, [train_size, val_size, test_size])

# 훈련 인자 설정
# Generate today's date string in YYYYMMDD format
from datetime import datetime
today_date_str = datetime.now().strftime('%Y%m%d')
exp_name = f"{today_date_str}_DinoV2Small_448_Sideview_Bert"
#exp_name = f"{today_date_str}_Debug"
batch_size = 8
num_train_epochs = 10
gradient_accumulation_steps = 4
warmup_steps = int(train_size * 0.2 // batch_size // gradient_accumulation_steps * num_train_epochs)
print(f"warmup_steps:{warmup_steps}")
training_args = TrainingArguments(
    output_dir=f'./log/{exp_name}/results',          # 모델 출력 디렉토리
    num_train_epochs=num_train_epochs,                             # 훈련 에포크 수
    per_device_train_batch_size=batch_size,                   # 훈련 배치 사이즈
    per_device_eval_batch_size=1,                    # 평가 배치 사이즈
    warmup_steps=warmup_steps,                               # 학습률 스케줄러를 위한 웜업 스텝 수
    weight_decay=0.01,                               # 가중치 감쇠
    logging_dir=f'./log/{exp_name}',                 # 로그 디렉토리
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
    # compute_metrics=compute_metrics, 
)

model_summary(model=model, max_depth=1)

# # Evaluate the model before training (Epoch 0)
# print("Evaluating the model at Epoch 0 (before training)...")
# metrics = trainer.evaluate(eval_dataset=val_dataset)
# print(f"Metrics at Epoch 0: {metrics}")

if 1:
    trainer.train()
else:
    # Resume training from the latest checkpoint
    trainer.train(resume_from_checkpoint=True)
trainer.save_model(f"./log/{exp_name}/results") # 모델 저장