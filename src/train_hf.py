import torch
from torch.utils.data import random_split, Dataset
from transformers import Trainer, TrainingArguments, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModel
from transformers import VisionEncoderDecoderModel, BertConfig, BertModel, ViTModel, AutoConfig, GPT2Config
import os
import argparse
from datetime import datetime
from transformers import TrainerCallback, TrainingArguments, Trainer
import numpy as np
from tqdm import tqdm

# Add . as a directory to import from
import sys
import re

# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, META_TOKEN
from plant_dataset import PlantDataset
from utils import model_summary

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

class CurriculumSubset(Dataset):
    """커리큘럼 러닝용 데이터셋 서브셋."""
    
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
        
    def __len__(self):
        return len(self.indices)
        
    def update_indices(self, indices):
        """접근 가능한 인덱스 업데이트."""
        self.indices = indices

class CurriculumLearningCallback(TrainerCallback):
    """커리큘럼 러닝을 위한 콜백."""
    
    def __init__(self, dataset, curriculum_steps=5, 
                 lr_strategy="decrease",
                 difficulty_scores=None,
                 sorted_indices=None,
                 initial_lr=5e-5, final_lr=1e-5):
        """
        Args:
            dataset: 기본 데이터셋
            curriculum_steps: 커리큘럼 단계 수
            lr_strategy: 학습률 조절 전략 ("decrease", "increase", "bell")
            initial_lr: 초기 학습률
            final_lr: 최종 학습률
        """
        super().__init__()
        self.dataset = dataset
        self.curriculum_steps = curriculum_steps
        self.lr_strategy = lr_strategy
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.difficulty_metric = 'complexity'
        self.original_length = len(dataset)
        
        # 난이도 점수 계산
        if difficulty_scores is None:
            self.difficulty_scores = self._calculate_difficulty_scores()
        else:
            self.difficulty_scores = difficulty_scores

        # 정렬된 인덱스 설정 - 수정된 로직
        if sorted_indices is None:
            self.sorted_indices = np.argsort(self.difficulty_scores)
        else:
            self.sorted_indices = sorted_indices
        
        # 현재 커리큘럼 단계
        self.current_step = 0
        
        # 접근 가능한 인덱스 업데이트
        self.update_curriculum(0)
        
    def _calculate_difficulty_scores(self):
        """각 샘플의 난이도 점수 계산."""
        difficulty_scores = []
        
        for idx in (range(len(self.dataset))):
            if self.difficulty_metric == 'complexity':
                # 샘플의 복잡도 계산 (예: 토큰 길이)
                sample = self.dataset[idx]
                difficulty = len(sample['labels'])
            elif self.difficulty_metric == 'stage':
                # 파일 이름에서 성장 단계 추출
                filename = self.dataset.dataset.image_files[idx] if hasattr(self.dataset, 'dataset') else ""
                match = re.search(r"day_(\d+)", filename)
                difficulty = int(match.group(1)) if match else 0
            else:
                difficulty = 0
                
            difficulty_scores.append(difficulty)
            
        return np.array(difficulty_scores)
    
    def update_curriculum(self, step):
        """커리큘럼 단계 업데이트."""
        self.current_step = min(step, self.curriculum_steps - 1)
        
        # 현재 단계에서 포함할 샘플 수 계산
        inclusion_ratio = (self.current_step + 1) / self.curriculum_steps
        n_samples = int(self.original_length * inclusion_ratio)
        
        # 포함할 샘플의 인덱스 결정 (쉬운 것부터)
        self.dataset.accessible_indices = self.sorted_indices[:n_samples].tolist()
        
        # 데이터셋의 __getitem__ 및 __len__ 메서드 수정
        self.dataset._original_getitem = self.dataset.__getitem__
        self.dataset._original_len = self.dataset.__len__
        
        def new_getitem(self, idx):
            if hasattr(self, 'accessible_indices'):
                original_idx = self.accessible_indices[idx]
                return self._original_getitem(original_idx)
            return self._original_getitem(idx)
        
        def new_len(self):
            if hasattr(self, 'accessible_indices'):
                return len(self.accessible_indices)
            return self._original_len()
        
        # 메서드 오버라이드
        self.dataset.__getitem__ = new_getitem.__get__(self.dataset)
        self.dataset.__len__ = new_len.__get__(self.dataset)
        
        print(f"Curriculum step {self.current_step}: Using {n_samples}/{self.original_length} samples")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """에폭 시작 시 호출되는 콜백 메서드."""
        # 에폭 기준으로 커리큘럼 단계 업데이트
        epochs_per_step = max(1, args.num_train_epochs // self.curriculum_steps)
        #new_step = min(state.epoch // epochs_per_step, self.curriculum_steps - 1)
        new_step = min(state.epoch, self.curriculum_steps - 1) # Use full data after self.curriculum_steps
        
        if new_step != self.current_step:
            self.update_curriculum(new_step)
            # 학습률 업데이트
            self.update_learning_rate(kwargs.get("trainer"), new_step)
            
    def update_learning_rate(self, trainer, step):
        """커리큘럼 단계에 따라 학습률 조정"""
        if not trainer or not hasattr(trainer, "optimizer"):
            return
            
        # 현재 단계에 따른 학습률 계산
        if self.lr_strategy == "decrease":
            # 복잡도 증가에 따라 학습률 감소
            progress = step / (self.curriculum_steps - 1)
            new_lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress
        elif self.lr_strategy == "increase":
            # 복잡도 증가에 따라 학습률 증가
            progress = step / (self.curriculum_steps - 1)
            new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        elif self.lr_strategy == "bell":
            # 종 모양 곡선: 중간에 최대값, 양쪽에서 낮음
            position = step / (self.curriculum_steps - 1)
            if position < 0.5:
                # 처음부터 중간까지 증가
                new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * (position * 2)
            else:
                # 중간부터 끝까지 감소
                new_lr = self.final_lr - (self.final_lr - self.initial_lr) * ((position - 0.5) * 2)
        else:
            # 기본값: 변화 없음
            return
        
        # 현재 스케줄러 상태 확인
        if hasattr(trainer, "lr_scheduler"):
            # 스케줄러의 base_lrs 속성 업데이트
            trainer.lr_scheduler.base_lrs = [new_lr] * len(trainer.optimizer.param_groups)

        # 옵티마이저의 학습률 업데이트
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        print(f"Curriculum Step {step}/{self.curriculum_steps-1}, lr: {new_lr:.6f}")
            
    def on_train_end(self, args, state, control, **kwargs):
        """훈련 종료 시 콜백 메서드."""
        # 모든 데이터로 마지막 단계 진행
        self.update_curriculum(self.curriculum_steps - 1)

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train the Image to Plant Architecture model')
    parser.add_argument('--image_size', type=int, default=448, help='Size of input images')
    parser.add_argument('--side_view', type=str, default='True', help='Use side view images')
    parser.add_argument('--preload', type=str, default='False', help='Preload dataset into memory')
    parser.add_argument('--encoder_checkpoint', type=str, default='facebook/dinov2-small', help='Encoder checkpoint to use')
    parser.add_argument('--decoder_checkpoint', type=str, default='gpt2', help='Decoder checkpoint to use')
    parser.add_argument('--dataset_path', type=str, default='data/2000_Plots_20241210_BetterQuantized', help='Path to the dataset')
    parser.add_argument('--today_date_str', type=str, default=datetime.now().strftime('%Y%m%d'), help='Date string for experiment naming')
    parser.add_argument('--log_dir', type=str, help='Main log directory')
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    args = parser.parse_args()

    # Convert string arguments to boolean
    args.side_view = args.side_view.lower() == 'true'
    args.preload = args.preload.lower() == 'true'

    # Use provided experiment name if available, otherwise construct one
    if args.exp_name:
        exp_name = args.exp_name
    else:
        #exp_name = f"{today_date_str}_{encoder_name}_{args.image_size}_{side_view_str}_{decoder_name}"
        exp_name = "debug"

    # Determine output directory
    if args.log_dir:
        output_base_dir = f"{args.log_dir}/{exp_name}"
    else:
        output_base_dir = f"./log/{exp_name}"

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    results_dir = f"{output_base_dir}/results"

    # Check for benchmark.txt in the specified log directory
    benchmark_file_path = os.path.join(output_base_dir, 'benchmark.txt')
    if os.path.exists(benchmark_file_path) and os.path.getsize(benchmark_file_path) > 0:
        print(f"Benchmark file exists at {benchmark_file_path}; exiting.")
        sys.exit(0)
        

    # 1. 디코더 설정 정의
    decoder_checkpoint = args.decoder_checkpoint
    if "google-bert/bert" in decoder_checkpoint:
        decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 2500  # 최대 시퀀스 길이 설정
        decoder_config.vocab_size = VOCAB_SIZE  # 토크나이저의 vocab 크기와 일치시킴
        decoder_config.add_cross_attention=True
        decoder_config.is_decoder=True
    elif "gpt2" in decoder_checkpoint:
        decoder_config = GPT2Config.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 2500  # 최대 시퀀스 길이 설정
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
        torch_dtype=torch.float16, 
    )

    # Freeze the encoder parameters
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 5. 모델 설정 업데이트
    model.config.decoder_start_token_id = SOS_TOKEN  # 디코더 시작 토큰
    model.config.bos_token_id = SOS_TOKEN  # Beginning of sequence 토큰
    model.config.pad_token_id = PAD_TOKEN  # 패딩 토큰
    model.config.eos_token_id = EOS_TOKEN  # End of sequence 토큰


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
    if 1:
        train_size = int(0.8 * len(plant_architecture_dataset))  # 80% for training
        val_size = int(0.1 * len(plant_architecture_dataset))    # 10% for validation
        test_size = len(plant_architecture_dataset) - train_size - val_size  # Remaining 10% for testing
    else:
        # Debugging
        train_size = int(0.01 * len(plant_architecture_dataset))  # 80% for training
        val_size = int(0.01 * len(plant_architecture_dataset))    # 10% for validation
        test_size = len(plant_architecture_dataset) - train_size - val_size  # Remaining 10% for testing
    print(f"train_size:{train_size}, val_size:{val_size}")
    
    # Use random_split with the seed set above
    train_dataset, val_dataset, test_dataset = random_split(plant_architecture_dataset, [train_size, val_size, test_size])

    # 커리큘럼 데이터셋 생성
    curriculum_steps = 10
    # 커리큘럼 콜백 설정
    curriculum_callback = CurriculumLearningCallback(
        train_dataset, 
        curriculum_steps=curriculum_steps,
        lr_strategy="bell",  # 복잡도 증가에 따라 학습률 감소
        initial_lr=5e-5,         # 초기 높은 학습률
        final_lr=5e-6            # 최종 낮은 학습률
    )

    # 훈련 인자 설정
    today_date_str = args.today_date_str
    encoder_name = args.encoder_checkpoint.split('/')[-1]
    decoder_name = args.decoder_checkpoint.split('/')[-1]
    side_view_str = "Sideview" if args.side_view else "TopView"

    batch_size = 4
    num_train_epochs = 10
    gradient_accumulation_steps = 4
    warmup_steps = int(train_size * 0.2 // batch_size // gradient_accumulation_steps * num_train_epochs)
    print(f"warmup_steps:{warmup_steps}")
    training_args = TrainingArguments(
        output_dir=f"{output_base_dir}/checkpoints",     # 모델 출력 디렉토리
        num_train_epochs=num_train_epochs,               # 훈련 에포크 수
        per_device_train_batch_size=batch_size,          # 훈련 배치 사이즈
        per_device_eval_batch_size=1,                    # 평가 배치 사이즈
        warmup_steps=warmup_steps,                       # 학습률 스케줄러를 위한 웜업 스텝 수 (or set the warmup_ratio)
        weight_decay=0.01,                               # 가중치 감쇠
        logging_dir=f"{output_base_dir}/logs",           # 로그 디렉토리
        logging_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",                          # 에폭마다 저장
        fp16=True,
    )

    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        callbacks=[curriculum_callback]  # 커리큘럼 콜백 추가
    )

    model_summary(model=model, max_depth=1)

    # Check if model is already trained
    if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
        print(f"Model checkpoint already exists at {results_dir}. Skipping training.")
        # Load the trained model to calculate metrics
        model = VisionEncoderDecoderModel.from_pretrained(results_dir)
    else:
        print("Model training...")
        # Check for existing checkpoints and resume training if they exist
        checkpoints_dir = os.path.join(output_base_dir, "checkpoints")
        if os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0:
            print(f"Model checkpoint already exists. Resuming training")
            trainer.train(resume_from_checkpoint=True)
        else:
            print("Training model from a scratch")
            trainer.train()
        trainer.save_model(results_dir) # 모델 저장

    print("Calculating metrics...")
    from calc_metric import calc_metric
    calc_metric(model, dataset_path, side_view=args.side_view)