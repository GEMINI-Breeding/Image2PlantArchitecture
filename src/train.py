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
import evaluate  # Add this import
from sklearn.metrics import accuracy_score, f1_score


# Add . as a directory to import from
import sys
import re
from calc_metric import calc_metric

# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, META_TOKEN
from plant_dataset import PlantDataset
from utils import model_summary
from models.model import PlantArchitectureModel

def custom_data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    
    max_label_length = max(len(f["labels"]) for f in features)
    
    # Step 1: First pad tokens using PAD_TOKEN
    padded_labels = torch.stack([
        torch.cat([
            torch.tensor(f["labels"], dtype=torch.long), 
            torch.full((max_label_length - len(f["labels"]),), PAD_TOKEN, dtype=torch.long)
        ])
        for f in features
    ])
    
    # Step 2: Create pad mask to create the decoder attention mask
    # True for actual tokens, False for padded tokens
    decoder_attention_mask = (padded_labels != PAD_TOKEN).long()
    
    # Step 3: Replace padded token values to -100 to ignore from loss calculation
    labels = padded_labels.clone()
    labels[padded_labels == PAD_TOKEN] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,  
    }

class CurriculumSubset(Dataset):
    """Dataset subset for curriculum learning."""
    
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
        
    def __len__(self):
        return len(self.indices)
        
    def update_indices(self, indices):
        """Update accessible indices."""
        self.indices = indices

class CurriculumLearningCallback(TrainerCallback):
    """Callback for curriculum learning."""
    
    def __init__(self, dataset, curriculum_steps=5, 
                 lr_strategy="decrease",
                 difficulty_scores=None,
                 sorted_indices=None,
                 initial_lr=5e-5, final_lr=1e-5):
        """
        Args:
            dataset: Base dataset
            curriculum_steps: Number of curriculum stages
            lr_strategy: Learning rate adjustment strategy ("decrease", "increase", "bell")
            initial_lr: Initial learning rate
            final_lr: Final learning rate
        """
        super().__init__()
        self.dataset = dataset
        self.curriculum_steps = curriculum_steps
        self.lr_strategy = lr_strategy
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.difficulty_metric = 'complexity'
        self.original_length = len(dataset)
        
        # Calculate difficulty scores
        if difficulty_scores is None:
            self.difficulty_scores = self._calculate_difficulty_scores()
        else:
            self.difficulty_scores = difficulty_scores

        # Set sorted indices - modified logic
        if sorted_indices is None:
            self.sorted_indices = np.argsort(self.difficulty_scores)
        else:
            self.sorted_indices = sorted_indices
        
        # Current curriculum step
        self.current_step = 0
        
        # Update accessible indices
        self.update_curriculum(0)
        
    def _calculate_difficulty_scores(self):
        """Calculate difficulty scores for each sample."""
        difficulty_scores = []
        
        for idx in (range(len(self.dataset))):
            if self.difficulty_metric == 'complexity':
                # Calculate sample complexity (e.g., token length)
                sample = self.dataset[idx]
                difficulty = len(sample['labels'])
            elif self.difficulty_metric == 'stage':
                # Extract growth stage from filename
                filename = self.dataset.dataset.image_files[idx] if hasattr(self.dataset, 'dataset') else ""
                match = re.search(r"day_(\d+)", filename)
                difficulty = int(match.group(1)) if match else 0
            else:
                difficulty = 0
                
            difficulty_scores.append(difficulty)
            
        return np.array(difficulty_scores)
    
    def update_curriculum(self, step):
        """Update curriculum stage."""
        self.current_step = min(step, self.curriculum_steps - 1)
        
        # Calculate number of samples to include in current stage
        inclusion_ratio = (self.current_step + 1) / self.curriculum_steps
        n_samples = int(self.original_length * inclusion_ratio)
        
        # Determine indices of samples to include (starting with easier ones)
        self.dataset.accessible_indices = self.sorted_indices[:n_samples].tolist()
        
        # Modify dataset's __getitem__ and __len__ methods
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
        
        # Override methods
        self.dataset.__getitem__ = new_getitem.__get__(self.dataset)
        self.dataset.__len__ = new_len.__get__(self.dataset)
        
        print(f"Curriculum step {self.current_step}: Using {n_samples}/{self.original_length} samples")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Callback method called at the beginning of each epoch."""
        # Update curriculum stage based on epoch
        epochs_per_step = max(1, args.num_train_epochs // self.curriculum_steps)
        #new_step = min(state.epoch // epochs_per_step, self.curriculum_steps - 1)
        new_step = min(state.epoch, self.curriculum_steps - 1) # Use full data after self.curriculum_steps
        
        if new_step != self.current_step:
            self.update_curriculum(new_step)
            # Update learning rate
            self.update_learning_rate(kwargs.get("trainer"), new_step)
            
    def update_learning_rate(self, trainer, step):
        """Adjust learning rate according to curriculum stage"""
        if not trainer or not hasattr(trainer, "optimizer"):
            return
            
        # Calculate learning rate based on current stage
        if self.lr_strategy == "decrease":
            # Decrease learning rate as complexity increases
            progress = step / (self.curriculum_steps - 1)
            new_lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress
        elif self.lr_strategy == "increase":
            # Increase learning rate as complexity increases
            progress = step / (self.curriculum_steps - 1)
            new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        elif self.lr_strategy == "bell":
            # Bell curve: maximum in the middle, lower at both ends
            position = step / (self.curriculum_steps - 1)
            if position < 0.5:
                # Increase from beginning to middle
                new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * (position * 2)
            else:
                # Decrease from middle to end
                new_lr = self.final_lr - (self.final_lr - self.initial_lr) * ((position - 0.5) * 2)
        else:
            # Default: no change
            return
        
        # Check current scheduler state
        if hasattr(trainer, "lr_scheduler"):
            # Update scheduler's base_lrs attribute
            trainer.lr_scheduler.base_lrs = [new_lr] * len(trainer.optimizer.param_groups)

        # Update optimizer's learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        print(f"Curriculum Step {step}/{self.curriculum_steps-1}, lr: {new_lr:.6f}")
            
    def on_train_end(self, args, state, control, **kwargs):
        """Callback method called at the end of training."""
        # Proceed to final stage with all data
        self.update_curriculum(self.curriculum_steps - 1)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1":f1}



def compute_metrics_for_training(eval_pred):
    """
    Compute F1 and accuracy metrics during training evaluation.
    Fast version using sklearn directly.
    """
    predictions, labels = eval_pred
    
    # Handle tuple predictions - take the first element (processed token IDs)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # If predictions are still logits (3D), convert to token IDs
    if len(predictions.shape) == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)

    # Fast vectorized filtering
    # Create masks for valid tokens (not -100 and not PAD)
    valid_label_mask = (labels != -100) & (labels != PAD_TOKEN)
    valid_pred_mask = (predictions != -100) & (predictions != PAD_TOKEN)
    
    # Combine masks - only keep positions where both label and prediction are valid
    combined_mask = valid_label_mask & valid_pred_mask
    
    # Extract all valid tokens at once
    all_pred_tokens = predictions[combined_mask]
    all_label_tokens = labels[combined_mask]
    
    # Fast sklearn computation
    if len(all_pred_tokens) > 0 and len(all_label_tokens) > 0:
        try:
            from sklearn.metrics import f1_score, accuracy_score
            
            # Convert to numpy arrays if they're tensors
            if hasattr(all_pred_tokens, 'cpu'):
                all_pred_tokens = all_pred_tokens.cpu().numpy()
            if hasattr(all_label_tokens, 'cpu'):
                all_label_tokens = all_label_tokens.cpu().numpy()
            
            # Calculate metrics using sklearn (much faster)
            micro_f1_score = f1_score(all_label_tokens, all_pred_tokens, average='weighted', zero_division=0)
            micro_accuracy_score = accuracy_score(all_label_tokens, all_pred_tokens)
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            micro_f1_score = 0.0
            micro_accuracy_score = 0.0
    else:
        micro_f1_score = 0.0
        micro_accuracy_score = 0.0

    return {
        'f1': micro_f1_score,
        'accuracy': micro_accuracy_score,
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits to ensure consistent shapes for metric computation.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Convert logits to predictions (argmax)
    predictions = torch.argmax(logits, dim=-1)
    
    # Pad predictions to match labels length if needed
    if predictions.shape[1] < labels.shape[1]:
        padding_size = labels.shape[1] - predictions.shape[1]
        padding = torch.full((predictions.shape[0], padding_size), PAD_TOKEN, 
                           device=predictions.device, dtype=predictions.dtype)
        predictions = torch.cat([predictions, padding], dim=1)
    elif predictions.shape[1] > labels.shape[1]:
        # Truncate predictions to match labels
        predictions = predictions[:, :labels.shape[1]]
    
    return predictions, labels

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train the Image to Plant Architecture model')
    parser.add_argument('--image_size', type=int, default=448, help='Size of input images')
    parser.add_argument('--side_view', type=str, default='True', help='Use side view images')
    parser.add_argument('--preload', type=str, default='False', help='Preload dataset into memory')
    parser.add_argument('--encoder_checkpoint', type=str, default='facebook/dinov2-small', help='Encoder checkpoint to use')
    parser.add_argument('--decoder_checkpoint', type=str, default='gpt2-medium', help='Decoder checkpoint to use')
    parser.add_argument('--dataset_path', type=str, default='/home/lion397/datasets/GEMINI/plant_architecture/20250311_Sideview_40Days', help='Path to the dataset')
    parser.add_argument('--today_date_str', type=str, default="20250430_TrainValTestByPlotMoreData", help='Date string for experiment naming')
    parser.add_argument('--exp_name', type=str, default="dinov2-small_448_Sideview_gpt2-medium", help='Experiment name')
    parser.add_argument('--curriculum', default='False', help='Use curriculum learning')
    parser.add_argument('--epoch', type=int, default=1, help='Number of traninig epochs')
    parser.add_argument('--grad_acc', type=int, default=4, help='gradient_accumulation_steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of traninig batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--color_jitter', type=str, default='False', help='Number of traninig epochs')
    parser.add_argument('--rnd_crop', type=str, default='False', help='Number of traninig epochs')
    parser.add_argument('--rnd_erase', type=str, default='False', help='Number of traninig epochs')
    parser.add_argument('--use_depth', type=str, default='False', help='Use Depth instead of RGB')
    parser.add_argument('--push_to_hub', type=str, default='True', help='Push model to huggingface hub')
    parser.add_argument('--debug', type=str, default='False', help='Use debug mode')



    args = parser.parse_args()

    # Convert string arguments to boolean
    args.side_view = args.side_view.lower() == 'true'
    args.preload = args.preload.lower() == 'true'
    args.curriculum = args.curriculum.lower() == 'true'
    args.color_jitter = args.color_jitter.lower() == 'true'
    args.rnd_crop = args.rnd_crop.lower() == 'true'
    args.rnd_erase = args.rnd_erase.lower() == 'true'
    args.use_depth = args.use_depth.lower() == 'true'
    args.debug = args.debug.lower() == 'true'
    args.push_to_hub = args.push_to_hub.lower() == 'true'


    # Use provided experiment name if available, otherwise construct one
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = "debug"

    # Determine output directory
    if args.today_date_str:
        output_base_dir = f"./log/{args.today_date_str}/{exp_name}"
    else:
        output_base_dir = f"./log/{exp_name}"

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    results_dir = f"{output_base_dir}/results"


    # 1. Define decoder configuration
    decoder_checkpoint = args.decoder_checkpoint
    if "google-bert/bert" in decoder_checkpoint:
        decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 2500  # Set maximum sequence length
        decoder_config.vocab_size = VOCAB_SIZE  # Match with tokenizer's vocabulary size
        decoder_config.add_cross_attention=True
        decoder_config.is_decoder=True
    elif "gpt2" in decoder_checkpoint:
        decoder_config = GPT2Config.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 4096*2 # Set maximum sequence length
        decoder_config.vocab_size = VOCAB_SIZE  # Match with tokenizer's vocabulary size
        decoder_config.add_cross_attention=True
        decoder_config.is_decoder=True
    elif "google/bigbird-roberta" in decoder_checkpoint:
        decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 4096*2  # Set maximum sequence length
        decoder_config.vocab_size = VOCAB_SIZE  # Match with tokenizer's vocabulary size
        decoder_config.add_cross_attention=True
        decoder_config.is_decoder=True
        decoder_config.attention_type='original_full'

    decoder_config.decoder_start_token_id = SOS_TOKEN
    decoder_config.bos_token_id = SOS_TOKEN  # Beginning of sequence token
    decoder_config.pad_token_id = PAD_TOKEN  # Padding token
    decoder_config.eos_token_id = EOS_TOKEN  # End of sequence token

    encoder_checkpoint = args.encoder_checkpoint
    image_size = args.image_size
    encoder_config = AutoConfig.from_pretrained(encoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size


    if 1:
        # Initialize distributed training for multi-GPU
        n_gpu = torch.cuda.device_count()
        print(f"Available GPUs: {n_gpu}")
        
        # Check if we're in a distributed environment
        if "RANK" in os.environ:
            # Distributed training setup
            import torch.distributed as dist
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Initialize distributed backend
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
            print(f"Initialized distributed training: rank={rank}, world_size={world_size}, device={device}")
        else:
            # Single node multi-GPU or single GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rank = 0
            print(f"Using device: {device}")

        model = PlantArchitectureModel.from_encoder_decoder_pretrained(
            encoder_checkpoint, decoder_checkpoint, 
            decoder_config=decoder_config, 
            encoder_config=encoder_config,
            decoder_ignore_mismatched_sizes=True,
            use_depth=args.use_depth,
            torch_dtype=torch.float16, 
            tp_plan="auto"
        )
        
        # Move model to device
        model = model.to(device)
        
        # Freeze the encoder parameters
        model.encoder.eval()
        for param in model.encoder.parameters():
            param.requires_grad = False

        # 5. Update model configuration
        model.config.decoder_start_token_id = SOS_TOKEN  # Decoder start token
        model.config.bos_token_id = SOS_TOKEN  # Beginning of sequence token
        model.config.pad_token_id = PAD_TOKEN  # Padding token
        model.config.eos_token_id = EOS_TOKEN  # End of sequence token

        # Resize token embeddings to match custom vocab size
        model.decoder.resize_token_embeddings(VOCAB_SIZE)
        # Don't wrap with DataParallel - let accelerate handle multi-GPU
        # No DataParallel or DDP wrapping here
    else:
        config = PlantArchitectureConfig(
            encoder_checkpoint=encoder_checkpoint,
            decoder_checkpoint=decoder_checkpoint,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            use_depth=True
        )
        model = PlantArchitectureModel(config)



    # Set a random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Create Dataset instance
    growth_stages = None # ["01"]
    dataset_path = args.dataset_path
    print("Loading Dataset...")
    if "debug" in exp_name or args.debug:
        train_ratio = 0.0008
        val_ratio = 0.0001
        test_ratio = 0.0001
    else:
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.01

    # Separate by plot number
    # Get the num plots from the last xml file
    xml_files = os.listdir(os.path.join(dataset_path, "xml"))
    xml_files.sort()
    num_plots = int(xml_files[-1].split("_")[1]) + 1

    train_end = int(num_plots * train_ratio)
    val_end = train_end + int(num_plots * val_ratio)
    test_end = min(num_plots, val_end + int(num_plots * test_ratio)) # Ensure total sums up to num_plots

    train_plots = [f"{plot:04d}" for plot in range(train_end)]
    val_plots = [f"{plot:04d}" for plot in range(train_end, val_end)]
    test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]

    train_dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                process_leaf=True, image_size=image_size,
                side_view=args.side_view,
                plot=train_plots,
                mode='train',
                preload=args.preload, add_sos_token=False,
                image_processor=image_processor,
                color_jitter = args.color_jitter,
                random_crop = args.rnd_crop,
                random_erase=args.rnd_erase)
    train_size = len(train_dataset)

    val_dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                process_leaf=True, image_size=image_size,
                side_view=args.side_view,
                plot=val_plots,
                image_processor=image_processor,
                mode='val',
                preload=args.preload, add_sos_token=False)
    val_size = len(val_dataset)

    test_dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                process_leaf=True, image_size=image_size,
                side_view=args.side_view,
                plot=test_plots,
                image_processor=image_processor,
                mode='test',
                preload=args.preload, add_sos_token=False)
    test_size = len(test_dataset)
        
    callbacks = []
    if args.curriculum:
        # Create curriculum dataset
        curriculum_steps = 10
        # Configure curriculum callback
        curriculum_callback = CurriculumLearningCallback(
            train_dataset, 
            curriculum_steps=curriculum_steps,
            lr_strategy="bell",  # Decrease learning rate as complexity increases
            initial_lr=5e-5,     # Initial high learning rate
            final_lr=5e-6        # Final low learning rate
        )
        callbacks.append(curriculum_callback)    

    # Set training arguments
    today_date_str = args.today_date_str
    encoder_name = args.encoder_checkpoint.split('/')[-1]
    decoder_name = args.decoder_checkpoint.split('/')[-1]
    side_view_str = "Sideview" if args.side_view else "TopView"

    batch_size = args.batch_size
    num_train_epochs = args.epoch
    gradient_accumulation_steps = 4
    eval_save_steps = 0.1 # or 1000
    #warmup_steps = int(train_size * 0.2 // batch_size // gradient_accumulation_steps * num_train_epochs)
    warmup_ratio = 0.2
    #print(f"warmup_steps:{warmup_steps}")
    training_args = TrainingArguments(
        output_dir=f"{output_base_dir}/checkpoints",     # Model output directory
        num_train_epochs=num_train_epochs,               # Number of training epochs
        per_device_train_batch_size=batch_size,          # Training batch size
        per_device_eval_batch_size=batch_size*2,         # Evaluation batch size
        warmup_ratio=warmup_ratio,                       # Number of warmup steps for learning rate scheduler (or set the warmup_ratio)
        weight_decay=0.01,                               # Weight decay
        logging_dir=f"{output_base_dir}/logs",           # Log directory
        logging_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",                           # Save at each epoch
        save_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model='f1',                      # Change from 'loss' to 'f1'
        greater_is_better=True,                          # Add this since F1 higher is better
        save_total_limit=5,
        learning_rate=1e-4,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers//n_gpu,
        fp16=True,
    )

    # Create Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics_for_training,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # Add this
        callbacks=callbacks  # Add curriculum callback
    )

    model_summary(model=model, max_depth=1)

    # Check if model is already trained
    if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
        print(f"Model checkpoint already exists at {results_dir}. Skipping training.")
        # Load the trained model to calculate metrics
        model = PlantArchitectureModel.from_pretrained(results_dir)
    else:
        # print("Test model saving...")
        # trainer.save_model(results_dir) # Save model
        print("Model training...")
        # Check for existing checkpoints and resume training if they exist
        checkpoints_dir = os.path.join(output_base_dir, "checkpoints")
        if os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0:
            print(f"Model checkpoint already exists. Resuming training")
            trainer.train(resume_from_checkpoint=True)
        else:
            print("Training model from a scratch")
            trainer.train()
        trainer.save_model(results_dir) # Save model

    if args.push_to_hub:
        import subprocess
        # Get Hugging Face username
        try:
            whoami_output = subprocess.check_output(["hf","auth","whoami"], text=True)
            username = whoami_output.strip().split("\n")[-1]
            if username.startswith("You are logged in as"):
                username = username.split("as")[-1].strip().split()[0]
        except Exception as e:
            print(f"Could not determine Hugging Face username: {e}")
            username = "your-username"
        # Use experiment name as model name
        model_name = exp_name if exp_name else "your-model-name"
        repo_id = f"{username}/{model_name}"
        print(f"Pushing model to Hugging Face Hub as {repo_id}")
        model.push_to_hub(repo_id)

    benchmark_folder = os.path.join(output_base_dir,"benchmark_results")
    benchmark_path = os.path.join(benchmark_folder, "benchmark.txt")
    if os.path.exists(benchmark_path) :
        print("Benchmark already exists")
    else:
        print("Calculating metrics...")
        model.eval()
        # Pass the model directly to calc_metric - accelerate will handle multi-GPU
        metrics = calc_metric(
            model=model,  # Pass model directly, no unwrapping needed
            test_dataset=test_dataset,
            log_path=benchmark_path,
            batch_size=batch_size,
            num_workers=args.num_workers//n_gpu,
            debug=args.debug,
            benchmark_folder=benchmark_folder
        )