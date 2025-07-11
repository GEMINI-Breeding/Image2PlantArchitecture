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
    # features is a list of samples returned from the dataset
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    
    # Padding processing: adjust labels' length to match the longest sequence
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
    Computes both macro-averaging (per-sample then averaged) and micro-averaging (corpus-level).
    """
    predictions, labels = eval_pred
    
    # Handle tuple predictions - take the second element (processed token IDs)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Take the processed token IDs, not the logits
    
    # Load metrics
    f1_metric = evaluate.load('f1')
    accuracy_metric = evaluate.load('accuracy')
    
    # If predictions are still logits (3D), convert to token IDs
    if len(predictions.shape) == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)
    
    # Initialize lists for macro averaging
    macro_f1_scores = []
    macro_accuracy_scores = []
    
    # Initialize lists for micro averaging (collect all tokens)
    all_pred_tokens = []
    all_label_tokens = []
    
    # Process each sample
    for pred, label in zip(predictions, labels):
        try:
            # Remove -100 tokens (ignored labels) and PAD tokens from both predictions and labels
            valid_mask = (label != -100) & (label != PAD_TOKEN)
            
            if valid_mask.sum() > 0:  # Check if there are any valid tokens
                pred_valid = pred[valid_mask]
                label_valid = label[valid_mask]
                
                # Also remove PAD tokens from predictions (in case they were predicted)
                final_mask = pred_valid != PAD_TOKEN
                if final_mask.sum() > 0:
                    pred_final = pred_valid[final_mask]
                    label_final = label_valid[final_mask]
                    
                    # For macro averaging: calculate per-sample metrics
                    if len(pred_final) > 0 and len(label_final) > 0:
                        # Ensure both sequences have the same length for fair comparison
                        min_len = min(len(pred_final), len(label_final))
                        pred_truncated = pred_final[:min_len]
                        label_truncated = label_final[:min_len]
                        
                        # Calculate per-sample metrics for macro averaging
                        f1_sample = f1_metric.compute(predictions=pred_truncated, references=label_truncated, average="weighted")
                        accuracy_sample = accuracy_metric.compute(predictions=pred_truncated, references=label_truncated)
                        
                        macro_f1_scores.append(f1_sample['f1'])
                        macro_accuracy_scores.append(accuracy_sample['accuracy'])
                        
                        # Collect tokens for micro averaging
                        all_pred_tokens.extend(pred_truncated.tolist())
                        all_label_tokens.extend(label_truncated.tolist())
                    else:
                        # Handle edge case where sequences are empty after filtering
                        macro_f1_scores.append(0.0)
                        macro_accuracy_scores.append(0.0)
                else:
                    # No valid predictions after filtering PAD tokens
                    macro_f1_scores.append(0.0)
                    macro_accuracy_scores.append(0.0)
            else:
                # No valid labels (all are -100 or PAD)
                macro_f1_scores.append(0.0)
                macro_accuracy_scores.append(0.0)
                
        except Exception as e:
            print(f"Error computing metrics for sample during training: {e}")
            macro_f1_scores.append(0.0)
            macro_accuracy_scores.append(0.0)
    
    # Calculate macro averages (average of per-sample scores)
    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0
    macro_accuracy = sum(macro_accuracy_scores) / len(macro_accuracy_scores) if macro_accuracy_scores else 0.0
    
    # Calculate micro averages (corpus-level metrics)
    if len(all_pred_tokens) > 0 and len(all_label_tokens) > 0:
        try:
            micro_f1 = f1_metric.compute(predictions=all_pred_tokens, references=all_label_tokens, average="weighted")
            micro_accuracy = accuracy_metric.compute(predictions=all_pred_tokens, references=all_label_tokens)
            
            micro_f1_score = micro_f1['f1']
            micro_accuracy_score = micro_accuracy['accuracy']
        except Exception as e:
            print(f"Error computing micro-level metrics: {e}")
            micro_f1_score = 0.0
            micro_accuracy_score = 0.0
    else:
        micro_f1_score = 0.0
        micro_accuracy_score = 0.0
    
    # Return both macro and micro metrics
    return {
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy,
        'micro_f1': micro_f1_score,
        'micro_accuracy': micro_accuracy_score,
        # Keep the original names for backward compatibility (using micro as default)
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
    if 1:
        parser.add_argument('--today_date_str', type=str, default=datetime.now().strftime('%Y%m%d'), help='Date string for experiment naming')
        parser.add_argument('--exp_name', type=str, help='Experiment name')
    else:
        parser.add_argument('--today_date_str', type=str, default="20250523_TrainOnFarm", help='Date string for experiment naming')
        parser.add_argument('--exp_name', type=str, default="dinov2-small_448_Sideview_gpt2-medium", help='Experiment name')

    parser.add_argument('--curriculum', default='False', help='Use curriculum learning')
    parser.add_argument('--epoch', type=int, default=1, help='Number of traninig epochs')
    parser.add_argument('--grad_acc', type=int, default=4, help='gradient_accumulation_steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of traninig batch_size')
    parser.add_argument('--color_jitter', type=str, default='False', help='Number of traninig epochs')
    parser.add_argument('--rnd_crop', type=str, default='False', help='Number of traninig epochs')
    parser.add_argument('--rnd_erase', type=str, default='False', help='Number of traninig epochs')
    parser.add_argument('--use_depth', type=str, default='False', help='Use Depth instead of RGB')


    args = parser.parse_args()

    # Convert string arguments to boolean
    args.side_view = args.side_view.lower() == 'true'
    args.preload = args.preload.lower() == 'true'
    args.curriculum = args.curriculum.lower() == 'true'
    args.color_jitter = args.color_jitter.lower() == 'true'
    args.rnd_crop = args.rnd_crop.lower() == 'true'
    args.rnd_erase = args.rnd_erase.lower() == 'true'
    args.use_depth = args.use_depth.lower() == 'true'


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

    encoder_checkpoint = args.encoder_checkpoint
    image_size = args.image_size
    encoder_config = AutoConfig.from_pretrained(encoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size


    if 1:
        model = PlantArchitectureModel.from_encoder_decoder_pretrained(
            encoder_checkpoint, decoder_checkpoint, 
            decoder_config=decoder_config, 
            encoder_config=encoder_config,
            decoder_ignore_mismatched_sizes=True,
            use_depth=args.use_depth,
            torch_dtype=torch.float16, 
        )
            # Freeze the encoder parameters
        model.encoder.eval()
        for param in model.encoder.parameters():
            param.requires_grad = False

        # 5. Update model configuration
        model.config.decoder_start_token_id = SOS_TOKEN  # Decoder start token
        model.config.bos_token_id = SOS_TOKEN  # Beginning of sequence token
        model.config.pad_token_id = PAD_TOKEN  # Padding token
        model.config.eos_token_id = EOS_TOKEN  # End of sequence token

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
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

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
    warmup_steps = int(train_size * 0.2 // batch_size // gradient_accumulation_steps * num_train_epochs)
    print(f"warmup_steps:{warmup_steps}")
    training_args = TrainingArguments(
        output_dir=f"{output_base_dir}/checkpoints",     # Model output directory
        num_train_epochs=num_train_epochs,               # Number of training epochs
        per_device_train_batch_size=batch_size,          # Training batch size
        per_device_eval_batch_size=batch_size*2,         # Evaluation batch size
        warmup_steps=warmup_steps,                       # Number of warmup steps for learning rate scheduler (or set the warmup_ratio)
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
        dataloader_num_workers=batch_size,
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
    if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0 and False:
        print(f"Model checkpoint already exists at {results_dir}. Skipping training.")
        # Load the trained model to calculate metrics
        model = VisionEncoderDecoderModel.from_pretrained(results_dir)
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

    print("Calculating metrics...")
    benchmark_folder = os.path.join(output_base_dir,"benchmark_results")
    benchmark_path = os.path.join(benchmark_folder, "benchmark.txt")
    # Use your custom data collator to handle variable-length sequences
    metrics = calc_metric(
        model=model,
        test_dataset=test_dataset,
        log_path=benchmark_path,
        batch_size=batch_size,
        num_workers=4,
        debug=False,
        benchmark_folder=benchmark_folder
    )