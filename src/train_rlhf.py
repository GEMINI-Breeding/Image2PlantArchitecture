import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import get_linear_schedule_with_warmup, CLIPModel, CLIPProcessor
from transformers import VisionEncoderDecoderModel, AutoModel, AutoImageProcessor
import os
import json
from tqdm import tqdm
import time

import argparse
from datetime import datetime
import subprocess
import cv2

# Import your modules
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE
from plant_dataset import PlantDataset, load_sideview_images

from string_to_xml_to_vec import vec2xml, recursive_to_linked, pretty_print_xml
from plant_tokenizer import token2vec
from image_process import process_leaf_image
import shutil

class ImageSimilarityRewardModel:
    """Computes rewards based on similarity between original and rendered images"""
    def __init__(
        self, 
        encoder,
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_processor=None
    ):
        # Load a pretrained vision model for computing image similarity
        self.device = device
        self.feature_extractor = encoder
        self.image_processor = image_processor

        # Set model to evaluation mode
        self.feature_extractor.eval()
    
    # Renderer for converting generated sequences back to images
    # This could be a separate model or a rule-based system
    def renderer(self, generated_sequences,
                output_path='temp', filename='rendered', 
                program_path="src/GenerateDataset/build",
                side_view=False, image_size=224,
                debug=False):
        """
        Render a batch of sequences to images - optimized for speed
        """
        # Initialize cache if not already present
        if not hasattr(self, '_render_cache'):
            self._render_cache = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        batch_size = generated_sequences.size(0) if torch.is_tensor(generated_sequences) else len(generated_sequences)
        rendered_images = [None] * batch_size
        render_jobs = []
        
        # Step 1: Check cache and identify which sequences need rendering
        for batch_idx in range(batch_size):
            sequence = generated_sequences[batch_idx] if torch.is_tensor(generated_sequences) else generated_sequences[batch_idx]
            # Create a hashable key for the cache (tuple of tokens)
            seq_key = tuple(sequence.cpu().numpy().tolist())
            
            # Use cached render if available
            if seq_key in self._render_cache:
                rendered_images[batch_idx] = self._render_cache[seq_key]
            else:
                # Queue for rendering
                batch_output_path = f"{output_path}/batch_{batch_idx}"
                batch_output_path = os.path.abspath(batch_output_path)
                batch_filename = f"{filename}_{batch_idx}.xml"
                render_jobs.append((batch_idx, sequence, batch_output_path, batch_filename, seq_key))
        
        # Step 2: Process rendering jobs in parallel
        if render_jobs:
            import concurrent.futures
            
            # Create output directories for all jobs upfront to avoid race conditions
            for _, _, batch_output_path, _, _ in render_jobs:
                os.makedirs(batch_output_path, exist_ok=True)
            
            # Define the worker function for a single rendering job
            def process_render_job(job):
                batch_idx, sequence, batch_output_path, batch_filename, seq_key = job
                
                # Skip if already rendered in another thread
                if seq_key in self._render_cache:
                    return batch_idx, self._render_cache[seq_key]
                
                # Save to XML
                try:
                    # Skip tokens 0-4 which might be special tokens
                    plant_vec = token2vec(sequence[5:])
                    plant_xml = vec2xml(plant_vec)
                    plant_xml_file_name = f"{batch_output_path}/{batch_filename}"
                    plant_xml = recursive_to_linked(plant_xml)
                    plant_xml_str = pretty_print_xml(plant_xml)
                    with open(plant_xml_file_name, "w") as f:
                        f.write(plant_xml_str)
                    
                    # Render jpeg - use environment variable once for all renders
                    image_name = batch_filename.split(".")[0]
                    os.environ["DISPLAY"] = ":11.0"
                    command = f"cd {program_path} && ./main -h 1.0 -o {batch_output_path} -name {image_name} -tile none -f {os.path.join(batch_output_path, batch_filename)}"
                    if side_view:
                        command += " -r"
                    
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    # if debug:
                    #     print(result.stdout)
                    #     print(result.stderr)
                    # Process image
                    if side_view:
                        img, _ = load_sideview_images(batch_output_path, batch_filename.replace("xml","jpeg"), image_size, True)
                    else:
                        img_path = plant_xml_file_name.replace("xml","jpeg")
                        img = cv2.imread(img_path)
                        if img is None:
                            if debug:
                                print(f"Warning: Failed to load image at {img_path}")
                            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Fast mode: skip complex image processing if not needed
                            leaf_area, plant_width, plant_height, leaf_img, _ = process_leaf_image(
                                img, sqaure_crop=True
                            )
                            img = cv2.resize(leaf_img, (image_size, image_size))
                    
                    # Convert to tensor
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    
                    # Cache the result
                    self._render_cache[seq_key] = img
                    return batch_idx, img, None
                    
                except Exception as e:
                    if debug:
                        print(f"Error processing image for batch {batch_idx}: {e}")
                    blank_img = torch.zeros((3, image_size, image_size), dtype=torch.float32)
                    return batch_idx, blank_img, e
            
            # Execute jobs with thread pool (I/O bound operations benefit from threads)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
                # Submit all jobs
                future_to_job = {executor.submit(process_render_job, job): job for job in render_jobs}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_job):
                    batch_idx, img, e = future.result()
                    rendered_images[batch_idx] = img
        
        # Ensure all images are processed (use those from cache)
        for batch_idx in range(batch_size):
            if rendered_images[batch_idx] is None:
                # This should not happen if the code is correct, but as a fallback
                blank_img = torch.zeros((3, image_size, image_size), dtype=torch.float32)
                rendered_images[batch_idx] = blank_img
        
        # Periodic cache cleanup to prevent memory issues
        if len(self._render_cache) > 1000:  # Adjust threshold as needed
            # Keep the 500 most recent entries
            cache_keys = list(self._render_cache.keys())
            for key in cache_keys[:-500]:
                del self._render_cache[key]
        
        # Stack all images into a single tensor
        return torch.stack(rendered_images)
    
    def compute_reward(self, original_images, generated_sequences, side_view=False, image_size=224, debug=False):
        """
        Compute similarity reward between original image and rendered generated sequence
        
        Args:
            original_images: Tensor of shape [batch_size, 3, H, W] - original plant images
            generated_sequences: Tensor of shape [batch_size, seq_len] - generated token sequences
            side_view: Whether to render side view
            image_size: Size of output images
            
        Returns:
            Tensor of shape [batch_size] with similarity scores (between 0 and 1)
        """
        # Ensure output path exists
        temp_dir = "temp_render"
        # Delete the previous folder
        # shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Render the generated sequences to images
        rendered_images = self.renderer(
            generated_sequences, 
            output_path=temp_dir,
            filename="rendered",
            side_view=side_view,
            image_size=image_size,
            debug=debug
        )
        
        # Process images for CLIP
        with torch.no_grad():
            # Convert format if needed - CLIP expects PIL images or tensors in [B, C, H, W] format
            if isinstance(original_images, torch.Tensor) and original_images.dim() == 4:
                # Already in the right format
                pass
            elif isinstance(original_images, list):
                original_images = torch.stack(original_images)
            
            # Normalize each image in the list using min-max normalization
            normalized_images = []
            for img in original_images:
                min_val = img.min()
                max_val = img.max()
                # Apply min-max normalization
                if max_val > min_val:  # Avoid division by zero
                    normalized_img = (img - min_val) / (max_val - min_val)
                else:
                    normalized_img = torch.zeros_like(img)
                normalized_images.append(normalized_img)
            
            original_images = torch.stack(normalized_images)
            if True:
                # Save original images too
                for idx, img in enumerate(original_images):
                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    debug_path = os.path.join(temp_dir, f"batch_{idx}/original_image_{idx}.jpeg")
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR before saving
                    cv2.imwrite(debug_path, img_np)

                    debug_path = os.path.join(temp_dir, f"batch_{idx}/rendered_image_{idx}.jpeg")
                    img_np = (rendered_images[idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_np= cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR before saving
                    cv2.imwrite(debug_path, img_np)

            # Get image embeddings from CLIP
            original_inputs = self.image_processor(images=original_images, return_tensors="pt").to(self.device)
            rendered_inputs = self.image_processor(images=rendered_images, return_tensors="pt").to(self.device)

            # Use this to get all token features:
            original_outputs = self.feature_extractor(**original_inputs)
            rendered_outputs = self.feature_extractor(**rendered_inputs)

            # Get the full sequence of embeddings (all tokens, not just CLS)
            original_embeddings = original_outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
            rendered_embeddings = rendered_outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

            # Normalize each token embedding
            original_embeddings = F.normalize(original_embeddings, p=2, dim=2)  # Normalize each token embedding
            rendered_embeddings = F.normalize(rendered_embeddings, p=2, dim=2)

            # Compute attention/similarity matrix between all tokens from both images
            # Shape: [batch_size, orig_seq_len, rend_seq_len]
            token_similarities = torch.bmm(original_embeddings, rendered_embeddings.transpose(1, 2))

            # Calculate the average diagonal similarity (for token-wise direct correspondence)
            batch_size, orig_seq_len, rend_seq_len = token_similarities.shape
            min_seq_len = min(orig_seq_len, rend_seq_len)

            # Extract diagonal elements up to the minimum sequence length
            diagonal_similarities = torch.zeros(batch_size, min_seq_len, device=self.device)
            for b in range(batch_size):
                for i in range(min_seq_len):
                    diagonal_similarities[b, i] = token_similarities[b, i, i]

            # Calculate mean diagonal similarity for each item in batch
            mean_diagonal_similarities = diagonal_similarities.mean(dim=1)  # Shape: [batch_size]

            # You can use this as another reward option:
            diagonal_rewards = (mean_diagonal_similarities + 1) / 2

            # Compare with the original reward calculation method:
            # Option 1: Maximum similarity for each token in original image
            max_similarities_per_token = token_similarities.max(dim=2)[0]  # Shape: [batch_size, orig_seq_len]
            mean_max_similarity = max_similarities_per_token.mean(dim=1)  # Shape: [batch_size]

            # Scale to 0-1 range
            rewards = (mean_max_similarity + 1) / 2

            # You could also combine both reward signals
            #combined_rewards = 0.5 * rewards + 0.5 * diagonal_rewards
            combined_rewards = diagonal_rewards
            
        return combined_rewards


class PlantRLTrainer:
    """Trainer for RL fine-tuning using image similarity rewards"""
    def __init__(
        self,
        model,
        ref_model,
        reward_model,
        optimizer,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        kl_coef=0.1,
        clip_range=0.2,
        value_clip_range=0.2,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        log_dir=None,
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # PPO hyperparameters
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Logging
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, "rl_training_log.jsonl")
        
        # Initialize value head for reward prediction
        hidden_size = self.model.decoder.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)
        self.value_optimizer = Adam(self.value_head.parameters(), lr=1e-4)
        
    def generate_sequences(self, batch, max_length=2500, num_return_sequences=1):
        """Generate sequences from the model for given images"""
        self.model.eval()
        pixel_values = batch["pixel_values"].to(self.device)
        plant_info = batch["plant_info"].to(self.device)
        labels = batch["labels"].to(self.device)
        with torch.no_grad():
            if 0:
                # encoder_outputs = self.model.encoder(pixel_values)
                # outputs = self.model.generate(
                #     encoder_outputs=encoder_outputs,
                #     max_length=max_length,
                #     do_sample=True,
                #     temperature=1.0,
                #     num_return_sequences=num_return_sequences,
                #     bos_token_id=self.model.config.bos_token_id,
                #     eos_token_id=self.model.config.eos_token_id,
                #     pad_token_id=self.model.config.pad_token_id
                # )

                # pixel_values = test_dataset[i]["pixel_values"].unsqueeze(0).to(model.device)
                # plant_info = test_dataset[i]["plant_info"]
                # plant_info = torch.tensor(plant_info, dtype=torch.long).to(model.device)  # Ensure plant_info is a tens
                outputs = self.model.generate(pixel_values,
                                        decoder_start_token_id=SOS_TOKEN,
                                        decoder_input_ids=plant_info,
                                        eos_token_id=EOS_TOKEN,
                                        max_length=max_length,
                                        use_cache=True,
                                        )
                
            else:
                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits

                # Get predictions
                outputs = torch.argmax(logits, dim=-1)

                sos_tokens = torch.full((outputs.size(0), 1), SOS_TOKEN, 
                                   dtype=outputs.dtype, 
                                   device=outputs.device)
                outputs = torch.cat([sos_tokens, outputs], dim=1)
                
        return outputs
    
    def train_value_head(self, dataloader, epochs=3):
        """Train the value head to predict similarity rewards"""
        value_losses = []
        
        # Set models to appropriate modes
        self.model.eval()
        self.value_head.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(dataloader, desc=f"Training value head epoch {epoch+1}/{epochs}"):
                pixel_values = batch["pixel_values"].to(self.device)
                plant_info = batch["plant_info"].to(self.device)
                labels = batch["labels"].to(self.device)
                # Generate sequences
                generated_sequences = self.generate_sequences(batch)
                
                # Compute rewards using the reward model
                rewards = self.reward_model.compute_reward(pixel_values, generated_sequences, 
                                                           image_size=pixel_values.size(-1),
                                                           debug=False)
                
                # Get encoder-decoder outputs for value prediction
                with torch.no_grad():
                    encoder_outputs = self.model.encoder(pixel_values)
                    encoder_hidden_states = encoder_outputs.last_hidden_state
                    # optionally project encoder_hidden_states
                    if (
                        self.model.encoder.config.hidden_size != self.model.decoder.config.hidden_size
                        and self.model.decoder.config.cross_attention_hidden_size is None
                    ):
                        encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)
                    decoder_outputs = self.model.decoder(
                        input_ids=generated_sequences,
                        encoder_hidden_states=encoder_hidden_states,
                        output_hidden_states=True
                    )
                    last_hidden_states = decoder_outputs.hidden_states[-1][:, -1, :]
                
                # Predict values
                predicted_values = self.value_head(last_hidden_states).squeeze(-1)
                
                # Value loss
                value_loss = F.mse_loss(predicted_values, rewards)
                
                # Backward pass
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                epoch_loss += value_loss.item()
                
            value_losses.append(epoch_loss / len(dataloader))
            print(f"Value head epoch {epoch+1}, loss: {value_losses[-1]:.4f}")
            
        return value_losses
    
    def rl_step(self, dataloader, epochs=4, train_value_head_first=True):
        """Run PPO-like training with image similarity rewards"""
        if train_value_head_first:
            self.train_value_head(dataloader)
        
        self.model.train()
        self.ref_model.eval()
        self.value_head.train()
        
        logs = []
        
        for epoch in range(epochs):
            policy_losses = []
            value_losses = []
            kl_divergences = []
            entropies = []
            rewards_history = []
            
            for batch in tqdm(dataloader, desc=f"RL epoch {epoch+1}/{epochs}"):
                pixel_values = batch["pixel_values"].to(self.device)
                plant_info = batch["plant_info"].to(self.device)

                # Generate sequences from current policy
                with torch.no_grad():
                    generated_sequences = self.generate_sequences(batch) # generated_sequences starts from <SOS>

                # Compute rewards
                rewards = self.reward_model.compute_reward(pixel_values, generated_sequences)
                rewards_history.extend(rewards.cpu().numpy().tolist())

                # Get encoder output from current model 
                with torch.no_grad():
                    encoder_outputs = self.model.encoder(pixel_values)
                    encoder_hidden_states = encoder_outputs.last_hidden_state
                    if (self.model.encoder.config.hidden_size != self.model.decoder.config.hidden_size
                        and self.model.decoder.config.cross_attention_hidden_size is None):
                        encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

                # Get policy outputs from current model
                policy_outputs = self.model.decoder(
                    input_ids=generated_sequences,
                    encoder_hidden_states=encoder_hidden_states,
                    output_hidden_states=True
                )
                logits = policy_outputs.logits[:, :-1] # Generated from decode, following SOS
                
                # Reference 모델 출력 계산 - 여전히 no_grad 컨텍스트 유지
                with torch.no_grad():
                    ref_outputs = self.ref_model.decoder(
                        input_ids=generated_sequences,
                        encoder_hidden_states=encoder_hidden_states,
                        output_hidden_states=True
                    )
                    ref_logits = ref_outputs.logits[:, :-1]

                # Value Head 예측 - no_grad 컨텍스트 밖으로 이동
                last_hidden_states = policy_outputs.hidden_states[-1][:, -1, :]
                values = self.value_head(last_hidden_states).squeeze(-1)
                
                # Compute log probs
                log_probs = F.log_softmax(logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                # Extract only the log probs for chosen tokens
                token_indices = generated_sequences[:, 1:].unsqueeze(-1) # generated_sequences starts from <SOS>
                chosen_log_probs = torch.gather(
                    log_probs, 
                    2, 
                    token_indices
                ).squeeze(-1)
                
                ref_chosen_log_probs = torch.gather(
                    ref_log_probs, 
                    2, 
                    token_indices
                ).squeeze(-1)
                
                # Create masks for sequence padding
                mask = (generated_sequences[:, 1:] != self.model.config.pad_token_id).float()
                
                # Advantage 계산 - 여기서 detach를 통해 policy 업데이트에만 영향
                advantages = rewards.unsqueeze(-1).expand_as(chosen_log_probs) - values.detach().unsqueeze(-1).expand_as(chosen_log_probs)
                
                # Compute PPO policy loss
                ratio = torch.exp(chosen_log_probs - ref_chosen_log_probs.detach())
                pg_loss1 = -advantages * ratio * mask
                pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mask
                pg_loss = torch.max(pg_loss1, pg_loss2).sum() / mask.sum().clamp(min=1e-5)
                
                # Compute value loss
                value_loss = F.mse_loss(values, rewards)
                
                # KL divergence
                kl = (ref_log_probs.detach() - log_probs) * F.softmax(ref_logits.detach(), dim=-1)
                kl = (kl.sum(-1) * mask).sum() / mask.sum().clamp(min=1e-5) * self.kl_coef
                
                # Entropy bonus
                entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1)
                entropy = (entropy * mask).sum() / mask.sum().clamp(min=1e-5) * self.entropy_coef
                
                # 분리된 손실 계산
                policy_loss = pg_loss + kl - entropy  # Policy 관련 손실만
                value_loss = F.mse_loss(values, rewards)  # Value 관련 손실만

                # 분리된 최적화 단계
                # Policy 최적화
                self.optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)  # 연산 그래프 유지
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Value 최적화
                self.value_optimizer.zero_grad()
                value_loss.backward()  
                torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                # Log metrics
                policy_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                kl_divergences.append(kl.item())
                entropies.append(entropy.item())
            
            # End of epoch logging
            log_data = {
                "epoch": epoch + 1,
                "policy_loss": np.mean(policy_losses),
                "value_loss": np.mean(value_losses),
                "kl_divergence": np.mean(kl_divergences),
                "entropy": np.mean(entropies),
                "mean_reward": np.mean(rewards_history),
                "timestamp": time.time()
            }
            
            logs.append(log_data)
            
            if self.log_dir:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_data) + "\n")
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Policy Loss: {log_data['policy_loss']:.4f}")
            print(f"  Value Loss: {log_data['value_loss']:.4f}")
            print(f"  KL Divergence: {log_data['kl_divergence']:.4f}")
            print(f"  Entropy: {log_data['entropy']:.4f}")
            print(f"  Mean Reward: {log_data['mean_reward']:.4f}")
        
        return logs


def run_rl_pipeline(
    model, 
    dataloader, 
    learning_rate=1e-5,
    rl_epochs=4,
    log_dir="./log/sim_rl",
    image_processor=None,
):
    """Full RL pipeline using image similarity rewards"""
    
    # Create reference model (copy of original model)
    ref_model = VisionEncoderDecoderModel.from_pretrained(model.config._name_or_path)
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Initialize reward model
    # Note: In a real implementation, you would need a proper renderer
    # that converts plant architecture sequences back to images
    reward_model = ImageSimilarityRewardModel(
        encoder=ref_model.encoder,
        image_processor=image_processor
    )
    
    # Initialize RL trainer
    rl_trainer = PlantRLTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        optimizer=optimizer,
        log_dir=log_dir,
    )
    
    # Run RL training
    print("Starting RL training...")
    rl_trainer.rl_step(dataloader, epochs=rl_epochs)
    
    return model



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fine-tune with image similarity RL')
    parser.add_argument('--model_path', type=str, default='log/20250325/dinov2-small_448_TopView-bert-base-uncased/results', help='Path to pretrained model')
    parser.add_argument('--dataset_path', type=str, default='data/2000_Plots_20241210_BetterQuantized', help='Path to dataset')
    parser.add_argument('--plot', type=str, default=[f"{i:04d}" for i in range(100)], help='Plots')
    parser.add_argument('--image_size', type=int, default=448, help='Image size')
    parser.add_argument('--side_view', type=str, default='False', help='Use side view')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--rl_epochs', type=int, default=4, help='Number of RL training epochs')
    parser.add_argument('--log_dir', type=str, default='./log/sim_rl', help='Log directory')
    args = parser.parse_args()

    # Convert string arguments to boolean
    args.side_view = args.side_view.lower() == 'true'

    # Load the pretrained model
    print(f"Loading pretrained model from {args.model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    image_processor = None
    for processor_attr in ['image_processor', 'feature_extractor', 'processor']:
        if hasattr(model.encoder, processor_attr):
            image_processor = getattr(model.encoder, processor_attr)
            break

    if image_processor is None:
        from transformers import AutoImageProcessor
        # Try to load the image processor from the encoder's config name
        encoder_name = model.encoder.config._name_or_path
        image_processor = AutoImageProcessor.from_pretrained(encoder_name)

    # Update image processor settings
    if hasattr(image_processor, 'size'):
        if isinstance(image_processor.size, dict):
            if 'shortest_edge' in image_processor.size:
                image_processor.size['shortest_edge'] = args.image_size
                image_processor.crop_size['width'] = args.image_size
                image_processor.crop_size['height'] = args.image_size

            if 'width' in image_processor.size and 'height' in image_processor.size:
                image_processor.size['width'] = args.image_size
                image_processor.size['height'] = args.image_size
        else:
            image_processor.size = args.image_size

    # Prepare dataset and dataloader
    print("Loading dataset...")

    # Set a random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    plant_architecture_dataset = PlantDataset(
        root_dir=args.dataset_path, 
        image_size=args.image_size,
        side_view=args.side_view,
        image_processor=image_processor,
        preload=False,
        process_leaf=True,
        add_sos_token=False,
        plot=args.plot,
        # stages=["00"]
    )
    # Split the dataset into Train, Validation, and Test sets
    train_size = int(0.8 * len(plant_architecture_dataset))  # 80% for training
    val_size = int(0.1 * len(plant_architecture_dataset))    # 10% for validation
    test_size = len(plant_architecture_dataset) - train_size - val_size  # Remaining 10% for testing

    # Use random_split with the seed set above
    train_dataset, val_dataset, test_dataset = random_split(plant_architecture_dataset, [train_size, val_size, test_size])


    # Make sure pixel_values are properly processed
    def custom_collator(features):
        # features는 데이터셋에서 반환된 샘플들의 리스트
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        plant_info = torch.stack([torch.tensor(f["plant_info"], dtype=torch.long) for f in features])
        # 패딩 처리: labels의 길이를 가장 긴 시퀀스에 맞춤
        max_label_length = max(len(f["labels"]) for f in features)
        labels = torch.stack([
            torch.cat([torch.tensor(f["labels"], dtype=torch.long), torch.full((max_label_length - len(f["labels"]),), PAD_TOKEN, dtype=torch.long)])
            for f in features
        ])

        # Plant info is integrated in labels, so don't need to return
        return {
            "pixel_values": pixel_values,
            "plant_info": plant_info,
            "labels": labels,
        }

    dataloader = DataLoader(
        plant_architecture_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collator
    )

    # Setup log directory
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rl_log_dir = f"{args.log_dir}/{timestamp}_SimRL"
    os.makedirs(rl_log_dir, exist_ok=True)

    # Run the RL pipeline
    print("Starting similarity-based RL process...")
    fine_tuned_model = run_rl_pipeline(
        model=model,
        dataloader=dataloader,
        rl_epochs=args.rl_epochs,
        log_dir=rl_log_dir,
        image_processor=image_processor
    )

    # Save the fine-tuned model
    fine_tuned_model.save_pretrained(f"{rl_log_dir}/fine_tuned_model")
    print(f"Similarity RL fine-tuning complete. Model saved to {rl_log_dir}/fine_tuned_model")

    # Optional: Calculate metrics after fine-tuning
    try:
        from calc_metric import calc_metric
        print("Calculating metrics after fine-tuning...")
        calc_metric(fine_tuned_model, args.dataset_path)
    except ImportError:
        print("calc_metric module not found. Skipping metric calculation.")