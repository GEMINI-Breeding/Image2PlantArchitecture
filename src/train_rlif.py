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
from typing import Optional
import subprocess
import cv2

# Import your modules
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE
from plant_dataset import PlantDataset, load_sideview_images

from string_to_xml_to_vec import vec2xml, recursive_to_linked, pretty_print_xml
from plant_tokenizer import token2vec
from image_process import process_leaf_image
import shutil


def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    else:
        pooled = x

    return pooled


# 패치 수준 정렬을 위한 추가 손실항
def patch_alignment_loss(original_patches, rendered_patches):
    # 패치 단위 정규화
    orig_norm = F.normalize(original_patches, p=2, dim=2)
    rend_norm = F.normalize(rendered_patches, p=2, dim=2)
    
    # 패치 간 정렬 손실 (상호 정보 최대화)
    sim_matrix = torch.bmm(orig_norm, rend_norm.transpose(1, 2))
    row_softmax = F.softmax(sim_matrix / 0.1, dim=2)
    col_softmax = F.softmax(sim_matrix / 0.1, dim=1)
    
    # 양방향 매칭 손실
    if 0:
        row_loss = -torch.log(row_softmax.diagonal(dim1=1, dim2=2)).mean()
        col_loss = -torch.log(col_softmax.diagonal(dim1=1, dim2=2)).mean()
    else:
        row_loss = (row_softmax.diagonal(dim1=1, dim2=2)).mean()
        col_loss = (col_softmax.diagonal(dim1=1, dim2=2)).mean()
    
    return (row_loss + col_loss) / 2

# 여러 레이어의 특성을 활용하여 다중 스케일 유사도 계산
def compute_multi_scale_similarity(original_hidden_states, rendered_hidden_states):
    # 주요 레이어 선택 (앞쪽, 중간, 뒤쪽)
    layer_indices = [0, len(original_hidden_states)//2, -1]
    multi_scale_sims = []
    
    for idx in layer_indices:
        orig = F.normalize(original_hidden_states[idx], p=2, dim=2)
        rend = F.normalize(rendered_hidden_states[idx], p=2, dim=2)
        sim = torch.bmm(orig, rend.transpose(1, 2))
        multi_scale_sims.append(sim.max(dim=2)[0].mean(dim=1))
    
    # 여러 레이어의 유사도 평균
    return torch.stack(multi_scale_sims).mean(dim=0)

class ImageSimilarityRewardModel:
    """Computes rewards based on similarity between original and rendered images"""
    def __init__(
        self, 
        encoder,
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_processor=None,
        side_view=False,
    ):
        # Load a pretrained vision model for computing image similarity
        self.device = device
        self.encoder = encoder
        self.image_processor = image_processor
        self.side_view = side_view
        # Set model to evaluation mode
        self.encoder.eval()
    
    # Renderer for converting generated sequences back to images
    # This could be a separate model or a rule-based system
    def renderer(self, generated_sequences,
                 output_path='temp', filename='rendered', 
                 program_path="src/GenerateDataset/build",
                 image_size=224,
                 debug=False):
        """
        Render a batch of sequences to images - single-threaded version
        """
        # Initialize cache if not already present
        if not hasattr(self, '_render_cache'):
            self._render_cache = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        batch_size = generated_sequences.size(0) if torch.is_tensor(generated_sequences) else len(generated_sequences)
        rendered_images = []
        
        # Process each sequence one by one
        for batch_idx in range(batch_size):
            sequence = generated_sequences[batch_idx] if torch.is_tensor(generated_sequences) else generated_sequences[batch_idx]
            # Create a hashable key for the cache
            seq_key = tuple(sequence.cpu().numpy().tolist())
            
            # Use cached render if available
            if seq_key in self._render_cache:
                rendered_images.append(self._render_cache[seq_key])
                continue
            
            # Setup paths for rendering
            batch_output_path = f"{output_path}/batch_{batch_idx}"
            batch_output_path = os.path.abspath(batch_output_path)
            batch_filename = f"{filename}_{batch_idx}.xml"
            os.makedirs(batch_output_path, exist_ok=True)
            
            try:
                # Skip tokens 0-4 which might be special tokens
                plant_vec = token2vec(sequence[5:])
                plant_xml = vec2xml(plant_vec)
                plant_xml_file_name = f"{batch_output_path}/{batch_filename}"
                plant_xml = recursive_to_linked(plant_xml)
                plant_xml_str = pretty_print_xml(plant_xml)
                with open(plant_xml_file_name, "w") as f:
                    f.write(plant_xml_str)
                
                # Render jpeg - use environment variable
                image_name = batch_filename.split(".")[0]
                os.environ["DISPLAY"] = ":11.0"
                command = f"cd {program_path} && ./main -h 1.0 -o {batch_output_path} -name {image_name} -tile none -f {os.path.join(batch_output_path, batch_filename)}"
                if self.side_view:
                    command += " -r"
                
                try:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
                    if debug:
                        print(f"Command output: {result.stdout}")
                        print(f"Command errors: {result.stderr}")
                except subprocess.TimeoutExpired:
                    if debug:
                        print(f"Warning: Rendering process timed out for batch {batch_idx}")
                    img = torch.zeros((3, image_size, image_size), dtype=torch.float32, device=self.device)
                    rendered_images.append(img)
                    self._render_cache[seq_key] = img
                    continue
                
                # Process image
                if self.side_view:
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
                        leaf_area, plant_width, plant_height, leaf_img, _ = process_leaf_image(
                            img, sqaure_crop=True
                        )
                        img = cv2.resize(leaf_img, (image_size, image_size))
                
                # Convert to tensor
                if isinstance(img, np.ndarray):
                    img = self.image_processor(images=img, return_tensors="pt").pixel_values[0].to(self.device)
                
                # Cache the result
                self._render_cache[seq_key] = img
                rendered_images.append(img)
                    
            except Exception as e:
                if debug:
                    print(f"Error processing image for batch {batch_idx}: {e}")
                img = torch.zeros((3, image_size, image_size), dtype=torch.float32, device=self.device)
                rendered_images.append(img)
                self._render_cache[seq_key] = img
        
        # Periodic cache cleanup to prevent memory issues
        if len(self._render_cache) > 2000:  # Adjust threshold as needed
            # Keep the 1000 most recent entries
            cache_keys = list(self._render_cache.keys())
            for key in cache_keys[:-1000]:
                del self._render_cache[key]
        
        # Stack all images into a single tensor
        return torch.stack(rendered_images)
    
    def compute_reward(self, original_images, generated_sequences, debug=False):
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
        image_size = original_images.size(-1)
        # Render the generated sequences to images
        rendered_images = self.renderer(
            generated_sequences, 
            output_path=temp_dir,
            filename="rendered",
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
            
                    
            # Use this to get all token features:
            original_outputs = self.encoder(original_images, output_hidden_states=True)
            rendered_outputs = self.encoder(rendered_images, output_hidden_states=True)

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
            diagonal_rewards = mean_diagonal_similarities

            # Compare with the original reward calculation method:
            # Option 1: Maximum similarity for each token in original image
            max_similarities_per_token = token_similarities.max(dim=2)[0]  # Shape: [batch_size, orig_seq_len]
            mean_max_similarity = max_similarities_per_token.mean(dim=1)  # Shape: [batch_size]

            # Scale to 0-1 range
            rewards = (mean_max_similarity + 1) / 2

            # Multi scale similarity
            multi_scale_similarity = compute_multi_scale_similarity(original_outputs.hidden_states,
                                                                   rendered_outputs.hidden_states)
            patch_alignment = patch_alignment_loss(original_outputs.hidden_states[-1],
                                                   rendered_outputs.hidden_states[-1])


            # 전체 이미지 수준의 유사도 추가 (CLS 토큰 또는 pooled 출력 사용)
            original_global = original_outputs.pooler_output  # [batch_size, hidden_dim]
            rendered_global = rendered_outputs.pooler_output  # [batch_size, hidden_dim]

            # L2 정규화 후 유사도 계산
            original_global = F.normalize(original_global, p=2, dim=1)
            rendered_global = F.normalize(rendered_global, p=2, dim=1)
            global_similarity = torch.sum(original_global * rendered_global, dim=1)

            # 0-1 범위로 조정
            global_similarity = (global_similarity + 1) / 2
            # You could also combine both reward signals
            #combined_rewards = 0.5 * rewards + 0.5 * diagonal_rewards
            combined_rewards = diagonal_rewards
            
            # Debug
            if True:
                # Save original images too
                for idx, img in enumerate(original_images):
                    debug_path = os.path.join(temp_dir, f"batch_{idx}/original_image_{idx}.jpeg")
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    image_vis = cv2.normalize(np.array(img_np), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) # Convert RGB to BGR before saving
                    cv2.imwrite(debug_path, image_vis)
                    
                    debug_path = os.path.join(temp_dir, f"batch_{idx}/rendered_image_{idx}.jpeg")
                    img_np = rendered_images[idx].permute(1, 2, 0).cpu().numpy()
                    rendered_image_vis = cv2.normalize(np.array(img_np), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    rendered_image_vis = cv2.cvtColor(rendered_image_vis, cv2.COLOR_RGB2BGR) # Convert RGB to BGR before saving
                    cv2.imwrite(debug_path, rendered_image_vis)

                    # Convert diagonal similarities to a heatmap
                    patch_size = int(np.sqrt(diagonal_similarities.size(-1)))
                    diagonal_similarities_vis = diagonal_similarities[idx][:-1].reshape(patch_size, patch_size).cpu().numpy()
                    heatmap = cv2.normalize(np.array(diagonal_similarities_vis), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap = cv2.resize(heatmap, (rendered_image_vis.shape[1], rendered_image_vis.shape[0]))
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    # Overlay the heatmap on the rendered image
                    mixed = cv2.addWeighted(rendered_image_vis, 0.5, image_vis, 0.5, 0)
                    overlay = cv2.addWeighted(mixed, 0.7, heatmap, 0.1, 0)
                    
                    # Save the overlay image
                    debug_path = os.path.join(temp_dir, f"batch_{idx}/similarities_{idx}.jpeg")
                    cv2.imwrite(debug_path, overlay)
                    
                    # Create a visualization of all similarity metrics
                    info_img = np.zeros((300, rendered_image_vis.shape[1], 3), dtype=np.uint8) + 255
                    
                    # Add text for each similarity metric
                    metrics = [
                        f"Diagonal Sim: {diagonal_rewards[idx]:.4f}",
                        f"Max Token Sim: {rewards[idx]:.4f}",
                        f"Multi-scale Sim: {multi_scale_similarity[idx]:.4f}",
                        f"Patch Alignment: {patch_alignment:.4f}",
                        f"Global Sim: {global_similarity[idx]:.4f}"
                    ]
                    
                    for i, text in enumerate(metrics):
                        cv2.putText(
                            info_img, text, (10, 30 + i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
                        )
                    
                    # Combine with the previous images
                    combined_vis = cv2.vconcat([overlay, info_img])
                    
                    # Save the combined visualization
                    debug_path = os.path.join(temp_dir, f"batch_{idx}/all_metrics_{idx}.jpeg")
                    cv2.imwrite(debug_path, combined_vis)

        return combined_rewards


class PlantRLTrainer:
    """Trainer for RL fine-tuning using image similarity rewards"""
    def __init__(
        self,
        model,
        ref_model,
        reward_model,
        optimizer,
        side_view,
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
            self.log_file = os.path.join(log_dir, "RL_log.txt")
        
        # Initialize value head for reward prediction
        hidden_size = self.model.decoder.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)
        self.value_optimizer = Adam(self.value_head.parameters(), lr=1e-4)

        self.side_view = side_view
        
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

                # Check if labels already start with SOS token
                if not torch.all(labels[:, 0] == SOS_TOKEN):
                    sos_tokens = torch.full((labels.size(0), 1), SOS_TOKEN, 
                                       dtype=labels.dtype, 
                                       device=labels.device)
                    labels = torch.cat([sos_tokens, labels], dim=1)

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
                        input_ids=labels[:,:-1], # Remove EOS
                        encoder_hidden_states=encoder_hidden_states,
                        output_hidden_states=True
                    )
                    logits = decoder_outputs.logits # Generated from decode, following SOS
                    generated_sequences = torch.argmax(logits, dim=-1) # Start with META token

                    # last_hidden_states = decoder_outputs.hidden_states[-1][:, -1, :]
                    last_hidden_states = text_global_pool(decoder_outputs.hidden_states[-1], generated_sequences)

                # Compute rewards using the reward model
                rewards = self.reward_model.compute_reward(pixel_values, generated_sequences, debug=False)
                
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
            
            # Log value head training loss
            if self.log_dir:
                with open(self.log_file, "a") as f:
                    log_data = {
                        "epoch": epoch + 1,
                        "value_head_loss": value_losses[-1],
                        "timestamp": time.time()
                    }
                    f.write(json.dumps(log_data) + "\n")
            
        return value_losses
    
    def rl_step(self, dataloader, epochs=4, train_value_head_first=True):
        """Run standard PPO training with image similarity rewards"""
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
            
            # 1. COLLECT EXPERIENCES (ROLLOUT PHASE)
            all_states = []
            all_actions = []
            all_logprobs = []
            all_rewards = []
            all_values = []
            all_masks = []
            all_advantages = []
            
            print("Collecting experiences...")
            for batch in tqdm(dataloader, desc=f"RL rollout {epoch+1}/{epochs}"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Ensure labels start with SOS token
                if not torch.all(labels[:, 0] == SOS_TOKEN):
                    sos_tokens = torch.full((labels.size(0), 1), SOS_TOKEN, 
                                       dtype=labels.dtype, 
                                       device=labels.device)
                    labels = torch.cat([sos_tokens, labels], dim=1)

                # Get encoder outputs
                with torch.no_grad():
                    encoder_outputs = self.model.encoder(pixel_values)
                    encoder_hidden_states = encoder_outputs.last_hidden_state
                    if (self.model.encoder.config.hidden_size != self.model.decoder.config.hidden_size
                        and self.model.decoder.config.cross_attention_hidden_size is None):
                        encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

                # Generate sequences with the current policy (sampling, not argmax)
                with torch.no_grad():
                    policy_outputs = self.model.decoder(
                        input_ids=labels[:,:-1],
                        encoder_hidden_states=encoder_hidden_states,
                        output_hidden_states=True
                    )
                    logits = policy_outputs.logits
                    
                    # Sample from the distribution instead of argmax
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    
                    # Get value predictions
                    last_hidden_states = text_global_pool(policy_outputs.hidden_states[-1], actions)
                    values = self.value_head(last_hidden_states).squeeze(-1)
                
                # Compute rewards for these actions
                rewards = self.reward_model.compute_reward(pixel_values, actions)
                rewards_history.extend(rewards.cpu().numpy().tolist())
                
                # Create masks for padding
                masks = (actions != self.model.config.pad_token_id).float()
                
                # Store experience
                all_states.append((pixel_values, encoder_hidden_states, labels[:,:-1]))
                all_actions.append(actions)
                all_logprobs.append(log_probs)
                all_rewards.append(rewards)
                all_values.append(values)
                all_masks.append(masks)
            
            # 2. COMPUTE ADVANTAGES AND RETURNS
            print("Computing advantages...")
            with torch.no_grad():
                # Simple advantage calculation (can be replaced with GAE)
                for i in range(len(all_rewards)):
                    advantages = all_rewards[i].unsqueeze(-1).expand_as(all_logprobs[i]) - all_values[i].unsqueeze(-1).expand_as(all_logprobs[i])
                    all_advantages.append(advantages)
            
            # 3. OPTIMIZE POLICY WITH MULTIPLE PASSES
            # Number of optimization epochs (standard PPO does multiple passes)
            n_opt_epochs = 4
            
            print("Optimizing policy...")
            for _ in range(n_opt_epochs):
                # Shuffle the experience indices
                indices = torch.randperm(len(all_states))
                
                # Process all collected experiences in mini-batches
                for idx in indices:
                    pixel_values, encoder_hidden_states, input_ids = all_states[idx]
                    actions = all_actions[idx]
                    old_log_probs = all_logprobs[idx]
                    rewards = all_rewards[idx]
                    advantages = all_advantages[idx]
                    masks = all_masks[idx]
                    
                    # Get current policy outputs
                    policy_outputs = self.model.decoder(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        output_hidden_states=True
                    )
                    logits = policy_outputs.logits
                    
                    # Get log probs of actions under current policy
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(actions)
                    
                    # Get logits from reference model (frozen version)
                    with torch.no_grad():
                        ref_outputs = self.ref_model.decoder(
                            input_ids=input_ids,
                            encoder_hidden_states=encoder_hidden_states,
                            output_hidden_states=True
                        )
                        ref_logits = ref_outputs.logits
                        ref_probs = F.softmax(ref_logits, dim=-1)
                        ref_dist = torch.distributions.Categorical(ref_probs)
                    
                    # Calculate KL divergence between current and reference policy
                    kl = torch.distributions.kl_divergence(dist, ref_dist) * masks
                    kl = kl.sum() / masks.sum().clamp(min=1e-8)
                    
                    # Calculate entropy
                    entropy = dist.entropy() * masks
                    entropy = entropy.sum() / masks.sum().clamp(min=1e-8)
                    
                    # Calculate surrogate objectives
                    ratio = torch.exp(new_log_probs - old_log_probs.detach())
                    surr1 = advantages * ratio * masks  # 부호 변경 (음수 제거)
                    surr2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * masks  # 부호 변경
                    pg_loss = -torch.min(surr1, surr2).sum() / masks.sum().clamp(min=1e-8)  # min 사용 및 음수화
                    
                    # Combined policy loss with KL penalty and entropy bonus
                    policy_loss = pg_loss + self.kl_coef * kl - self.entropy_coef * entropy
                    
                    # Optimize policy network
                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    # Get current value predictions
                    last_hidden_states = text_global_pool(policy_outputs.hidden_states[-1], actions)
                    values = self.value_head(last_hidden_states.detach()).squeeze(-1)
                    
                    # Value loss
                    value_loss = F.mse_loss(values, rewards)
                    
                    # Optimize value network
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
                    self.value_optimizer.step()
                    
                    # Record losses
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    kl_divergences.append(kl.item())
                    entropies.append(entropy.item())
                    
                    if self.scheduler:
                        self.scheduler.step()
            
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
    side_view,
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
        image_processor=image_processor,
        side_view=side_view
    )
    
    # Initialize RL trainer
    rl_trainer = PlantRLTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        optimizer=optimizer,
        log_dir=log_dir,
        side_view=side_view,
    )
    
    # Run RL training
    print("Starting RL training...")
    rl_trainer.rl_step(dataloader, epochs=rl_epochs)
    
    return model




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fine-tune with image similarity RL')
    parser.add_argument('--model_path', type=str, default='log/20250327/dinov2-small_224_Sideview_gpt2/results', help='Path to pretrained model')
    parser.add_argument('--dataset_path', type=str, default='data/2000_Plots_20241210_BetterQuantized', help='Path to dataset')
    parser.add_argument('--plot', type=str, default=[f"{i:04d}" for i in range(1)], help='Plots')
    #parser.add_argument('--plot', type=str, default=None, help='Plots')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--side_view', type=str, default='True', help='Use side view')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--rl_epochs', type=int, default=10, help='Number of RL training epochs')
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
    print(f"Setting up log directories...")
    # Extract parent directory of model_path (go one level up from results directory)
    parent_dir = os.path.dirname(args.model_path)

    # Create file/directory paths relative to parent directory
    log_file = os.path.join(parent_dir, "RL_log.txt")
    results_dir = os.path.join(parent_dir, "RL_results")
    benchmark_file = os.path.join(parent_dir, "RL_benchmark.txt")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Run the RL pipeline
    print("Starting similarity-based RL process...")
    fine_tuned_model = run_rl_pipeline(
        model=model,
        dataloader=dataloader,
        rl_epochs=args.rl_epochs,
        log_dir=parent_dir,  # 모델 경로의 상위 폴더를 기준으로 설정
        image_processor=image_processor,
        side_view=args.side_view
    )

    # Save the fine-tuned model
    fine_tuned_model.save_pretrained(results_dir)
    print(f"Similarity RL fine-tuning complete. Model saved to {results_dir}")

    # Optional: Calculate metrics after fine-tuning
    try:
        from calc_metric import calc_metric
        print("Calculating metrics after fine-tuning...")
        benchmark_path = os.path.join(parent_dir, "RL_benchmark.txt")
        calc_metric(fine_tuned_model, args.dataset_path, log_path=benchmark_path, side_view=args.side_view)
    except ImportError:
        print("calc_metric module not found. Skipping metric calculation.")