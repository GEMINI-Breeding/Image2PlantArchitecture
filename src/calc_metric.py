import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
from plant_dataset import PlantDataset
from plant_tokenizer import token2vec, token_ids_to_base64_like, PAD_TOKEN
from typing import Dict, Any, Optional, Tuple, List
import os
from string_to_xml_to_vec import vec2xml, pretty_print_xml, recursive_to_linked


def collate_fn(features):
    """Collate function for DataLoader - updated to match batch generation format."""
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    
    # Get plant_info from the labels (assuming it's the first 5 tokens after SOS)
    plant_info = []
    labels_list = []
    
    for f in features:
        labels = f["labels"]
        if len(labels) > 6:  # SOS + 5 plant_info tokens
            plant_info.append(labels[1:6])  # Extract plant_info (skip SOS)
        else:
            plant_info.append([PAD_TOKEN] * 5)  # Fallback if sequence is too short
        labels_list.append(labels)
    
    # Convert to numpy arrays first, then to tensors
    plant_info_array = np.array(plant_info, dtype=np.int64)
    plant_info = torch.from_numpy(plant_info_array)
    
    # Padding processing: adjust labels' length to match the longest sequence
    max_label_length = max(len(labels) for labels in labels_list)
    
    # Create padded labels array using numpy first
    labels_padded = np.full((len(labels_list), max_label_length), PAD_TOKEN, dtype=np.int64)
    for i, labels in enumerate(labels_list):
        labels_padded[i, :len(labels)] = labels
    
    # Convert to tensor
    labels = torch.from_numpy(labels_padded)
    
    plant_vec = [np.concatenate(f["plant_vec"]) for f in features]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "plant_info": plant_info,  # Add this for compatibility
        "plant_vec": plant_vec
    }


def prepare_dataset(
    dataset_path: str,
    image_processor: Any,
    image_size: int = 448,
    side_view: bool = False,
    growth_stages: Optional[List[str]] = None,
    test_split: float = 0.1,
    val_split: float = 0.1,
    preload: bool = False,
    seed: int = 42
) -> Any:
    """Prepare and split the dataset for evaluation."""
    # Configure image processor
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size
    
    xml_files = os.listdir(os.path.join(dataset_path, "xml"))
    xml_files.sort()
    num_plots = int(xml_files[-1].split("_")[1]) + 1

    train_end = int(num_plots * (1-val_split-test_split))
    val_end = train_end + int(num_plots * val_split)
    test_end = min(num_plots, val_end + int(num_plots * test_split))
    
    test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]

    test_dataset = PlantDataset(
        root_dir=dataset_path, 
        stages=growth_stages, 
        process_leaf=True, 
        image_size=image_size,
        side_view=side_view,
        plot=test_plots,
        mode='val',
        preload=preload, 
        image_processor=image_processor, 
        add_sos_token=False
    )

    return test_dataset


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    debug: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Run model inference on test dataset with batch processing using generation."""
    from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    
    model.eval()
    all_predictions = []
    all_labels = []
    gt_plant_vecs = []
    

    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating with Generation"):
            if debug and len(all_predictions) > 10:
                break

            # Prepare inputs
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            plant_vecs = batch["plant_vec"]
            
            batch_size = pixel_values.shape[0]
            
            # Extract plant_info from labels (first 5 tokens after SOS)
            plant_info_batch = labels[:, 0:5]  # Extract plant info (5 tokens after SOS)
            
            # Set dynamically the max length from the labels
            max_length = len(labels[0]) * 2
            try:
                # Generate predictions for the entire batch at once
                generated_ids = model.generate(
                    pixel_values,
                    decoder_start_token_id=SOS_TOKEN,
                    decoder_input_ids=plant_info_batch,
                    eos_token_id=EOS_TOKEN,
                    pad_token_id=PAD_TOKEN,
                    max_length=max_length,
                    use_cache=True,
                    do_sample=False,  # Use greedy decoding for reproducible results
                    num_beams=1       # Greedy search
                )
                
                # Process each sample in the batch
                for i in range(batch_size):
                    try:
                        # Extract individual prediction from batch results
                        single_generated = generated_ids[i].cpu().numpy()
                        single_label = labels[i].cpu().numpy()
                        single_plant_vec = plant_vecs[i]
                        
                        # Remove the input plant_info part (first 6 tokens: SOS + plant_info)
                        if len(single_generated) > 6:
                            prediction = single_generated[6:]
                        else:
                            prediction = np.array([])
                        
                        # Store results
                        all_predictions.append(prediction)
                        all_labels.append(single_label)
                        gt_plant_vecs.append(single_plant_vec)
                        
                    except Exception as e:
                        print(f"Error processing sample {i} in batch: {e}")
                        # Fallback to empty prediction
                        all_predictions.append(np.array([]))
                        all_labels.append(labels[i].cpu().numpy())
                        gt_plant_vecs.append(plant_vecs[i])
                        
            except Exception as e:
                print(f"Error generating batch: {e}")
                # Fallback: process each sample individually if batch generation fails
                for i in range(batch_size):
                    try:
                        single_pixel_values = pixel_values[i:i+1]
                        single_plant_info = plant_info_batch[i:i+1]
                        
                        generated_ids = model.generate(
                            single_pixel_values,
                            decoder_start_token_id=SOS_TOKEN,
                            decoder_input_ids=single_plant_info,
                            eos_token_id=EOS_TOKEN,
                            pad_token_id=PAD_TOKEN,
                            max_length=max_length,
                            use_cache=True,
                            do_sample=False,
                            num_beams=1
                        )
                        
                        generated_sequence = generated_ids.squeeze().cpu().numpy()
                        if len(generated_sequence.shape) > 0 and len(generated_sequence) > 6:
                            prediction = generated_sequence[6:]
                        else:
                            prediction = np.array([])
                            
                        all_predictions.append(prediction)
                        all_labels.append(labels[i].cpu().numpy())
                        gt_plant_vecs.append(plant_vecs[i])
                        
                    except Exception as inner_e:
                        print(f"Error generating for individual sample {i}: {inner_e}")
                        all_predictions.append(np.array([]))
                        all_labels.append(labels[i].cpu().numpy())
                        gt_plant_vecs.append(plant_vecs[i])
    
    return all_predictions, all_labels, gt_plant_vecs

def compute_metrics(
    predictions: List[np.ndarray],
    labels: List[np.ndarray],
) -> Dict[str, float]:
    """Compute corpus-based BLEU scores for the model predictions using generation results."""
    # Load metrics
    bleu_metric = evaluate.load('sacrebleu')
    
    # Collect all predictions and references for corpus-level evaluation
    all_pred_ascii = []
    all_label_ascii = []

    # Process all samples to build corpus
    for pred, label in zip(predictions, labels):
        try:
            # Remove PAD tokens from predictions and labels
            pred_no_pad = pred[pred != PAD_TOKEN] if len(pred) > 0 else np.array([])
            # For labels, skip the first 6 tokens (SOS + plant_info) and remove PAD tokens
            label_sequence = label[5:] if len(label) > 5 else label
            label_no_pad = label_sequence[label_sequence != PAD_TOKEN]
            
            # Convert token IDs to semantic base64-like encoding for BLEU calculation
            if len(pred_no_pad) > 0 and len(label_no_pad) > 0:
                pred_ascii = token_ids_to_base64_like(pred_no_pad)
                label_ascii = token_ids_to_base64_like(label_no_pad)
                
                all_pred_ascii.append(pred_ascii)
                all_label_ascii.append(label_ascii)
            else:
                # Handle empty sequences - add empty strings to maintain alignment
                all_pred_ascii.append("")
                all_label_ascii.append("")
                
        except Exception as e:
            print(f"Error processing sample for corpus BLEU: {e}")
            # Add empty strings to maintain alignment
            all_pred_ascii.append("")
            all_label_ascii.append("")
    
    # Compute corpus-level BLEU
    if all_pred_ascii and all_label_ascii:
        # For corpus-level BLEU, we pass all predictions and all references at once
        corpus_bleu_result = bleu_metric.compute(
            predictions=all_pred_ascii, 
            references=[[ref] for ref in all_label_ascii]  # Each reference needs to be in a list
        )
        corpus_bleu_score = corpus_bleu_result['score']
    else:
        corpus_bleu_score = 0.0
    
    # Compute averages
    metrics = {
        'bleu': corpus_bleu_score
    }

    return metrics


def calc_metric(
    model: torch.nn.Module, 
    test_dataset: PlantDataset, 
    log_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    debug: bool = False,
    benchmark_folder: str = "benchmark",
) -> Dict[str, float]:
    """Calculate metrics for a model on a given dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create benchmark folders
    pred_folder = os.path.join(benchmark_folder, "pred")
    gt_folder = os.path.join(benchmark_folder, "gt")
    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)
    
    # Create DataLoader for batch processing
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Evaluate model with batch processing
    predictions, labels, gt_plant_vecs = evaluate_model(
        model, 
        dataloader, 
        device,
        debug,
    )
    
    # Process predictions and save XMLs
    print("Converting predictions to plant vectors and saving XMLs...")
    for idx, (pred, label, gt_plant_vec) in enumerate(tqdm(zip(predictions, labels, gt_plant_vecs), 
                                                            desc="Saving XMLs", 
                                                            total=len(predictions))):
        try:
            # Save predicted XML - convert predicted tokens to plant vector
            pred_tokens = pred
            est_plant_vec = token2vec(pred_tokens)
            
            if est_plant_vec:
                pred_xml = vec2xml(est_plant_vec, plant_id=idx)
                pred_xml = recursive_to_linked(pred_xml)
                pred_xml_str = pretty_print_xml(pred_xml)
                
                pred_xml_path = os.path.join(pred_folder, f"plant_{idx:04d}.xml")
                with open(pred_xml_path, "w") as f:
                    f.write(pred_xml_str)
            
            # Save ground truth XML - convert ground truth tokens to plant vector
            gt_tokens = label[5:] if len(label) > 5 else label  # Skip SOS + plant_info
            gt_tokens = gt_tokens[gt_tokens != PAD_TOKEN]  # Remove PAD tokens
            gt_plant_vec_from_tokens = token2vec(gt_tokens)
            
            if gt_plant_vec_from_tokens:
                gt_xml = vec2xml(gt_plant_vec_from_tokens, plant_id=idx)
                gt_xml = recursive_to_linked(gt_xml)
                gt_xml_str = pretty_print_xml(gt_xml)
                
                gt_xml_path = os.path.join(gt_folder, f"plant_{idx:04d}.xml")
                with open(gt_xml_path, "w") as f:
                    f.write(gt_xml_str)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")
    
    print(f"\nXML files saved to:")
    print(f"  Predictions: {os.path.abspath(pred_folder)}")
    print(f"  Ground Truth: {os.path.abspath(gt_folder)}")
    
    # Save results to log file
    with open(log_path, "w") as log_file:
        log_file.write("EVALUATION RESULTS\n")
        log_file.write("="*50 + "\n")
        for name, value in metrics.items():
            log_file.write(f"{name.upper()}: {value:.4f}\n")
        
        log_file.write(f"\nXML files saved to:\n")
        log_file.write(f"  Predictions: {os.path.abspath(pred_folder)}\n")
        log_file.write(f"  Ground Truth: {os.path.abspath(gt_folder)}\n")
        log_file.write(f"  Total samples processed: {len(predictions)}\n")
    
    return metrics