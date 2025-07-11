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
from accelerate import Accelerator, DistributedDataParallelKwargs


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


def evaluate_model_with_accelerator(
    model: torch.nn.Module,
    dataloader: Any,
    accelerator: Accelerator,
    debug: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Run model inference on test dataset with batch processing using generation with accelerator."""
    from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    
    print(f"Using accelerate with {accelerator.num_processes} processes")
    
    model.eval()
    all_predictions = []
    all_labels = []
    gt_plant_vecs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating with Generation", disable=not accelerator.is_local_main_process):
            if debug and len(all_predictions) > 10:
                break

            # Prepare inputs
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            plant_vecs = batch["plant_vec"]
            
            batch_size = pixel_values.shape[0]
            
            # Extract plant_info from labels (first 5 tokens after SOS)
            plant_info_batch = labels[:, 0:5]
            
            # Set a more reasonable max_length
            max_length = min(512, max(len(label) for label in labels) + 50)
            
            try:
                # Generate predictions for the entire batch at once
                with torch.cuda.amp.autocast():
                    # Use the underlying model for generation (unwrap from accelerator)
                    unwrapped_model = accelerator.unwrap_model(model)
                    generated_ids = unwrapped_model.generate(
                        pixel_values,
                        decoder_start_token_id=SOS_TOKEN,
                        decoder_input_ids=plant_info_batch,
                        eos_token_id=EOS_TOKEN,
                        pad_token_id=PAD_TOKEN,
                        max_length=max_length,
                        min_length=6,
                        use_cache=True,
                        do_sample=False,
                        num_beams=1,
                        return_dict_in_generate=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        repetition_penalty=1.1,
                    )
                    
                # Process locally without gathering to avoid deadlock
                for i in range(batch_size):
                    try:
                        # Extract individual prediction from batch results
                        single_generated = generated_ids[i].cpu().numpy()
                        single_label = labels[i].cpu().numpy()
                        single_plant_vec = plant_vecs[i] if i < len(plant_vecs) else []
                        
                        # Remove the input plant_info part (first 6 tokens: SOS + plant_info)
                        if len(single_generated) > 6:
                            prediction = single_generated[6:]
                        else:
                            prediction = np.array([])
                        
                        # Remove EOS token if present at the end
                        if len(prediction) > 0 and prediction[-1] == EOS_TOKEN:
                            prediction = prediction[:-1]
                        
                        # Store results
                        all_predictions.append(prediction)
                        all_labels.append(single_label)
                        gt_plant_vecs.append(single_plant_vec)
                        
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        # Fallback to empty prediction
                        all_predictions.append(np.array([]))
                        all_labels.append(labels[i].cpu().numpy() if i < len(labels) else np.array([]))
                        gt_plant_vecs.append(plant_vecs[i] if i < len(plant_vecs) else [])
                        
            except Exception as e:
                print(f"Error generating batch: {e}")
                # In case of failure, add empty results for this batch
                for i in range(batch_size):
                    all_predictions.append(np.array([]))
                    all_labels.append(labels[i].cpu().numpy())
                    gt_plant_vecs.append(plant_vecs[i] if i < len(plant_vecs) else [])
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Gather results from all processes at the end
    if accelerator.num_processes > 1:
        # Convert to tensors for gathering
        all_predictions_gathered = []
        all_labels_gathered = []
        gt_plant_vecs_gathered = []
        
        # Gather predictions
        for pred in all_predictions:
            if len(pred) > 0:
                pred_tensor = torch.from_numpy(pred).to(accelerator.device)
                gathered_pred = accelerator.gather(pred_tensor)
                if accelerator.is_main_process:
                    all_predictions_gathered.extend([p.cpu().numpy() for p in gathered_pred])
            elif accelerator.is_main_process:
                all_predictions_gathered.append(np.array([]))
        
        # Gather labels 
        for label in all_labels:
            label_tensor = torch.from_numpy(label).to(accelerator.device)
            gathered_label = accelerator.gather(label_tensor)
            if accelerator.is_main_process:
                all_labels_gathered.extend([l.cpu().numpy() for l in gathered_label])
        
        # For plant_vecs, just collect from main process for now
        if accelerator.is_main_process:
            gt_plant_vecs_gathered = gt_plant_vecs
        
        if accelerator.is_main_process:
            return all_predictions_gathered, all_labels_gathered, gt_plant_vecs_gathered
        else:
            return [], [], []
    else:
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
    
    # Accelerator를 한 번만 생성
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # 메인 프로세스에서만 폴더 생성
    if accelerator.is_main_process:
        pred_folder = os.path.join(benchmark_folder, "pred")
        gt_folder = os.path.join(benchmark_folder, "gt")
        os.makedirs(pred_folder, exist_ok=True)
        os.makedirs(gt_folder, exist_ok=True)
    
    # 모든 프로세스가 폴더 생성 완료까지 대기
    accelerator.wait_for_everyone()
    
    # DataLoader 생성
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 모델과 데이터로더 준비
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # 추론 실행 (accelerator를 파라미터로 전달)
    predictions, labels, gt_plant_vecs = evaluate_model_with_accelerator(
        model, 
        dataloader, 
        accelerator,  # 동일한 accelerator 사용
        debug,
    )
    
    # 메인 프로세스에서만 후처리
    if accelerator.is_main_process:
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
    else:
        # Return empty metrics for non-main processes
        return {}