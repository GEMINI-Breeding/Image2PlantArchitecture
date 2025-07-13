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
import sacrebleu, rouge_score


def collate_fn(features):
    """Collate function for DataLoader - updated to match batch generation format."""
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    
    # Get plant_info from the labels (assuming it's the first 5 tokens after SOS)
    plant_info = []
    labels_list = []
    
    for f in features:
        labels = f["labels"]
        plant_info.append(f["plant_info"])
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
    
    plant_vec = [f["plant_vec"] for f in features]

    decoder_attention_mask = (labels != PAD_TOKEN).long()

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "plant_info": plant_info,  # Add this for compatibility
        "plant_vec": plant_vec,
        "decoder_attention_mask": decoder_attention_mask,  
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
    benchmark_folder: str = "benchmark",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Run model inference on test dataset with batch processing using generation with accelerator."""
    from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    import pickle
    import glob
    import os
    
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
                        
                        # Remove the label plant_info part (first 5 tokens: plant_info)
                        if len(single_label) > 5:
                            single_label = single_label[5:]
                        else:
                            single_label = np.array([])

                        
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
    
    # SIMPLE SOLUTION: Save each process results directly to benchmark folder
    
    # Save local results to temporary files with unique names directly in benchmark folder
    temp_file = os.path.join(benchmark_folder, f'eval_results_rank_{accelerator.process_index}_{os.getpid()}.pkl')
    
    # Ensure the benchmark folder exists for all processes
    os.makedirs(benchmark_folder, exist_ok=True)
    
    with open(temp_file, 'wb') as f:
        pickle.dump({
            'predictions': all_predictions,
            'labels': all_labels, 
            'gt_plant_vecs': gt_plant_vecs,
            'process_index': accelerator.process_index,
            'num_samples': len(all_predictions)
        }, f)
    
    print(f"Rank {accelerator.process_index}: Saved {len(all_predictions)} samples to {temp_file}")
    
    accelerator.wait_for_everyone()
    
    # Main process collects all results
    if accelerator.is_main_process:
        collected_predictions = []
        collected_labels = []
        collected_gt_plant_vecs = []
        
        # Collect from all processes - look for all rank files
        rank_files = glob.glob(os.path.join(benchmark_folder, f"eval_results_rank_*_{os.getpid()}.pkl"))
        
        # Also look for files from other processes (different PIDs)
        all_rank_files = glob.glob(os.path.join(benchmark_folder, "eval_results_rank_*.pkl"))
        
        print(f"Main process found {len(all_rank_files)} result files: {all_rank_files}")
        
        for rank_file in sorted(all_rank_files):  # Sort to ensure consistent order
            try:
                with open(rank_file, 'rb') as f:
                    rank_data = pickle.load(f)
                    print(f"Loading {rank_data['num_samples']} samples from rank {rank_data['process_index']}")
                    collected_predictions.extend(rank_data['predictions'])
                    collected_labels.extend(rank_data['labels'])
                    collected_gt_plant_vecs.extend(rank_data['gt_plant_vecs'])
                
                # Clean up the file after loading
                os.unlink(rank_file)
                
            except Exception as e:
                print(f"Error loading rank file {rank_file}: {e}")
        
        print(f"Collected total samples: {len(collected_predictions)}")
        return collected_predictions, collected_labels, collected_gt_plant_vecs
    else:
        # Non-main processes return empty
        return [], [], []


def compute_metrics(
    predictions: List[np.ndarray],
    labels: List[np.ndarray],
) -> Dict[str, float]:
    """Compute corpus-based BLEU and ROUGE scores for the model predictions using generation results."""
    
    # Collect all predictions and references for corpus-level evaluation
    all_pred_ascii = []
    all_label_ascii = []
    all_pred_ascii_arch_only = []  # For architecture tokens only (0-23)
    all_label_ascii_arch_only = []
    
    # 디버깅용 카운터
    debug_count = 0
    debug_samples = 5

    # Process all samples to build corpus
    for pred, label in zip(predictions, labels):
        try:
            # Remove PAD tokens
            pred_no_pad = pred[pred != PAD_TOKEN] if len(pred) > 0 else np.array([])
            label_no_pad = label[label != PAD_TOKEN]
            
            # 토큰 ID를 직접 문자열로 변환 (1:1 매핑 보장)
            if len(pred_no_pad) > 0 and len(label_no_pad) > 0:
                pred_ascii = " ".join(map(str, pred_no_pad))
                label_ascii = " ".join(map(str, label_no_pad))
                
                all_pred_ascii.append(pred_ascii)
                all_label_ascii.append(label_ascii)
                
                # Architecture tokens only (0-23)
                pred_arch_only = pred_no_pad[(pred_no_pad >= 0) & (pred_no_pad <= 23)]
                label_arch_only = label_no_pad[(label_no_pad >= 0) & (label_no_pad <= 23)]
                
                pred_ascii_arch = " ".join(map(str, pred_arch_only)) if len(pred_arch_only) > 0 else ""
                label_ascii_arch = " ".join(map(str, label_arch_only)) if len(label_arch_only) > 0 else ""
                
                all_pred_ascii_arch_only.append(pred_ascii_arch)
                all_label_ascii_arch_only.append(label_ascii_arch)
                
                # 디버깅
                if debug_count < debug_samples:
                    print(f"Tokens→ASCII: {pred_no_pad[:5]} → '{pred_ascii[:20]}...'")
                    print(f"Split check: {len(pred_no_pad)} tokens → {len(pred_ascii.split())} strings")
                    debug_count += 1
            else:
                # Handle empty sequences - add empty strings to maintain alignment
                all_pred_ascii.append("")
                all_label_ascii.append("")
                all_pred_ascii_arch_only.append("")
                all_label_ascii_arch_only.append("")
                
        except Exception as e:
            print(f"Error processing sample for corpus metrics: {e}")
            # Add empty strings to maintain alignment
            all_pred_ascii.append("")
            all_label_ascii.append("")
            all_pred_ascii_arch_only.append("")
            all_label_ascii_arch_only.append("")
    
    # 아키텍처 토큰 통계 출력
    non_empty_pred_arch = [s for s in all_pred_ascii_arch_only if s]
    non_empty_label_arch = [s for s in all_label_ascii_arch_only if s]
    
    print(f"\n=== ARCHITECTURE TOKENS STATISTICS ===")
    print(f"Total samples: {len(all_pred_ascii_arch_only)}")
    print(f"Non-empty pred arch sequences: {len(non_empty_pred_arch)}")
    print(f"Non-empty label arch sequences: {len(non_empty_label_arch)}")
    
    if len(non_empty_pred_arch) > 0:
        print(f"Sample pred arch sequences: {non_empty_pred_arch[:3]}")
    if len(non_empty_label_arch) > 0:
        print(f"Sample label arch sequences: {non_empty_label_arch[:3]}")
    
    # Compute corpus-level BLEU (full sequence)
    if all_pred_ascii and all_label_ascii:
        corpus_bleu_result = sacrebleu.corpus_bleu(
            hypotheses=all_pred_ascii, 
            references=[[ref] for ref in all_label_ascii]
        )
        corpus_bleu_score = corpus_bleu_result.score
    else:
        corpus_bleu_score = 0.0
    
    # Compute corpus-level BLEU (architecture tokens only)
    if all_pred_ascii_arch_only and all_label_ascii_arch_only:
        # 빈 문자열 제거
        filtered_pred_arch = [s for s in all_pred_ascii_arch_only if s.strip()]
        filtered_label_arch = [s for s in all_label_ascii_arch_only if s.strip()]
        
        print(f"Filtered arch sequences - pred: {len(filtered_pred_arch)}, label: {len(filtered_label_arch)}")
        
        if filtered_pred_arch and filtered_label_arch:
            corpus_bleu_arch_result = sacrebleu.corpus_bleu(
                hypotheses=filtered_pred_arch,
                references=[[ref] for ref in filtered_label_arch]
            )
            corpus_bleu_arch_score = corpus_bleu_arch_result.score
        else:
            corpus_bleu_arch_score = 0.0
            print("No valid architecture sequences found for BLEU calculation!")
    else:
        corpus_bleu_arch_score = 0.0
    
    # Fix: Use the correct rouge_score function
    # Compute ROUGE scores (full sequence)
    if all_pred_ascii and all_label_ascii:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(all_pred_ascii, all_label_ascii):
                if pred.strip() and ref.strip():  # Only score non-empty sequences
                    scores = scorer.score(ref, pred)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
            
            rouge1_score = np.mean(rouge1_scores) if rouge1_scores else 0.0
            rouge2_score = np.mean(rouge2_scores) if rouge2_scores else 0.0
            rougeL_score = np.mean(rougeL_scores) if rougeL_scores else 0.0
            
        except Exception as e:
            print(f"Error computing ROUGE scores: {e}")
            rouge1_score = rouge2_score = rougeL_score = 0.0
    else:
        rouge1_score = rouge2_score = rougeL_score = 0.0
    
    # Compute ROUGE scores (architecture tokens only)
    if all_pred_ascii_arch_only and all_label_ascii_arch_only:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_arch_scores = []
            rouge2_arch_scores = []
            rougeL_arch_scores = []
            
            for pred, ref in zip(all_pred_ascii_arch_only, all_label_ascii_arch_only):
                if pred.strip() and ref.strip():  # Only score non-empty sequences
                    scores = scorer.score(ref, pred)
                    rouge1_arch_scores.append(scores['rouge1'].fmeasure)
                    rouge2_arch_scores.append(scores['rouge2'].fmeasure)
                    rougeL_arch_scores.append(scores['rougeL'].fmeasure)
            
            rouge1_arch_score = np.mean(rouge1_arch_scores) if rouge1_arch_scores else 0.0
            rouge2_arch_score = np.mean(rouge2_arch_scores) if rouge2_arch_scores else 0.0
            rougeL_arch_score = np.mean(rougeL_arch_scores) if rougeL_arch_scores else 0.0
            
        except Exception as e:
            print(f"Error computing ROUGE architecture scores: {e}")
            rouge1_arch_score = rouge2_arch_score = rougeL_arch_score = 0.0
    else:
        rouge1_arch_score = rouge2_arch_score = rougeL_arch_score = 0.0
    
    # Compute averages
    metrics = {
        'bleu_full': corpus_bleu_score,
        'bleu_arch': corpus_bleu_arch_score,
        'rouge1_full': rouge1_score,
        'rouge2_full': rouge2_score,
        'rougeL_full': rougeL_score,
        'rouge1_arch': rouge1_arch_score,
        'rouge2_arch': rouge2_arch_score,
        'rougeL_arch': rougeL_arch_score,
        'bleu': corpus_bleu_score  # Keep original for compatibility
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
        gt_folder = os.path.join(benchmark_folder, "gt_quantized")
        gt_raw_folder = os.path.join(benchmark_folder, "gt_raw")
        os.makedirs(pred_folder, exist_ok=True)
        os.makedirs(gt_folder, exist_ok=True)
        os.makedirs(gt_raw_folder, exist_ok=True)
        os.makedirs(benchmark_folder, exist_ok=True)  # Ensure benchmark folder exists
    
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
    
    # 추론 실행 (accelerator와 benchmark_folder를 파라미터로 전달)
    # It should return squenence start from "0", without SOS and METADATA tokens and end with EOS token
    predictions, labels, gt_plant_vecs = evaluate_model_with_accelerator(
        model, 
        dataloader, 
        accelerator,  # 동일한 accelerator 사용
        debug,
        benchmark_folder,  # Pass benchmark folder
    )
    
    # 메인 프로세스에서만 후처리
    if accelerator.is_main_process:
        print("Converting predictions to plant vectors and saving XMLs...")
        for idx, (pred, label, gt_plant_vec) in enumerate(tqdm(zip(predictions, labels, gt_plant_vecs), 
                                                                desc="Saving XMLs", 
                                                                total=len(predictions))):
            try:
                # Save predicted XML - convert predicted tokens to plant vector
                pred_tokens = pred[pred != PAD_TOKEN]  # Remove PAD tokens
                est_plant_vec = token2vec(pred_tokens)
                
                if est_plant_vec:
                    pred_xml = vec2xml(est_plant_vec, plant_id=idx)
                    pred_xml = recursive_to_linked(pred_xml)
                    pred_xml_str = pretty_print_xml(pred_xml)
                    
                    pred_xml_path = os.path.join(pred_folder, f"plant_{idx:04d}.xml")
                    with open(pred_xml_path, "w") as f:
                        f.write(pred_xml_str)
                
                # Save ground truth XML - convert ground truth tokens to plant vector
                gt_tokens = label
                gt_tokens = gt_tokens[gt_tokens != PAD_TOKEN]  # Remove PAD tokens
                gt_plant_vec_from_tokens = token2vec(gt_tokens)
                
                if gt_plant_vec_from_tokens:
                    gt_xml = vec2xml(gt_plant_vec_from_tokens, plant_id=idx)
                    gt_xml = recursive_to_linked(gt_xml)
                    gt_xml_str = pretty_print_xml(gt_xml)
                    
                    gt_xml_path = os.path.join(gt_folder, f"plant_{idx:04d}.xml")
                    with open(gt_xml_path, "w") as f:
                        f.write(gt_xml_str)


                if len(gt_plant_vec) > 0:
                    gt_xml = vec2xml(gt_plant_vec, plant_id=idx)
                    gt_xml = recursive_to_linked(gt_xml)
                    gt_xml_str = pretty_print_xml(gt_xml)
                    
                    gt_xml_path = os.path.join(gt_raw_folder, f"plant_{idx:04d}.xml")
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
        print("FULL SEQUENCE METRICS:")
        print(f"  BLEU: {metrics['bleu_full']:.4f}")
        print(f"  ROUGE-1: {metrics['rouge1_full']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2_full']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL_full']:.4f}")
        print("\nARCHITECTURE TOKENS ONLY (0-23):")
        print(f"  BLEU: {metrics['bleu_arch']:.4f}")
        print(f"  ROUGE-1: {metrics['rouge1_arch']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2_arch']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL_arch']:.4f}")
        
        print(f"\nXML files saved to:")
        print(f"  Predictions: {os.path.abspath(pred_folder)}")
        print(f"  Ground Truth: {os.path.abspath(gt_folder)}")
        
        # Save results to log file
        with open(log_path, "w") as log_file:
            log_file.write("EVALUATION RESULTS\n")
            log_file.write("="*50 + "\n")
            log_file.write("FULL SEQUENCE METRICS:\n")
            log_file.write(f"  BLEU: {metrics['bleu_full']:.4f}\n")
            log_file.write(f"  ROUGE-1: {metrics['rouge1_full']:.4f}\n")
            log_file.write(f"  ROUGE-2: {metrics['rouge2_full']:.4f}\n")
            log_file.write(f"  ROUGE-L: {metrics['rougeL_full']:.4f}\n")
            log_file.write("\nARCHITECTURE TOKENS ONLY (0-23):\n")
            log_file.write(f"  BLEU: {metrics['bleu_arch']:.4f}\n")
            log_file.write(f"  ROUGE-1: {metrics['rouge1_arch']:.4f}\n")
            log_file.write(f"  ROUGE-2: {metrics['rouge2_arch']:.4f}\n")
            log_file.write(f"  ROUGE-L: {metrics['rougeL_arch']:.4f}\n")
            
            log_file.write(f"\nXML files saved to:\n")
            log_file.write(f"  Predictions: {os.path.abspath(pred_folder)}\n")
            log_file.write(f"  Ground Truth: {os.path.abspath(gt_folder)}\n")
            log_file.write(f"  Total samples processed: {len(predictions)}\n")
        
        return metrics
    else:
        # Return empty metrics for non-main processes
        return {}