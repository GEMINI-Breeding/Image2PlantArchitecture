import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoImageProcessor
from torch.utils.data import random_split, DataLoader
from plant_dataset import PlantDataset
from typing import Dict, Any, Optional, Tuple, List
import os

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
) -> Tuple[Any, Any, Any]:
    """
    Prepare and split the dataset for evaluation.
    
    Args:
        dataset_path: Path to the dataset
        image_processor: Image processor for the model
        image_size: Size of images
        growth_stages: List of growth stages to include, or None for all
        test_split: Fraction of data to use for testing
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Configure image processor
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size
    
    xml_files = os.listdir(os.path.join(dataset_path, "xml"))
    xml_files.sort()
    num_plots = int(xml_files[-1].split("_")[1]) + 1

    train_end = int(num_plots * (1-val_split-test_split))
    val_end = train_end + int(num_plots * val_split)
    test_end = min(num_plots, val_end + int(num_plots * test_split)) # Ensure total sums up to num_plots
    
    train_plots = [f"{plot:04d}" for plot in range(train_end)]
    val_plots = [f"{plot:04d}" for plot in range(train_end, val_end)]
    test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]


    test_dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                process_leaf=True, image_size=image_size,
                side_view=side_view,
                plot=test_plots,
                mode='val',
                preload=preload, image_processor=image_processor, add_sos_token=False)

    return test_dataset

def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run model inference on test dataset with batch processing.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing test samples
        device: Device to run inference on
        
    Returns:
        Tuple of (all_predictions, all_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare inputs
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results - handle full batches
            for pred, label in zip(predictions, labels):
                all_predictions.append(pred.cpu().numpy())
                all_labels.append(label.cpu().numpy())
    
    return all_predictions, all_labels


def compute_metrics(
    predictions: List[np.ndarray],
    labels: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model predictions.
    
    Args:
        predictions: List of model predictions
        labels: List of ground truth labels
        
    Returns:
        Dictionary of metric name to value
    """
    # Load metrics
    bleu_metric = evaluate.load('sacrebleu')
    f1_metric = evaluate.load('f1')
    accuracy_metric = evaluate.load('accuracy')
    
    # Initialize score lists
    bleu_scores = []
    f1_scores = []
    accuracy_scores = []
    
    # Calculate per-sample metrics
    for pred, label in zip(predictions, labels):
        # Convert to strings for BLEU
        pred_str = " ".join([str(x) for x in pred])
        label_str = " ".join([str(x) for x in label])
        
        # Calculate BLEU
        bleu_result = bleu_metric.compute(predictions=[pred_str], references=[[label_str]])
        bleu_scores.append(bleu_result['score'])
        
        # Calculate token-level metrics
        f1 = f1_metric.compute(predictions=pred, references=label, average="weighted")
        accuracy = accuracy_metric.compute(predictions=pred, references=label)
        
        f1_scores.append(f1['f1'])
        accuracy_scores.append(accuracy['accuracy'])
    
    # Compute averages
    return {
        'bleu': sum(bleu_scores) / len(bleu_scores),
        'f1': sum(f1_scores) / len(f1_scores),
        'accuracy': sum(accuracy_scores) / len(accuracy_scores)
    }


def calc_metric(
    model: torch.nn.Module, 
    test_dataset: PlantDataset, 
    log_path: str,
    collate_fn: Any,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Calculate metrics for a model on a given dataset.
    
    Args:
        model: The model to evaluate
        test_dataset: The dataset to evaluate on
        log_path: Path to save log results
        collate_fn: Custom collate function for DataLoader
        batch_size: Batch size for evaluation
        num_workers: Number of worker processes for DataLoader
        
    Returns:
        Dictionary of metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
        
    # Create DataLoader for batch processing
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Evaluate model with batch processing - remove extra parameters
    predictions, labels = evaluate_model(
        model, 
        dataloader, 
        device
    )
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels)
    # Print results
    for name, value in metrics.items():
        print(f"Average {name.upper()}: {value:.4f}")
    
    # Save results to log file
    with open(log_path, "w") as log_file:
        for name, value in metrics.items():
            log_file.write(f"Average {name.upper()}: {value:.4f}\n")
    
    return metrics