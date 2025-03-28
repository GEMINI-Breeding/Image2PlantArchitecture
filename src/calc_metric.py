import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoImageProcessor
from torch.utils.data import random_split, DataLoader
from plant_dataset import PlantDataset
from typing import Dict, Any, Optional, Tuple, List


def prepare_dataset(
    dataset_path: str,
    image_processor: Any,
    image_size: int = 448,
    side_view: bool = False,
    growth_stages: Optional[List[str]] = None,
    test_split: float = 0.1,
    val_split: float = 0.1,
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
    
    # Create dataset
    dataset = PlantDataset(
        root_dir=dataset_path, 
        stages=growth_stages,
        process_leaf=True, 
        image_size=image_size,
        side_view=side_view,
        preload=False, 
        image_processor=image_processor, 
        add_sos_token=False
    )
    
    # Split dataset
    torch.manual_seed(seed)
    train_size = int((1.0 - test_split - val_split) * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    return random_split(dataset, [train_size, val_size, test_size])


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: Any,
    device: torch.device
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run model inference on test dataset.
    
    Args:
        model: The model to evaluate
        test_dataset: Dataset containing test samples
        device: Device to run inference on
        
    Returns:
        Tuple of (all_predictions, all_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataset, desc="Evaluating"):
            # Prepare inputs
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device)
            labels = torch.tensor(batch["labels"]).unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_predictions.append(predictions.squeeze().cpu().numpy())
            all_labels.append(labels.squeeze().cpu().numpy())

            # # Dry run
            # if len(all_labels) > 10:
            #     break
    
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


def calc_metric(model: torch.nn.Module, dataset_path: str, image_size: int = 448, side_view=False) -> Dict[str, float]:
    """
    Calculate metrics for a model on a given dataset.
    
    Args:
        model: The model to evaluate
        dataset_path: Path to the dataset
        image_size: Size of images
        
    Returns:
        Dictionary of metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get image processor
    encoder_name = model.encoder.config._name_or_path
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    
    # Prepare dataset
    _, _, test_dataset = prepare_dataset(
        dataset_path=dataset_path,
        image_processor=image_processor,
        image_size=image_size,
        side_view=side_view
    )
    
    # Evaluate model
    predictions, labels = evaluate_model(model, test_dataset, device)
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels)
    
    # Print results
    for name, value in metrics.items():
        print(f"Average {name.upper()}: {value:.4f}")
    
    return metrics