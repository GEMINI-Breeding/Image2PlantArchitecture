import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoImageProcessor
from torch.utils.data import random_split



def calc_metric(model, dataset_path):
    # Load the metrics
    bleu_metric = evaluate.load('sacrebleu')
    f1_metric = evaluate.load('f1')
    accuracy_metric = evaluate.load('accuracy')

    # Ensure the model is in evaluation mode
    model.eval()

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize lists to store predictions and labels
    all_predictions = []
    all_labels = []


    # Dataset 인스턴스 생성
    if 0:
        growth_stages = ["01"]
    else:
        growth_stages = None

    image_size = 448
    encoder_checkpoint = "facebook/dinov2-small"
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size

    dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                        process_leaf=True, image_size=image_size,
                        side_view=True,
                        preload=False, image_processor=image_processor, add_sos_token=False)

    # Split the dataset into Train, Validation, and Test sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing

    # Use random_split with the seed set above
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Evaluation loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataset, desc="Evaluating")):
            # Move inputs and labels to the device
            pixel_values = batch["pixel_values"].unsqueeze(0).to(model.device)
            labels = torch.tensor(batch["labels"]).unsqueeze(0).to(model.device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Store predictions and labels
            all_predictions.append(predictions.squeeze().cpu().numpy())
            all_labels.append(labels.squeeze().cpu().numpy())

            # # Dry run
            # if i > 10:
            #     break

    # Flatten predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    # Compute metrics
    f1 = f1_metric.compute(predictions=all_predictions, references=all_labels, average="weighted")
    print(f"F1 Score: {f1['f1']}")
    accuracy = accuracy_metric.compute(predictions=all_predictions, references=all_labels)
    print(f"Accuracy: {accuracy['accuracy']}")
    predicted_string = " ".join([f"{x }" for x in all_predictions])
    all_labels_string = " ".join([f"{x }" for x in all_labels])
    bleu = bleu_metric.compute(predictions=[predicted_string], references=[[all_labels_string]])
    print(f"BLEU: {bleu['score']}")

    # Print metrics