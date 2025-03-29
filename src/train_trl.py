import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import random_split
import argparse
from plant_dataset import PlantDataset, load_sideview_images
from transformers import VisionEncoderDecoderModel, AutoModel, AutoImageProcessor

from src.train_rl_if import ImageSimilarityRewardModel

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

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collator
    )

    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

    reward_model = ImageSimilarityRewardModel(
        encoder=model.encoder,
        image_processor=image_processor
    )

    training_args = GRPOConfig(output_dir=args.log_dir, logging_steps=10)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_model.compute_reward,
        train_dataset=train_dataset,
    )
    trainer.train()