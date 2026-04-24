"""
Paper Figure Generation Script - Journal Quality Figures
Generates high-quality side-by-side comparison figures for plant architecture predictions.
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
import shutil
import subprocess
from PIL import Image

from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, token2vec
from models.model import PlantArchitectureModel
from plant_dataset import PlantDataset, load_sideview_images
from image_process import process_leaf_image
from string_to_xml_to_vec import vec2xml, recursive_to_linked, pretty_print_xml
from transformers import AutoImageProcessor

# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
})

CACHE_DIR = "paper_figures_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def collate_fn(batch):
    """Collate function for batching samples."""
    images = [f['pixel_values'] for f in batch]
    plant_info = [f['plant_info'] for f in batch]
    out = [f['labels'] for f in batch]
    lens = [len(f['labels']) for f in batch]
    max_length = max(lens)
    out_padded = np.ones([len(out), max_length]) * PAD_TOKEN
    for i, seq in enumerate(out):
        out_padded[i, :len(seq)] = seq
    images = torch.stack(images)
    plant_info = np.array(plant_info)
    plant_info = torch.tensor(plant_info)
    out_tensor = torch.tensor(out_padded, dtype=torch.long)

    return {
        "pixel_values": images,
        "labels": out_tensor,
        "plant_info": plant_info
    }


program_path = os.path.join(os.path.dirname(__file__), "../CowpeaSimulator/build")


def re_render_xml(output_path, filename, rotation=True, debug=False, use_cache=True):
    """Re-render XML file to generate plant images."""
    os.environ["DISPLAY"] = ":1.0"
    image_name = filename.split("/")[-1].split(".")[0]
    
    if use_cache and os.path.exists(os.path.join(output_path, f"{image_name}.jpeg")):
        if debug:
            print(f"Cache hit: {image_name}.jpeg already exists. Skipping.")
        return None
        
    command = f"cd {program_path} && ./main -h 1.0 -o {output_path} -name {image_name} -tile none -f {os.path.join(output_path, filename)}"
    if rotation:
        command += " -r"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if debug:
        print(result.stdout)
        print(result.stderr)
    return result


def create_journal_figure(checkpoint_path, dataset_path, output_path="paper_figures_journal.png",
                          n_samples=5, growth_stages=None, use_cache=True):
    """
    Create a publication-quality figure with side-by-side GT vs Prediction comparison.
    
    Layout: Each row is a different growth stage
            Within each stage: Row of GTs, then Row of Predictions
    """
    if growth_stages is None:
        growth_stages = ["09", "19", "29", "39"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = PlantArchitectureModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16).to(device)
    model.eval()
    
    # Determine image size from checkpoint path
    if "224" in checkpoint_path:
        image_size = 224
    elif "448" in checkpoint_path:
        image_size = 448
    else:
        image_size = 224
    
    max_length = 4096 * 2
    is_sideview_model = "Sideview" in checkpoint_path
    
    # Setup image processor
    encoder_name = model.encoder.config._name_or_path
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size
    
    # Get test plots
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    xml_files = [f for f in os.listdir(os.path.join(dataset_path, "xml")) if f.endswith(".xml")]
    xml_files.sort()
    num_plots = int(xml_files[-1].split("_")[1]) + 1
    
    train_end = int(num_plots * train_ratio)
    val_end = train_end + int(num_plots * val_ratio)
    test_end = min(num_plots, val_end + int(num_plots * test_ratio))
    test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]
    
    # Create figure with professional layout
    n_stages = len(growth_stages)
    # Aggressively reduced figure height to eliminate white space
    fig = plt.figure(figsize=(10, 2.8 * n_stages + 2.5)) 
    # Increased top margin to avoid title occlusion and left margin for labels
    gs = GridSpec(n_stages, 1, figure=fig, hspace=0.1, left=0.15, right=0.98, top=0.88, bottom=0.02)
    
    # Main title
    fig.text(0.5, 0.98, "Cowpea Plant Architecture Prediction: Temporal Comparison", 
             ha='center', va='top', fontsize=24, fontweight='bold', color='#1a1a1a')
    
    # Subtitle or description
    fig.text(0.5, 0.955, "Comparing Ground Truth (GT) and Predicted architectures across growth stages", 
             ha='center', va='top', fontsize=16, style='italic', color='#555555')
    
    for stage_idx, stage in enumerate(growth_stages):
        print(f"Processing growth stage: {stage}")
        day = int(stage)
        
        # Create sub-grid for this stage: 2 rows (GT row, Prediction row)
        stage_gs = gs[stage_idx].subgridspec(2, 1, hspace=0.02)
        gt_gs = stage_gs[0].subgridspec(1, n_samples, wspace=0.05)
        pred_gs = stage_gs[1].subgridspec(1, n_samples, wspace=0.05)
        
        pos = gs[stage_idx].get_position(fig)
        
        # Load dataset for this stage
        test_dataset = PlantDataset(
            root_dir=dataset_path, stages=[stage],
            process_leaf=True, image_size=image_size,
            side_view=is_sideview_model, plot=test_plots,
            mode='test', preload=False, 
            image_processor=image_processor, add_sos_token=False
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=n_samples, 
            shuffle=False, collate_fn=collate_fn
        )
        
        # Get first batch
        batch_data = next(iter(test_dataloader))
        batch_images = batch_data["pixel_values"].to(device, dtype=torch.float16)
        batch_labels = batch_data["labels"].to(device)
        batch_plant_info = batch_data["plant_info"].to(device)
        
        # Generate predictions with caching
        checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path))
        pred_cache_file = os.path.join(CACHE_DIR, f"pred_{checkpoint_name}_stage{stage}_{n_samples}s.npy")
        
        if use_cache and os.path.exists(pred_cache_file):
            print(f"  Using cached predictions for stage {stage}")
            batch_result = np.load(pred_cache_file)
        else:
            print(f"  Running inference for stage {stage}...")
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    batch_result = model.generate(
                        batch_images,
                        decoder_start_token_id=SOS_TOKEN,
                        decoder_input_ids=batch_plant_info,
                        eos_token_id=EOS_TOKEN,
                        pad_token_id=PAD_TOKEN,
                        max_length=max_length,
                        use_cache=True,
                    )
                batch_result = batch_result.cpu().numpy()
            if use_cache:
                np.save(pred_cache_file, batch_result)
        
        # Process each sample
        for i in range(n_samples):
            ax_gt = fig.add_subplot(gt_gs[i])
            ax_pred = fig.add_subplot(pred_gs[i])
            
            # Add sample labels ONLY on the first growth stage
            if stage_idx == 0:
                ax_gt.set_title(f"Sample {i + 1}", fontsize=14, fontweight='bold', pad=15)
            
            out = batch_labels[i]
            ground_truth = out.cpu().numpy()
            
            # Generate GT visualization (always use sideview synthesis for the figure)
            plant_xml_file_name = f"{CACHE_DIR}/plant_{stage}_{i}_gt.xml"
            plant_jpeg_file_name = plant_xml_file_name.replace(".xml", ".jpeg")
            
            if not (use_cache and os.path.exists(plant_jpeg_file_name)):
                plant_vec = token2vec(ground_truth[5:])
                plant_xml = vec2xml(plant_vec)
                plant_xml = recursive_to_linked(plant_xml)
                plant_xml_str = pretty_print_xml(plant_xml)
                with open(plant_xml_file_name, "w") as f:
                    f.write(plant_xml_str)
                re_render_xml(os.path.abspath(CACHE_DIR), os.path.abspath(plant_xml_file_name), rotation=True, use_cache=use_cache)
            
            gt_img, _ = load_sideview_images(
                CACHE_DIR, 
                plant_jpeg_file_name, 
                image_size, True
            )
            
            # Generate Prediction visualization
            result = batch_result[i]
            plant_xml_file_name_est = f"{CACHE_DIR}/plant_{stage}_{i}_est.xml"
            plant_jpeg_file_name_est = plant_xml_file_name_est.replace(".xml", ".jpeg")
            
            if not (use_cache and os.path.exists(plant_jpeg_file_name_est)):
                plant_vec = token2vec(result[6:])
                plant_xml = vec2xml(plant_vec)
                plant_xml = recursive_to_linked(plant_xml)
                plant_xml_str = pretty_print_xml(plant_xml)
                with open(plant_xml_file_name_est, "w") as f:
                    f.write(plant_xml_str)
                re_render_xml(os.path.abspath(CACHE_DIR), os.path.abspath(plant_xml_file_name_est), rotation=True, use_cache=use_cache)
            
            pred_img, _ = load_sideview_images(
                CACHE_DIR, 
                plant_jpeg_file_name_est, 
                image_size, True
            )
            
            # Display images
            ax_gt.imshow(gt_img)
            ax_pred.imshow(pred_img)
            
            # Remove axes but keep borders
            for ax in [ax_gt, ax_pred]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
            
            # Professional Slate Grey borders for GT for better definition
            for spine in ax_gt.spines.values():
                spine.set_color('#34495e') # Slate grey
                spine.set_linewidth(1.2)
            
            # Prominent MAROON-RED borders for Prediction
            for spine in ax_pred.spines.values():
                spine.set_color('#8b0000') # Maroon red
                spine.set_linewidth(2.0)
        
        # Add stage label on the left side
        fig.text(0.02, pos.y0 + pos.height/2,
                 f"Day {day}", fontsize=22, fontweight='bold',
                 rotation=90, va='center', ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfdfd', edgecolor='#cccccc', alpha=1.0))
    
    # Add row labels (Ground Truth / Prediction) for each stage
    for stage_idx in range(n_stages):
        pos = gs[stage_idx].get_position(fig)
        # Rotated 90 degrees for a more compact and professional scientific look
        fig.text(0.10, pos.y0 + pos.height*0.75, "Ground Truth", fontsize=11, 
                ha='center', va='center', color='#34495e', fontweight='bold', rotation=90)
        fig.text(0.10, pos.y0 + pos.height*0.25, "Prediction", fontsize=11, 
                ha='center', va='center', color='#8b0000', fontweight='bold', rotation=90)
    
    # Save figure with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved figure to {output_path}")
    plt.close()


def create_comparison_figure(checkpoint_path, dataset_path, output_path="paper_figures_comparison.png"):
    """
    Alternative layout: 2x2 grid showing one sample per growth stage
    with larger, more detailed views.
    """
    growth_stages = ["09", "19", "29", "39"]
    n_samples = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PlantArchitectureModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16).to(device)
    model.eval()
    
    if "224" in checkpoint_path:
        image_size = 224
    elif "448" in checkpoint_path:
        image_size = 448
    else:
        image_size = 224
    
    max_length = 4096 * 2
    is_sideview_model = "Sideview" in checkpoint_path
    
    encoder_name = model.encoder.config._name_or_path
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    image_processor.crop_size['width'] = image_size
    image_processor.crop_size['height'] = image_size
    image_processor.size['shortest_edge'] = image_size
    
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    xml_files = [f for f in os.listdir(os.path.join(dataset_path, "xml")) if f.endswith(".xml")]
    xml_files.sort()
    num_plots = int(xml_files[-1].split("_")[1]) + 1
    
    train_end = int(num_plots * train_ratio)
    val_end = train_end + int(num_plots * val_ratio)
    test_end = min(num_plots, val_end + int(num_plots * test_ratio))
    test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(10, 18))
    fig.suptitle("Plant Architecture Growth Progression: Ground Truth vs. Prediction", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add column labels
    fig.text(0.28, 0.96, "Ground Truth", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.78, 0.96, "Generated", ha='center', fontsize=12, fontweight='bold')
    
    for stage_idx, stage in enumerate(growth_stages):
        print(f"Processing growth stage: {stage}")
        day = int(stage)
        
        ax_gt = axes[stage_idx, 0]
        ax_pred = axes[stage_idx, 1]
        
        # Load dataset
        test_dataset = PlantDataset(
            root_dir=dataset_path, stages=[stage],
            process_leaf=True, image_size=image_size,
            side_view=is_sideview_model, plot=test_plots,
            mode='test', preload=False,
            image_processor=image_processor, add_sos_token=False
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, collate_fn=collate_fn
        )
        
        batch_data = next(iter(test_dataloader))
        batch_images = batch_data["pixel_values"].to(device, dtype=torch.float16)
        batch_labels = batch_data["labels"].to(device)
        batch_plant_info = batch_data["plant_info"].to(device)
        
        # Generate predictions
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                batch_result = model.generate(
                    batch_images,
                    decoder_start_token_id=SOS_TOKEN,
                    decoder_input_ids=batch_plant_info,
                    eos_token_id=EOS_TOKEN,
                    pad_token_id=PAD_TOKEN,
                    max_length=max_length,
                    use_cache=True,
                )
            batch_result = batch_result.cpu().numpy()
        
        # Process
        out = batch_labels[0]
        ground_truth = out.cpu().numpy()
        result = batch_result[0]
        
        # Generate GT
        plant_xml_file_name = f"{CACHE_DIR}/comp_{stage}_gt.xml"
        plant_jpeg_file_name = plant_xml_file_name.replace(".xml", ".jpeg")
        
        if not os.path.exists(plant_jpeg_file_name):
            plant_vec = token2vec(ground_truth[5:])
            plant_xml = vec2xml(plant_vec)
            plant_xml = recursive_to_linked(plant_xml)
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)
            re_render_xml(os.path.abspath(CACHE_DIR), os.path.abspath(plant_xml_file_name), rotation=True)
        
        gt_img, _ = load_sideview_images(
            CACHE_DIR,
            plant_jpeg_file_name,
            image_size, True
        )
        
        # Generate Pred
        plant_xml_file_name_est = f"{CACHE_DIR}/comp_{stage}_est.xml"
        plant_jpeg_file_name_est = plant_xml_file_name_est.replace(".xml", ".jpeg")
        
        if not os.path.exists(plant_jpeg_file_name_est):
            plant_vec = token2vec(result[6:])
            plant_xml = vec2xml(plant_vec)
            plant_xml = recursive_to_linked(plant_xml)
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name_est, "w") as f:
                f.write(plant_xml_str)
            re_render_xml(os.path.abspath(CACHE_DIR), os.path.abspath(plant_xml_file_name_est), rotation=True)
        
        pred_img, _ = load_sideview_images(
            CACHE_DIR,
            plant_jpeg_file_name_est,
            image_size, True
        )
        
        # Display
        ax_gt.imshow(gt_img)
        ax_pred.imshow(pred_img)
        
        ax_gt.axis('off')
        ax_pred.axis('off')
        
        # Add day label
        ax_gt.set_ylabel(f"Day {day}", fontsize=12, fontweight='bold', 
                        rotation=0, ha='right', va='center', labelpad=15)
        
        # Add subtle borders
        for ax in [ax_gt, ax_pred]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved figure to {output_path}")
    plt.close()


def main():
    """Main function to generate figures."""
    dataset_path = "/home/lion397/datasets/GEMINI/plant_architecture/20250311_Sideview_40Days"
    checkpoint_path = "/home/lion397/codes/Image2PlantArchitecture/log/20250713_TrainOnFarm/dinov2-base_448_RGB_TopView_gpt2-medium/results"
    
    # Generate journal-quality figure with side-by-side comparison
    print("\n=== Generating journal figure (5 samples) ===")
    create_journal_figure(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_path="paper_figures_journal.png",
        n_samples=5,
        growth_stages=["09", "19", "29", "39"]
    )
    
    # Also generate the alternative 2x2 comparison figure
    print("\n=== Generating comparison figure ===")
    create_comparison_figure(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_path="paper_figures_comparison.png"
    )


if __name__ == "__main__":
    main()
