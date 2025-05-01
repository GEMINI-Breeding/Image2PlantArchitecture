# %%
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel

import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from PIL import Image, ImageFile
from tqdm import tqdm
import xml.etree.ElementTree as ET


import re

# Add ../ as a directory to import from
import sys
sys.path.append('../')

from plant_dataset import load_sideview_images
from image_process import process_leaf_image
from plant_tokenizer import vec2token, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE
from string_to_xml_to_vec import xml2vec, linked_to_recursive
from plant_dataset import PlantDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
from transformers import VisionEncoderDecoderModel
from utils import model_summary
from transformers import AutoImageProcessor, AutoModel
if 0:
    checkpoint_path = "../log/20250411_Curriculum10_Fulldata10/dinov2-small_224_TopView_gpt2/results"
    side_view = False
    image_size = 224
else:
    checkpoint_path = "../log/20250411_Curriculum10_Fulldata10/dinov2-base_224_TopView_gpt2-medium/results"
    side_view = False
    image_size = 224
print(checkpoint_path)
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(device)
# Set the model to evaluation mode
model.eval()

# Try to load the image processor from the encoder's config name
encoder_name = model.encoder.config._name_or_path
image_processor = AutoImageProcessor.from_pretrained(encoder_name)

image_processor.crop_size['width'] = image_size
image_processor.crop_size['height'] = image_size
image_processor.size['shortest_edge'] = image_size

# %%
model_summary(model)

# %%
from torch.utils.data import random_split
# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Dataset 인스턴스 생성
if 1:
    growth_stages = ["01"]
else:
    growth_stages = None

dataset_path = "../data/2000_Plots_20241210_BetterQuantized"
dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                       process_leaf=True, image_size=image_size,
                       side_view=side_view,
                       preload=False, image_processor=image_processor, add_sos_token=False)

# Split the dataset into Train, Validation, and Test sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = int(0.1 * len(dataset))    # 10% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing

# Use random_split with the seed set above
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])



# %%
# Path to the program for re-rendering XML files
program_path = "../src/GenerateDataset/build"

import os
import subprocess
# Function to re-render a single XML file
def re_render_xml(output_path, filename, rotation=True, debug=False):
    image_name = filename.split("/")[-1].split(".")[0]
    os.environ["DISPLAY"] = ":11.0"
    command = f"cd {program_path} && ./main -h 1.0 -o {output_path} -name {image_name} -tile none -f {os.path.join(output_path, filename)}"
    if rotation:
        command += " -r"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if debug:
        print(result.stdout)
        print(result.stderr)
    return result


# %%
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN
from models.model import get_tgt_mask
from plant_dataset import PlantDataset, load_sideview_images
from image_process import process_leaf_image
from plantstring2model import plantstring2model
from string_to_xml_to_vec import vec2xml, recursive_to_linked
from plant_tokenizer import token2vec
import shutil
from string_to_xml_to_vec import string2vec, vec2string, vec2xml, pretty_print_xml

  
# Dataset 인스턴스 생성
if 1:
    growth_stages = ["01"]
else:
    growth_stages = None


dataset = PlantDataset(root_dir=dataset_path, stages=growth_stages, 
                       process_leaf=True, image_size=image_size,
                       side_view=side_view,
                       preload=False, image_processor=image_processor, add_sos_token=False)

# Split the dataset into Train, Validation, and Test sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = int(0.1 * len(dataset))    # 10% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing

# Use random_split with the seed set above
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Prepare the figure
n_figures = 5
fig, axes = plt.subplots(2, n_figures, figsize=(20, 8))

# Create temp folder
temp_folder = "temp"
shutil.rmtree(temp_folder, ignore_errors=True)
os.makedirs(temp_folder, exist_ok=True)



# %%
# Inference all the test images and save
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import platform
import pandas as pd
import re
from plant_tokenizer import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from plant_dataset import PlantDataset, load_sideview_images
from image_process import process_leaf_image
from plantstring2model import plantstring2model
from string_to_xml_to_vec import vec2xml, recursive_to_linked, pretty_print_xml
from plant_tokenizer import token2vec
import shutil
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from concurrent.futures import ThreadPoolExecutor

def parse_output(output):
    """Parse the output from the subprocess to extract plant parameters."""
    try:
        plant_height = float(re.search(r'Plant Height: ([\d.-]+)', output).group(1))
        stem_height = float(re.search(r'Stem Height: ([\d.-]+)', output).group(1))
        leaf_count = int(re.search(r'Leaf count: (\d+)', output).group(1))

        leaf_area_match = re.search(r'Leaf area: ([\d.eE+-]+)', output)
        if leaf_area_match:
            leaf_area_str = leaf_area_match.group(1)
            if leaf_area_str == '-' or np.isnan(float(leaf_area_str)) or np.isneginf(float(leaf_area_str)):
                leaf_area = -1
            else:
                leaf_area = float(leaf_area_str)
        else:
            leaf_area = -1
            
        leaf_inclination = list(map(float, re.findall(r'Leaf inclination: ([\d. -]+)', output)[0].split()))
        return {
            'Plant Height': plant_height, 
            'Stem Height': stem_height, 
            'Leaf Count': leaf_count,
            'Leaf Area': leaf_area,
            'Leaf Inclination': leaf_inclination
        }
    except Exception as e:
        print(f"Error parsing measurement output: {e}")
        return None

def calculate_vegetation_metrics(image):
    """Calculate ExG average and Vegetation Fraction from an image."""
    try:
        # Calculate ExG
        green = image[:, :, 1].astype(float)
        red = image[:, :, 2].astype(float)
        blue = image[:, :, 0].astype(float)
        exg = 2 * green - red - blue
        exg_avg = np.mean(exg)
        
        # Normalize Image for thresholding
        exg_norm = cv2.normalize(exg, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply Otsu's threshold to calculate Vegetation Fraction
        _, binary = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vegetation_fraction = np.sum(binary > 0) / binary.size
        
        return exg_avg, vegetation_fraction
    except Exception as e:
        print(f"Error calculating vegetation metrics: {e}")
        return 0.0, 0.0

def re_render_xml_with_measurements(output_path, xml_path, program_path, rotation=True, debug=False):
    """Render XML file and return plant measurements."""
    image_name = os.path.basename(xml_path).split(".")[0]
    
    # Set display for rendering
    if platform.system() != "Darwin":
        os.environ["DISPLAY"] = ":11.0"
        
    command = f"cd {program_path} && ./main -h 1.0 -o {output_path} -name {image_name} -tile none -f {xml_path}"
    if rotation:
        command += " -r"
        
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if debug:
        print(f"Command: {command}")
        print(result.stdout)
        print(result.stderr)
        
    # Parse measurements from output
    measurements = None
    if result.returncode == 0:
        measurements = parse_output(result.stdout)
        
    return measurements

# Create Dataset instance
print("Loading Dataset...")
plant_architecture_dataset = PlantDataset(
    root_dir=dataset_path, 
    stages=None,  # Use all growth stages 
    process_leaf=True, 
    image_size=image_size,
    side_view=side_view,
    preload=False, 
    image_processor=image_processor, 
    add_sos_token=False
)

# Split the dataset into Train, Validation, and Test sets
train_size = int(0.8 * len(plant_architecture_dataset))
val_size = int(0.1 * len(plant_architecture_dataset))
test_size = len(plant_architecture_dataset) - train_size - val_size
print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Use random_split with fixed seed for reproducibility
torch.manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    plant_architecture_dataset, [train_size, val_size, test_size]
)

# Create dataloader for test set
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create output folder
exp_name = checkpoint_path.split("/")[-2]
output_folder = checkpoint_path.replace("results","test")
output_temp_folder = os.path.join(output_folder, "temp")
program_path = "../src/GenerateDataset/build"

# shutil.rmtree(output_folder, ignore_errors=True)
# shutil.rmtree(output_temp_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_temp_folder, exist_ok=True)
os.makedirs(f"{output_folder}/xml", exist_ok=True)
os.makedirs(f"{output_folder}/images", exist_ok=True)

# Set model to evaluation mode
model.eval()
device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize data collection for measurements
measurement_data = []

# Check if csv file exists
csv_path = os.path.join(output_folder, 'plant_measurements_comparison.csv')
if os.path.exists(csv_path):
    # Load csv
    measurements_df = pd.DataFrame(measurement_data)
    measurement_data = list(measurements_df)

# Process test dataset
print("Generating predictions for test dataset...")
for idx, data in enumerate(tqdm(test_dataloader)):
    # Extract data
    image = data["pixel_values"]
    out = data["labels"]
    plant_info = data["plant_info"]
    
    # Ensure correct dimensions
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    out = torch.tensor(out).to(device)
    
    # Get ground truth
    ground_truth = out.squeeze(0).cpu().numpy()
    
    # Create entry for measurements
    entry = {'File': f"plant_{idx:04d}"}

    # Check if entry is already exist in measurements_df
    if 'measurements_df' in locals() and not measurements_df.empty and measurements_df['File'].eq(entry['File']).any():
        print(f"Entry for {entry['File']} already exists, skipping...")
        continue

    
    # Save input image and calculate vegetation metrics in one step
    input_img = image[0].permute(1, 2, 0).cpu().numpy()
    input_img = cv2.normalize(input_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Calculate vegetation metrics while we have the image in memory
    exg_avg, vegetation_fraction = calculate_vegetation_metrics(input_img)
    entry['ExG_Avg'] = exg_avg
    entry['Vegetation_Fraction'] = vegetation_fraction
    
    # Save the image
    cv2.imwrite(f"{output_folder}/images/plant_{idx:04d}_input.jpg", input_img)
    
    # Save ground truth XML
    try:
        plant_vec = token2vec(ground_truth[5:])  # Skip metadata tokens
        plant_xml = vec2xml(plant_vec)
        plant_xml = recursive_to_linked(plant_xml)
        plant_xml_str = pretty_print_xml(plant_xml)
        gt_xml_path = f"{output_folder}/xml/plant_{idx:04d}_gt.xml"
        with open(gt_xml_path, "w") as f:
            f.write(plant_xml_str)
            
        # Render and measure ground truth
        gt_measurements = re_render_xml_with_measurements(
            os.path.abspath(output_temp_folder), 
            os.path.abspath(gt_xml_path),
            program_path,
            rotation=True,
            debug=True,
        )
        print(gt_measurements)
        if gt_measurements:
            # Add ground truth measurements to entry
            for k, v in gt_measurements.items():
                entry[f'{k}_gt'] = v
                
        # Process the rendered ground truth image
        if side_view:
            gt_img, _ = load_sideview_images(output_temp_folder, f"plant_{idx:04d}_gt.jpeg", image_size, True)
        else:
            gt_img = cv2.imread(f"{output_temp_folder}/plant_{idx:04d}_gt.jpeg")
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
        # Save the processed ground truth image
        cv2.imwrite(f"{output_folder}/images/plant_{idx:04d}_gt_render.jpg", cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
            
    except Exception as e:
        print(f"Error processing ground truth for idx {idx}: {e}")
    
    # Generate prediction
    try:
        with torch.no_grad():
            plant_info_tensor = torch.tensor(plant_info, dtype=torch.long).to(device)
            result = model.generate(
                image,
                decoder_start_token_id=SOS_TOKEN,
                decoder_input_ids=plant_info_tensor,
                eos_token_id=EOS_TOKEN,
                pad_token_id=PAD_TOKEN,
                max_length=2500
            )
            result = result.squeeze().cpu().numpy()[6:]  # Skip SOS token and metadata
            # print(result)
        
    except Exception as e:
        print(f"Error generating prediction for idx {idx}: {e}")
        
    # Save prediction XML
    plant_vec = token2vec(result)
    plant_xml = vec2xml(plant_vec)
    plant_xml = recursive_to_linked(plant_xml)
    plant_xml_str = pretty_print_xml(plant_xml)
    pred_xml_path = f"{output_folder}/xml/plant_{idx:04d}_pred.xml"
    with open(pred_xml_path, "w") as f:
        f.write(plant_xml_str)
        
    # Render and measure prediction
    pred_measurements = re_render_xml_with_measurements(
        os.path.abspath(output_temp_folder), 
        os.path.abspath(pred_xml_path),
        program_path,
        rotation=True
    )
    
    if pred_measurements:
        # Add prediction measurements to entry
        for k, v in pred_measurements.items():
            entry[f'{k}_pred'] = v
            
    # Process the rendered prediction image
    if side_view:
        pred_img, _ = load_sideview_images(output_temp_folder, f"plant_{idx:04d}_pred.jpeg", image_size, True)
    else:
        pred_img = cv2.imread(f"{output_temp_folder}/plant_{idx:04d}_pred.jpeg")
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        
    # Save the processed prediction image
    cv2.imwrite(f"{output_folder}/images/plant_{idx:04d}_pred_render.jpg", cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        
    
    # Calculate difference metrics if we have both ground truth and prediction measurements
    if 'Plant Height_gt' in entry and 'Plant Height_pred' in entry:
        entry['Height_Diff'] = abs(entry['Plant Height_gt'] - entry['Plant Height_pred'])
        if entry['Plant Height_gt'] != 0:
            entry['Height_Rel_Diff'] = entry['Height_Diff'] / entry['Plant Height_gt'] * 100
    
    if 'Stem Height_gt' in entry and 'Stem Height_pred' in entry:
        entry['Stem_Height_Diff'] = abs(entry['Stem Height_gt'] - entry['Stem Height_pred'])
        if entry['Stem Height_gt'] != 0:
            entry['Stem_Height_Rel_Diff'] = entry['Stem_Height_Diff'] / entry['Stem Height_gt'] * 100
    
    if 'Leaf Count_gt' in entry and 'Leaf Count_pred' in entry:
        entry['Leaf_Count_Diff'] = abs(entry['Leaf Count_gt'] - entry['Leaf Count_pred'])
        if entry['Leaf Count_gt'] != 0:
            entry['Leaf_Count_Rel_Diff'] = entry['Leaf_Count_Diff'] / entry['Leaf Count_gt'] * 100
            
    if 'Leaf Area_gt' in entry and 'Leaf Area_pred' in entry and entry['Leaf Area_gt'] > 0:
        entry['Leaf_Area_Diff'] = abs(entry['Leaf Area_gt'] - entry['Leaf Area_pred'])
        entry['Leaf_Area_Rel_Diff'] = entry['Leaf_Area_Diff'] / entry['Leaf Area_gt'] * 100
    
    # Add entry to measurement data
    measurement_data.append(entry)

    # print(f"Inference complete. Results saved to {output_folder}")
    
    # Update CSV every 100 steps
    if idx % 10 == 0 or idx == len(test_dataloader) -1:
        # Create and save measurements DataFrame
        measurements_df = pd.DataFrame(measurement_data)
        measurements_df.to_csv(csv_path, index=False)

# Print summary statistics
print("\nSummary Statistics:")
if 'Height_Diff' in measurements_df.columns:
    print(f"Average Plant Height Difference: {measurements_df['Height_Diff'].mean():.2f}")
if 'Stem_Height_Diff' in measurements_df.columns:
    print(f"Average Stem Height Difference: {measurements_df['Stem_Height_Diff'].mean():.2f}")
if 'Leaf_Count_Diff' in measurements_df.columns:
    print(f"Average Leaf Count Difference: {measurements_df['Leaf_Count_Diff'].mean():.2f}")

if 'Height_Rel_Diff' in measurements_df.columns:
    print(f"Average Plant Height Relative Difference: {measurements_df['Height_Rel_Diff'].mean():.2f}%")
if 'Stem_Height_Rel_Diff' in measurements_df.columns:
    print(f"Average Stem Height Relative Difference: {measurements_df['Stem_Height_Rel_Diff'].mean():.2f}%")
if 'Leaf_Count_Rel_Diff' in measurements_df.columns:
    print(f"Average Leaf Count Relative Difference: {measurements_df['Leaf_Count_Rel_Diff'].mean():.2f}%")

print(f"Measurement comparison complete. Results saved to {csv_path}")

# %%
# Create comparison plots
if True:
    # Set up figure for comparison plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plant Height comparison
    if 'Plant Height_gt' in measurements_df.columns and 'Plant Height_pred' in measurements_df.columns:
        axs[0, 0].scatter(measurements_df['Plant Height_gt'], measurements_df['Plant Height_pred'], alpha=0.5)
        max_height = max(measurements_df['Plant Height_gt'].max(), measurements_df['Plant Height_pred'].max()) * 1.1
        axs[0, 0].plot([0, max_height], [0, max_height], 'r--')
        axs[0, 0].set_xlabel('Ground Truth Plant Height')
        axs[0, 0].set_ylabel('Predicted Plant Height')
        axs[0, 0].set_title('Plant Height Comparison')
    
    # Stem Height comparison
    if 'Stem Height_gt' in measurements_df.columns and 'Stem Height_pred' in measurements_df.columns:
        axs[0, 1].scatter(measurements_df['Stem Height_gt'], measurements_df['Stem Height_pred'], alpha=0.5)
        max_stem_height = max(measurements_df['Stem Height_gt'].max(), measurements_df['Stem Height_pred'].max()) * 1.1
        axs[0, 1].plot([0, max_stem_height], [0, max_stem_height], 'r--')
        axs[0, 1].set_xlabel('Ground Truth Stem Height')
        axs[0, 1].set_ylabel('Predicted Stem Height')
        axs[0, 1].set_title('Stem Height Comparison')
    
    # Leaf Count comparison
    if 'Leaf Count_gt' in measurements_df.columns and 'Leaf Count_pred' in measurements_df.columns:
        axs[1, 0].scatter(measurements_df['Leaf Count_gt'], measurements_df['Leaf Count_pred'], alpha=0.5)
        max_leaf_count = max(measurements_df['Leaf Count_gt'].max(), measurements_df['Leaf Count_pred'].max()) * 1.1
        axs[1, 0].plot([0, max_leaf_count], [0, max_leaf_count], 'r--')
        axs[1, 0].set_xlabel('Ground Truth Leaf Count')
        axs[1, 0].set_ylabel('Predicted Leaf Count')
        axs[1, 0].set_title('Leaf Count Comparison')
    
    # Leaf Area comparison
    if 'Leaf Area_gt' in measurements_df.columns and 'Leaf Area_pred' in measurements_df.columns:
        valid_leaf_area = measurements_df[(measurements_df['Leaf Area_gt'] > 0) & (measurements_df['Leaf Area_pred'] > 0)]
        if len(valid_leaf_area) > 0:
            axs[1, 1].scatter(valid_leaf_area['Leaf Area_gt'], valid_leaf_area['Leaf Area_pred'], alpha=0.5)
            max_leaf_area = max(valid_leaf_area['Leaf Area_gt'].max(), valid_leaf_area['Leaf Area_pred'].max()) * 1.1
            axs[1, 1].plot([0, max_leaf_area], [0, max_leaf_area], 'r--')
            axs[1, 1].set_xlabel('Ground Truth Leaf Area')
            axs[1, 1].set_ylabel('Predicted Leaf Area')
            axs[1, 1].set_title('Leaf Area Comparison')
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/measurement_comparison.png")
    
    print(f"Comparison plots saved to {output_folder}")

print(f"All processing complete for {output_folder}")