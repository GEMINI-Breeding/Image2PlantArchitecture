# %%
import os
import shutil
import sys
import subprocess

# %%
program_path = "src/GenerateDataset/build_release"

# Dataset Path
dataset_path = "data/generated_Nov14_20224"

# Clean the dataset directory
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)

# Create the dataset directory
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Create images directory
images_path = dataset_path + "/images"
if not os.path.exists(images_path):
    os.makedirs(images_path)

# Create xml files directory
xml_path = dataset_path + "/xml"
if not os.path.exists(xml_path):
    os.makedirs(xml_path)

# %%
output_path = "output"
if 1:
    n_iter = 1

    # Get absolute path
    output_path = os.path.abspath(output_path)

    # Remove and recreate output folder
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    # Generate dataset
    for i in range(n_iter):
        print("Generating image: ", i)
        seed = i
        image_name = f"cowpea_{i:04d}" 
        # Generate image 
        # Construct the command
        command = ""
        command += f"cd {program_path} && ./main " 
        command += f"-h 1.0 -o {output_path} -seed {seed} -name {image_name} -xml -tile black"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        # print("Command executed successfully")
        # print(result.stdout)  # Print the standard output
        pass
    else:
        print(result.stdout)  # Print the standard output
        print(result.stderr)  # Print the error output
        raise("Command failed")
        pass

# %%
# List jpg and xml files and move them to the dataset directory
for filename in os.listdir(output_path):
    if filename.endswith(".jpeg"):
        shutil.move(os.path.join(output_path, filename), images_path)
    elif filename.endswith(".xml"):
        shutil.move(os.path.join(output_path, filename), xml_path)

# %%
# Test loading the generated dataset
from plant_dataset import PlantDataset 
# Show some images
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from plant_dataset import collate_fn
transform = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
train_dataset = PlantDataset(dataset_path, load_depth=False, preload=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
import cv2
n = 5
for i in range(n):
    #image, vecs, _ = train_dataset[-i-1]
    image, vecs, _ = train_dataset[i]
    #image, vecs, _ = next(iter(train_loader))
    image = image.permute(1, 2, 0)
    image_rgb = image[:, :, :3]
    img = cv2.normalize(np.array(image_rgb.cpu()), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(5, 2, 2*i+1)
    plt.imshow(img)
    plt.title(f"Image {i}")
    plt.axis('off')

    if train_dataset.load_depth:
        image_depth = image[:, :, 3]
        img = cv2.normalize(np.array(image_depth.cpu()), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(img, cmap='gray')
        plt.title("Depth")
        plt.axis('off')

# %%



