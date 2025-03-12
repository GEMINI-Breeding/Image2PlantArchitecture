import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from PIL import Image, ImageFile
from tqdm import tqdm

# Add . as a directory to import from
import sys
# Get the parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from image_process import process_leaf_image
from plant_tokenizer import SOS_token, EOS_token, PAD_token, EOS_vec_padded, SOS_vec_padded
from string_to_xml_to_vec import string2vec, vec2string, vec2xml, pretty_print_xml, xml2vec, linked_to_recursive
import xml.etree.ElementTree as ET
from plant_tokenizer import vec2token as vec2token
import re
import joblib

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_sideview_images(images_dir, image_file_name, img_size, process_leaf):

    # Load side view images and combine them into 2x2
    # angles = [0, 90, 180, 270]
    angles = [-1, 0, 120, 240]
    image_name = image_file_name.split("/")[-1].split(".")[0]
    # Make a empty image
    total_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    total_plant_info = []
    for i, angle in enumerate(angles):
        try:
            if angle == -1:
                image = Image.open(os.path.join(images_dir, f"{image_name}.jpeg"))
            else:
                image = Image.open(os.path.join(images_dir, f"{image_name}_{angle}.jpeg"))
        except:
            print(f"Error loading {image_file_name}")
            return None, None, None
        leaf_area, plant_width, plant_height, processed_img, (x,y,w,h) = process_leaf_image(np.array(image), 
                                                                            normalize=True, debug=False, sqaure_crop=True)
        plant_info = [leaf_area, plant_width, plant_height]
        if process_leaf:
            # Preprocess image
            leaf_img = cv2.resize(processed_img, (img_size//2, img_size//2))
        else:
            leaf_img = cv2.resize(np.array(image), (img_size//2, img_size//2))

        # Add to the empty image
        if i == 0:
            total_img[:img_size//2, :img_size//2] = leaf_img
        elif i == 1:
            total_img[:img_size//2, img_size//2:] = leaf_img
        elif i == 2:
            total_img[img_size//2:, :img_size//2] = leaf_img
        elif i == 3:
            total_img[img_size//2:, img_size//2:] = leaf_img
        
        total_plant_info.append(plant_info)
        # Debug
        # cv2.imshow("Total", total_img)
        # cv2.waitKey(0)
        
    leaf_img = total_img
    # Average the plant info
    plant_info = np.mean(total_plant_info, axis=0)

    return leaf_img, plant_info



class PlantDataset(Dataset):
    def __init__(self, root_dir, plot=None, stages=None, transform=None, 
                 image_size=224, load_depth=False, preload=True, side_view=False,
                 process_leaf=False):

        self.root_dir = root_dir
        self.load_depth = load_depth          
        # images_path
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(root_dir, 'images')
        self.depth_image_dir = os.path.join(root_dir, 'depth')
        # plant_string_path
        self.plant_xml_dir = os.path.join(root_dir, 'xml')
        self.preload = preload
        # Get list of plant strings
        self.plant_xml_files = os.listdir(self.plant_xml_dir)
        # Get list of images
        self.image_files = [x.replace('.xml', '.jpeg') for x in self.plant_xml_files]
        if load_depth:
            self.depth_images = os.listdir(self.depth_image_dir)
            self.depth_images.sort()

        # Sort the lists
        self.image_files.sort()
        self.plant_xml_files.sort()

        self.img_size = image_size
        # Filter with statges
        # Regular expression to extract plot and day numbers
        pattern = r"cowpea_(\d+)_day_(\d+)"
        if stages:
            self.image_files = [x for x in self.image_files if re.match(pattern, x).group(2) in stages]
            self.plant_xml_files = [x for x in self.plant_xml_files if re.match(pattern, x).group(2) in stages]
            if self.load_depth:
                self.depth_images = [x for x in self.depth_images if re.match(pattern, x).group(2) in stages]

        if plot:
            self.image_files = [x for x in self.image_files if re.match(pattern, x).group(1) in plot]
            self.plant_xml_files = [x for x in self.plant_xml_files if re.match(pattern, x).group(1) in plot]
            if self.load_depth:
                self.depth_images = [x for x in self.depth_images if re.match(pattern, x).group(1) in plot]
                
        self.transform = transform

        self.process_leaf = process_leaf
        self.side_view = side_view
        self.plant_string_raw = ""
        
        print(f"Total {len(self.image_files)} images and plant strings loaded")
        
        # self.param_scaler = joblib.load(os.path.join(self.current_script_dir,'scaler.pkl'))

        if self.preload:
            # Pre-load data
            self.images = []
            self.vec = []
            self.plant_infos = []
            print("Pre-loading data")
            for i in tqdm(range(len(self.image_files))):
                image, plant_info, vec = self.getitem(i)
                if image is not None:
                    self.images.append(image)
                    self.vec.append(vec)
                    self.plant_infos.append(plant_info)

    def __len__(self):
        return len(self.image_files)
    
    def getitem(self, idx):
        # Load image
        if self.side_view:
            try:
                leaf_img, plant_info = load_sideview_images(self.image_dir, self.image_files[idx], self.img_size, process_leaf=self.process_leaf)
            except:
                print(f"Error loading {self.image_files[idx]}")
                return None, None, None          
        else:
            try:
                image = Image.open(os.path.join(self.image_dir, self.image_files[idx]))
                # Convert to numpy array
                image = np.array(image)
            except:
                print(f"Error loading {self.image_files[idx]}")
                return None, None, None
            
            leaf_area, plant_width, plant_height, processed_img, (x,y,w,h) = process_leaf_image(np.array(image), 
                                                                                normalize=True, debug=False, sqaure_crop=True)
            plant_info = [leaf_area, plant_width, plant_height]
            if self.process_leaf:
                # Preprocess image
                leaf_img = cv2.resize(processed_img, (self.img_size, self.img_size))
            else:
                leaf_img = cv2.resize(np.array(image), (self.img_size, self.img_size))

            if self.load_depth:
                # Convert depth to grayscale
                depth = Image.open(os.path.join(self.depth_image_dir, self.depth_images[idx]))
                depth = np.array(depth)
                # Convert to grayscale if not already
                if len(depth.shape) > 2:
                    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

                if self.process_leaf:
                    # Crop the depth image
                    depth = depth[y:y+h, x:x+w]
                    
                # Normalize depth image to 0-255
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth = depth.astype(np.uint8)

                # Resize the images
                depth = cv2.resize(depth, (self.img_size, self.img_size))

                # Add depth channel
                leaf_img = np.concatenate((leaf_img, depth[:, :, np.newaxis]), axis=2)

    

        # Load XML file
        # Load and parse the XML file
        try:
            xml_file = os.path.join(self.plant_xml_dir, self.plant_xml_files[idx])
            tree = ET.parse(xml_file)
        except Exception as e:
            print(f"Error Reading {xml_file}")
            print(e)

        # Get the root element
        root = tree.getroot()

        root = linked_to_recursive(root)
        plant_array = []
        xml2vec(root[0], plant_array) # Assume single plant


        return leaf_img, plant_info, plant_array
    
    def __getitem__(self, idx):

        if self.preload:
            image = self.images[idx]
            vec = self.vec[idx]
            plant_info = self.plant_infos[idx]
        else:
            image, plant_info, vec = self.getitem(idx)
            

        if self.transform:
            # Check if the image is a PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = self.transform(image)

            if isinstance(image, Image.Image):
                image = np.array(image)

        if vec:
            # Tokenize the plant structure
            out = vec2token(vec)

            if 0:
                # Scale the token
                out[:,1:] = self.param_scaler.transform(out[:,1:])
                
            # Add SOS and EOS tokens
            out = np.concatenate(([SOS_vec_padded], out, [EOS_vec_padded]))
            out_len = len(out)
        else:
            out = None
            out_len = 0

        # Conver to tensor
        image = torch.tensor(image)
        # Permute the image tensor
        image = image.permute(2, 0, 1)

        return image, plant_info, out, out_len



if __name__ == "__main__":
    # Load plant dataset
    dataset = PlantDataset("data/generated_Nov22_2024", preload=False)

    # iterate over samples
    max_len = -1
    max_vec = []
    for i in range(len(dataset)):
        image, vec, vec_len = dataset[i]
        if max_len < vec_len:
            max_vec = vec
            max_len = vec_len

    print(max_len)