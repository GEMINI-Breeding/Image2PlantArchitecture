from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

from image_process import process_leaf_image
from plant_tokenizer import SOS_token, EOS_token, PAD_token, vec2token, params_EOS_token_padded, params_SOS_token_padded
from string_to_xml_to_vec import string2vec, vec2string, vec2xml, pretty_print_xml


class PlantDataset(Dataset):
    def __init__(self, root_dir, plot=None, stages=None, transform=None, img_size=224, use_depth=True, preload=True, dry_run=False, process_leaf=False):

        self.root_dir = root_dir
        self.use_depth = use_depth          
        # images_path
        self.images_path = os.path.join(root_dir, 'images')
        self.depth_path = os.path.join(root_dir, 'depth')
        # plant_string_path
        self.plant_string_path = os.path.join(root_dir, 'plantstrings')
        self.preload = preload
        # Get list of images
        self.image_paths = os.listdir(self.images_path)
        self.depth_images = os.listdir(self.depth_path)

        # Get list of plant strings
        self.plant_strings = [x.replace('.jpeg', '.txt') for x in self.image_paths]

        # Sort the lists
        self.image_paths.sort()
        self.plant_strings.sort()
        self.depth_images.sort()

        # Filter with statges
        if stages:
            self.image_paths = [x for x in self.image_paths if x.split('_')[2] in stages]
            self.plant_strings = [x for x in self.plant_strings if x.split('_')[2] in stages]
            self.depth_images = [x for x in self.depth_images if x.split('_')[2] in stages]

        if plot:
            self.image_paths = [x for x in self.image_paths if x.split('_')[3] in plot]
            self.plant_strings = [x for x in self.plant_strings if x.split('_')[3] in plot]
            self.depth_images = [x for x in self.depth_images if x.split('_')[3] in plot]
                
        self.transform = transform

        self.process_leaf = process_leaf

        print(f"Total {len(self.image_paths)} images and plant strings loaded")
        
        if self.preload:
            # Pre-load data
            self.images = []
            self.out = []
            print("Pre-loading data")
            for i in tqdm(range(len(self.image_paths))):
                image, out, out_len = self.getitem(i)
                if image is not None:
                    self.images.append(image)
                    self.out.append(out)

    def __len__(self):
        return len(self.image_paths)
    
    def getitem(self, idx):
        # Load image
        try:
            image = Image.open(os.path.join(self.images_path, self.image_paths[idx]))
        except:
            print(f"Error loading {self.image_paths[idx]}")
            return None, None, None
        if self.process_leaf:
            # Preprocess image
            leaf_area, plant_width, plant_height, leaf_img, (x,y,w,h) = process_leaf_image(np.array(image), 
                                                                                normalize=True, debug=False, sqaure_crop=True)
            leaf_img = cv2.resize(leaf_img, (224, 224))
        else:
            leaf_img = cv2.resize(np.array(image), (224, 224))

        if self.use_depth:
            # Convert depth to grayscale
            depth = Image.open(os.path.join(self.depth_path, self.depth_images[idx]))
            depth = np.array(depth)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

            if self.process_leaf:
                # Crop the depth image
                depth = depth[y:y+h, x:x+w]
                
            # Normalize depth image to 0-255
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth = depth.astype(np.uint8)

            # Resize the images
            depth = cv2.resize(depth, (224, 224))

            # Add depth channel
            leaf_img = np.concatenate((leaf_img, depth[:, :, np.newaxis]), axis=2)

        image = leaf_img
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        
        # Load plant string
        with open(os.path.join(self.plant_string_path, self.plant_strings[idx]), 'r') as f:
            plant_string = f.read()
        
        vec = string2vec(plant_string)[0]

        # Tokenize the plant structure
        out = vec2token(vec)

            
        # Add SOS and EOS tokens
        out = np.concatenate(([params_SOS_token_padded], out, [params_EOS_token_padded]))

        return image, out, len(out)

    
    def __getitem__(self, idx):

        if self.preload:
            image = self.images[idx]
            out = self.out[idx]
            out_len = len(out)
        else:
            image, out, out_len = self.getitem(idx)

        return image, out, out_len
        
                
