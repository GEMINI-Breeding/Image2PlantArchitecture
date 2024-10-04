import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from PIL import Image, ImageFile
from tqdm import tqdm

from image_process import process_leaf_image
from plant_tokenizer import SOS_token, EOS_token, PAD_token, params_EOS_token_padded, params_SOS_token_padded
from string_to_xml_to_vec import string2vec, vec2string, vec2xml, pretty_print_xml

from plant_tokenizer import vec2token_new as vec2token

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PlantDataset(Dataset):
    def __init__(self, root_dir, plot=None, stages=None, transform=None, 
                 image_size=224, load_depth=True, preload=True, 
                 dry_run=False, process_leaf=False):

        self.root_dir = root_dir
        self.load_depth = load_depth          
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
        self.plant_string_files = [x.replace('.jpeg', '.txt') for x in self.image_paths]

        # Sort the lists
        self.image_paths.sort()
        self.plant_string_files.sort()
        self.depth_images.sort()

        self.img_size = image_size
        # Filter with statges
        if stages:
            self.image_paths = [x for x in self.image_paths if x.split('_')[2] in stages]
            self.plant_string_files = [x for x in self.plant_string_files if x.split('_')[2] in stages]
            self.depth_images = [x for x in self.depth_images if x.split('_')[2] in stages]

        if plot:
            self.image_paths = [x for x in self.image_paths if x.split('_')[3] in plot]
            self.plant_string_files = [x for x in self.plant_string_files if x.split('_')[3] in plot]
            self.depth_images = [x for x in self.depth_images if x.split('_')[3] in plot]
                
        self.transform = transform

        self.process_leaf = process_leaf

        self.plant_string_raw = ""
        
        print(f"Total {len(self.image_paths)} images and plant strings loaded")
        
        if self.preload:
            # Pre-load data
            self.images = []
            self.vec = []
            print("Pre-loading data")
            for i in tqdm(range(len(self.image_paths))):
                image, vec = self.getitem(i)
                if image is not None:
                    self.images.append(image)
                    self.vec.append(vec)

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
            leaf_img = cv2.resize(leaf_img, (self.img_size, self.img_size))
        else:
            leaf_img = cv2.resize(np.array(image), (self.img_size, self.img_size))

        if self.load_depth:
            # Convert depth to grayscale
            depth = Image.open(os.path.join(self.depth_path, self.depth_images[idx]))
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

        image = leaf_img

        # Load plant string
        with open(os.path.join(self.plant_string_path, self.plant_string_files[idx]), 'r') as f:
            self.plant_string_raw = f.read()
        
        vec = string2vec(self.plant_string_raw)[0]

        return image, vec

    
    def __getitem__(self, idx):

        if self.preload:
            image = self.images[idx]
            vec = self.vec[idx]
        else:
            image, vec = self.getitem(idx)

        # Convert image to PIL and apply transforms
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # Tokenize the plant structure
        out = vec2token(vec)
            
        # Add SOS and EOS tokens
        out = np.concatenate(([params_SOS_token_padded], out, [params_EOS_token_padded]))
        out_len = len(out)

        return image, out, out_len
        
                

def collate_fn(batch):
    images, vectors, lengths = zip(*batch)
    max_length = max(lengths)
    # Check if the vectors are 1 dimensional
    if len(vectors[0].shape) == 1:
        vectors_padded = np.ones((len(vectors), max_length), dtype=int) * PAD_token
    else:
        # vectors_padded = np.ones((len(vectors), max_length, 1+5+3+2+4)) * PAD_token
        vectors_padded = np.ones((len(vectors), max_length, vectors[0].shape[-1])) * PAD_token # Bacth samples are padded with PAD_token
    
        # Should not reset the param space PAD_token because of the masked loss
        if 0:
            # Reset param space
            vectors_padded[:,:,1:] = 0
        
    for i, vector in enumerate(vectors):
        end = lengths[i]
        vectors_padded[i, :end] = vector
    images = torch.stack(images)
    vectors_padded = torch.tensor(vectors_padded,dtype=torch.float32)
    return images, vectors_padded, lengths
