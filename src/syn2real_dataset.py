import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
import os



'''
bbox_plants: class, x, y, w, h. Plants are 0, weeds are 1. x and y are the center of the bounding box. Only visible plants are marked.
plantstring: Displays all plants within the plot. The number at the front is the Plant ID.
plantIDmap: Displays the ID of the plant.
'''
def generate_color_palette(num_colors):

    # Create a grayscale image with as many pixels as colors needed
    grayscale_palette = np.linspace(0, 255, num_colors, dtype=np.uint8).reshape(-1, 1)

    # Apply a colormap to the grayscale image to get a color palette
    color_palette = cv2.applyColorMap(grayscale_palette, cv2.COLORMAP_JET)

    return color_palette

modalities = {'bbox':['_closedflowers.txt', '_leaves.txt', '_openflowers.txt', '_plants.txt', '_pods.txt'],
                           'depth':['.txt'],
                           'plantIDmap':['.txt'],
                            'depth':['.txt'],
                            'plantstring':['.txt'],
                            'RGB':['.jpeg']
                           }

# Generate a color palette for 10 bounding boxes
color_palette = generate_color_palette(len(modalities['bbox'])+1)
#colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]


class Syn2RealDataset(Dataset):

    def __init__(self, root_dir, transform=None, use_depth=False):
        self.root_dir = root_dir
        if transform==None:
            # Define the default transformation
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.transform = transform
        else:
            self.transform = transform

        self.modalities = modalities
        
        self.label_ID = {'background':0, 'plants':1, 'weeds':2, 'closedflowers':3, 'leaves':4, 'openflowers':5, 'pods':6}
        
        self.base_names = self.read_dataset(root_dir)
        self.stages = [base_name.split('_')[2] for base_name in self.base_names]
        self.use_depth = use_depth

    def get_label_ID(self, label:list):
        label_ID = []
        for l in label:
            label_ID.append(self.label_ID[l])
        return label_ID
        
    
    def modality_checker(self, base_name):
        all_exist = True
        for modality in self.modalities.keys():
            for modality_ext in self.modalities[modality]:
                file = os.path.join(self.root_dir, f"{base_name}_{modality}{modality_ext}")
                if not os.path.exists(file):
                    all_exist = False
        return all_exist
        
    def read_dataset(self, root_dir):
        files = os.listdir(root_dir)
        base_names = [self.parse_base_name(file) for file in files]
        # Get unique base names
        base_names = list(set(base_names))
        base_names = sorted(base_names)

        # Get only the base names that have all modalities
        filtered_base_names = [name for name in base_names if self.modality_checker(name)]
        return filtered_base_names

    def parse_base_name(self, file):
        name_parts = file.split('_')
        return '_'.join(name_parts[:5])
        
    
    def __len__(self):
        return len(self.base_names)

    def getitem(self, idx):
        item = dict()
        item['base_name'] = self.base_names[idx]
        item['bbox'] = self.read_bbox(item['base_name'])
        
        item['stage'] = item['base_name'].split('_')[2] # 003, 010, 016, 023
        item['plot'] = item['base_name'].split('_')[3] #  000, 001, 002, 003, 004
        if self.use_depth:
            item['depth'] = self.read_depth(item['base_name'])
            
        #item['plantIDmap'] = self.read_plantIDmap(item['base_name'])
        item['plantstring'] = self.read_plantstring(item['base_name'])
        item['RGB'] = self.read_RGB(item['base_name'])

        # Process bbox
        item['id_bbox_string'] = get_plant_ids_bbox_strings(item)
        return item
    
    def __getitem__(self, idx):
        item = self.getitem(idx)
        if self.transform:
            item['RGB'] = self.transform(item['RGB'])
            item['depth'] = self.transform(item['depth'])
            item['plantIDmap'] = self.transform(item['plantIDmap'])
        return item


    def read_bbox(self, base_name):
        # Create an empty bbox
        bbox_total = np.zeros((0, 5)) # class, x, y, w, h
        for obj_label in self.modalities['bbox']:
            label_name = obj_label.split('_')[1].split('.')[0]
            label_idx = self.label_ID[label_name]
            file = os.path.join(self.root_dir, f"{base_name}_bbox{obj_label}")
            if os.path.exists(file) and os.path.getsize(file) > 0:
                bbox = np.loadtxt(file)
                # If bbox is not empty
                if bbox.size != 0:
                    if len(bbox.shape) == 1:
                        bbox = bbox.reshape(1, -1)
                    # Convert the bbox from center to top-left
                    bbox[:,1] = bbox[:,1] - bbox[:,3]/2
                    bbox[:,2] = bbox[:,2] - bbox[:,4]/2

                    # Assign class label. 0 is background
                    if obj_label == '_plants.txt':
                        # Brain accdently make weed to 0 and 1.
                        plant_bbox = bbox[bbox[:,0] == 0]
                        weed_bbox = bbox[bbox[:,0] == 1]

                        if 0:
                            # Remove the weed bbox from the plant bbox
                            for i in range(len(weed_bbox)):
                                for j in range(len(plant_bbox)):
                                    # Check if the weed bbox is the same as the plant bbox
                                    if np.all(weed_bbox[i,1:] == plant_bbox[j,1:]):
                                        # Then remove the plant bbox
                                        plant_bbox = np.delete(plant_bbox, j, axis=0)
                                        break
                            bbox = np.concatenate((plant_bbox, weed_bbox), axis=0)
                        else:
                            # Remove the plant box back from the number of weeds
                            weed_cnt = len(weed_bbox)
                            plant_cnt = len(plant_bbox)
                            plant_bbox = plant_bbox[:plant_cnt-weed_cnt]
                            bbox = np.concatenate((plant_bbox, weed_bbox), axis=0)
                        # Assign the class label
                        bbox[:,0] = bbox[:,0] + label_idx # label_idx or plants, label_idx+1 for weeds
                    else:    
                        bbox[:,0] = label_idx
                    # Concatenate the bbox
                    bbox_total = np.concatenate((bbox_total, bbox), axis=0)


        return bbox_total
    
    def read_depth(self, base_name):
        file = os.path.join(self.root_dir, f"{base_name}_depth.txt")
        depth = np.loadtxt(file)
        return depth
    
    def read_plantIDmap(self, base_name):
        file = os.path.join(self.root_dir, f"{base_name}_plantIDmap.txt")
        plantIDmap = np.loadtxt(file)
        # Set nan to 0
        plantIDmap = np.nan_to_num(plantIDmap, nan=-1) # 0 to 255, nan -> -1 is background
        # to int
        #plantIDmap = plantIDmap.astype(np.uint8) # 0-255
        plantIDmap = plantIDmap.astype(np.int16)
        return plantIDmap
    
    def read_plantstring(self, base_name):
        file = os.path.join(self.root_dir, f"{base_name}_plantstring.txt")
        plantstring = []
        with open(file, 'r') as f:
            for line in f:
                plantstring.append(line.strip())

        return plantstring
    
    def read_RGB(self, base_name):
        file = os.path.join(self.root_dir, f"{base_name}_RGB.jpeg")
        image = cv2.imread(file)
        return image

def get_plant_bbox_from_idmap(plantIDmap):
    map_plantIDs = np.unique(plantIDmap)
    # Remove the background
    map_plantIDs = map_plantIDs[map_plantIDs >= 0]
    id_bboxes = []
    for plantID in map_plantIDs:
        plant = np.zeros_like(plantIDmap)
        plant[plantIDmap == (int(plantID))] = 255
        plant = plant.astype(np.uint8)
        x, y, w, h = cv2.boundingRect(plant)
        id_bboxes.append((plantID, x, y, w, h))

    return id_bboxes

def get_plant_ids_bbox_strings(item):
    
    if 0:
        # Get all unique plant IDs
        plantIDs = np.unique(item['plantIDmap'])

        # Remove the background
        plantIDs = plantIDs[plantIDs >= 0]
    else:
        if 0:
            # 먼저 Plant ID map에서 식물의 bounding box와 ID를 찾는다
            # 그리고 item['bbox']에서 식물의 bounding box를 찾는다
            # 두 bounding box가 겹치는지 확인하고 겹치는 경우에만 ID와 Bounding box를 출력한다.
            out = []
            id_bboxes = get_plant_bbox_from_idmap(item['plantIDmap'])
            # Find plant bounding box from bbox
            plant_bbox = item['bbox'][item['bbox'][:,0] == 1] # 1 is plant
            for id_bbox in id_bboxes:
                plantID, x, y, w, h = id_bbox
                id_map_matched = False
                for bbox in plant_bbox:
                    # Check overlap between plantID and bbox is higher than 0.95
                    x1, y1, w1, h1 = bbox[1:]
                    img_h, img_w = item['RGB'].shape[:2]
                    x1 = x1 * img_w
                    y1 = y1 * img_h
                    w1 = w1 * img_w
                    h1 = h1 * img_h
                    # Convert to integer by multiplying by the image size
                    x2, y2, w2, h2 = x, y, w, h
                    # Calculate the overlap
                    x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                    y_overlap = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    overlap_area = x_overlap * y_overlap
                    bbox_area = w1 * h1
                    overlap = overlap_area/bbox_area
                    if overlap > 0.95:
                        id_map_matched = True
                        break
                if id_map_matched:
                    out.append([plantID, x, y, w, h])
        else:
            # 그냥 bbox에서 식물만 찾는다.
            out = []
            plant_bbox = item['bbox'][item['bbox'][:,0] == 1] # 1 is plant
            for plantID, bbox in enumerate(plant_bbox):
                x, y, w, h = bbox[1:]
                img_h, img_w = item['RGB'].shape[:2]
                x = int(x * img_w)
                y = int(y * img_h)
                w = int(w * img_w)
                h = int(h * img_h)
                out.append([plantID, x, y, w, h])

        # Get the plant string
        plant_string = item['plantstring']
        plant_string_ids = [x.split(' {')[0] for x in plant_string]
        for i, plant in enumerate(out):
            plant_id = plant[0]
            idx = plant_string_ids.index(str(plant_id))
            out[i].append(plant_string[idx])

            
    return out

def get_plant(item, plantID, debug=True):
    plantID_fromString = [x.split(' {')[0] for x in item['plantstring']]
    
    idx = plantID_fromString.index(str(plantID))
    plant_string = item['plantstring'][idx]

    # Subset the plantIDmap with the plantID
    plantIDmap = item['plantIDmap']
    plant = np.zeros_like(plantIDmap)
    plant[plantIDmap == (int(plantID))] = 255
    plant = plant.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(plant)

    if debug:
        scale = 1/6
        # Crop the plant
        rgb = item['RGB']
        plant_rgb = rgb[y:y+h, x:x+w,:]

        # Display the plant id on the image
        cv2.rectangle(item['RGB'], (x, y), (x+w, y+h), (255,0,0), 1)
        # Draw the class label
        cv2.putText(item['RGB'], f"{str(int(plantID))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        plant_disp = cv2.resize(item['RGB'], (0,0), fx=scale, fy=scale)
        cv2.imshow("Plant", plant_rgb)
        cv2.waitKey(0)

    return (x, y, w, h, plant_string)



def draw_bbox(img, bbox, selected_labels="all"):
    for i, box in enumerate(bbox):
        class_label, x, y, w, h = box
        if selected_labels == "all":
            pass
        else:
            # Check if class label
            if class_label not in selected_labels:
                continue

        # Convert to actual size by multiplying image size
        x = int(x * img.shape[1])
        y = int(y * img.shape[0])
        w = int(w * img.shape[1])
        h = int(h * img.shape[0])

        # Generate color from class
        color = tuple(map(int, color_palette[int(class_label)].tolist()[0]))
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)

        # Draw the class label
        #cv2.putText(img, str(int(class_label)), (x-w//2, y-h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if class_label == 1:
            cv2.putText(img, f"Plant", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        elif class_label == 2:
            cv2.putText(img, f"Weed", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        

    return img





if __name__ == '__main__':
    
    # Load the dataset
    # syn2real_dataset = Syn2RealDataset(root_dir='/home/lion397/codes/cvae-real2syn/Syn2Real_cowpea', transform=transform)
    syn2real_dataset = Syn2RealDataset(root_dir='/home/lion397/codes/cvae-real2syn/Syn2Real_cowpea', transform=[])
    #item = syn2real_dataset[-1]
    
    syn2real_dataset[637]
    # Iteration test
    if 1:
        print("Test read all the dataset")
        try:
            for i, item in enumerate(tqdm(syn2real_dataset)):
                pass
        except Exception as e:
            print(e)
            
    item = syn2real_dataset[i+1]

    # Resize the images
    scale = 1/6
    rgb_resized = cv2.resize(item['RGB'], (0,0), fx=scale, fy=scale)
    draw_bbox(rgb_resized, item['bbox'],selected_labels=syn2real_dataset.get_label_ID(['plants', 'weeds']))
    #draw_bbox(rgb_resized, item['bbox'])

    if 1:
        # Apply colormap to the depth image and then resize
        depth_colored = cv2.applyColorMap(cv2.normalize(item['depth'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colored, (0,0), fx=scale, fy=scale)

        # Apply colormap to the plantIDmap and then resize
        plantIDmap_colored = cv2.applyColorMap(cv2.normalize(item['plantIDmap'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
        out = get_plant_ids_bbox_strings(item)
        plant_ids = [x[0] for x in out]
        for plant_id in plant_ids:
            plantIDmap = item['plantIDmap']
            plant = np.zeros_like(plantIDmap)
            plant[plantIDmap == (int(plant_id))] = 255
            plant = plant.astype(np.uint8)
            x, y, w, h = cv2.boundingRect(plant)
            if plant_id >= 10:
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(plantIDmap_colored, (x, y), (x+w, y+h), color, int(1/scale))
            cv2.putText(plantIDmap_colored, f"{str(int(plant_id))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5/scale, color, int(1/scale))

        plantIDmap_resized = cv2.resize(plantIDmap_colored, (0,0), fx=scale, fy=scale)

        # Concatenate the images
        #concatenated = cv2.hconcat([rgb_resized, depth_resized, plantIDmap_resized])
        concatenated = cv2.hconcat([rgb_resized, plantIDmap_resized])

        # Display the concatenated image
        cv2.imshow('Concatenated', concatenated)
        key = cv2.waitKey(-1)
        if key == 'q':
            cv2.destroyAllWindows()

    # Test plandID
    out = get_plant_ids_bbox_strings(item)