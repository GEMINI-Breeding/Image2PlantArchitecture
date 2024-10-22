import os, sys, copy
import shutil
import subprocess
import cv2
import torch

# 경로 설정
script_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_file_dir)
from string_to_xml_to_vec import save_plant_string
from image_process import process_leaf_image
from plant_tokenizer import token2vec
from utils import coordinates_to_angle

class plantstring2model:

    def __init__(self, program_path, program_name, background_path=None, display=":10.0", height=2.0, verbose=False):
        

        self.program_path = program_path
        self.program_name = program_name

        self.display = display

        self.background_path = background_path
        self.height = height
        os.environ["DISPLAY"] = self.display

        self.verbose = verbose

        self.output_path_name = "output"

        # Create the output directory
        os.makedirs(self.output_path_name, exist_ok=True)

    def run(self, in_plantstring_path, output_path=None):
        
        plantstring_name = os.path.splitext(os.path.basename(in_plantstring_path))[0]
        if output_path:
            output_file_name = output_path
        else:
            output_file_name = f"{plantstring_name}_top.jpeg"
        # Construct the command
        command = ""
        command += f"cd {self.program_path} && ./{self.program_name} "
        # If self.background_path exists
        if self.background_path:
            file_name = os.path.abspath(self.background_path)
            # Copy the background tile to build
            command += f"-tile {file_name} "
            
        # Add height
        command += f"-h {self.height} "
        
        # Add the plantstring path
        command += f"-f {in_plantstring_path} "
        command += f"-o {output_file_name} "

        if self.verbose == False:
            command += " > log.txt 2>&1"
            
        # Run the command using os.system
        # os.system(f"{command}")
        # Replace os.system(f"{command}") with subprocess.run
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # print("Command executed successfully")
            # print(result.stdout)  # Print the standard output
            pass
        else:
            # print("Command failed")
            # print(result.stderr)  # Print the error output
            pass
    
    def plant_vec_to_image(self, plant_vec, idx, suffix="", image_size=224):
        output_path = f"temp/output_{suffix}_{idx}"
        #output_path = f"/dev/shm/output_{suffix}_{idx}"  # Use RAM disk
        plant_string_file_name = save_plant_string(plant_vec, output_path, idx, suffix)
        self.run(in_plantstring_path=os.path.abspath(plant_string_file_name), 
                                    output_path=os.path.abspath(output_path))
        
        generated_image_path = f"{output_path}/plant_string_{suffix}_{idx}_top.jpeg"
        img = cv2.imread(generated_image_path)
        leaf_area, plant_width, plant_height, leaf_img, _ = process_leaf_image(img, sqaure_crop=True, thr=0.2)
        leaf_img = cv2.normalize(leaf_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        leaf_img = cv2.resize(leaf_img, (image_size, image_size))

        return leaf_img
        
    def generate_image(self, batch_idx, tokens, image_size, suffix):
        plant_vec = token2vec(tokens[batch_idx].squeeze().squeeze().tolist())
        
        # Generate image
        img = self.plant_vec_to_image(plant_vec, idx=batch_idx, suffix=suffix, image_size=image_size)
        # img_tensor = torch.tensor(img).to(tokens.device).permute(2, 0, 1)  # (C, H, W)
        return batch_idx, img
            
# Test 
if __name__ == "__main__":
    p2m = plantstring2model("src/PlantString2Model/build", "PlantString2Model", display=":11.0")
    p2m.run("/home/lion397/codes/Image2PlantArchitecture/src/plant_string.txt")