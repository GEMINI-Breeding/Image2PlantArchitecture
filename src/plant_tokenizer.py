import numpy as np
import math
import random
import os, sys
# Path Settings
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(project_dir)

from src.plant_architecture_utils import euler_to_quaternion, quaternion_to_euler
from src.plant_architecture_utils import coordinates_to_angle, angle_to_coordinates
from scipy.spatial.distance import cdist

from sklearn.cluster import MiniBatchKMeans, KMeans

import torch
import torch.nn.functional as F
from typing import List, Union
import pandas as pd

# Create a dict convert plant structure to token
# Depth, organ type | Token
# 0,0               | 0 # Shoot     
# 0,1               | 1 # Internode
# 0,2               | 2 # Petiole
# 0,3               | 3 # Leaf 0
# 0,4               | 4 # Leaf 1
# 0,5               | 5 # Leaf 2
# 1,0               | 6 # Shoot
# 1,1               | 7
# 1,2               | 8
# 1,3               | 9  # Leaf 0
# 1,4               | 10 # Leaf 1
# 1,5               | 11 # Leaf 2
# 2,0               | 12
# ...
# 3,0               | 18
# 3,1               | 19
# 3,2               | 20
# 3,3               | 21
# 3,4               | 22
# 3,5               | 23

# And then, paramter quantization comes,
predetermined_centers = np.unique(np.concatenate([
            np.array([0, 10, -10, 15, -15, 20, 40, 90]),  # Some special angles
            np.linspace(-360, 360, 18+1),  # angles
            np.array([0.9, 1.0]),  # Some special float values
            np.array([1,  3]),  # Some special integer values
            np.linspace(0, 1.0, 11),
            np.linspace(0, 0.1, 11),
            np.linspace(0, 0.01, 11),  # float values for lengths
            np.linspace(0, 0.001, 11)  # float values for lengths
        ])).reshape(-1, 1)
# SOS               | 23 + len(predetermined_centers) + 1 # Start of sentence
# PAD               | 23 + len(predetermined_centers) # Padding
# EOS               | 23 + len(predetermined_centers) # End of sentence


# 4*6 + 3 => Max nested depth is 3
N_DEPTH = 4
N_ORGAN = 6
NUM_PSTR_TOKEN = N_DEPTH * N_ORGAN
SOS_TOKEN = NUM_PSTR_TOKEN + len(predetermined_centers) + 0
PAD_TOKEN = NUM_PSTR_TOKEN + len(predetermined_centers) + 1
EOS_TOKEN = NUM_PSTR_TOKEN + len(predetermined_centers) + 2
NUM_TOTAL_TOKENS = EOS_TOKEN + 1

def vec2token(vec: List[np.ndarray]) -> np.ndarray:
    """
    Convert vec to tokens
    vec: converted plant vector from plant string. (depth, organ, [params])
    2025.01.07 Just cat params without scaler. token scaler will be added after this
    2025.03.18 Totally changed the logic. It will tokenize the depth and organ, and param to make a single sequence
    """
    tokens = []
    for x in vec:
        # Append depth and organ token
        depth_organ = x[0] * N_ORGAN + x[1]
        tokens.append(depth_organ)
        
        # Append param tokens
        params = np.array(x[2:]).reshape(-1,1)
       
        distances = cdist(params, predetermined_centers)  # Compute pairwise distances
        quantized_index = np.argmin(distances, axis=1)    # Find the closest center
        for i in range(len(quantized_index)):
            tokens.append(NUM_PSTR_TOKEN + quantized_index[i])

    return np.array(tokens)


def token2vec(tokens: np.ndarray) -> List[np.ndarray]:
    vec = []
    vec_line = None
    for token in tokens:
        if token == SOS_TOKEN:
            pass
        elif token == EOS_TOKEN or token == PAD_TOKEN:
            break
        else:
            # Check if structure token
            if token < NUM_PSTR_TOKEN:
                depth = token // 6
                organ = token % 6
                if vec_line:
                    vec.append(vec_line)
                vec_line = [depth,organ]
            else:
                value = predetermined_centers[token - NUM_PSTR_TOKEN][0]
                vec_line.append(value)
    # Add last params (unclosed)
    vec.append(vec_line)

    return vec

def get_shoot_params(params: np.ndarray, token: np.ndarray) -> None:
    """
    Shoot Parameters:
    Shoot parameter 0: max = 59.9308, min = 0.0
    Shoot parameter 1: max = 359.98, min = -19.997
    Shoot parameter 2: max = 359.721, min = 0.051564
    Shoot parameter 3: max = 19.0, min = 0.0
    Shoot parameter 4: max = 3.0, min = 1.0
    """
    params[0] = token[1]     # shoot_base_rotation_pitch
    params[1] = token[2]     # shoot_base_rotation_yaw
    params[2] = token[3]     # shoot_base_rotation_roll
    params[3] = token[4]     # plant_age
    params[4] = 1.0 if abs(1.0 - token[5]) < abs(3.0 - token[5]) else 3.0 # shoot_type

def get_internode_params(params: np.ndarray, token: np.ndarray) -> None:
    """
    Internode Parameters:
    Internode parameter 0: max = 0.03, min = 0.000249986
    Internode parameter 1: max = 0.00317704, min = 0.0005
    Internode parameter 2: max = 20.0, min = 0.0
    Internode parameter 3: max = 214.997, min = 145.001
    """
    params[0] = max(token[6], 0.0002)     # internode_length
    params[1] = max(token[7], 0.0005)     # internode_radius
    params[2] = token[8]     # internode_pitch
    params[3] = token[9]     # phyllotactic angle

def get_petiole_params(params: np.ndarray, token: np.ndarray) -> None:
    """
    Petiole Parameters:
    Petiole parameter 0: max = 0.099999, min = 1e-06
    Petiole parameter 1: max = 0.0018, min = 4e-06
    Petiole parameter 2: max = 79.9864, min = 45.0005
    Petiole parameter 3: max = -50.0003, min = -333.113
    Petiole parameter 4: max = 1.0, min = 0.9
    """
    params[0] = max(token[10], 1e-7)     # petiole_length
    params[1] = max(token[11], 4e-06)    # petiole radius
    params[2] = token[12]    # petiole_pitch
    params[3] = token[13]    # petiole_curvature
    params[4] = token[14]    # leaflet_scale

def get_leaf_params(params: np.ndarray, token: np.ndarray) -> None:
    """
    Leaf Parameters:
    Leaf parameter 0: max = 0.107999, min = 0.0002
    Leaf parameter 1: max = 37.7116, min = -43.7484
    Leaf parameter 2: max = 10.0, min = 0.0
    Leaf parameter 3: max = -15.0, min = -15.0
    """
    params[0] = max(token[15], 0.0002)    # leaf_scale
    params[1] = token[16]    # leaf pitch
    params[2] = token[17]    # leaf yaw
    params[3] = token[18]    # leaf roll
    

def generate_noise_plant_tokens(tokens, noise_level=0.1, mode='train'):
    noise_token = torch.zeros_like(tokens)
    for batch_idx in range(len(tokens)):
        for idx in range(len(tokens[batch_idx])):
            label = tokens[batch_idx][idx][0]
            if label == SOS_TOKEN:
                #structure.append(SOS_word)
                # Do not append SOS token
                # break
                pass
            elif label == EOS_TOKEN or label == PAD_TOKEN:
                #structure.append(EOS_word)
                # Do not append EOS token
                break
            else:
                i = label // 4
                j = label % 4
                # Scale the params to match the original scale
    
                if j == 0:
                    # Shoot
                    noise_token[batch_idx][idx][1] = torch.randn(1).squeeze() * noise_level 
                    noise_token[batch_idx][idx][2] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][3] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][4] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][5] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][6] = torch.randn(1).squeeze() * noise_level
                    # params_padded[3] = token[7] * 100 # shoot_gravitropic_curvature
                    # params_padded[4] = token[8] # shoot_type
                elif j == 1:
                    # Internode
                    # params_padded[0] = token[9] / 100 # internode_length
                    # params_padded[1] = token[10] / 100 # internode_radius
                    # params_padded[2] = token[11] * 180 / math.pi # internode_pitch
                    # params_padded[3] = token[12] * 180 / math.pi # phyllotactic angle, random.uniform(130, 145)
                    pass
                elif j == 2:
                    # Petiole
                    # params_padded[0] = token[13] / 100 # petiole_length
                    # params_padded[1] = token[14] / 100 # petiole radius, random.uniform(0.00075, 0.00125)
                    # params_padded[2] = token[15] * 180 / math.pi # petiole_pitch
                    pass
                elif j == 3:
                    # Leaf
                    # params_padded[0] = token[16] / 100 # leaf_scale
                    # params_padded[1] = coordinates_to_angle(token[17], token[18])
                    # params_padded[2] = coordinates_to_angle(token[19], token[20])
                    # params_padded[3] = coordinates_to_angle(token[21], token[22])
                    noise_token[batch_idx][idx][17] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][18] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][19] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][20] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][21] = torch.randn(1).squeeze() * noise_level
                    noise_token[batch_idx][idx][22] = torch.randn(1).squeeze() * noise_level
                else:
                    raise ValueError(f"Invalid organ type {j}")
    # Make the noise tensor requires_grad
    if mode == 'train':
        noise_token.requires_grad = True
    return noise_token


if __name__ == "__main__":


    from plant_dataset import PlantDataset
    dataset_dir = "data/2000_Plots_20241210"
    train_dataset = PlantDataset(dataset_dir, load_depth=False, preload=False,
                                 process_leaf=False,
                                 image_size=224)
    from tqdm import tqdm
    for image, plant_info, out, out_len in tqdm(train_dataset):
        # # Get the first_shoot
        # first_shoot = out[1]

        # pitch = first_shoot[1]
        # yaw = first_shoot[2]
        # roll = first_shoot[3]
        # q = rpydeg2quat(roll, pitch, yaw)
        # roll_, pitch_, yaw_ = quat2rpydeg(q)
        # print(f"roll={roll}, pitch={pitch}, yaw={yaw}")
        # print("q=", q)
        # print(f"roll_={roll_}, pitch_={pitch_}, yaw_={yaw_}")

        # # check if the conversion is correct
        # assert abs(roll - roll_) < 1e-6
        # assert abs(pitch - pitch_) < 1e-6
        # assert abs(yaw - yaw_) < 1e-6
        vec = token2vec(out)
        tokens = vec2token(vec)

        # print(out)
        # print(tokens)
        # print(vec)
        
        # # check if the conversion is correct
        if not np.allclose(out[1:-1], np.array(tokens)):
            # Check line by line
            for i, (a, b) in enumerate(zip(out[1:-1], tokens)):
                if not np.allclose(a, b):
                    print(f"line {i}: {a} != {b}")

    pass