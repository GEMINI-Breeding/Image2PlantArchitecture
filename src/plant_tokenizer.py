import numpy as np
import math
import random

from plant_architecture_utils import euler_to_quaternion, quaternion_to_euler
from plant_architecture_utils import coordinates_to_angle, angle_to_coordinates

import torch

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
# SOS               | 24 # Start of sentence
# PAD               | 25 # Padding
# EOS               | 26 # End of sentence


# 4*6 + 3 => Max nested depth is 3
SOS_token = 4*6
PAD_token = 4*6 + 1
EOS_token = 4*6 + 2

N_PARAMS = 18

if 0:
    SOS_vec_padded = np.ones(15)*SOS_token
    EOS_vec_padded = np.ones(15)*EOS_token
else:
    if 1:
        # Zero padded params. SOS and EOS token are zero padded
        SOS_vec_padded = np.zeros(N_PARAMS+1)
        EOS_vec_padded = np.zeros(N_PARAMS+1)
    else:
        # Padding with PAD_token
        SOS_vec_padded = np.ones(N_PARAMS+1)*PAD_token
        EOS_vec_padded = np.ones(N_PARAMS+1)*PAD_token
    
    SOS_vec_padded[0] = SOS_token
    EOS_vec_padded[0] = EOS_token



def vec2token(vec, n_params=N_PARAMS):
    """
    Convert vec to tokens
    vec: converted plant vector from plant string. (depth, organ, [params])
    2025.01.07 Just cat params without scaler. token scaler will be added after this
    """
    tokens = []
    for x in vec:
        depth_organ = x[0]*6 + x[1]
        if 1:
            token = np.zeros(n_params+1) # padding zeros to match the desired length
        else:
            token = np.ones(n_params+1) * PAD_token # If use PAD_token, unmatched params's loss will be ignored

        token[0] = depth_organ
        # Scale the params to radians, centimeters, etc.
        params = x[2:]
        if x[1] == 0:
            # Shoot params
            # x[2]: shoot_base_rotation_pitch
            # x[3]: shoot_base_rotation_yaw
            # x[4]: shoot_base_rotation_roll
            # x[5]: plant_age
            # x[6]: shoot type
            # token[1],token[2] = angle_to_coordinates(x[2])
            # token[3],token[4] = angle_to_coordinates(x[3])
            # token[5],token[6] = angle_to_coordinates(x[4])
            token[1] = params[0]     # shoot_base_rotation_pitch
            token[2] = params[1]     # shoot_base_rotation_yaw
            token[3] = params[2]     # shoot_base_rotation_roll
            token[4] = params[3]     # plant_age
            token[5] = params[4]     # shoot_type
        elif x[1] == 1:
            # Internode params
            token[6] = params[0]     # internode_length
            token[7] = params[1]     # internode_radius
            token[8] = params[2]     # internode_pitch
            token[9] = params[3]     # phyllotactic angle
        elif x[1] == 2:
            # Petiole params
            token[10] = params[0]    # petiole_length
            token[11] = params[1]    # petiole_radius
            token[12] = params[2]    # petiole_pitch
            token[13] = params[3]    # petiole_curvature
            token[14] = params[4]    # leaflet_scale
        elif x[1] == 3 or x[1] == 4 or x[1] == 5:
            # Leaf params
            token[15] = params[0]    # leaf_scale
            token[16] = params[1]    # leaf pitch
            token[17] = params[2]    # leaf yaw
            token[18] = params[3]    # leaf roll

            # token[2],token[3] = angle_to_coordinates(x[3]) # leaf pitch
            # token[4],token[5] = angle_to_coordinates(x[4]) # leaf yaw
            # token[6],token[7] = angle_to_coordinates(x[5]) # leaf roll
        else:
            raise ValueError(f"Invalid organ type {x[1]}")
        
        tokens.append(token)

    return np.array(tokens)

def token2vec(tokens):
    vec = []
    for token in tokens:
        label = token[0]
        if label == SOS_token:
            #structure.append(SOS_word)
            # Do not append SOS token
            pass
        elif label == EOS_token or label == PAD_token:
            #structure.append(EOS_word)
            # Do not append EOS token
            break
        else:
            i = label // 6
            j = label % 6
            # Scale the params to match the original scale
            params = np.zeros(6)
            if j == 0:
                # Shoot
                params[0] = token[1]     # shoot_base_rotation_pitch
                params[1] = token[2]     # shoot_base_rotation_yaw
                params[2] = token[3]     # shoot_base_rotation_roll
                params[3] = token[4]     # plant_age
                params[4] = 1.0 if abs(1.0 - token[5]) < abs(3.0 - token[5]) else 3.0 # shoot_type
            elif j == 1:
                # Internode
                params[0] = token[6]     # internode_length
                params[1] = token[7]     # internode_radius
                params[2] = token[8]     # internode_pitch
                params[3] = token[9]     # phyllotactic angle, random.uniform(130, 145)
            elif j == 2:
                # Petiole
                params[0] = token[10]    # petiole_length
                params[1] = token[11]    # petiole radius, random.uniform(0.00075, 0.00125)
                params[2] = token[12]    # petiole_pitch
                params[3] = token[13]    # petiole_curvature
                params[4] = token[14]    # leaflet_scale
            elif j == 3 or j == 4 or j == 5:
                # Leaf
                params[0] = token[15]    # leaf_scale
                params[1] = token[16]    # leaf pitch
                params[2] = token[17]
                params[3] = token[18]

                # Convert 0-360 to -180-180
                for k in range(1, 4):
                    if params[k] > 180:
                        params[k] -= 360
            else:
                raise ValueError(f"Invalid organ type {j}")

            # Make 1x6 array with i, j and params
            vec.append(np.concatenate(([i, j], params),axis=0))
    return vec

def generate_noise_plant_tokens(tokens, noise_level=0.1, mode='train'):
    noise_token = torch.zeros_like(tokens)
    for batch_idx in range(len(tokens)):
        for idx in range(len(tokens[batch_idx])):
            label = tokens[batch_idx][idx][0]
            if label == SOS_token:
                #structure.append(SOS_word)
                # Do not append SOS token
                # break
                pass
            elif label == EOS_token or label == PAD_token:
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
    dataset_dir = "data/generated_Nov22_20224"
    train_dataset = PlantDataset(dataset_dir, load_depth=False, preload=False,
                                 process_leaf=False,
                                 image_size=224)
    from tqdm import tqdm
    for image, out, length in tqdm(train_dataset):
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