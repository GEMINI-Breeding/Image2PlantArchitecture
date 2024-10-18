import numpy as np
import math
import random

from utils import euler_to_quaternion, quaternion_to_euler
from utils import coordinates_to_angle, angle_to_coordinates

import torch

# Create a dict convert plant structure to token
# Structure | Token
# 00        | 0
# 01        | 1
# 02        | 2
# 03        | 3
# 10        | 4
# 11        | 5
# 12        | 6
# 13        | 7
# 20        | 8
# ...
# 90        | 36
# 91        | 37
# 92        | 38
# 93        | 39
# SOS token | 40
# EOS token | 41
# PAD token | 42 

SOS_token = 40
PAD_token = 41
EOS_token = 42

if 0:
    params_SOS_token_padded = np.ones(15)*SOS_token
    params_EOS_token_padded = np.ones(15)*EOS_token
else:
    # Zero padded params. SOS and EOS token are zero padded
    params_SOS_token_padded = np.zeros(23)
    params_SOS_token_padded[0] = SOS_token
    params_EOS_token_padded = np.zeros(23)
    params_EOS_token_padded[0] = EOS_token

def vec2token(vec, n_params=23):
    """
    Convert vec to tokens
    vec: converted plant vector from plant string. (depth, organ, [params])
    """
    tokens = []
    for x in vec:
        depth_organ = x[0]*4 + x[1]
        if 1:
            token = np.zeros(n_params) # padding zeros to match the desired length
        else:
            token = np.ones(n_params) * PAD_token # If use PAD_token, unmatched params's loss will be ignored

        token[0] = depth_organ
        # Scale the params to radians, centimeters, etc.

        if x[1] == 0:
            # Shoot params
            # x[2]: shoot_base_pitch
            # x[3]: shoot_base_yaw
            # x[4]: shoot_base_roll
            token[1],token[2] = angle_to_coordinates(x[2])
            token[3],token[4] = angle_to_coordinates(x[3])
            token[5],token[6] = angle_to_coordinates(x[4])
            token[7] = x[5] / 100           # shoot_gravitropic_curvature
            token[8] = x[6]                 # shoot_type
        elif x[1] == 1:
            # Internode params
            token[9] = x[2] * 100 # internode_length
            token[10] = x[3] * 100 # internode_radius
            token[11] = x[4] / 180 * math.pi # internode_pitch
            token[12] = x[5] / 180 * math.pi # phyllotactic angle
        elif x[1] == 2:
            # Petiole params
            token[13] = x[2] * 100 # petiole_length
            token[14] = x[3] * 100 # petiole_radius
            token[15] = x[4] / 180 * math.pi # petiole_pitch
        elif x[1] == 3:
            # Leaf params
            token[16] = x[2] * 100 # leaf_scale
            token[17],token[18] = angle_to_coordinates(x[3]) # leaf pitch
            token[19],token[20] = angle_to_coordinates(x[4]) # leaf yaw
            token[21],token[22] = angle_to_coordinates(x[5]) # leaf roll
        else:
            raise ValueError(f"Invalid organ type {x[1]}")
        
        tokens.append(token)
    return tokens

def token2vec(tokens):
    vec = []
    for token in tokens:
        label = token[0]
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
            params_padded = np.zeros(5)
            # Scale the params to match the original scale
   
            if j == 0:
                # Shoot
                params_padded[0] = coordinates_to_angle(token[1], token[2])
                params_padded[1] = coordinates_to_angle(token[3], token[4])
                params_padded[2] = coordinates_to_angle(token[5], token[6])
                params_padded[3] = token[7] * 100 # shoot_gravitropic_curvature
                params_padded[4] = token[8] # shoot_type
            elif j == 1:
                # Internode
                params_padded[0] = token[9] / 100 # internode_length
                params_padded[1] = token[10] / 100 # internode_radius
                params_padded[2] = token[11] * 180 / math.pi # internode_pitch
                params_padded[3] = token[12] * 180 / math.pi # phyllotactic angle, random.uniform(130, 145)
            elif j == 2:
                # Petiole
                params_padded[0] = token[13] / 100 # petiole_length
                params_padded[1] = token[14] / 100 # petiole radius, random.uniform(0.00075, 0.00125)
                params_padded[2] = token[15] * 180 / math.pi # petiole_pitch
            elif j == 3:
                # Leaf
                params_padded[0] = token[16] / 100 # leaf_scale
                params_padded[1] = coordinates_to_angle(token[17], token[18])
                params_padded[2] = coordinates_to_angle(token[19], token[20])
                params_padded[3] = coordinates_to_angle(token[21], token[22])
            else:
                raise ValueError(f"Invalid organ type {j}")

            # Make 1x6 array with i, j and params
            vec.append(np.concatenate(([i, j], params_padded),axis=0))
    return np.array(vec)

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
    dataset_dir = "/home/lion397/codes/Image2PlantArchitecture/data/generated_dataset_Sep22_black"
    train_dataset = PlantDataset(dataset_dir, plot=["000", "001", "002",], load_depth=True, preload=False,
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


"""
PhytomerParameters::PhytomerParameters( std::minstd_rand0 *generator ) {

    //--- internode ---//
    internode.pitch.initialize( 20, generator );
    internode.phyllotactic_angle.initialize(137.5, generator );
    internode.color = RGB::forestgreen;
    internode.length_segments = 1;
    internode.radial_subdivisions = 7;

    //--- petiole ---//
    petiole.petioles_per_internode = 1;
    petiole.pitch.initialize( 90, generator );
    petiole.radius.initialize( 0.001, generator );
    petiole.length.initialize( 0.05, generator );
    petiole.curvature.initialize(0, generator);
    petiole.taper.initialize( 0, generator );
    petiole.color = RGB::forestgreen;
    petiole.length_segments = 1;
    petiole.radial_subdivisions = 7;

    //--- leaf ---//
    leaf.leaves_per_petiole.initialize( 1, generator);
    leaf.pitch.initialize( 0, generator );
    leaf.yaw.initialize( 0, generator );
    leaf.roll.initialize( 0, generator );
    leaf.leaflet_offset.initialize( 0, generator );
    leaf.leaflet_scale = 1;
    leaf.prototype_scale.initialize(0.05,generator);
    leaf.subdivisions = 1;
    leaf.unique_prototypes = 1;

    //--- peduncle ---//
    peduncle.length.initialize(0.05,generator);
    peduncle.radius.initialize(0.001, generator);
    peduncle.pitch.initialize(0,generator);
    peduncle.roll.initialize(0,generator);
    peduncle.curvature.initialize(0,generator);
    peduncle.length_segments = 3;
    peduncle.radial_subdivisions = 7;

    //--- inflorescence ---//
    inflorescence.flowers_per_rachis.initialize(1, generator);
    inflorescence.flower_offset.initialize(0, generator);
    inflorescence.flower_arrangement_pattern = "alternate";
    inflorescence.pitch.initialize(0,generator);
    inflorescence.roll.initialize(0,generator);
    inflorescence.flower_prototype_scale.initialize(0.0075,generator);
    inflorescence.fruit_prototype_scale.initialize(0.0075,generator);
    inflorescence.fruit_gravity_factor_fraction.initialize(0, generator);
    inflorescence.unique_prototypes = 1;

}
"""