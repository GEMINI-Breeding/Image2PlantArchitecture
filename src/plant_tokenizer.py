import numpy as np
import math
import random

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
    params_SOS_token_padded = np.zeros(19)
    params_SOS_token_padded[0] = SOS_token
    params_EOS_token_padded = np.zeros(19)
    params_EOS_token_padded[0] = EOS_token

# def token2vec(tokens):
#     vec = []
#     for token in tokens:
#         label = token[0]
#         if label == SOS_token:
#             #structure.append(SOS_word)
#             # Do not append SOS token
#             # break
#             pass
#         elif label == EOS_token or label == PAD_token:
#             #structure.append(EOS_word)
#             # Do not append EOS token
#             break
#         else:
#             i = label // 4
#             j = label % 4
#             params_padded = np.zeros(5)
#             if j == 0:
#                 # Shoot
#                 params_padded[0] = token[1] * 180 / math.pi # shoot_base_pitch
#                 params_padded[1] = token[2] * 180 / math.pi # shoot_base_yaw
#                 params_padded[2] = token[3] * 180 / math.pi # shoot_base_roll
#                 params_padded[3] = token[4] * 100 # shoot_gravitropic_curvature
#                 params_padded[4] = token[5] # shoot_type
#             elif j == 1:
#                 # Internode
#                 params_padded[0] = token[6] / 100 # internode_length
#                 params_padded[1] = token[7] / 100 # internode_radius
#                 params_padded[2] = token[8] * 180 / math.pi # internode_pitch
#                 # Hardcored phyllotactic angle because it is not given from Brian yet
#                 # params_padded[3] = 137.5 # TODO phyllotactic angle. The most common angle is the golden angle, or Fibonacci angle, which is approximately 137.5°.
#                 params_padded[3] = random.uniform(130, 145)
#             elif j == 2:
#                 # Petiole
#                 params_padded[0] = token[9] / 100 # petiole_length
#                 #params_padded[1] = 0.001 # TODO petiole radius is not given
#                 params_padded[1] = random.uniform(0.00075, 0.00125) # TODO petiole radius is not given
#                 params_padded[2] = token[10] * 180 / math.pi # petiole_pitch
#             elif j == 3:
#                 # Leaf
#                 params_padded[0] = token[11] / 100 # leaf_scale
#                 params_padded[1] = token[12] * 180 / math.pi # leaf_pitch
#                 params_padded[2] = token[13] * 180 / math.pi # leaf_yaw
#                 params_padded[3] = token[14] * 180 / math.pi # leaf_roll
#             else:
#                 raise ValueError(f"Invalid organ type {j}")

#             # Make 1x6 array with i, j and params
#             vec.append(np.concatenate(([i, j], params_padded),axis=0))
#     return np.array(vec)

# def vec2token(vec, n_params=15):
#     tokens = []
#     for x in vec:
#         depth_organ = x[0]*4 + x[1]
#         if 1:
#             token = np.zeros(n_params) # padding zeros to match the desired length
#         else:
#             token = np.ones(n_params) * PAD_token

#         token[0] = depth_organ
        
#         if x[1] == 0:
#             # Shoot params    
#             token[1] = x[2] / 180 * math.pi # shoot_base_pitch
#             token[2] = x[3] / 180 * math.pi # shoot_base_yaw
#             token[3] = x[4] / 180 * math.pi # shoot_base_roll
#             token[4] = x[5] / 100           # shoot_gravitropic_curvature
#             token[5] = x[6]                 # shoot_type
#         elif x[1] == 1:
#             # Internode params
#             token[6] = x[2] * 100 # internode_length
#             token[7] = x[3] * 100 # internode_radius
#             token[8] = x[4] / 180 * math.pi # internode_pitch
#         elif x[1] == 2:
#             # Petiole params
#             token[9] = x[2] * 100 # petiole_length
#             token[10] = x[3] / 180 * math.pi # petiole_pitch
#         elif x[1] == 3:
#             # Leaf params
#             token[11] = x[2] * 100 # leaf_scale
#             token[12] = x[3] / 180 * math.pi # leaf_pitch
#             token[13] = x[4] / 180 * math.pi # leaf_yaw
#             token[14] = x[5] / 180 * math.pi # leaf_roll
#         else:
#             raise ValueError(f"Invalid organ type {x[1]}")
        
#         tokens.append(token)
#     return tokens

def rpydeg2quat(roll, pitch, yaw):
    # Convert degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    
    # Compute quaternion
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = np.zeros(4)
    q[0] = cr * cp * cy + sr * sp * sy
    q[1] = sr * cp * cy - cr * sp * sy
    q[2] = cr * sp * cy + sr * cp * sy
    q[3] = cr * cp * sy - sr * sp * cy

    # w가 항상 양수가 되도록 정규화하기도 합니다.
    # 비록 q와 -q가 같은 회전을 나타내지만, 연속적인 회전 시퀀스에서는 일관성을 위해 한 가지 표현을 선택하여 사용하는 것이 좋습니다.
    if q[0] < 0:
        q = -q

    return q

def quat2rpydeg(q):
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Compute roll, pitch, yaw
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert radians to degrees
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    # Normalize to [0, 360)
    roll = roll % 360
    pitch = pitch % 360
    yaw = yaw % 360

    return roll, pitch, yaw



def vec2token_new(vec, n_params=19):
    tokens = []
    for x in vec:
        depth_organ = x[0]*4 + x[1]
        if 0:
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
            q = rpydeg2quat(roll=x[4], pitch=x[2], yaw=x[3])
            token[1] = q[0]
            token[2] = q[1]
            token[3] = q[2]
            token[4] = q[3]
            
            token[5] = x[5] / 100           # shoot_gravitropic_curvature
            token[6] = x[6]                 # shoot_type
        elif x[1] == 1:
            # Internode params
            token[7] = x[2] * 100 # internode_length
            token[8] = x[3] * 100 # internode_radius
            token[9] = x[4] / 180 * math.pi # internode_pitch
            token[10] = x[5] / 180 * math.pi # phyllotactic angle
        elif x[1] == 2:
            # Petiole params
            token[11] = x[2] * 100 # petiole_length
            token[12] = x[3] * 100 # petiole_radius
            token[13] = x[4] / 180 * math.pi # petiole_pitch
        elif x[1] == 3:
            # Leaf params
            token[14] = x[2] * 100 # leaf_scale

            q = rpydeg2quat(roll=x[5], pitch=x[3], yaw=x[4])
            # x[3]: leaf_pitch
            # x[4]: leaf_yaw
            # x[5]: leaf_roll
            token[15] = q[0]
            token[16] = q[1]
            token[17] = q[2]
            token[18] = q[3]
        else:
            raise ValueError(f"Invalid organ type {x[1]}")
        
        tokens.append(token)
    return tokens

def token2vec_new(tokens):
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
                q = np.array([token[1], token[2], token[3], token[4]])
                roll, pitch, yaw = quat2rpydeg(q)
                params_padded[0] = pitch
                params_padded[1] = yaw
                params_padded[2] = roll
                params_padded[3] = token[5] * 100 # shoot_gravitropic_curvature
                params_padded[4] = token[6] # shoot_type
            elif j == 1:
                # Internode
                params_padded[0] = token[7] / 100 # internode_length
                params_padded[1] = token[8] / 100 # internode_radius
                params_padded[2] = token[9] * 180 / math.pi # internode_pitch
                params_padded[3] = token[10] * 180 / math.pi # phyllotactic angle, random.uniform(130, 145)
            elif j == 2:
                # Petiole
                params_padded[0] = token[11] / 100 # petiole_length
                params_padded[1] = token[12] / 100 # petiole radius, random.uniform(0.00075, 0.00125)
                params_padded[2] = token[13] * 180 / math.pi # petiole_pitch
            elif j == 3:
                # Leaf
                params_padded[0] = token[14] / 100 # leaf_scale
                q = np.array([token[15], token[16], token[17], token[18]])
                roll, pitch, yaw = quat2rpydeg(q)
                params_padded[1] = pitch
                params_padded[2] = yaw
                params_padded[3] = roll
            else:
                raise ValueError(f"Invalid organ type {j}")

            # Make 1x6 array with i, j and params
            vec.append(np.concatenate(([i, j], params_padded),axis=0))
    return np.array(vec)

if __name__ == "__main__":

    # Test rpy2quat and quat2rpy

    pitch = 4.940316
    yaw = 307.362274
    roll = 104.781784
    
    q = rpydeg2quat(roll, pitch, yaw)
    roll_, pitch_, yaw_ = quat2rpydeg(q)
    print(f"roll={roll}, pitch={pitch}, yaw={yaw}")
    print("q=", q)
    print(f"roll_={roll_}, pitch_={pitch_}, yaw_={yaw_}")

    # Test rpy2quat and quat2rpy    
    roll = 359
    pitch = 150
    yaw = 260
    
    q = rpydeg2quat(roll, pitch, yaw)
    roll_, pitch_, yaw_ = quat2rpydeg(q)
    print(f"roll={roll}, pitch={pitch}, yaw={yaw}")
    print("q=", q)
    print(f"roll_={roll_}, pitch_={pitch_}, yaw_={yaw_}")

    from plant_dataset import PlantDataset
    dataset_dir = "/home/lion397/codes/Image2PlantArchitecture/data/generated_dataset_Sep22_black"
    train_dataset = PlantDataset(dataset_dir, plot=["000", "001", "002",], use_depth=True, preload=False,
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
        vec = token2vec_new(out)
        tokens = vec2token_new(vec)

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