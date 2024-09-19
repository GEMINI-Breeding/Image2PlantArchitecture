import numpy as np
import math


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
EOS_token = 41
PAD_token = 42

if 0:
    params_SOS_token_padded = np.ones(15)*SOS_token
    params_EOS_token_padded = np.ones(15)*EOS_token
else:
    # Zero padded params
    params_SOS_token_padded = np.zeros(15)
    params_SOS_token_padded[0] = SOS_token
    params_EOS_token_padded = np.zeros(15)
    params_EOS_token_padded[0] = EOS_token



def vec2token(vec, n_params=15):
    tokens = []
    for x in vec:
        depth_organ = x[0]*4 + x[1]
        if 1:
            token = np.zeros(n_params) # padding zeros to match the desired length
        else:
            token = np.ones(n_params) * PAD_token

        token[0] = depth_organ
        
        if x[1] == 0:
            # Shoot params    
            token[1] = x[2] / 180 * math.pi # shoot_base_pitch
            token[2] = x[3] / 180 * math.pi # shoot_base_yaw
            token[3] = x[4] / 180 * math.pi # shoot_base_roll
            token[4] = x[5] / 100           # shoot_gravitropic_curvature
            token[5] = x[6]                 # shoot_type
        elif x[1] == 1:
            # Internode params
            token[6] = x[2] * 100 # internode_length
            token[7] = x[3] * 100 # internode_radius
            token[8] = x[4] / 180 * math.pi # internode_pitch
        elif x[1] == 2:
            # Petiole params
            token[9] = x[2] * 100 # petiole_length
            token[10] = x[3] / 180 * math.pi # petiole_pitch
        elif x[1] == 3:
            # Leaf params
            token[11] = x[2] * 100 # leaf_scale
            token[12] = x[3] / 180 * math.pi # leaf_pitch
            token[13] = x[4] / 180 * math.pi # leaf_yaw
            token[14] = x[5] / 180 * math.pi # leaf_roll
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
            if j == 0:
                # Shoot
                params_padded[0] = token[1] * 180 / math.pi # shoot_base_pitch
                params_padded[1] = token[2] * 180 / math.pi # shoot_base_yaw
                params_padded[2] = token[3] * 180 / math.pi # shoot_base_roll
                params_padded[3] = token[4] * 100 # shoot_gravitropic_curvature
                params_padded[4] = token[5] # shoot_type
            elif j == 1:
                # Internode
                params_padded[0] = token[6] / 100 # internode_length
                params_padded[1] = token[7] / 100 # internode_radius
                params_padded[2] = token[8] * 180 / math.pi # internode_pitch
                # Hardcored phyllotactic angle because it is not given from Brian yet
                params_padded[3] = 137.5 # TODO phyllotactic angle. The most common angle is the golden angle, or Fibonacci angle, which is approximately 137.5°.
            elif j == 2:
                # Petiole
                params_padded[0] = token[9] / 100 # petiole_length
                params_padded[1] = 0.001 # TODO petiole radius is not given
                params_padded[2] = token[10] * 180 / math.pi # petiole_pitch
            elif j == 3:
                # Leaf
                params_padded[0] = token[11] / 100 # leaf_scale
                params_padded[1] = token[12] * 180 / math.pi # leaf_pitch
                params_padded[2] = token[13] * 180 / math.pi # leaf_yaw
                params_padded[3] = token[14] * 180 / math.pi # leaf_roll
            else:
                raise ValueError(f"Invalid organ type {j}")

            # Make 1x6 array with i, j and params
            vec.append(np.concatenate(([i, j], params_padded),axis=0))
    return np.array(vec)



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