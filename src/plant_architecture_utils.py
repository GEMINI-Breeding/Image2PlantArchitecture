import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

def plot_image(image):
    # Plot the image
    image_vis = image.permute(0, 2, 3, 1).cpu().numpy()
    img_rgb = image_vis[0, :, :, :3]
    img_depth = image_vis[0, :, :, 3]
    # Normalize img_rgb to 0-255 per channels
    for i in range(3):
        img_rgb[:, :, i] = (img_rgb[:, :, i] - img_rgb[:, :, i].min()) / (img_rgb[:, :, i].max() - img_rgb[:, :, i].min()) * 255
    img_rgb = img_rgb.astype(np.uint8)
    # BGR to RGB
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(img_depth)
    plt.show()


def visualize_attention(image, attention_weights, words, word_index, layer_index, interpolation=cv2.INTER_CUBIC):
    """
    Visualize the attention map for a specific word in the sequence from a specific layer.
    
    Args:
    - image (np.array): The original image.
    - attention_weights (torch.Tensor): The attention weights from the Transformer Decoder.
    - words (list): The list of words in the sequence.
    - word_index (int): The index of the word to visualize.
    - layer_index (int): The index of the layer to visualize.
    """
    # Get the attention map for the specific word and layer
    attention_map = attention_weights[layer_index].squeeze()[word_index].detach().cpu().numpy()
    
    # Remove the CLS token
    # Check if the attention map is n*n+1
    if not np.sqrt(attention_map.shape[0]).is_integer():
        # Remove the CLS token
        attention_map = attention_map[1:]
    # Reshape the attention map to sqaure image
    feature_size = int(np.sqrt(attention_map.shape[0]))
    attention_map = attention_map.reshape(feature_size, feature_size)

    # Reshape the attention map to the size of the image
    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]), interpolation=interpolation)
    
    # Normalize the attention map
    attention_map = attention_map / attention_map.max()
    
    # Overlay the attention map on the image
    overlay = image.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
    
    # # Plot the original image and the attention map
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # # Original image
    # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0].set_title("Original Image")
    # axes[0].axis('off')
    
    # # Attention map
    # axes[1].imshow(overlay)
    # axes[1].set_title(f"Layer {layer_index} - Attention Map for '{words[word_index]}'")
    # axes[1].axis('off')
    
    # plt.show()

    return overlay

def euler_to_quaternion(roll, pitch, yaw, degrees=True, order='xyz'):
    """
    Convert Roll-Pitch-Yaw angles (in radians) to Quaternion using SciPy.
    :param roll: Roll angle (φ) in radians
    :param pitch: Pitch angle (θ) in radians
    :param yaw: Yaw angle (ψ) in radians
    :return: Quaternion as a numpy array [q_w, q_x, q_y, q_z]
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    # Create a rotation object from Euler angles (RBZ convention)
    if order == 'xyz':
        rotation = R.from_euler(order, [roll, pitch, yaw])
    elif order == 'yzx':
        rotation = R.from_euler(order, [pitch, yaw, roll])
    else:
        rotation = R.from_euler(order, [roll, pitch, yaw])

    # Convert the rotation object to a quaternion
    quaternion = rotation.as_quat()  # Returns [q_x, q_y, q_z, q_w]
    
    # Rearranging to [q_w, q_x, q_y, q_z]
    return np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

def quaternion_to_euler(q, degrees=True, order='xyz'):
    """
    Convert Quaternion to Roll-Pitch-Yaw angles (in radians).
    :param q: Quaternion as a numpy array [q_w, q_x, q_y, q_z]
    :param degrees: Whether to return angles in degrees
    :param order: Rotation order for Euler angles. Default is 'xyz' (Roll-Pitch-Yaw), you can also use 'yzx' (Pitch-Yaw-Roll) etc.
    :return: Roll, Pitch, Yaw in radians as a tuple (roll, pitch, yaw)
    """
    rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # SciPy expects [q_x, q_y, q_z, q_w]
    if order == 'xyz':
        roll, pitch, yaw = rotation.as_euler(order)  # Returns roll, pitch, yaw in radians
    elif order == 'yzx':
        pitch, yaw, roll = rotation.as_euler(order)
    else:
        roll, pitch, yaw = rotation.as_euler(order)
        
    if degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
    return roll, pitch, yaw

def test_conversion(roll, pitch, yaw):
    # Step 1: Convert RPY to Quaternion
    quaternion = euler_to_quaternion(roll, pitch, yaw, degrees=True)
    print(f"Original RPY: (Roll: {roll}, Pitch: {pitch}, Yaw: {yaw})")
    print(f"Converted Quaternion: {quaternion}")

    # Step 2: Convert Quaternion back to RPY
    converted_roll, converted_pitch, converted_yaw = quaternion_to_euler(quaternion, degrees=True)
    
    print(f"Converted back to RPY: (Roll: {converted_roll}, Pitch: {converted_pitch}, Yaw: {converted_yaw})")
    
    # Check if the original and converted values are close
    roll_diff = converted_roll - roll
    pitch_diff = converted_pitch - pitch
    yaw_diff = converted_yaw - yaw
    
    print(f"Differences: (Roll: {roll_diff}, Pitch: {pitch_diff}, Yaw: {yaw_diff})")
    print("---------------------------------------------------")


def angle_to_coordinates(theta, degrees=True):
    """
    Convert an angle (in degrees) to Cartesian coordinates (x, y).
    :param theta: Angle in degrees
    :return: Tuple (x, y), where x = cos(theta), y = sin(theta)
    """
    if degrees:
        # Convert to radians for calculation
        radians = np.radians(theta)
    radians = radians % (2 * np.pi)

    x = np.cos(radians)
    y = np.sin(radians)
    return x, y


def coordinates_to_angle(x, y, degrees=True, angle_min=None, angle_max=None):
    """
    Convert Cartesian coordinates (x, y) back to an angle (in degrees).
    :param x: x coordinate
    :param y: y coordinate
    :return: Angle in degrees
    """
    angle = np.arctan2(y, x)  # Get angle in degrees     
    if angle < 0:
        angle += 2 * np.pi
    if degrees:
        angle = np.degrees(angle)

    # Check if the angle is within the specified range
    if angle_min is not None:
        if angle < angle_min:
            angle += 360

    # Check if the angle is within the specified range
    if angle_max is not None:
        if angle > angle_max:
            angle -= 360
    
    return angle


def test_angle_conversion(theta):
    print(f"Testing angle: {theta} degrees")
    
    # Convert angle to coordinates
    x, y = angle_to_coordinates(theta)
    print(f"Coordinates (x, y): ({x}, {y})")

    # Convert back to angle
    converted_angle = coordinates_to_angle(x, y)
    print(f"Converted back to angle: {converted_angle} degrees")
    
    # Check for consistency
    assert np.isclose(converted_angle, theta % 360), "Converted angle does not match the original angle!"
    print("Test passed!\n")

if __name__ == "__main__":
    # Test cases with angles in degrees
    test_angles = [0, 90, 180, 270, 360, 45, 135, 225, 315, 720]

    for angle in test_angles:
        test_angle_conversion(angle)
        test_angle_conversion(-angle)

    from plant_dataset import PlantDataset
    dataset_dir = "/home/lion397/codes/Image2PlantArchitecture/data/generated_dataset_Sep22_black"
    train_dataset = PlantDataset(dataset_dir, plot=["000", "001", "002",], load_depth=True, preload=False,
                                 process_leaf=False,
                                 image_size=224)
    from tqdm import tqdm
    for i, data in tqdm(enumerate(train_dataset)):
        vec = train_dataset.getitem(i)[1]
        test_angles = vec[0][2:5]
        for angle in test_angles:
            test_angle_conversion(angle)



    test_cases = [
        (30, 45, 60),
        (90, 0, 0),
        (120, 45, 30),
        (0, 90, 0),
        (0, 0, 180),
        (45, 45, 45),
        (30, 60, 90),
        (60, 30, 90),
        (15, 15, 30),
        (45, 90, 180)
    ]

    for roll, pitch, yaw in test_cases:
        test_conversion(roll, pitch, yaw)