import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
