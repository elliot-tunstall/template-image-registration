import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2

def plot_image_grid(*images):
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, image in enumerate(images):
        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_image_and_mask(image, masks: list, alpha=0.5, pixelMap=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(image, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', cmap='gray')
    for mask in masks:
        cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))
        ax.imshow(mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', cmap=cmap, alpha=alpha)
    ax.axis('off')
    plt.show()

def draw_masks_fromList(image, masks_generated, labels, colors) :
  masked_image = image.copy()
  for i in range(len(masks_generated)) :
    masked_image = np.where(np.repeat(masks_generated[i][:, :, np.newaxis], 3, axis=2),
                            np.asarray(colors[int(labels[i][-1])], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

  return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)
