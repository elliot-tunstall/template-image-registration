import sys
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase2/utils')
from loading import load_inputs as load
from plotting import show_image
from segmentation_toolkit import Segmentation
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# Define a function to format the tick values
def format_ticks(value, _):
    return f'{value * 0.077}'

def round_to_sf(arr, sf):
    """
    Rounds a NumPy array to the specified number of significant figures.
    
    Parameters:
    arr (np.array): The array to round.
    sf (int): Number of significant figures.
    
    Returns:
    np.array: The rounded array.
    """
    # Compute the order of magnitude
    order_of_magnitude = np.floor(np.log10(np.abs(arr)))
    
    # Calculate the scaling factor
    scaling_factor = np.power(10, order_of_magnitude - sf + 1)
    
    # Round the array
    rounded_array = np.round(arr / scaling_factor) * scaling_factor
    
    return rounded_array

path = os.path.join('/Users/elliottunstall/Desktop/Imperial/FYP/', 'Carotid dataset/dr_dataset001_10__2022_07_08/')

fixed, moving, pxm = load('soft', 0, 20, path)

plt.figure()
plt.imshow(fixed, cmap='gray')
plt.show(block=False)

# Segment the fixed image
seg_fixed = Segmentation(fixed, method='otsu')
# seg_fixed.kmeans(mrf=5)
# seg_fixed.otsu_thresholding()
seg_fixed.use_regions()
# seg_fixed.apply_smoothing(method='morph_closing', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_fixed.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_fixed.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
seg_fixed.apply_smoothing(method='fill_holes', region='tissue')
seg_fixed.normalise_background()
fixed_mask = seg_fixed.masks['tissue'][0].mask

# Segment the fixed image
seg_moving = Segmentation(moving, method='otsu')
# seg_fixed.kmeans(mrf=5)
# seg_fixed.otsu_thresholding()
seg_moving.use_regions()
# seg_fixed.apply_smoothing(method='morph_closing', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_moving.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_moving.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
seg_moving.apply_smoothing(method='fill_holes', region='tissue')
seg_moving.normalise_background()

plt.figure()
plt.imshow(seg_fixed.regions['tissue'])
plt.show()

fig, axs = plt.subplots(1,2, figsize=(12, 4))
seg_fixed.plot_axis(overlay_image=True, alpha=0.4, axis=axs[0])
seg_moving.plot_axis(overlay_image=True, alpha=0.4, axis=axs[1])

# Create custom legend handles
tissue_patch = mpatches.Patch(color='green', label='Tissue')
background_patch = mpatches.Patch(color='blue', label='Background')

# Define your scale factor
scale_factor = pxm['dx']*1000
print(scale_factor)

# Scale the axes of each subplot
for ax in axs:
    # Get the current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # # Set the new limits
    # ax.set_xlim([xlim[0]*scale_factor, xlim[1]*scale_factor])
    # ax.set_ylim([ylim[0]*scale_factor, ylim[1]*scale_factor])

# Apply the function to the x and y axis
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    ax.set_xticks(np.linspace(0, fixed.shape[1], num=5))
    ax.set_yticks(np.linspace(0, fixed.shape[0], num=5))

    ax.set_xticklabels(np.round(np.linspace(-0.5 * fixed.shape[1]*0.077, 0.5 * fixed.shape[1]*0.077, num=5)))
    ax.set_yticklabels(np.round(np.linspace(0, fixed.shape[0]*0.077, num=5)))

    ax.set_xlabel('Lateral Displacement (mm)')
    ax.set_ylabel('Depth (mm)')

fig.legend(handles=[tissue_patch, background_patch], loc='upper center', ncol=2)

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5)
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Carotid/segmented_10.png', dpi=1200)
plt.show()
