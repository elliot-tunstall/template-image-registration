import sys
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase2/utils')
from loading import load_inputs as load
from segmentation_toolkit import Mask, Segmentation
from useful_functions import load_inputs, show_image
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import time

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD


def plot_segmentation(dataset, phase, dataset_number, frame):
    start_time = time.process_time()
    if dataset == "cardiac":
        path = '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/'
        if phase == "diastole":
            fixed, moving, pxm = load_inputs(frame, 1)
        else:
            fixed, moving, pxm = load_inputs(frame+35, 35)

        seg_fixed = Segmentation(fixed, method='kmeans')
        seg_fixed.apply_smoothing(method='morph_closing')
        seg_fixed.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
        seg_fixed.masks['background'] = []
        fixed_mask = seg_fixed.masks['tissue'][0].mask

        # load the algorithm
        algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[44, 44, 44, 44, 44, 44, 44], 
                                                    ss_sigma_factor=0.1818, opt_tol=0.001041168)

    elif dataset == "soft":
        path = f'/Users/elliottunstall/Desktop/Imperial/FYP/Carotid dataset/dr_dataset001_{dataset_number}__2022_07_08/'
        fixed, moving, pxm = load('soft', 0, frame, path)
        seg_fixed = Segmentation(fixed, method='otsu')
        seg_fixed.use_regions()
        seg_fixed.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
        seg_fixed.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
        seg_fixed.apply_smoothing(method='fill_holes', region='tissue')
        seg_fixed.normalise_background()
        seg_fixed.masks['background'] = []
        fixed_mask = seg_fixed.masks['tissue'][0].mask

        algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[169, 169], 
                                                    ss_sigma_factor=0.36756503579153, opt_tol=0.000581157573915277)

    if dataset == "cardiac":
        seg_moving = Segmentation(moving, method='kmeans')
        seg_moving.apply_smoothing(method='morph_closing')
        seg_moving.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
        seg_moving.masks['background'] = []

    elif dataset == "soft":
        seg_moving = Segmentation(moving, method='otsu')
        seg_moving.use_regions()
        seg_moving.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
        seg_moving.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
        seg_moving.apply_smoothing(method='fill_holes', region='tissue')
        seg_moving.normalise_background()
        seg_moving.masks['background'] = []

    
    mapping = algorithm.optimize(fixed.astype(float), moving.astype(float))
    deform_forward1 = mapping.forward
    deform_backward1 = mapping.backward
    output_image1 = mapping.transform(moving, 'linear')
    moving_mask = mapping.transform_inverse(fixed_mask, 'linear')

    execution_time1 = time.process_time() - start_time

    print(f"Frame {frame} processed in {execution_time1} seconds")

    return seg_moving, moving_mask

if __name__ == "__main__":

    dataset_number = 1
    frames = [2, 6, 10, 14, 18]
    path = '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/'
    _, _, pxm = load_inputs(1, 1)

    # Create a new figure
    fig, axs = plt.subplots(3, 5, figsize=(12, 6))

    # # Adjust the spacing between the subplots
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Remove the spacing around the figure and between the subplots
    plt.subplots_adjust(left=0.005, right=0.995, bottom=0, top=1, wspace=0.05, hspace=0.05)

    # Plot the first 15 images in a 5x3 grid
    for i in range(5):
        for j in range(3):

            # Calculate the index of the image
            if j == 0:
                dataset = "cardiac"
                phase = "diastole"
            elif j == 1:
                dataset = "cardiac"
                phase = "systole"
            elif j == 2:
                dataset = "soft"
                phase = "None"

            seg_moving, moving_mask = plot_segmentation(dataset, phase, dataset_number, frames[i])
            # Plot the image
            if j == 2:
                seg_moving.plot_axis(overlay_image=True, alpha=0.5, overlay_mask=moving_mask, axis = axs[j, i])
            else:
                seg_moving.plot_axis(pixelMap = pxm, overlay_image=True, alpha=0.5, overlay_mask=moving_mask, axis = axs[j, i])

            # # Remove the axis
            # axs[j, i].axis('off')

            # Remove the tick labels but keep the axis
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

    # Show the figure

    # Add lines between the subplots
    # for i in range(1, 2):
    #     fig.add_artist(lines.Line2D([0, 1], [i/2, i/2], color='black', transform=fig.transFigure))
    for i in range(1, 5):
        fig.add_artist(lines.Line2D([i/5, i/5], [0, 1], color='black', transform=fig.transFigure))
    
    plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Results/SAS_figures_live.png', dpi=1200)
    plt.show()
    

