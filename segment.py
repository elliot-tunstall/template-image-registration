import time
from utils.segmentation_toolkit import Mask, Segmentation
from utils.useful_functions import load_inputs, show_image
import matplotlib.pyplot as plt
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

cmap = ListedColormap(['white', 'green'])

fixed, moving, pxm = load_inputs(55, 1)
print(pxm['X'][0][0].min())
print(pxm['X'][0][0].max())

plt.imshow(moving, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap='gray')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/moving_09.png', dpi=1200)
plt.show()

plt.imshow(fixed, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap='gray')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/fixed_09.png', dpi=1200)
plt.show()


seg_fixed = Segmentation(fixed, method='kmeans')
seg_moving = Segmentation(moving, method='kmeans')
# seg_moving.kmeans(mrf=0.25)
# seg_fixed.kmeans(mrf=0.15)
seg_fixed.apply_smoothing(method='morph_closing')
# seg_moving.kmeans(mrf=0.15)
seg_moving.apply_smoothing(method='morph_closing')
# print(seg_fixed.masks['tissue'])
seg_fixed.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
ignoreRegion = seg_fixed.masks['background'][0].mask
seg_fixed.masks['background'] = [seg_fixed.masks['background'][1], seg_fixed.masks['background'][2]]
seg_moving.masks['tissue'] = [seg_moving.masks['tissue'][0]]
seg_moving.masks['background'] = [seg_moving.masks['background'][1], seg_moving.masks['background'][2]]

# seg_fixed.masks['background'] = [seg_fixed.masks['background'][1], seg_fixed.masks['background'][2]]
# seg_fixed.masks['tissue'][0].show(pixelMap = pxm)
# seg_fixed.masks['background'][0].show(pixelMap = pxm, alpha=0.5)

fig, axs = plt.subplots(1,2)

pxm['X'][0][0] = pxm['X'][0][0]*1000
pxm['Z'][0][0] = pxm['Z'][0][0]*1000

tissue_patch = mpatches.Patch(color='green', label='Tissue')
background_patch = mpatches.Patch(color='blue', label='Background')

seg_fixed.plot_axis(pixelMap = pxm, overlay_image=True, alpha=0.4, axis=axs[0])
seg_moving.plot_axis(pixelMap = pxm, overlay_image=True, alpha=0.4, axis=axs[1])

# Define your scale factor
scale_factor = 1000

# Scale the axes of each subplot
for ax in axs:
    # Get the current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # ax.set_xticks(np.linspace(0, fixed.shape[1], num=5))
    # ax.set_yticks(np.linspace(0, fixed.shape[0], num=5))

    # ax.set_xticklabels(np.round(np.linspace(pxm['X'][0][0].min(), pxm['X'][0][0].max(), num=5), decimals=3))
    # ax.set_yticklabels(np.round(np.linspace(pxm['Z'][0][0].min(), pxm['Z'][0][0].max(), num=5), decimals=3))

    ax.set_xlabel('Lateral Displacement (mm)')
    ax.set_ylabel('Depth (mm)')

fig.legend(handles=[tissue_patch, background_patch], loc='upper center', ncol=2)
plt.subplots_adjust(wspace=0.5)
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/segmented_09.png', dpi=1200)
plt.show()


fixed_mask = seg_fixed.masks['tissue'][0].mask

algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[44, 44, 44, 44, 44, 44, 44], 
                                                ss_sigma_factor=0.1818, opt_tol=0.001041168)

start_time = time.process_time()
mapping = algorithm.optimize(fixed.astype(float), moving.astype(float))
execution_time1 = time.process_time() - start_time
deform_forward1 = mapping.forward
deform_backward1 = mapping.backward
output_image1 = mapping.transform(moving, 'linear')
moving_mask = mapping.transform_inverse(fixed_mask, 'linear')

seg_moving.masks['tissue'] = [Mask(moving_mask, 0, 'tissue')]
seg_fixed.masks['background'] = []
seg_moving.masks['background'] = []

plt.figure()
seg_fixed.show(pixelMap = pxm, overlay_image=True, alpha=0.4, axis=True)
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/fixed_seg_09.png', dpi=1200)
plt.show()

plt.imshow(fixed_mask, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap=cmap)
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/fixed_mask_09.png', dpi=1200)
plt.show()

plt.imshow(deform_backward1[:,:,0], extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/df_Z_09.png', dpi=1200)
plt.show()

plt.imshow(deform_backward1[:,:,1], extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/df_x_09.png', dpi=1200)
plt.show()

plt.figure()
seg_moving.show(pixelMap = pxm, overlay_image=True, alpha=0.4, axis=True)
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/moving_seg_09.png', dpi=1200)
plt.show()

plt.imshow(moving_mask, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap=cmap)
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Depth (mm)')
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/Cardiac/moving_mask_09.png', dpi=1200)
plt.show()

