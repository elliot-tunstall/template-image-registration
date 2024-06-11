import sys
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase2/utils')
sys.path.append('/Users/elliottunstall/Desktop')
from loading import load_soft_gts
from loading import load_inputs as load_inputs2
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD
from segmentation_toolkit import Segmentation
import numpy as np
from dipy_custom_imwarp.dipy2.align.metrics import SSDMetric as SSD2
from dipy_custom_imwarp.dipy2.align.imwarp import SymmetricDiffeomorphicRegistration as custom_reg

mpl.rcParams['xtick.labelsize'] = 12  # fontsize for x-axis tick labels
mpl.rcParams['ytick.labelsize'] = 12  # fontsize for y-axis tick labels
mpl.rcParams['figure.titlesize'] = 16  # fontsize for the overall plot title
mpl.rcParams['axes.labelsize'] = 14  # fontsize for x-axis and y-axis labels

path = '/Users/elliottunstall/Desktop/Imperial/FYP/Carotid dataset/dr_dataset001_10__2022_07_08/'
# path = '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/'
framenums = [0, 20]

gt = load_soft_gts(path, framenums[1])
fixed,moving,pxm = load_inputs2('soft',framenums[0],framenums[1],path)

# seg_fixed = Segmentation(fixed, method='kmeans')
# seg_fixed.apply_smoothing(method='morph_closing')
# seg_fixed.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
# seg_fixed.masks['background'] = [seg_fixed.masks['background'][1], seg_fixed.masks['background'][2]]
# fixed_mask = seg_fixed.masks['tissue'][0].mask

# # load the algorithm
# algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[44, 44, 44, 44, 44, 44, 44], 
#                                         ss_sigma_factor=0.1818, opt_tol=0.001041168)
# custom_algorithm = custom_reg(metric=SSD2(2, alpha=0, beta=0), level_iters=[44, 44, 44, 44, 44, 44, 44], ss_sigma_factor=0.1818, opt_tol=0.001041168)

# Segment the fixed image
seg_fixed = Segmentation(fixed, method='otsu')
seg_fixed.use_regions()
# seg_fixed.apply_smoothing(method='morph_closing', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_fixed.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_fixed.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
seg_fixed.apply_smoothing(method='fill_holes', region='tissue')
seg_fixed.normalise_background()
fixed_mask = seg_fixed.masks['tissue'][0].mask

algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[169, 169], 
                                                    ss_sigma_factor=0.36756503579153, opt_tol=0.000581157573915277)
custom_algorithm = custom_reg(metric=SSD2(2, alpha=0, beta=1), level_iters=[169, 169], 
                                                    ss_sigma_factor=0.36756503579153, opt_tol=0.000581157573915277)


mapping = algorithm.optimize(fixed.astype(float), moving.astype(float))
deform_forward = mapping.forward
deform_backward = mapping.backward
output_image1 = mapping.transform(moving, 'linear')

moving_mask = mapping.transform_inverse(fixed_mask, 'linear')

error = gt - deform_forward
error[:,:,0] = error[:,:,0]*pxm['dx']
error[:,:,1] = error[:,:,1]*pxm['dz']

# Define the scale factor
scale_factor_x = pxm['dx'] * fixed.shape[0] * 500
scale_factor_z = pxm['dz'] * fixed.shape[1] * 500

# Calculate the extent of the image
extent = (-scale_factor_x, scale_factor_x, -scale_factor_z, scale_factor_z)

# Create an axis for the colorbar
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.figure();plt.imshow(gt[:,:,0]*pxm['dx']*1000000, extent=extent, aspect='auto');plt.title('Induced Motion (\u03BCm)', fontsize=16, fontweight='bold'); plt.colorbar();plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/1.png', dpi=1200)
# plt.figure();plt.imshow(gt[:,:,1]);plt.colorbar();plt.title('GT z')
plt.figure();plt.imshow(deform_forward[:,:,0]*pxm['dx']*1000000, extent=extent, aspect='auto');plt.title('Baseline Est. (\u03BCm)', fontsize=16, fontweight='bold');plt.colorbar(); plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/2.png', dpi=1200)
# plt.figure();plt.imshow(deform_forward[:,:,1]);plt.colorbar();plt.title('est z')
plt.figure();plt.imshow(error[:,:,0]*1000000, cmap='inferno', extent=extent, aspect='auto');plt.title('Absolute Difference (\u03BCm)', fontsize=16, fontweight='bold');plt.colorbar();plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/3.png', dpi=1200)



mapping = custom_algorithm.optimize(fixed.astype(float), moving.astype(float), fixed_mask, moving_mask)
deform_forward = mapping.forward
output_image2 = mapping.transform(moving, 'linear')

error = gt - deform_forward
error[:,:,0] = error[:,:,0]*pxm['dx']
error[:,:,1] = error[:,:,1]*pxm['dz']

# plt.figure();plt.imshow(gt[:,:,0]);plt.colorbar();plt.title('GT x')
# plt.figure();plt.imshow(gt[:,:,1]);plt.colorbar();plt.title('GT z')
plt.figure();plt.imshow(deform_forward[:,:,0]*pxm['dx']*1000000, extent=extent, aspect='auto');plt.colorbar();plt.title('Co-Registration Est. (\u03BCm)', fontsize=16, fontweight='bold'); plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/4.png', dpi=1200)
# plt.figure();plt.imshow(deform_forward[:,:,1]);plt.colorbar();plt.title('est z')
plt.figure();plt.imshow(error[:,:,0]*1000000, cmap='inferno', extent=extent, aspect='auto');plt.colorbar();plt.title('Absolute Difference (\u03BCm)', fontsize=16, fontweight='bold'); plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/5.png', dpi=1200)
plt.figure();plt.imshow(output_image1, cmap='grey', extent=extent, aspect='auto');plt.title('Output Image', fontsize=16, fontweight='bold');plt.colorbar(); plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/6.png', dpi=1200)
plt.figure();plt.imshow(output_image2, cmap='grey', extent=extent, aspect='auto');plt.title('Output Image', fontsize=16, fontweight='bold'); plt.colorbar();plt.xlabel('Lateral (mm)'); plt.ylabel('Depth (mm)'); 
plt.savefig('/Users/elliottunstall/Desktop/Imperial/FYP/Figures/deformation field/7.png', dpi=1200)
plt.show()