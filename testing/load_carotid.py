import sys
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase2/utils')
from loading import load_inputs as load
from plotting import show_image
from segmentation_toolkit import Segmentation
import os
import matplotlib.pyplot as plt

path = os.path.join('/Users/elliottunstall/Desktop/Imperial/FYP/', 'Carotid dataset/dr_dataset001_1__2022_07_08/')

fixed, moving, pxm = load('soft', 0, 15, path)

plt.figure()
plt.imshow(fixed)
plt.show(block=False)

# Segment the fixed image
seg_fixed = Segmentation(fixed, method='otsu')
# seg_fixed.kmeans(mrf=5)
# seg_fixed.otsu_thresholding()
seg_fixed.use_regions()
# seg_fixed.apply_smoothing(method='morph_closing', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_fixed.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
seg_fixed.apply_smoothing(method='gaussian', kernel_size=5, shape="[]", sigma=1, region='tissue')
seg_fixed.apply_smoothing(method='fill_holes', kernel_size=5, shape="[]", sigma=0.6, iterations=1, region='tissue')
seg_fixed.normalise_background()
fixed_mask = seg_fixed.masks['tissue'][0].mask

plt.figure()
plt.imshow(seg_fixed.regions['tissue'])
plt.show()
seg_fixed.show(alpha=0.4)

