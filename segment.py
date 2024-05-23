import time
from utils.segmentation_toolkit import Mask, Segmentation
from utils.useful_functions import load_inputs, show_image
import matplotlib.pyplot as plt
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['white', 'green'])

fixed, moving, pxm = load_inputs(50)
plt.imshow(moving, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap='gray')
plt.show()

seg_fixed = Segmentation(fixed, method='kmeans')
seg_moving = Segmentation(moving, method='kmeans')
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
seg_fixed.show(pixelMap = pxm, overlay_image=True, alpha=0.4)
seg_moving.show(pixelMap = pxm, overlay_image=True, alpha=0.4)
plt.imshow(ignoreRegion)
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
seg_fixed.show(pixelMap = pxm, overlay_image=True, alpha=0.4)
plt.imshow(fixed_mask, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap=cmap)
plt.show()
plt.imshow(deform_backward1[:,:,0], extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper')
plt.show()
plt.imshow(deform_backward1[:,:,1], extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper')
plt.show()
seg_moving.show(pixelMap = pxm, overlay_image=True, alpha=0.4)
plt.imshow(moving_mask, extent=(pxm['X'][0][0].min(), pxm['X'][0][0].max(), pxm['Z'][0][0].max(), pxm['Z'][0][0].min()), origin='upper', cmap=cmap)
plt.show()

