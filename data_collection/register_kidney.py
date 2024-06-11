import sys
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase2/utils')
sys.path.append('/Users/elliottunstall/Desktop')
import mat73
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.util import img_as_ubyte
from segmentation_toolkit import Segmentation, Mask
import cv2
import time
from scipy.io import savemat

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD
from dipy_custom_imwarp.dipy2.align.imwarp import SymmetricDiffeomorphicRegistration as custom_reg
from dipy_custom_imwarp.dipy2.align.metrics import SSDMetric as SSD2

print("Loading data...")
kidney_images = mat73.loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/elliot_kidney_data.mat')
print("Data loaded")

fixed = abs(kidney_images['ImgData'][:,:,0,0])

# Load an image
image = cv2.imread('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/frame_rescaled_001_markup.jpg')

# Define lower and upper thresholds for red color
lower_red = 130
upper_red = 135

# initialize results
shape = kidney_images['ImgData'].shape
results = np.zeros((shape[0], shape[1], 2, shape[2], 2))
output_images = np.zeros((shape[0], shape[1], shape[2], 2))
print(np.shape(results))

# load the algorithm
algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[44, 44, 44, 44, 44, 44, 44], 
                                        ss_sigma_factor=0.1818, opt_tol=0.001041168)
custom_algorithm = custom_reg(metric=SSD2(2, alpha=0, beta=1), level_iters=[44, 44, 44, 44, 44, 44, 44], ss_sigma_factor=0.1818, opt_tol=0.001041168)

# Threshold the image to get only red colors
mask = cv2.inRange(image[:,:,0], lower_red, upper_red)

for i in range(0, np.shape(kidney_images['ImgData'])[2]):

    bmode = abs(kidney_images['ImgData'][:,:,i,0])
    am = abs(kidney_images['ImgData'][:,:,i,1])
    
    # Define the range of intensities you want to enhance
    v_min, v_max = np.percentile(bmode, (2,98))

    # Rescale the intensities to the range 0-1
    image_rescaled = exposure.rescale_intensity(bmode, in_range=(v_min, v_max))

    # Run the algorithm                                         
    start_time = time.process_time()
    mapping = algorithm.optimize(fixed.astype(float), bmode.astype(float))
    execution_time1 = time.process_time() - start_time
    deform_forward1 = mapping.forward
    deform_backward1 = mapping.backward
    output_image1 = mapping.transform(bmode, 'linear')
    moving_mask = mapping.transform_inverse(mask, 'linear')

    start_time = time.process_time()
    mapping = custom_algorithm.optimize(fixed.astype(float), bmode.astype(float), mask.astype(float), moving_mask.astype(float))
    execution_time2 = time.process_time() - start_time
    deform_forward2 = mapping.forward
    deform_backward2 = mapping.backward
    output_image2 = mapping.transform(bmode, 'linear')

    # store results 
    results[:,:,:,i,0] = deform_forward1
    results[:,:,:,i,1] = deform_forward2
    output_images[:,:,i,0] = output_image1
    output_images[:,:,i,1] = output_image2

    print(f"Image {i} processed of {len(abs(kidney_images['ImgData'][0,0,:,0]))}")
    print("Execution time for dipy:", execution_time1)
    print("Execution time for custom:", execution_time2)
    print("estimated tiem remaining:", (execution_time1 + execution_time2) * (len(abs(kidney_images['ImgData'][0,0,:,0])) - i))

np.save('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/field_results.npy', results)
savemat('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/field_results.mat', {'deformField': results})
np.save('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/output_images.npy', output_images)
savemat('/Users/elliottunstall/Desktop/Imperial/FYP/Kidney dataset/output_images.mat', {'outputImages': output_images})