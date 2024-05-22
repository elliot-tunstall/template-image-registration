## Testing registration propagation

import sys;
import time
import numpy as np
from skimage import metrics 
from tensorflow.keras.losses import BinaryCrossentropy
sys.path.append('/Users/elliottunstall/Desktop')
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase/utils')
from segmentation_toolkit import Mask, Segmentation
from useful_functions import load_inputs, show_image
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import mat73
from metrics_evaluation import metrics_evaluation
from field_evaluation import field_evaluation
from eval_toolkit import delete_outsideFOV, calc_pts_error
from Parameters import Parameters
from plotting import show_image, show_image_pts

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD
from dipy.viz import regtools
from dipy_custom_imwarp.dipy2.align.imwarp import SymmetricDiffeomorphicRegistration as custom_reg
from dipy_custom_imwarp.dipy2.align.metrics import SSDMetric as SSD2

from dipy.align.imaffine import AffineMap, MutualInformationMetric, AffineRegistration

bce = BinaryCrossentropy()

def run(frame):
        
    # Load the images
    # frame = 80
    fixed, moving, pxm = load_inputs(frame)

    # Segment the fixed image
    seg_fixed = Segmentation(fixed)
    seg_fixed.kmeans(mrf=0.15)
    seg_fixed.apply_smoothing(method='morph_closing')
    seg_fixed.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
    seg_fixed.masks['background'] = [seg_fixed.masks['background'][1], seg_fixed.masks['background'][2]]
    fixed_mask = seg_fixed.masks['tissue'][0].mask

    # load the algorithm
    p = Parameters('dipy', type='random', space_mult=10)
    p.use_with_defaults()
    p.set_manually({'grad_step': 0.3485268567702533, 'metric': 'SSD', 'num_iter': 308, 
                    'num_pyramids': 7, 'opt_end_thresh': 0.0010411677260804102, 'num_dim': 2, 
                    'smoothing': 0.18183433601302146})
    algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[44, 44, 44, 44, 44, 44, 44], 
                                                ss_sigma_factor=0.1818, opt_tol=0.001041168)

    # Run the algorithm                                         
    start_time = time.process_time()
    mapping = algorithm.optimize(fixed.astype(float), moving.astype(float))
    execution_time1 = time.process_time() - start_time
    deform_forward1 = mapping.forward
    deform_backward1 = mapping.backward
    output_image1 = mapping.transform(moving, 'linear')
    moving_mask = mapping.transform_inverse(fixed_mask, 'linear')

    seg_moving = Segmentation(moving)
    seg_moving.kmeans(mrf=0.15)
    seg_moving.apply_smoothing(method='morph_closing')
    seg_moving.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
    true_mask = seg_moving.masks['tissue'][0].mask
    dice = dice_coefficient(moving_mask, true_mask)
    hd = metrics.hausdorff_distance(moving_mask, true_mask)
    loss = bce(moving_mask, true_mask)
    Bce = loss.numpy()

    # print(execution_time1)
    # show_image(fixed, pxm)
    # show_image(output_image1, pxm)

    # Calculate the metrics
    errors = metrics_evaluation(output_image1, fixed, moving, execution_time1, 
                                "1-70_cardiac", '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/')
    newpts = field_evaluation(deform_forward1, [1, frame], p, 'cardiac', 
                            '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/', 
                            savename='', savedir='', output_image=output_image1)

    # show_image_pts(output_image1,pxm,newpts,1);plt.title('Resulting image')

    gts = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/eval_pts_'+str(frame)+'.mat')['pts']   # <- change path. used for evaluation  

    gts2, newpts2 = delete_outsideFOV(gts.copy(), newpts.copy())                                                # delete points outside of FOV

    mag_error1, sd_error1, mean_error1 = calc_pts_error(newpts2.copy(), gts2.copy())

    # load the custom algorithm
    p_custom = Parameters('dipy_custom', type='random', space_mult=10)
    p_custom.use_with_defaults()
    p_custom.set_manually({'grad_step': 0.3485268567702533, 'metric': 'SSD', 'num_iter': 308, 'num_pyramids': 7, 'opt_end_thresh': 0.0010411677260804102, 'num_dim': 2, 'smoothing': 0.18183433601302146})
    custom_algorithm = custom_reg(metric=SSD2(2), level_iters=[44, 44, 44, 44, 44, 44, 44], ss_sigma_factor=0.1818, opt_tol=0.001041168)

    # Run the custom algorithm                                         
    start_time = time.process_time()
    mapping = custom_algorithm.optimize(fixed.astype(float), moving.astype(float), fixed_mask.astype(float), moving_mask.astype(float))
    execution_time2 = time.process_time() - start_time
    deform_forward2 = mapping.forward
    deform_backward2 = mapping.backward
    output_image2 = mapping.transform(moving, 'linear')

    # Calculate the metrics
    errors_custom = metrics_evaluation(output_image2, fixed, moving, execution_time2, "1-70_cardiac_custom", '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/')
    newpts_custom = field_evaluation(deform_forward2, [1, frame], p_custom, 'cardiac', '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/', savename='', savedir='', output_image=output_image2)

    # show_image_pts(output_image2,pxm,newpts_custom,1);plt.title('Resulting image')

    gts = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/eval_pts_'+str(frame)+'.mat')['pts']   # <- change path. used for evaluation  

    gts2, newpts2 = delete_outsideFOV(gts.copy(), newpts_custom.copy())                                                # delete points outside of FOV

    mag_error2, sd_error2, mean_error2 = calc_pts_error(newpts2.copy(), gts2.copy())

    return errors, mag_error1, sd_error1, mean_error1, errors_custom, mag_error2, sd_error2, mean_error2, execution_time1, execution_time2, dice, hd, Bce


def dice_coefficient(static_mask, moving_mask):
        r"""Computes the Dice coefficient between two binary masks

        Computes the Dice coefficient between two binary masks. The Dice
        coefficient is a measure of the overlap between two binary masks and is
        defined as:

        .. math::

            D = \frac{2|A \cap B|}{|A| + |B|}

        where A and B are the binary masks and |.| denotes the number of
        elements in the set.

        Parameters
        ----------
        static_mask : array, shape (R, C) or (S, R, C)
            the static binary mask
        moving_mask : array, shape (R, C) or (S, R, C)
            the moving binary mask

        Returns
        -------
        dice : float
            the Dice coefficient between the two binary masks
        """
        intersection = np.sum(static_mask * moving_mask)
        union = np.sum(static_mask) + np.sum(moving_mask)
        return 2.0 * intersection / union



if __name__ == '__main__':
    errors, mag_error1, sd_error1, mean_error1, errors_custom, mag_error2, sd_error2, mean_error2, execution_time1, execution_time2 = run(40)
    print(errors)
    print(mag_error1)
    print(sd_error1)
    print(mean_error1)
    print(errors_custom)
    print(mag_error2)
    print(sd_error2)
    print(mean_error2)