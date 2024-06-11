## Testing registration propagation

import sys;
import time
import numpy as np
from skimage import metrics 
from tensorflow.keras.losses import BinaryCrossentropy
sys.path.append('/Users/elliottunstall/Desktop')
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase2/utils')
from loading import load_inputs as load
from segmentation_toolkit import Mask, Segmentation
from useful_functions import load_inputs, show_image
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import mat73
from metrics_evaluation import metrics_evaluation
from field_evaluation import field_evaluation
from eval_toolkit import delete_outsideFOV, delete_outside_mask, calc_pts_error, calc_df_error
from Parameters import Parameters
from plotting import show_image, show_image_pts

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric as SSD
from dipy.viz import regtools
from dipy_custom_imwarp.dipy2.align.imwarp import SymmetricDiffeomorphicRegistration as custom_reg
from dipy_custom_imwarp.dipy2.align.metrics import SSDMetric as SSD2

from dipy.align.imaffine import AffineMap, MutualInformationMetric, AffineRegistration

bce = BinaryCrossentropy()

def run(frame, dataset='cardiac', fixed_frame=1, dataset_number=1, alpha=0, beta=1):

    if dataset == 'cardiac':
        # Load the images
        path = '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/'
        fixed, moving, pxm = load_inputs(frame, fixed_frame)

        # Segment the fixed image
        seg_fixed = Segmentation(fixed, method='kmeans')
        seg_fixed.apply_smoothing(method='morph_closing')
        seg_fixed.masks['tissue'] = [seg_fixed.masks['tissue'][0]]
        seg_fixed.masks['background'] = [seg_fixed.masks['background'][1], seg_fixed.masks['background'][2]]
        fixed_mask = seg_fixed.masks['tissue'][0].mask

        # load the algorithm parameters
        p = Parameters('dipy', type='random', space_mult=10)
        p.use_with_defaults()
        p.set_manually({'grad_step': 0.3485268567702533, 'metric': 'SSD', 'num_iter': 308, 
                        'num_pyramids': 7, 'opt_end_thresh': 0.0010411677260804102, 'num_dim': 2, 
                        'smoothing': 0.18183433601302146})
        
        # load the custom algorithm parameters
        p_custom = Parameters('dipy_custom', type='random', space_mult=10)
        p_custom.use_with_defaults()
        p_custom.set_manually({'grad_step': 0.3485268567702533, 'metric': 'SSD', 'num_iter': 308, 'num_pyramids': 7, 'opt_end_thresh': 0.0010411677260804102, 'num_dim': 2, 'smoothing': 0.18183433601302146})
        
        # load the algorithm
        algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[44, 44, 44, 44, 44, 44, 44], 
                                                ss_sigma_factor=0.1818, opt_tol=0.001041168)
        custom_algorithm = custom_reg(metric=SSD2(2, alpha=alpha, beta=beta), level_iters=[44, 44, 44, 44, 44, 44, 44], ss_sigma_factor=0.1818, opt_tol=0.001041168)

    elif dataset == 'soft':
         # Load the images
        path = f'/Users/elliottunstall/Desktop/Imperial/FYP/Carotid dataset/dr_dataset001_{dataset_number}__2022_07_08/'
        fixed, moving, pxm = load('soft', fixed_frame, frame, path)

        # Segment the fixed image
        seg_fixed = Segmentation(fixed, method='otsu')
        seg_fixed.use_regions()
        # seg_fixed.apply_smoothing(method='morph_closing', kernel_size=3, shape="+", iterations=3, region='tissue')
        seg_fixed.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
        seg_fixed.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
        seg_fixed.apply_smoothing(method='fill_holes', region='tissue')
        seg_fixed.normalise_background()
        fixed_mask = seg_fixed.masks['tissue'][0].mask

        # load the algorithm parameters
        p = Parameters('dipy', type='random', space_mult=10)
        p.use_with_defaults()
        p.set_manually({'grad_step': 0.3619964484722210, 'metric': 'SSD', 'num_iter': 338, 
                        'num_pyramids': 2, 'opt_end_thresh': 0.000581157573915277, 'num_dim': 2, 
                        'smoothing': 0.36756503579153})
        
        # load the custom algorithm parameters
        p_custom = Parameters('dipy_custom', type='random', space_mult=10)
        p_custom.use_with_defaults()
        p_custom.set_manually({'grad_step': 0.3619964484722210, 'metric': 'SSD', 'num_iter': 338, 
                        'num_pyramids': 2, 'opt_end_thresh': 0.000581157573915277, 'num_dim': 2, 
                        'smoothing': 0.36756503579153})
        

        # load the algorithm
        algorithm = SymmetricDiffeomorphicRegistration(metric=SSD(2), level_iters=[169, 169], 
                                                ss_sigma_factor=0.36756503579153, opt_tol=0.000581157573915277)
        custom_algorithm = custom_reg(metric=SSD2(2, alpha=alpha, beta=beta), level_iters=[169, 169], 
                                                ss_sigma_factor=0.36756503579153, opt_tol=0.000581157573915277)


    # Run the algorithm                                         
    start_time = time.process_time()
    mapping = algorithm.optimize(fixed.astype(float), moving.astype(float))
    execution_time1 = time.process_time() - start_time
    deform_forward1 = mapping.forward
    deform_backward1 = mapping.backward
    output_image1 = mapping.transform(moving, 'linear')
    moving_mask = mapping.transform_inverse(fixed_mask, 'linear')

    if dataset == 'cardiac':
        seg_moving = Segmentation(moving, method='kmeans')
        seg_moving.apply_smoothing(method='morph_closing')
        seg_moving.masks['tissue'] = [seg_moving.masks['tissue'][0]]

    elif dataset == 'soft':
         # Segment the fixed image
        seg_moving = Segmentation(moving, method='otsu')
        seg_moving.use_regions()
        # seg_fixed.apply_smoothing(method='morph_closing', kernel_size=3, shape="+", iterations=3, region='tissue')
        seg_moving.apply_smoothing(method='binary_dilation', kernel_size=3, shape="+", iterations=3, region='tissue')
        seg_moving.apply_smoothing(method='gaussian', kernel_size=5, sigma=1, region='tissue')
        seg_moving.apply_smoothing(method='fill_holes', region='tissue')
        seg_moving.normalise_background()

    true_mask = seg_moving.masks['tissue'][0].mask
    dice = dice_coefficient(moving_mask, true_mask)
    hd = metrics.hausdorff_distance(moving_mask, true_mask)
    loss = bce(moving_mask, true_mask)
    Bce = loss.numpy()

    # print(execution_time1)
    # show_image(fixed, pxm)
    # show_image(output_image1, pxm)

    # Calculate the metrics
    errors = metrics_evaluation(output_image1, fixed, moving, execution_time1, '', '')

    if dataset == 'cardiac':
        newpts = field_evaluation(deform_forward1, [fixed_frame, frame], p, dataset, 
                            path, savename='', savedir='', output_image=output_image1)
        # show_image_pts(output_image1,pxm,newpts,1);plt.title('Resulting image')

        gts = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/eval_pts_'+str(frame)+'.mat')['pts']   # <- change path. used for evaluation  

        gts2, newpts2 = delete_outsideFOV(gts.copy(), newpts.copy())                                                # delete points outside of FOV

        mag_error1, sd_error1, mean_error1 = calc_pts_error(newpts2.copy(), gts2.copy())

    elif dataset == 'soft':
        error = field_evaluation(deform_forward1, [fixed_frame, frame], p, dataset, path,
                            savename='', savedir='', output_image=output_image1)
        
        # error2 = delete_outside_mask(error.copy(), true_mask) 
         
        mag_error1, sd_error1, mean_error1 = calc_df_error(error.copy())


    # Run the custom algorithm                                         
    start_time = time.process_time()
    mapping = custom_algorithm.optimize(fixed.astype(float), moving.astype(float), fixed_mask.astype(float), true_mask.astype(float))
    execution_time2 = time.process_time() - start_time
    deform_forward2 = mapping.forward
    deform_backward2 = mapping.backward
    output_image2 = mapping.transform(moving, 'linear')

    # Calculate the metrics
    errors_custom = metrics_evaluation(output_image2, fixed, moving, execution_time2, "1-70_cardiac_custom", '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/')
    
    if dataset == 'cardiac':
        newpts_custom = field_evaluation(deform_forward2, [fixed_frame, frame], p_custom, 'cardiac', '/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/', savename='', savedir='', output_image=output_image2)

        # show_image_pts(output_image2,pxm,newpts_custom,1);plt.title('Resulting image')

        gts = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/eval_pts_'+str(frame)+'.mat')['pts']   # <- change path. used for evaluation  

        gts2, newpts2 = delete_outsideFOV(gts.copy(), newpts_custom.copy())                                                # delete points outside of FOV

        mag_error2, sd_error2, mean_error2 = calc_pts_error(newpts2.copy(), gts2.copy())

    elif dataset == 'soft':
        error = field_evaluation(deform_forward2, [fixed_frame, frame], p_custom, dataset, path,
                            savename='', savedir='', output_image=output_image2)
        
        # error2 = delete_outside_mask(error.copy(), true_mask)
        
        mag_error2, sd_error2, mean_error2 = calc_df_error(error.copy())
    
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
    errors, mag_error1, sd_error1, mean_error1, errors_custom, mag_error2, sd_error2, mean_error2, execution_time1, execution_time2, dice, hd, Bce = run(10, dataset='cardiac')
    print(errors)
    print(mag_error1)
    print(sd_error1)
    print(mean_error1)
    print(errors_custom)
    print(mag_error2)
    print(sd_error2)
    print(mean_error2)