

## Author: Clara Rodrigo Gonzalez 
# This is me trying to figure out bayesian optimization
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------------------- Evaluation metrics --------------------------- #
def mean_squared_error(fixed, moving):
    # ROI = (moving != 0).astype(float)
    return np.mean((fixed - moving)**2)

def sum_squared_diff(fixed, moving):
    # ROI = (moving != 0).astype(float)
    return np.sum((fixed - moving)**2)

def mutual_information(img1, img2, bins=20):
    """
    measure the mutual information of the given two images

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    calculated mutual information: float

    """
    # ROI = (img2 != 0).astype(float)
    # img1 = img1[ROI]

    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)

    # convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal x over y
    py = np.sum(pxy, axis=0)  # marginal y over x
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals

    # now we can do the calculation using the pxy, px_py 2D arrays
    nonzeros = pxy > 0  # filer out the zero values
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def norm_cross_corr(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    # ROI = (data1 != 0).astype(float)
    # data0 = data0[ROI]

    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def joint_entropy(fixed, moving, bins):
    
    # ROI = (moving != 0).astype(float)
    # fixed = fixed[ROI]

    binned_dist = np.histogram2d(fixed.ravel(), moving.ravel(), bins)[0]

    # normalize to give probabilities
    probs = binned_dist / float(np.sum(binned_dist))

    # get rid of bins with 0
    probs = probs[np.nonzero(probs)]

    # calculate joint entropy
    joint_entropy = np.sum(probs + np.log2(probs))

    return joint_entropy

# ---------------------------- Evaluation function --------------------------- #
def metrics_evaluation(output_image, fixed, moving, execution_time, savename, savedir):

    ROI = (fixed != 0)
    fixed = fixed[ROI]
    moving = moving[ROI]
    output_image = output_image[ROI]
    
    # Ensure no nan values 
    output_image = np.nan_to_num(output_image, nan=0)
    
    errors = np.zeros([6,2])    # [number of similarity metrics, 0: abs value 1: proportional value]

    errors[0,0] = sum_squared_diff(output_image, fixed)
    errors[0,1] = errors[0,0]/sum_squared_diff(moving, fixed)
    
    errors[1,0] = mean_squared_error(output_image, fixed)
    errors[1,1] = errors[1,0]/mean_squared_error(moving, fixed)

    errors[2,0] = mutual_information(output_image, fixed)
    errors[2,1] = errors[2,0]/mutual_information(moving, fixed)

    errors[3,0] = structural_similarity(output_image, fixed, data_range=fixed.max()-fixed.min())
    errors[3,1] = errors[3,0]/structural_similarity(moving, fixed, data_range=fixed.max()-fixed.min())

    errors[4,0] = norm_cross_corr(output_image, fixed)
    errors[4,1] = errors[4,0]/norm_cross_corr(moving, fixed)

    errors[5,0] = joint_entropy(output_image, fixed, 20)
    errors[5,1] = errors[3,0]/joint_entropy(moving, fixed, 20)

    # print(errors[0,1])
    # if savename != '':
    #     try:
    #         data = pd.read_csv(savename+"_metrics.csv")
    #     except:
    #         data = pd.DataFrame(columns=['savename','time','SSD','pSSD','MSE','pMSE','MI','pMI','SSIM','pSSIM','NCC','pNCC','JE','pJE'])
                
    #     new_data = {'savename': savename,
    #                 'time':     execution_time,
    #                 'SSD':      errors[0,0],
    #                 'pSSD':     errors[0,1],
    #                 'MSE':      errors[1,0],
    #                 'pMSE':     errors[1,1],
    #                 'MI':       errors[2,0],
    #                 'pMI':      errors[2,1],
    #                 'SSIM':     errors[3,0],
    #                 'pSSIM':    errors[3,1],
    #                 'NCC':      errors[4,0],
    #                 'pNCC':     errors[4,1],
    #                 'JE':       errors[5,0],
    #                 'pJE':      errors[5,1]
    #                 }
        
    #     data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)     # add new data to old dataframe
    #     data = data.loc[:, ~data.columns.str.contains('^Unnamed')]              # delete any empty columns created
        
    #     data.to_csv(savedir+savename+"_metrics.csv", index=True)                        # save

    return errors
        
    
    