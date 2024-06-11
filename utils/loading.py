## Author: Clara Rodrigo Gonzalez

import scipy
from scipy.io import loadmat
import numpy as np
from scipy.signal import hilbert
from data_loader.sampler import samples
# from sampler import samples

def load_inputs(dataset, frame0, frame1, path):
    if dataset == 'cardiac':
        frame0 = int(frame0)
        frame1 = int(frame1)

        data0 = loadmat(path + 'bmode_f'+str(frame0)+'.mat')

        data1 = loadmat(path + 'bmode_f'+str(frame1)+'.mat')
        fixed = data0['blurry']
        moving = data1['blurry']
        tmp_pxm = data0['pxm']

        # Downsample z-axis by 1/2
        fixed = fixed[0:fixed.shape[0]:2,:]
        moving = moving[0:moving.shape[0]:2,:]

        pxm = dict()
        pxm['Z'] = tmp_pxm['Z'][0][0][:,0:tmp_pxm['Z'][0][0].shape[1]:2][0,:]
        pxm['X'] = tmp_pxm['X'][0][0][0,:]
        pxm['dx'] = tmp_pxm['dx'].item()[0][0]
        pxm['dz'] = tmp_pxm['dz'].item()[0][0]

        return fixed, moving, pxm

    elif dataset == 'soft':
    
        US_data_type = {'representation': 'bmode', 'dynamic_range': 0.01} 
        frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, US_data_type = US_data_type, N_pixels_desired = None, network_reduction_factor = 1, pad_cval_image = -1, pad_cval_deform = 0)

        fixed = frames[frame0,0,:,:]
        moving = frames[frame1,0,:,:]        

        pxm = {'dx': dx,
               'dz': dz}

        return fixed, moving, pxm
    
def load_soft_gts(path, frame):
    
        US_data_type = {'representation': 'bmode', 'dynamic_range': 0.01} 
        frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, US_data_type = US_data_type, N_pixels_desired = None, network_reduction_factor = 1, pad_cval_image = -1, pad_cval_deform = 0)

        deform_forward = np.zeros([frames.shape[2], frames.shape[3],2])

        deform_forward[:,:,0] = XX_deform/dx
        deform_forward[:,:,1] = ZZ_deform/dz
        
        deform_forward = deform_forward*scales[frame]
        
        return deform_forward
