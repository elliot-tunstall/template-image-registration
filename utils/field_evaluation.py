import numpy as np
import subprocess
import mat73
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import subprocess
import nibabel as nib
from scipy.ndimage import zoom

from plotting import show_image_pts, show_image
from loading import load_soft_gts
from loading import load_inputs as load_inputs2
# from pipeline import load_inputs

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
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def field_evaluation(deform_forward, framenums, p, dataset, path, savename, savedir, output_image):
    
    if dataset == 'cardiac': 
        _,_,pxm = load_inputs2('cardiac',framenums[0],framenums[1],path)

        # ------------------------ Calculate deformation field ----------------------- #
        deform_forward[:,:,0] = deform_forward[:,:,0]*pxm['dz']
        deform_forward[:,:,1] = deform_forward[:,:,1]*pxm['dx']

        # -------------------------- Generate ground truths -------------------------- #
        # pts = subprocess.run(["matlab", "-batch", f"get_eval_pts(frame,'PLA',savedir+'new/','');"])
        pts0 = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/eval_pts_'+str(0)+'.mat')['pts']       #framenums[0]
        pts = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/eval_pts_'+str(framenums[1])+'.mat')['pts']
        
        x = pxm['X']
        z = pxm['Z']
        
        newpts = np.zeros(pts0.shape)
        
        for i in range(pts0.shape[0]):
            if(np.abs(pts[i,0])<np.abs(np.min(x))):
                # Forward
                xind = np.where(np.abs(x - pts[i,0]) == min(np.abs(x - pts[i,0])))
                zind = np.where(np.abs(z - pts[i,2]) == min(np.abs(z - pts[i,2])))
                dx = deform_forward[zind,xind,1]
                dz = deform_forward[zind,xind,0]
            
                newpts[i,0] = pts[i,0] + dx
                newpts[i,1] = pts[i,2] + dz
                
                # Backward - not doing it anymore
                # xind = np.where(np.abs(x - pts0[i,0]) == min(np.abs(x - pts0[i,0])))
                # zind = np.where(np.abs(z - pts0[i,2]) == min(np.abs(z - pts0[i,2])))
                # dx = deform_backward[zind,xind,1]
                # dz = deform_backward[zind,xind,0]
            
                # newpts[i,2] = pts0[i,0] + dx
                # newpts[i,3] = pts0[i,2] + dz

        # a=30
        # plt.figure();plt.scatter(pts0[0:a,0],-pts0[0:a,2],c='b');plt.scatter(newpts[0:a,0],-newpts[0:a,1],c='r')
        # ------------------------------- Save results ------------------------------- #
        if savename != '':
            try:
                data = pd.read_csv(savename + '_field.csv')
            except:
                data = pd.DataFrame(columns=['savename', 'params', 'newpts'])
                    
            new_data = {'savename': savename,
                        'params':   [p.params],
                        'newpts':   [newpts.tolist()]
                        }
            
            data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=False)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            
            data.to_csv(savename + '_field.csv')
        return newpts
    
    else:
        gt = load_soft_gts(path, framenums[1])
        _,_,pxm = load_inputs2('soft',framenums[0],framenums[1],path)
        
        error = gt - deform_forward
        error[:,:,0] = error[:,:,0]*pxm['dx']
        error[:,:,1] = error[:,:,1]*pxm['dz']
        
        if 0:
            plt.figure();plt.imshow(gt[:,:,0]);plt.colorbar();plt.title('GT x')
            plt.figure();plt.imshow(gt[:,:,1]);plt.colorbar();plt.title('GT z')
            plt.figure();plt.imshow(deform_forward[:,:,0]);plt.colorbar();plt.title('est x')
            plt.figure();plt.imshow(deform_forward[:,:,1]);plt.colorbar();plt.title('est z')
        
        # ------------------------------- Save results ------------------------------- #
        if savename != '':
            try:
                data = pd.read_csv(savename + '_field.csv')
            except:
                data = pd.DataFrame(columns=['savename', 'params', 'error', 'shape'])
                    
            new_data = {'savename': savename,
                        'params':   [p.params],
                        'error':    [error.tolist()],
                        'shape':    [gt.shape]
                        }
            
            data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=False)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            
            data.to_csv(savename + '_field.csv')
        
        return 

def field_evaluation_simple(deform_forward, output_image):
    pts0 = loadmat('eval_pts/eval_pts_'+str(0)+'.mat')['pts']       #framenums[0]
    pts = loadmat('eval_pts/eval_pts_'+str(40)+'.mat')['pts']
    fixed, moving, pxm = load_inputs('cardiac',1,40,'/media/clararg/8TB HDD/Data/STRAUS/simulations/patient1/')

    # pxm = mat73.loadmat('/media/clararg/8TB HDD/Data/STRAUS/simulations/patient1/old/new_bmode_f1.mat')['pixelMap']

    # ------------------------ Calculate deformation field ----------------------- #
    deform_forward[:,:,0] = deform_forward[:,:,0] * pxm['dx']
    deform_forward[:,:,1] = deform_forward[:,:,1] * pxm['dz']
    
    x = pxm['X']
    z = pxm['Z']

    pts0[:,2] = pts0[:,2]
    pts[:,2] = pts[:,2]
    gts = [pts0, pts]

    newpts=np.zeros(pts.shape)
    
    for i in range(pts.shape[0]):
        # Forward
        xind = np.where(np.abs(x - pts[i,0]) == min(np.abs(x - pts[i,0])))
        zind = np.where(np.abs(z - pts[i,2]) == min(np.abs(z - pts[i,2])))
        dx = deform_forward[zind,xind,1]
        dz = deform_forward[zind,xind,0]
    
        newpts[:,0] = pts[:,0] + dz
        newpts[:,2] = pts[:,2] + dx

    error = calc_pts_error(newpts2, gts2)


    return np.sum(error[0]) 
