
import numpy as np
from scipy.io import savemat, loadmat

def delete_outsideFOV(gts, newpts):   
    """
    Deletes scatterers that are outside of our sector field of view. 
    """ 
    # delete pts that have value 0
    ind = np.where(newpts[:,0]==0)[0].tolist()
    gts = np.delete(gts,ind,0)
    newpts = np.delete(newpts,ind,0)
    
    tmp = loadmat('/Users/elliottunstall/Desktop/Imperial/FYP/Example_cardiac_dataset/bmode_f1.mat')['pxm']
    pxm = {'X': tmp['X'][0][0][0,:],
           'Z': tmp['Z'][0][0][:,0:tmp['Z'][0][0].shape[1]:2][0,:],
           'bottomRight': tmp['bottomRight'][0][0][0]}
    
    ind = []    
    for i in range(gts.shape[0]):
        xcoord = gts[i,0]
        zcoord = gts[i,2] + 0.0108                              # these numbers come from the focus of the ultrasound
        
        theta = np.arctan(np.abs(xcoord)/np.abs(zcoord))
        r = np.abs(zcoord)/np.cos(theta)
        
        if theta > 45*np.pi/180 :
            ind.append(i)
        if r > pxm['bottomRight'][2] + 0.0108:
            ind.append(i)
        if xcoord < np.min(newpts[:,0]):
            ind.append(i)
        if xcoord < -0.045:
            ind.append(i)
                
    gts = np.delete(gts,ind,0)
    newpts = np.delete(newpts,ind,0)
                
    return gts, newpts

def delete_outside_mask(error, mask):
    # print(f"the shape of error is {np.shape(error)}")
    # print(f"the shape of mask is {np.shape(mask)}")
    for i in range(np.shape(error)[0]):
        for j in range(np.shape(error)[1]):
            if mask[i,j] != 1:
                error[i,j,0] = 0
                error[i,j,1] = 0

    return error

def calc_magnitude_1D(vectors, components):    
    x_component = vectors[:, components[0]]
    y_component = vectors[:, components[1]]
    
    # Calculate magnitude using vectorized operations
    magnitude = np.sqrt(x_component**2 + y_component**2)
    return magnitude

def calc_magnitude_2D(vectors, components):    
    x_component = vectors[:,:, components[0]]
    y_component = vectors[:,:, components[1]]
    
    # Calculate magnitude using vectorized operations
    magnitude = np.sqrt(x_component**2 + y_component**2)
    return magnitude

def calc_pts_error(newpts, gts):
    
    ae = np.zeros([gts.shape[0], 4])
    ae[:,0] = gts[:,0] - newpts[:,0]       
    ae[:,1] = gts[:,2] - newpts[:,1]       
    
    mag_error = calc_magnitude_1D(ae[:,0:2],[0,1])      # magnitude of error of each scatterer
    sd_error = np.std(mag_error)                         # standard deviation of the magnitude error
    mean_error = np.mean(mag_error)
    
    return mag_error, sd_error, mean_error

def calc_df_error(error):
    mag_error = (calc_magnitude_2D(error, [0, 1])).flatten()
    mag_error = mag_error[mag_error != 0]                 # remove zero values
    sd_error = np.std(mag_error)                         # standard deviation of the magnitude error
    mean_error = np.mean(mag_error)

    return mag_error, sd_error, mean_error



