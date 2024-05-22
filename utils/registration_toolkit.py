## Registration Toolkit

import ants
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def registration(fixed, moving, type_of_transform='SyN', syn_metric='mattes', plot_before=False, plot_after=False):
    if plot_before:
        fixed.plot(overlay=moving, title='Before Registration')

    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform=type_of_transform, syn_metric=syn_metric)
    warped_moving = mytx['warpedmovout']
   
    if plot_after:
        fixed.plot(overlay=warped_moving,title='After Registration')

    return warped_moving

def plot_image_grid(image_data: np.ndarray, rows, cols, number_of_figures = 1, title_string = "", title_array = False):

    """ Plot an image grid from 3D np.ndarray
        - rows: int
        - cols: int 
        - title_string: string included in individual plot title
        - title_array: list of values to be in the title of each image"""

    # assert rows%number_of_figures == 0, "Number of rows is not directly divisable by number of figures"
    # assert cols

    for fig_num in range(number_of_figures):
        
        # Create the figure and axis objects
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

        # Set the padding between subplots
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.1)

        # Loop through each row and column to plot images
        for i in range(0, rows):
            for j in range(0, cols):
                index = j + i*cols + fig_num*rows*cols
                # Plot the image
                axs[i, j].imshow(image_data[index, :, :], cmap='gray', vmin=0, vmax=255)
                axs[i, j].axis('off')
                
                # Add label
                if title_array:
                    axs[i, j].set_title(f"Image {index} - {title_string}: {title_array[index]} ")
                    
        # Show the plot
        plt.show()

def registration_plot_from_np(image_data: np.ndarray, rows, cols, number_of_figures = 1):

    """ Plot a 4D np.ndarray of images into an image grid. """

    assert rows%number_of_figures == 0, "Number of rows is not directly divisable by number of figures"
    num_rows = int(rows/number_of_figures)

    for fig_num in range(number_of_figures):
        
        # Create the figure and axis objects
        fig, axs = plt.subplots(num_rows, cols, figsize=(12, 8))

        # Set the padding between subplots
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.1)

        # Loop through each row and column to plot images
        for i in range(0, num_rows):
            for j in range(0, cols):
                # Plot the image
                axs[i, j].imshow(image_data[i + (fig_num*num_rows), j, :, :], cmap='gray', vmin=0, vmax=255)
                axs[i, j].axis('off')
                
                # Add label to the first image of each row
                if j == 0:
                    axs[i, j].set_title("Fixed Image")
                    
        # Show the plot
        plt.show()

def motion_correction():
    # We illustrate the steps below by building a 3D "functional" image and then "motion correcting" 
    # just as we would do with functional MRI or any other dynamic modality.
    image = ants.image_read(ants.get_ants_data('r16')) # 
    image2 = ants.image_read(ants.get_ants_data('r64'))
    ants.set_spacing( image, (2,2) )
    ants.set_spacing( image2, (2,2) )
    imageTar = ants.make_image( ( *image2.shape, 2 ) )
    ants.set_spacing( imageTar, (2,2,2) )
    fmri = ants.list_to_ndimage( imageTar, [image,image2] )

    # Now we motion correct this image just using the first slice as target.
    ants.set_direction( fmri, np.eye( 3 ) * 2 )
    images_unmerged = ants.ndimage_to_list( fmri )
    motion_corrected = list()
    for i in range( len( images_unmerged ) ):
        areg = ants.registration( images_unmerged[0], images_unmerged[i], "SyN" )
        motion_corrected.append( areg[ 'warpedmovout' ] )

    # Merge the resuling list back to a 3D image.
        motCorr = ants.list_to_ndimage( fmri, motion_corrected )
        # ants.image_write( motCorr, '/tmp/temp.nii.gz' )

    return motCorr

def register_collection(data: np.ndarray, fixed_image: np.ndarray, type_of_transform: str, syn_metric: str):
    registration_collection = []; moving_collection = []
    fixed_ants = ants.from_numpy(fixed_image)

    for i in range(np.shape(data)[3]):
        moving = ants.from_numpy(data[:,:,0,i])
        moving_collection.append(np.asarray(data[:,:,0,i]))

        warped_image = registration(fixed_ants, moving, type_of_transform=type_of_transform, syn_metric=syn_metric)
        np_reg = np.asarray(warped_image.numpy())
        registration_collection.append(np_reg)
    
    return np.asarray(moving_collection), np.asarray(registration_collection)

def registration_success_measure(registration_collection: np.ndarray, fixed_image: np.ndarray, success_metric = 'abs_diff'):

    # Normalise the images for fair comparison
    # registration_collection = registration_collection/registration_collection.max()
    # fixed_image = fixed_image/fixed_image.max()

    ## Success Metric - absolute difference
    if success_metric == 'abs_diff':
        metric_images = []
        metric_values = []
        for i in range(np.shape(registration_collection)[0]):
            metric_images.append(np.asarray(abs(registration_collection[i,:,:] - fixed_image)))
            metric_values.append(round(np.sum(metric_images[-1])))

    ## Success Metric - mean squared error
    elif success_metric == 'mse':
        metric_images = []
        metric_values = []
        for i in range(np.shape(registration_collection)[0]):

            metric_images.append(np.asarray((registration_collection[i,:,:].astype("float") - fixed_image.astype("float")) ** 2))
            err = np.sum(metric_images[-1])
            err /= float(registration_collection[i,:,:].shape[0] * fixed_image.shape[1])
            metric_values.append(round(err))

    elif success_metric == 'ssim':
        metric_values = []

        for i in range(np.shape(registration_collection)[0]):
            metric_values.append(ssim(registration_collection[i,:,:], fixed_image, data_range=1.0))

        metric_images = None

    return metric_images, metric_values

def transform_2_displacement_field(fixed, ants_transforms):

    # _______Create the deformation field________

    # Create a grid of points that matches the fixed image
    x = np.arange(0, fixed.shape[0])
    y = np.arange(0, fixed.shape[1])
    xx, yy = np.meshgrid(x, y)

    # Create a DataFrame from the grid of points
    points = pd.DataFrame({
        'x': xx.flatten(),
        'y': yy.flatten()
    })

    # Apply the transforms to the points
    warped_points = ants.apply_transforms_to_points(2, points, transformlist=ants_transforms)

    # The deformation field is the difference between the warped points and the original points
    deformation_field = warped_points[['x', 'y']].values - points[['x', 'y']].values

    # Reshape the deformation field back to the shape of the original image
    deformation_field = deformation_field.reshape(fixed.shape[0], fixed.shape[1], 2)

    return deformation_field