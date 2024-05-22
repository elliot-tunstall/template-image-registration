## Testing ANTS registration Library

from cgitb import grey
from multiprocessing.connection import wait
from turtle import delay
import ants
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def registration(fixed, moving):
    #fixed.plot(overlay=moving, title='Before Registration')
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN' )
    warped_moving = mytx['warpedmovout']
    #fixed.plot(overlay=warped_moving,title='After Registration')

    return warped_moving

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

def reshape_to_grid(images, rows, cols):
    """
    Reshape a 3D array of images into a 4D grid of images.

    Parameters:
    - images: 3D NumPy array of shape (num_images, height, width)
    - rows: Number of rows in the grid
    - cols: Number of columns in the grid

    Returns:
    - grid_images: 4D NumPy array of shape (rows, cols, height, width)
    """
    height, width, num_images = images.shape
    assert num_images == rows * cols, "Number of images does not match grid size"

    grid_images = images.reshape(height, width, rows, cols)
    return grid_images

def registration_plot_from_np(image_data: np.array, rows, cols, number_of_figures = 1):

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


if __name__ == '__main__':
    print('Hello World!')

    path = 'Clinical BMode heart acquisitions/1_Clinical_20220225_121444_CW_VERASONICS_BMode_IM_0022.avi' # change path 
    cap = cv2.VideoCapture(path)
    
    data = np.zeros([600,800,3,75])
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()  
        if(ret):
            data[:,:,:,i] = frame
            i += 1
        else:
            break

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):

              # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()
             
            break

    # motCorr = motion_correction()
    # motCorr.plot(axis=2, nslices=6)

    # Image transformations
    # data = np.flip(data, axis=1)
    # data = np.rot90(data, axes=(0,1))

    #cv2.imshow("First Frame", data[:,:,0,0])
    #cv2.imshow("Tenth Frame", data[:,:,0,10])

    # # Create ANTSpy images
    # fixed = ants.from_numpy(data[:,:,0,0])
    # moving = ants.from_numpy(data[:,:,0,1])    # you can use any frames you want 

    # fixed.plot()
    # moving.plot()

    # Process the ANTsPy image
    # warped_image = registration(fixed, moving)
    # warped_nparray = np.asarray(warped_image.numpy())
    # registration_collection = np.zeros((*np.shape(warped_nparray), 75))
    # registration_collection[:,:,0] = warped_nparray
    # #warped_image.plot()
    
## What we aim to do here is do batch image registration:
    # Register 4 images batches to every 4th image (2 either side) and then compare with the fixed images.
    registration_collection = []

    for i in range(2,73,5):
        image_row = []
        # image_row = np.append(image_row, data[:,:,0,i])
        image_row.append(data[:,:,0,i])
        fixed = ants.from_numpy(data[:,:,0,i])

        for j in range(0,5):
            if j != 2:
                moving = ants.from_numpy(data[:,:,0,i-2+j])
                warped_image = registration(fixed, moving)
                np_reg = np.asarray(warped_image.numpy())
                # image_row = np.append(image_row, np_reg)
                image_row.append(np_reg)
        registration_collection.append(image_row)
        # registration_collection = np.append(registration_collection, image_row)

    registration_collection = np.array(registration_collection)
    # image_grid = reshape_to_grid(registration_collection, 15, 5)
    registration_plot_from_np(registration_collection, 15, 5, number_of_figures=3)

    # DDD_registrations = ants.from_numpy(registration_collection[:,:,:])
    # DDD_registrations.plot(axis=2, nslices=16)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter('ANTS/registration_videos/video_1.mp4', fourcc, 1.0, (800, 600))
    # for i in range(0, 3):
    #     frame = np.uint8(registration_collection[:,:,i])
    #     video_writer.write(frame)
    #     #cv2.imread(f"registration {i+1}", registration_collection[:,:,i])

    # # Release the VideoWriter object
    # video_writer.release()

