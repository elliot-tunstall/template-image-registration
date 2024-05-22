import ants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

path = '/Users/elliottunstall/Desktop/Imperial/FYP/Clinical BMode heart acquisitions/12_Clinical_20230421_114731_VERASONICS_HFR016_BMode_IM_0197.avi' # change path 
cap = cv2.VideoCapture(path)

data = np.zeros([600,800,3,36])
i = 0
while cap.isOpened():
    ret, frame = cap.read()  
    if ret:
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

fixed = ants.from_numpy(data[:,:,0,9])
moving = ants.from_numpy(data[:,:,0,15])
fixed.plot(overlay=moving, title='Before Registration')
mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN', syn_metric='demons' )

warped_moving = mytx['warpedmovout']
transforms = mytx['fwdtransforms']
print(transforms)
def_field = ants.image_read(transforms[0]).numpy()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot the deformation field for x-axis displacement
axs[0].imshow(def_field[:,:,0])
axs[0].set_title('Deformation Field (X-axis)')

# Plot the deformation field for y-axis displacement
axs[1].imshow(def_field[:,:,1])
axs[1].set_title('Deformation Field (Y-axis)')

plt.tight_layout()
plt.show()

mywarpedgrid = ants.create_warped_grid( moving, grid_directions=(True,True),
                        transform=mytx['fwdtransforms'], fixed_reference_image=fixed )
mywarpedgrid.plot(title='Warped Grid')
fixed.plot(overlay=warped_moving,
           title='After Registration')

dg = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ) )
plt.imshow(dg[1], cmap='gray')
plt.show()

print(mytx['fwdtransforms'])

# Read the transform
transform = ants.image_read(mytx['fwdtransforms'][0])

# _____Create the jacobean matrix________
jac = ants.create_jacobian_determinant_image(fixed, transform, do_log=False, geom=True)[...]


# The deformation field is now an ANTs image where each pixel's value represents the displacement of that pixel from the original image to the transformed image.
# We can convert this to a numpy array and visualize it using matplotlib.
print(np.shape(jac))
plt.imshow(jac)
plt.show()

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
warped_points = ants.apply_transforms_to_points(2, points, transformlist=mytx['fwdtransforms'])

# The deformation field is the difference between the warped points and the original points
deformation_field = warped_points[['x', 'y']].values - points[['x', 'y']].values

# Reshape the deformation field back to the shape of the original image
deformation_field = deformation_field.reshape(fixed.shape[0], fixed.shape[1], 2)

# Plot the deformation field

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot the deformation field for x-axis displacement
axs[0].imshow(deformation_field[..., 0])
axs[0].set_title('Deformation Field (X-axis)')

# Plot the deformation field for y-axis displacement
axs[1].imshow(deformation_field[..., 1])
axs[1].set_title('Deformation Field (Y-axis)')

# Plot the magnitude of the deformation field
magnitude = np.sqrt(np.sum(deformation_field**2, axis=2))
axs[2].imshow(magnitude)
axs[2].set_title('Magnitude of Deformation Field')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()



