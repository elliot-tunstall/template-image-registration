## Author: Clara Rodrigo Gonzalez

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os


def show_image(img, pixelMap):
    plt.figure()
    plt.imshow(img, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', cmap = 'grey')
    plt.colorbar()

def load_inputs(frame, fixed_frame):
	# I wrote this in a txt file so you may have to change the indents
	dirname = os.path.dirname(__file__)
	path = os.path.join('/Users/elliottunstall/Desktop/Imperial/FYP/', 'Example_cardiac_dataset')   # <- CHANGE PATH HERE
	data0 = loadmat(os.path.join(path, 'bmode_f'+str(fixed_frame)+'.mat'))
	data1 = loadmat(os.path.join(path, 'bmode_f'+str(frame)+'.mat'))
	fixed = data0['blurry']
	moving = data1['blurry']
	pxm = data0['pxm']

	return fixed, moving, pxm

def resize_image(image, pixel_map):
	resized_image = np.zeros_like(image)

	x_min, x_max, z_max, z_min = pixel_map['X'][0][0].min(), pixel_map['X'][0][0].max(), pixel_map['Z'][0][0].max(), pixel_map['Z'][0][0].min()

	for i in range(image[0][0].shape[0]):
		for j in range(image[0][0].shape[1]):
			new_i = int((pixel_map['X'][i, j] - x_min) / (x_max - x_min) * image.shape[0])
			new_j = int((pixel_map['Z'][i, j] - z_min) / (z_max - z_min) * image.shape[1])
			resized_image[new_i, new_j] = image[i, j]

	return resized_image
