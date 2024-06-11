import numpy as np
from skimage.measure import label, regionprops
import cv2
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
sys.path.append('/Users/elliottunstall/Desktop/Imperial/FYP/codebase/venv/segmentation-mask-overlay')
#from src.segmentation_mask_overlay import overlay_masks

def connected_components(mask, smoothing_iter=4):
    # Perform connected components labeling
    labeled_mask = label(mask, connectivity=2, background=0)
    
    # Get region properties of each connected component
    regions = regionprops(labeled_mask)
    
    # Sort regions by area in descending order
    regions.sort(key=lambda x: x.area, reverse=True)
    
    # Create a dictionary to store the binary images of each component
    binary_images = {}
    
    # Generate binary images for each component
    for i, region in enumerate(regions):
        binary_image = np.zeros_like(mask)
        binary_image[tuple(region.coords.T)] = 1
        
        # binary_image = fill_holes(binary_image)
        closed_mask = smooth_mask(binary_image, method='morphological', kernel_size=10, iterations=1)
        smoothed_mask = smooth_mask(closed_mask, method='gaussian', kernel_size=5, iterations=2)

        # smoothed_mask = create_smooth_binary_mask(binary_image, epsilon_factor=0.001)
        binary_images[i+1] = smoothed_mask
    
    return binary_images

def create_mask(binary_image):
    # Create a new image with three channels
    color_image = np.zeros((binary_image.shape[0], binary_image.shape[1]), dtype=np.uint8)

    # Set the color of the new image based on the binary image
    color_image[binary_image > 0] = 1

    return color_image

class Mask():
    "custom defined class for masks"

    def __init__(self, mask: np.ndarray, props: regionprops, region: str):
        self.properties = props
        self.region = region

        # Define a custom colormap: 0 -> blue, 1 -> blue, 2 -> green
        # self.cmap_tissue = ListedColormap(['black', 'green'])
        # self.camp_background = ListedColormap(['black', 'blue'])

        if self.region == 'tissue':
            self.cmap = ListedColormap(['black', 'green'])
        elif self.region == 'background':
            self.cmap = ListedColormap(['black', 'blue'])

        # self.mask = create_mask(mask)
        # self.cmap = self.cmap.set_under('black', alpha=0)
        self.mask = mask.astype(np.uint8)

    def show(self, pixelMap=None, axis=False, alpha=1.0, multiplot=False):
        "Show the mask"

        mask = np.ma.masked_where(self.mask < 0.9, self.mask)
        
        if multiplot == False:
            if pixelMap:
                plt.imshow(mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=alpha, cmap=self.cmap, clim=[0.9, 1], interpolation='none')
            else:
                plt.imshow(mask, alpha=alpha, cmap=self.cmap, clim=[0.9, 1.1], interpolation='none')
            
            plt.show()
        
        else:
            if pixelMap:
                plt.imshow(mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=alpha, cmap=self.cmap, clim=[0.9, 1], interpolation='none')
            else:
                plt.imshow(mask, alpha=alpha, cmap=self.cmap, clim=[0.9, 1], interpolation='none')

        return axis
    
    def show_axis(self, pixelMap=None, axis=False, alpha=1.0, multiplot=False):
        "Show the mask"

        mask = np.ma.masked_where(self.mask < 0.9, self.mask)
        
        if multiplot == False:
            if pixelMap:
                axis.imshow(mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=alpha, cmap=self.cmap, clim=[0.9, 1], interpolation='none')
            else:
                axis.imshow(mask, alpha=alpha, cmap=self.cmap, clim=[0.9, 1.1], interpolation='none')
            
            plt.show()
        
        else:
            if pixelMap:
                axis.imshow(mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=alpha, cmap=self.cmap, clim=[0.9, 1], interpolation='none')
            else:
                axis.imshow(mask, alpha=alpha, cmap=self.cmap, clim=[0.9, 1], interpolation='none')

    def transparency(self, alpha):
        # Ensure the image has three channels
        if len(self.mask.shape) == 2:
            mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)

        # Add an alpha channel to the image
        b, g, r = cv2.split(mask)
        rgba = [b, g, r, alpha * np.ones(b.shape, dtype=np.uint8)]
        mask = cv2.merge(rgba, 4)

        return mask
    
    def smooth_mask(self, method='gaussian', kernel_size=5, sigma=3, shape="[]", iterations=1):
        """
        Smooth a binary mask.
        
        Parameters:
            mask (np.array): Binary mask (numpy array).
            method (str): Smoothing method ('gaussian' or 'morph_closing' or 'morph_opening).
            kernel_size (int): Size of the kernel used for smoothing.
            
        Returns:
            np.array: Smoothed mask.
        """
        # # Ensure mask is in the correct format for OpenCV operations
        # mask = np.uint8(mask * 255)

        if method == 'gaussian':
            # Applying Gaussian Blur
            for _ in range(iterations):
                smoothed = cv2.GaussianBlur(self.mask.astype(float), (kernel_size, kernel_size), sigma)

            # Apply a threshold to convert the smoothed mask back to a binary mask
            _, self.mask = cv2.threshold(smoothed, 0.3, 1, cv2.THRESH_BINARY)

        elif method == 'morph_opening':
            # Creating a kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
        # Applying morphological operations
            # smoothed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations)

            for _ in range(iterations):

                # Perform an erosion operation
                eroded = cv2.erode(self.mask.astype(np.uint8), kernel, iterations=1)

                # Perform a dilation operation
                dilated = cv2.dilate(eroded.astype(np.uint8), kernel, iterations=1)

                self.mask = dilated

        elif method == 'morph_closing':
            # Creating a kernel
            if shape == '+':

                def create_plus_matrix(size):
                    matrix = np.zeros((size, size))
                    matrix[size // 2, :] = 1
                    matrix[:, size // 2] = 1
                    return matrix
                
                kernel = create_plus_matrix(kernel_size).astype(np.uint8)
                
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Applying morphological operations
            # smoothed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations)

            for _ in range(iterations):

                # Perform a dilation operation
                dilated = cv2.dilate(self.mask.astype(np.uint8), kernel, iterations=1)

                # Perform an erosion operation
                eroded = cv2.erode(dilated.astype(np.uint8), kernel, iterations=1)

                self.mask = eroded

        elif method == 'binary_dilation':
            # Creating a kernel
            if shape == '+':

                def create_plus_matrix(size):
                    matrix = np.zeros((size, size))
                    matrix[size // 2, :] = 1
                    matrix[:, size // 2] = 1
                    return matrix
                
                kernel = create_plus_matrix(kernel_size).astype(np.uint8)
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Applying morphological operations
            for _ in range(iterations):
                # Perform a dilation operation
                dilated = cv2.dilate(self.mask.astype(np.uint8), kernel, iterations=1)
                self.mask = dilated  

        elif method == 'binary_erosion':
            # Creating a kernel
            if shape == '+':

                def create_plus_matrix(size):
                    matrix = np.zeros((size, size))
                    matrix[size // 2, :] = 1
                    matrix[:, size // 2] = 1
                    return matrix
                
                kernel = create_plus_matrix(kernel_size).astype(np.uint8)
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Applying morphological operations
            for _ in range(iterations):
                # Perform a dilation operation
                eroded = cv2.erode(self.mask.astype(np.uint8), kernel, iterations=1)
                self.mask = eroded      

        elif method == 'fill_holes':
            # Fill holes in the mask
            self.mask = binary_fill_holes(self.mask).astype(np.uint8)

        else:
            raise ValueError("Unsupported smoothing method. Use 'gaussian', 'morph_opening' or 'morph_closing'.")
        
        if np.all(self.mask == 0) or np.all(self.mask == False):
            return False


    def fill_holes(self):
        """
        Fill holes in a binary mask.
        
        Parameters:
            mask (np.array): Binary mask (numpy array).
            
        Returns:
            np.array: Mask with holes filled.
        """
        # Ensure mask is in the correct format for OpenCV operations
        self.mask = np.uint8(self.mask * 255)
        
        # Fill holes in the mask
        self.mask = binary_fill_holes(self.mask).astype(np.uint8)
        
        return self.mask

    def smooth_contour_mask(self, epsilon_factor=0.01):
        """
        Create a smooth binary mask from an input mask by finding and smoothing contours.
        
        Parameters:
            mask (np.array): Input binary mask, expected to be in grayscale or uint8 format.
            epsilon_factor (float): Factor to control the approximation accuracy of the contour smoothing.
            
        Returns:
            np.array: Smoothed and filled binary mask.
        """
        # Ensure mask is in uint8 format, adjusting if it's in boolean format or otherwise
        if self.mask.dtype != np.uint8:
            mask = np.uint8(self.mask * 255)
        
        # Make sure the mask is contiguous in memory, as expected by OpenCV
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

        # Apply threshold to ensure it's a binary mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare an image to draw the smooth contours, ensuring it's compatible with cv::Mat
        smooth_mask = np.zeros_like(mask, dtype=np.uint8)  # Explicit dtype specification

        # Make sure the smooth_mask is contiguous as well
        smooth_mask = np.ascontiguousarray(smooth_mask)

        for contour in contours:
            # Approximate the contour to smooth it
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Fill the smoothed contour on the mask
            cv2.fillPoly(smooth_mask, [approx], 255)

        self.mask = smooth_mask
        return self.mask


    def smooth_mask_with_median_filter(self, kernel_size=5):
        """
        Smooth a binary mask using a median filter.
        
        Parameters:
            mask (np.array): Binary mask (numpy array), expected to be in binary format (0s and 1s).
            kernel_size (int): Size of the median filter kernel. Must be an odd number.
            
        Returns:
            np.array: Smoothed binary mask.
        """
        # Ensure the mask is in uint8 format and scaled to 0-255
        mask_uint8 = np.uint8(self.mask * 255) if np.max(self.mask) <= 1 else self.mask
        
        # Apply the median filter for smoothing
        smoothed_mask = cv2.medianBlur(mask_uint8, kernel_size)
        
        # Convert back to binary format
        _, smoothed_binary_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)
        
        self.mask = smoothed_binary_mask / 255
        return self.mask


class Segmentation():
    "custom defined class for segmententation"

    def __init__(self, image: np.ndarray, method='otsu'):
        self.image = image

        if method not in ['otsu', 'kmeans']:
            raise ValueError("The 'method' parameter must be either 'otsu' or 'kmeans'.")

        if method == 'otsu':
            self.tissue, self.background = self.otsu_thresholding()
        elif method == 'kmeans':
            self.tissue, self.background = self.kmeans()

        # self.create_masks()

    def create_masks(self):
        "Create masks for the segmented regions."
        self.regions = {}
        self.regions['background'] = self.background
        self.regions['tissue'] = self.tissue
        self.masks = {}

        for key, value in self.regions.items():

            # Perform connected components labeling
            labeled_mask = label(value, connectivity=2, background=0)
            
            # Get region properties of each connected component
            regions = regionprops(labeled_mask)

            # Sort regions by area in descending order
            regions.sort(key=lambda x: x.area, reverse=True)

            # initialse variables
            binary_images = []
            
            # Generate binary images for each component
            for i, region in enumerate(regions):
                binary_image = np.zeros_like(self.image)
                binary_image[tuple(region.coords.T)] = 1
                binary_images.append(Mask(binary_image, props=region, region=key))

            self.masks[key] = binary_images

    def kmeans(self, mrf=0.15):
        "Kmeans segmentation using ANTsPy."

        import ants

        kmask = (self.image != 0).astype(float)
        kmask = ants.from_numpy(kmask)
        ants_image = ants.from_numpy(self.image)

        seg = ants.kmeans_segmentation(ants_image, 3, kmask=kmask, mrf=mrf)
        self.background = seg['segmentation'].numpy() == 1
        self.tissue = seg['segmentation'].numpy() == 2

        self.background = self.background / np.max(self.background)
        self.tissue = self.tissue / np.max(self.tissue)

        self.create_masks()

        return self.tissue, self.background

    def otsu_thresholding(self):
        "Otsu thresholding for segmentation."

        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(self.image)
        self.tissue = self.image > threshold

        threshold = threshold_otsu(self.image)
        self.background = self.image < threshold

        self.background = self.background / np.max(self.background)
        self.tissue = self.tissue / np.max(self.tissue)

        self.create_masks()

        return self.tissue, self.background

    def apply_smoothing(self, method='gaussian', kernel_size=5, sigma=3, shape="[]", region='all', iterations=1):
        """
        Apply a smoothing operation to all binary masks.

        Parameters:
            method (str): Smoothing method ('gaussian' or 'morph_closing' or 'morph_opening).
            
        Returns:
            np.array: Smoothed binary mask.
        """

        if not isinstance(region, str):
            raise TypeError("The 'region' parameter must be a string.")
        elif region != 'all' and region not in self.masks:
            raise ValueError("The 'region' parameter is not valid.")

        if region == 'all':
            for key in self.masks:
                for i in self.masks[key]:
                    valid = i.smooth_mask(method=method, kernel_size=kernel_size, sigma=sigma, shape=shape, iterations=iterations)
                    
                    if valid == False:
                        self.masks[key].remove(i)  
        else:
            for i in self.masks[region]:
                valid = i.smooth_mask(method=method, kernel_size=kernel_size, sigma=sigma, shape=shape, iterations=iterations)

                if valid == False:
                        self.masks[key].remove(i) 

        return self.masks
    
    def select_masks(self, region, size):
        """
        Select the largest masks in a region.

        Parameters:
            region (str): Region to select masks from.
            size (int): Number of masks to select.
            
        Returns:
            dict: Dictionary of selected masks.
        """
        if not isinstance(region, str):
            raise TypeError("The 'region' parameter must be a string.")
        elif region not in self.masks:
            raise ValueError("The 'region' parameter is not valid.")
        elif not isinstance(size, int):
            raise TypeError("The 'size' parameter must be an integer.")
        elif size < 1:
            raise ValueError("The 'size' parameter must be greater than 0.")
        
        # Sort the masks in the region by area in descending order
        for mask in self.masks[region]:
            if mask.properties.area < size:
                self.masks[region].remove(mask)
        
    def use_regions(self):
        """
        Use entire region as single mask object.
        """
        
        self.masks['tissue'] = [Mask(self.tissue, props=None, region='tissue')]
        self.masks['background'] = [Mask(self.background, props=None, region='background')]

    def normalise_background(self, region='tissue'):
        """
        Normalise the background mask to the tissue mask.

        Parameters:
            region (str): Region to normalise. (optional) = 'tissue'.
        """
        masks = []

        if region not in self.masks:
            raise ValueError("The region does not exist.")
        
        for i in self.masks[region]:
            masks.append(i.mask)
        
        combined_mask = np.maximum.reduce(masks)
        inverse_mask = (1 != combined_mask)
        self.masks['background'] = [Mask(inverse_mask, props=None, region='background')]
    
    def show(self, pixelMap=None, overlay_image=True, axis=False, alpha=0.7, region='all', overlay_mask=False):
        # I want to chnage this function to create a image of by appending transparent maks to the overlayed_image.
        # It should create a new attribute - mask.plot
        """
        Show all the masks as a matplotlib plot.

        Parameters:
            pixelMap (dict): Pixel map for the image. (optional) = None.
            overlay_on_image (np.ndarray): Image to overlay the masks onto. (optional) = False.
            axis (plt axis): use for a matplotlib subplot functionality. (optional) = False.
        """
        # fig, axis = plt.subplots(1, 1, figsize=(6, 4))
        multiplot = True

        if not isinstance(overlay_image, bool):
            raise TypeError("The 'overlayed_image' parameter must be of type bool")

        if overlay_image == False:
            overlayed_image = np.zeros_like(self.image)
            print("Overlay image is False")

        else:
            overlayed_image = self.image
        
        if len(np.shape(overlayed_image)) != 2:
            raise ValueError("The overlayed image must be a 2D array.")
            # overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_GRAY2BGR)

        # overlayed_image = overlayed_image.astype(np.uint8)
        # overlayed_image = overlayed_image / np.max(overlayed_image)

        # plt.imshow(overlayed_image, cmap='gray')
        # plt.show()

            # for key in self.masks:
            #     for idx, mask in enumerate(self.masks[key]):
            #         # i.change_transparency(alpha)
            #         axis = mask.show(pixelMap=pixelMap, alpha=alpha, multiplot=multiplot)
            #         print(f"Completed {key}: {idx+1} of {len(self.masks[key])} masks")
        # else:

        if overlay_image == True: 
            if pixelMap is None:
                plt.imshow(overlayed_image, cmap='gray', alpha=1)
            else:
                plt.imshow(overlayed_image, cmap='gray', extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=1)


        if region == 'all':
            for key in self.masks:
                for i in self.masks[key]:
                    if np.shape(overlayed_image) != np.shape(i.mask):
                        raise ValueError(f"The shape of the overlayed image must match the shape of the mask. Shapes {np.shape(overlayed_image)} - {np.shape(i.mask)}")
                    
                    i.show(pixelMap=pixelMap, alpha=alpha, multiplot=True)
        
        else:
            if region not in self.masks:
                raise ValueError(f"The region {region} does not exist.")
            
            for i in self.masks[region]:
                if np.shape(overlayed_image) != np.shape(i.mask):
                    raise ValueError(f"The shape of the overlayed image must match the shape of the mask. Shapes {np.shape(overlayed_image)} - {np.shape(i.mask)}")
                
                i.show(pixelMap=pixelMap, alpha=alpha, multiplot=True)

        if overlay_mask == True:
            cmap = ListedColormap(['black', 'red'])

            overlay_mask = np.ma.masked_where(overlay_mask < 0.9, overlay_mask)
        
            if pixelMap:
                plt.imshow(overlay_mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=alpha, cmap=cmap, clim=[0.9, 1], interpolation='none')
            else:
                plt.imshow(overlay_mask, alpha=alpha, cmap=cmap, clim=[0.9, 1.1], interpolation='none')
        
        if axis == False:     
            plt.show()

    def plot_axis(self, pixelMap=None, overlay_image=False, axis=False, alpha=0.7, region='all', overlay_mask=None):
        # I want to chnage this function to create a image of by appending transparent maks to the overlayed_image.
        # It should create a new attribute - mask.plot
        """
        Show all the masks as a matplotlib plot.

        Parameters:
            pixelMap (dict): Pixel map for the image. (optional) = None.
            overlay_on_image (np.ndarray): Image to overlay the masks onto. (optional) = False.
            axis (plt axis): use for a matplotlib subplot functionality. (optional) = False.
        """
        # fig, axis = plt.subplots(1, 1, figsize=(6, 4))
        multiplot = True

        if not isinstance(overlay_image, bool):
            raise TypeError("The 'overlayed_image' parameter must be of type bool")

        if overlay_image == False:
            overlayed_image = np.zeros_like(self.image)
            print("Overlay image is False")

        else:
            overlayed_image = self.image
        
        if len(np.shape(overlayed_image)) != 2:
            raise ValueError("The overlayed image must be a 2D array.")

        if overlay_image == True: 
            if pixelMap is None:
                axis.imshow(overlayed_image, cmap='gray', alpha=1)
            else:
                axis.imshow(overlayed_image, cmap='gray', extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=1)


        if region == 'all':
            for key in self.masks:
                for i in self.masks[key]:
                    if np.shape(overlayed_image) != np.shape(i.mask):
                        raise ValueError(f"The shape of the overlayed image must match the shape of the mask. Shapes {np.shape(overlayed_image)} - {np.shape(i.mask)}")
                    
                    i.show_axis(pixelMap=pixelMap, alpha=alpha, multiplot=True, axis=axis)
        
        else:
            if region not in self.masks:
                raise ValueError(f"The region {region} does not exist.")
            
            for i in self.masks[region]:
                if np.shape(overlayed_image) != np.shape(i.mask):
                    raise ValueError(f"The shape of the overlayed image must match the shape of the mask. Shapes {np.shape(overlayed_image)} - {np.shape(i.mask)}")
                
                i.show_axis(pixelMap=pixelMap, alpha=alpha, multiplot=True, axis=axis)

        if overlay_mask is not None:
            cmap = ListedColormap(['black', 'darkorange'])

            overlay_mask = np.ma.masked_where(overlay_mask < 0.9, overlay_mask)
        
            if pixelMap:
                axis.imshow(overlay_mask, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper', alpha=alpha, cmap=cmap, clim=[0.9, 1], interpolation='none')
            else:
                axis.imshow(overlay_mask, alpha=alpha, cmap=cmap, clim=[0.9, 1.1], interpolation='none')

        return axis    
        

    def show_select(self, mask_dict, pixelMap=None, overlay_on_image=False, alpha=0.5):
        """
        Show a selection of masks as a matplotlib plot.

        Parameters:
            mask_dict (dict): Dictionary of masks to show.
            pixelMap (dict): Pixel map for the image. (optional) = None.
            overlay_on_image (np.ndarray): Image to overlay the masks onto. (optional) = False.
            alpha (float): Transparency of the masks. (optional) = 0.5.
        """
        if overlay_on_image is not False:
            plt.imshow(overlay_on_image, cmap='gray', extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper')

        for key, value in mask_dict.items():
                self.mask[key][value].show(pixelMap=pixelMap, alpha=alpha)

        plt.show()
    