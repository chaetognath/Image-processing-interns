###################################
#                                 #
#          Final Project          #
#                                 #
###################################

# Complete the segmentation of of lung in the CT imageusing thresholding, edge detection, and morphological operations.


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.filters as filters
import scipy.ndimage as ndi

image = io.imread('images/lungCT.PNG', as_gray=True)
print(image)

# Apply binary thresholding
thresh = filters.threshold_otsu(image)
binary_image = (image < thresh).astype(np.float64)

# Morphological operation filters
B_sq = np.ones((3,3), dtype = np.uint8)


# Apply morphological opening : remove small objects
erosion_image = morph.binary_erosion(binary_image, B_sq, mode = 'min')
open_image = morph.binary_dilation(erosion_image, B_sq, mode= 'min')

# Apply morphological closing : fill small holes
dilation_image = morph.binary_dilation(open_image, footprint = B_sq, mode = 'min')
close_image = morph.binary_erosion(dilation_image, footprint = B_sq, mode = 'min')

# Apply Region Filling
fill_image = ndi.binary_fill_holes(close_image, structure = np.ones((5,5))).astype(np.uint8) * 255

# Find edges using morphological gradient
dilated = morph.binary_dilation(fill_image, footprint = B_sq, mode = 'min').astype(np.uint8) * 255
eroded = morph.binary_erosion(fill_image, footprint = B_sq, mode = 'min').astype(np.uint8) * 255
edge_image = dilated - eroded

# Add edges as red linees to the original image
edge_colored = image.copy()
edge_colored[edge_image == 255] = 1  # Set edge pixels to white (1 in grayscale)



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('CT Image')
plt.imshow(image, cmap='grey')
plt.subplot(1,2,2)
plt.title('Lung Segmentated')
plt.imshow(fill_image, cmap='grey')

plt.show()