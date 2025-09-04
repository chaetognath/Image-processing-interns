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
binary_image = morph.binary_opening(binary_image, mode = 'min')


# Apply morphological closing : fill small holes
binary_image = morph.binary_closing(binary_image, np.ones((7,7)), mode = 'max')

# Apply Region Filling
binary_image = ndi.binary_fill_holes(binary_image, structure = np.ones((3,3))).astype(np.uint8) * 255

# Find connected components
lung = morph.label(binary_image, connectivity=1)

lung_mask = ((lung == 4) | (lung == 3)).astype(np.uint8)


# Find edges using morphological gradient
dilated = morph.binary_dilation(lung_mask, footprint = B_sq, mode = 'min').astype(np.uint8) * 255
eroded = morph.binary_erosion(lung_mask, footprint = B_sq, mode = 'min').astype(np.uint8) * 255
edge_image = dilated - eroded

# Convert the original grayscale image to an RGB image so we can add color
# We'll normalize it to 0-1 range for proper display with matplotlib
image_rgb = plt.cm.gray(image)[:, :, :3] # Take only RGB channels, discard alpha if present

# Overlay the red edges
# Where edge_image is 255 (white), set the red channel to 1 (full red) and others to 0
# Where edge_image is 0 (black), leave the original pixel color
image_with_edges = image_rgb.copy()
image_with_edges[edge_image == 255, 0] = 1.0  # Set Red channel to max
image_with_edges[edge_image == 255, 1] = 0.0  # Set Green channel to min
image_with_edges[edge_image == 255, 2] = 0.0  # Set Blue channel to min




plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('CT Image')
plt.imshow(image, cmap='grey')
plt.subplot(1,2,2)
plt.title('Lung Segmentated')
plt.imshow(image_with_edges, cmap='grey')

plt.show()