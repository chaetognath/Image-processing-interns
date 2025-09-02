import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.filters as filters

image = io.imread('images/lung.jpg')

# Apply thresholding to create a binary image
thresh = filters.threshold_otsu(image)
thresh_image = (image < thresh).astype(np.uint8) * 255

struct_elem = np.ones((3,3), dtype=np.uint8)
dilation_image = morph.binary_dilation(thresh_image, struct_elem, mode='min').astype(np.uint8) * 255
erosion_image = morph.binary_erosion(dilation_image, struct_elem, mode='min').astype(np.uint8) * 255

external_boundary = dilation_image - thresh_image
internal_boundary = dilation_image - erosion_image


plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 4, 2)
plt.title('After Thresholding')
plt.imshow(thresh_image, cmap='gray')
plt.subplot(1, 4, 3)
plt.title('After Dilation')
plt.imshow(dilation_image, cmap='gray')
plt.subplot(1, 4, 4)
plt.title('After Erosion')
plt.imshow(erosion_image, cmap='gray')
plt.show()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('External Boundary')
plt.imshow(external_boundary, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Internal Boundary')
plt.imshow(internal_boundary, cmap='gray')
plt.show()