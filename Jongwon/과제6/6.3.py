import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.filters as filters
import scipy.ndimage as ndi

image = io.imread('images/hand.jpg')

erosion_image = morph.erosion(image, footprint= np.ones((3,3)),mode='min')
internal = image - erosion_image

filled = ndi.binary_fill_holes(internal, np.ones((3,3)), mode='min').astype(np.uint8) * 255

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Internal Boundary')
plt.imshow(internal, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Filled Image')
plt.imshow(filled, cmap='gray')
plt.show()