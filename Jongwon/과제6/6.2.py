import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.filters as filters

# Opening and Closing operations
image = io.imread('images/lung.jpg')

# Add salt & pepper noise to the image (noise density: 0.02)
noisy_image = image.copy()
noise_density = 0.02
num_salt = np.ceil(noise_density * image.size * 0.5)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
noisy_image[coords] = 255
image = noisy_image


# Apply thresholding to create a binary image
thresh = filters.threshold_otsu(image)
thresh_image = (image < thresh).astype(np.uint8) * 255

# Cross SE
struct_elem = np.ones((3,3), dtype=np.uint8)

dilation_image = morph.binary_dilation(thresh_image, struct_elem, mode='min').astype(np.uint8) * 255
erosion_image = morph.binary_erosion(thresh_image, struct_elem, mode='min').astype(np.uint8) * 255

opening_image = morph.binary_erosion(image, struct_elem, mode='min').astype(np.uint8) * 255
closing_image = morph.binary_dilation(image, struct_elem, mode='min').astype(np.uint8) * 255

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Noisy Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('After Opening')
plt.imshow(opening_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('After Closing')
plt.imshow(closing_image, cmap='gray')
plt.show()