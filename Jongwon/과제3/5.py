import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

image = np.float64(io.imread('images/chest.png'))

# 3x3 Laplacian filter with center value = 4
laplacian_filter = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

laplacian_image = ndi.convolve(image, laplacian_filter)

sharper_image = image - laplacian_image

min_val = np.min(sharper_image)
max_val = np.max(sharper_image)
sharper_image_rescaled = (sharper_image - min_val) / (max_val - min_val)
sharper_image_rescaled *= 255


plt.figure(figsize = (12,6))
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(image, cmap = 'gray')
plt.subplot(1,3,2)
plt.title('Laplacian Image')
plt.imshow(laplacian_image, cmap = 'gray')
plt.subplot(1,3,3)
plt.title('Sharper Image')
plt.imshow(sharper_image_rescaled, cmap = 'gray')
plt.show()
