import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

image = np.float64(io.imread('images/chest.png')) # type = float64
smoothed_image = ndi.gaussian_filter(image, sigma=1.3)

# 3x3 Laplacian filter with center value = 4
laplacian_filter = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

laplacian_image = ndi.convolve(image, laplacian_filter, mode='constant') # type = float64

sharper_image = smoothed_image + laplacian_image

plt.figure(figsize = (12,6))
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(image, cmap = 'gray')
plt.subplot(1,3,2)
plt.title('Laplacian Image')
plt.imshow(laplacian_image, cmap = 'gray')
plt.subplot(1,3,3)
plt.title('Sharper Image')
plt.imshow(sharper_image, cmap = 'gray')
plt.show()