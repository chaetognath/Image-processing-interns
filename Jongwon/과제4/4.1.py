import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

image = plt.imread('images/brain_whitenoise.jpg') #uint8
print(image.dtype)

laplacian_filter = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [1, 2, -16, 2, 1],
    [0, 1, 2, 1, 0],
    [0, 0, 1, 0, 0]
])

guassian_image = ndi.gaussian_filter(image, sigma = 3.0)
LoG_image = ndi.convolve(guassian_image, laplacian_filter)


plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(image, cmap='grey')
plt.title('Original Image')
plt.subplot(1,3,2)
plt.imshow(ndi.convolve(image, laplacian_filter), cmap='grey')
plt.title('Laplacian Filter')
plt.subplot(1,3,3)
plt.imshow(LoG_image, cmap='grey')
plt.title('Gaussian + Laplacian Filter')
plt.show()