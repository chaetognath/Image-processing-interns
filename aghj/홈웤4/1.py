import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi
import gaussian as gau

i = plt.imread('images/brain_whitenoise.jpg')

laplacian_mask2d = np.array([
            [ 0, 0, -1, 0, 0],
            [ 0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0 , 0]
        ])

lapul_image = ndi.convolve(i,laplacian_mask2d)

size = 5
sigma = 3
gmask = gau.make_gaussian_mask2d(size,sigma)
gau_image = ndi.convolve(i,gmask)

final = ndi.convolve(gau_image, laplacian_mask2d)

plt.subplot(1,3,3)
plt.title('LoG')
plt.imshow(final, cmap='gray')

plt.subplot(1,3,2)
plt.title('laplacian~')
plt.imshow(lapul_image, cmap='gray')

plt.subplot(1,3,1)
plt.title('original')
plt.imshow(i, cmap='gray')
plt.show()

