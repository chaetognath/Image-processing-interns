
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi

i = np.float64(io.imread('images/chest.png'))
'''
plt.figure()
plt.imshow(i, cmap='gray')
plt.show()

'''

laplacian_mask2d = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)

applied_laplacian = ndi.convolve(i, laplacian_mask2d, mode='constant')
sharpppp = applied_laplacian + i

plt.subplot(1,3,1)
plt.imshow(i, cmap='gray')

plt.subplot(1,3,2)
plt.imshow(applied_laplacian, cmap='gray')

plt.subplot(1,3,3)
plt.imshow(sharpppp, cmap='gray')
plt.show()