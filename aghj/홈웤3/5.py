
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

plt.figure()
plt.imshow(applied_laplacian, cmap='gray')
plt.show()

plt.figure()
plt.imshow(sharpppp, cmap='gray')
plt.show()