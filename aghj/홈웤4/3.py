#median filter
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi
import gaussian as gau
import skimage.transform as tr

i = io.imread('images/body.jpg')
bilinear = tr.rescale(i, 0.25, order = 1)

original_bilinear = tr.rescale(bilinear, 4, order=0)
original_nearest = tr.rescale(bilinear, 4, order=1)
original_bicubic = tr.rescale(bilinear,4,order = 3)

print(i.shape)
plt.figure()
plt.subplot(1,4,1)
plt.imshow(bilinear, cmap='gray')

plt.subplot(1,4,2)
plt.title('bilinear')
plt.imshow(original_bilinear,cmap='gray')

plt.subplot(1,4,3)
plt.title('nearest')
plt.imshow(original_nearest, cmap='gray')

plt.subplot(1,4,4)
plt.title('bicubic')
plt.imshow(original_bicubic, cmap='gray')
plt.show()