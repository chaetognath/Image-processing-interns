#Rotation
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi
import gaussian as gau
import skimage.transform as tr

i = io.imread('images/body.jpg')

deg30_near = tr.rotate(i,30, order =0)
deg30_bicubic = tr.rotate(i,30, order = 3)

plt.subplot(1,2,1)
plt.title('30degrees rotate using nearest')
plt.imshow(deg30_near, cmap='gray')

plt.subplot(1,2,2)
plt.title('30degrees rotate using bicubic')
plt.imshow(deg30_bicubic, cmap='gray')
plt.show()