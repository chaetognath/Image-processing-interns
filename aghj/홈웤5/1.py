#Rotation
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi
import skimage.transform as tr

im_p1 = io.imread('images/p1.jpg')
im_p2 = io.imread('images/p2.jpg')

plt.subplot(1,2,1)
fig = plt.figure() ;fig.show(io.imshow(im_p1<50))

plt.subplot(1,2,2)
fig2 = plt.figure(); fig2.show(io.imshow(im_p2<50))
plt.show()
