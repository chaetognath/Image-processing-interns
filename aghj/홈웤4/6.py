#Affine transform
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi
import gaussian as gau
import skimage.transform as tr

im = io.imread('images/img.jpg')
af = io.imread('images/affined_img.bmp')

print(im.shape)
print(af.shape)

tform = tr.AffineTransform(shear=(-0.47,0),rotation=-0.05, translation=(-150,0))
shear_x = tr.warp(af, tform, order = 3)
final = tr.rescale(shear_x, (1,0.35), order =3)

plt.subplot(1,4,1)
plt.imshow(im, cmap='gray')

plt.subplot(1,4,2)
plt.imshow(af, cmap='gray')

plt.subplot(1,4,3)
plt.imshow(shear_x, cmap='gray')

plt.subplot(1,4,4)
plt.imshow(final, cmap='gray')
plt.show()