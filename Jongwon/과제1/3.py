from skimage import io
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as tr

image = plt.imread('images/lena.bmp')
m1 = tr.rescale(image, 0.5)
m2 = tr.rescale(image, 0.25)
m3 = tr.rescale(image, 0.125)

m1 = tr.rescale(m1, 2)
m2 = tr.rescale(m2, 4)
m3 = tr.rescale(m3, 8)

plt.figure()
plt.subplot(1,4,1)
plt.title('Original Image')
plt.imshow(image, cmap = 'gray')
plt.subplot(1,4,2)
plt.title('0.5')
plt.imshow(m1, cmap = 'gray')
plt.subplot(1,4,3)
plt.title('0.25')
plt.imshow(m2, cmap = 'gray')
plt.subplot(1,4,4)
plt.title('0.125')
plt.imshow(m3, cmap = 'gray')
plt.show()