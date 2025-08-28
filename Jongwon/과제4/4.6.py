import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as tf
import skimage.io as io

image = io.imread('images/img.jpg')
affined_image = io.imread('images/affined_img.bmp')

tf_matrix = tf.AffineTransform(shear=(np.deg2rad(-30), np.deg2rad(3.5)), translation=(-175, 0))

altered_image = tf.warp(affined_image, tf_matrix, order = 3, preserve_range=True)

plt.figure(figsize = (11,5))
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(image, cmap = 'gray')
plt.subplot(1,3,2)
plt.title('Affined Image')
plt.imshow(affined_image, cmap = 'gray')
plt.subplot(1,3,3)
plt.title('Altered Image')
plt.imshow(altered_image, cmap='grey')

plt.show()
