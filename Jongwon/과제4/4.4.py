import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as tf
import skimage.io as io

# Rotation

image = plt.imread('images/body.jpg')

rotated_image_NN = tf.rotate(image, 30, order = 0)
rotated_image_BC = tf.rotate(image, 30, order = 2)
difference_image = np.abs(rotated_image_BC - rotated_image_NN)
binary = (difference_image > 127).astype(np.uint8)

plt.figure(figsize = (11,6))
plt.subplot(1,3,1)
plt.title('Nearest neighbor')
plt.imshow(rotated_image_NN, cmap = 'grey')
plt.subplot(1,3,2)
plt.title('Bicubic')
plt.imshow(rotated_image_BC, cmap = 'grey')
plt.subplot(1,3,3)
plt.title('Difference Image')
plt.imshow(binary, cmap = 'binary')
plt.show()