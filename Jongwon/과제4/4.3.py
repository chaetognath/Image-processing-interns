import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as tf

image = plt.imread('images/body.jpg')
print(image.shape)

resize_image = tf.rescale(image, 0.25, order = 1) # Bilinear interpolation
shrink_image_NN = tf.rescale(resize_image, 4, order = 0) # Nearest neighbor
shrink_image_BL = tf.rescale(resize_image, 4, order = 1) # Bilinear
shrink_image_BC = tf.rescale(resize_image, 4, order = 2) # Bicubic

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(2,2,2)
plt.title('Shrink Image (Nearest Neighbor)')
plt.imshow(shrink_image_NN, cmap='gray')
plt.subplot(2,2,3)
plt.title('Shrink Image (Bilinear)')
plt.imshow(shrink_image_BL, cmap='gray')
plt.subplot(2,2,4)
plt.title('Shrink Image (Bicubic)')
plt.imshow(shrink_image_BC, cmap='gray')

plt.show()

plt.title('Difference Image')
plt.imshow(np.abs(shrink_image_BC-shrink_image_BL), cmap='grey')
plt.show()
