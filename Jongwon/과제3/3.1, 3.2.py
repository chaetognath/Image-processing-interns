import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

image = io.imread('images/roundImage.png')

# Make 15x15 average filter
avg_filter = 1/225*np.ones((15,15))

conv_image = ndi.convolve(image, avg_filter)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap = 'grey')
plt.subplot(1,2,2)
plt.title('Conv Image')
plt.imshow(conv_image, cmap = 'grey')
plt.show()

# Plot the center row of the image and conv_image
plt.plot(image[image.shape[0]//2], 'b-', label = 'Input')
plt.plot(conv_image[conv_image.shape[0]//2] ,'y--', label = 'Avg Filter')
plt.legend(loc = 'best')
plt.show()