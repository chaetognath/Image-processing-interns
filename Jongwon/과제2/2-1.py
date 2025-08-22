import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex

im = np.uint8(plt.imread('images/x-ray.png')*255)

lut2 = np.uint8(np.arange(256)*0.5)

new_image = lut2[im]

ph = ex.equalize_hist(new_image)

plt.figure(figsize=(12, 6))

plt.subplot(2,3,1)
plt.imshow(im, cmap='grey', vmin=0, vmax=255)
plt.title('Original Image')
plt.subplot(2,3,4)
plt.hist(im.ravel(), bins=256)
plt.ylim(0, 4000)
plt.title('Histogram of Original Image')

plt.subplot(2,3,2)
plt.imshow(new_image, cmap='grey', vmin=0, vmax=255)
plt.title('Image Adjusted with LUT')
plt.subplot(2,3,5)
plt.hist(new_image.ravel(), bins=256, range=(0, 255))
plt.ylim(0, 4000)
plt.title('Histogram of Adjusted Image')

plt.subplot(2,3,3)
plt.title('Histogram Equalization')
plt.imshow(ph, cmap='grey')
plt.subplot(2,3,6)
plt.title('Histogram of Equalized Image')
plt.hist(ph.ravel()*255, bins=256)
plt.ylim(0, 4000)

plt.show()