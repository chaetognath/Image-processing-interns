import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

image = np.float64(io.imread('images/roundImage.png')) # image is now float64

# 3x3 Prewitt filter for vertical edges
prewitt_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


# 3x3 Sobel filter for vertical edges
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

prewitt_transposed = np.transpose(prewitt_vertical)
sobel_transposed = np.transpose(sobel_vertical)

def HorizontalCenterProfile(im):
    center = im.shape[0]//2
    return im[center]

def VerticalCenterProfile(im):
    center = im.shape[1]//2
    return [item[center] for item in im]

# main
prewitt_image = ndi.convolve(image, prewitt_vertical)
sobel_image = ndi.convolve(image, sobel_vertical)
transposed_prewitt_image = ndi.convolve(image, prewitt_transposed)
transposed_sobel_image = ndi.convolve(image, sobel_transposed)


# Display images and profiles
plt.figure(figsize = (12,6))
plt.subplot(2,3,1)
plt.imshow(image, cmap = 'gray')
plt.subplot(2,3,4)
plt.plot(HorizontalCenterProfile(image))
plt.subplot(2,3,2)
plt.imshow(prewitt_image, cmap='gray')
plt.subplot(2,3,5)
plt.plot(HorizontalCenterProfile(prewitt_image))
plt.subplot(2,3,3)
plt.imshow(sobel_image, cmap='gray')
plt.subplot(2,3,6)
plt.plot(HorizontalCenterProfile(sobel_image))

plt.show()

# Show vertical center profiles
plt.subplot(1,2,1)
plt.plot(VerticalCenterProfile(prewitt_image))
plt.subplot(1,2,2)
plt.plot(VerticalCenterProfile(sobel_image))

plt.show()

# Show transposed image and vertical center profile
plt.subplot(2,2,1)
plt.imshow(transposed_prewitt_image, cmap='gray')
plt.subplot(2,2,3)
plt.plot(VerticalCenterProfile(transposed_prewitt_image))
plt.subplot(2,2,2)
plt.imshow(transposed_sobel_image, cmap='gray')
plt.subplot(2,2,4)
plt.plot(VerticalCenterProfile(transposed_sobel_image))

plt.show()


# Show omni-directional edge detection
omni_prewitt = np.abs(prewitt_image) + np.abs(transposed_prewitt_image)
omni_sobel = np.abs(sobel_image) + np.abs(transposed_sobel_image)

plt.subplot(1,2,1)
plt.imshow(omni_prewitt, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(omni_sobel, cmap='gray')

plt.show()

# Show omni-directional edge absolute difference
plt.imshow(np.abs(omni_prewitt - omni_sobel), cmap='viridis')
plt.show()