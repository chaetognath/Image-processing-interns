import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import skimage.filters as fl

p1 = io.imread('images/p1.jpg')
p2 = io.imread('images/p2.jpg')

p1_center = p1[p1.shape[0]//2]
p2_center = p2[p2.shape[0]//2]


def adaptive_threshold(image):
    block_size = 80
    binary = np.zeros(image.shape, dtype=np.uint8)
    starts = range(0, image.shape[1]-1, block_size)
    ends = range(block_size, image.shape[1]+1, block_size)

    for start, end in zip(starts, ends):
        block = image[:, start:end]
        thresh = fl.threshold_otsu(block)
        binary[:, start:end] = (block > thresh) * 255
        
    return binary.astype(np.uint8)


plt.figure(figsize = (10, 8), tight_layout = True)

plt.subplot(3,2,1)
plt.title('p1')
plt.imshow(p1, cmap = 'gray')

plt.subplot(3,2,3)
plt.title('p1 histogram')
plt.plot(p1_center)

plt.subplot(3,2,5)
plt.title('p2 binary')
plt.imshow(adaptive_threshold(p1), cmap = 'gray')

plt.subplot(3,2,2)
plt.title('p2')
plt.imshow(p2, cmap = 'gray')

plt.subplot(3,2,4)
plt.plot(p2_center)
plt.title('p2 histogram')

plt.subplot(3,2,6)
plt.title('p2 binary')
plt.imshow(adaptive_threshold(p2), cmap = 'gray')

plt.show()
