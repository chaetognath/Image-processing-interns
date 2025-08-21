import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

im = np.uint8(plt.imread('images/x-ray.png')*255)

arr = np.arange(256)

lut1 = np.uint8(np.clip(arr*2, 0, 255))
lut2 = np.uint8(np.arange(256)*0.5)
lut3 = np.uint8(-np.arange(256)+255)

t1 = np.arange(97)*(32/96)
t2 = (160/104)*(np.arange(97, 201)-96)+32
t3 = (63/55)*(np.arange(201, 256)-200)+192

lut4 = np.uint8(np.floor(np.concatenate((t1, t2, t3))))

LUTS = [lut1, lut2, lut3, lut4]

def ImageAdjustLut(a, b) :
    new_image = b[a]
    return new_image

for i in range(4) :
    plt.subplot(2,2,i+1)
    plt.imshow(ImageAdjustLut(im, LUTS[i]), cmap = 'gray', vmin=0, vmax=255)

plt.show()

