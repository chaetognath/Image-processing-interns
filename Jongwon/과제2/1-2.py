import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

im = plt.imread('images/x_ray2.png')

arr = np.arange(256)

lut1 = np.uint8(np.clip(arr*2, 0, 255))
lut2 = np.uint8(np.arange(256)/2)
lut3 = np.uint8(-np.arange(256)+255)

t1 = np.arange(97)*(32/96)
t2 = (160/104)*(np.arange(97, 201)-96)+32
t3 = (63/55)*(np.arange(201, 256)-200)+192

lut4 = np.uint8(np.floor(np.concatenate((t1, t2, t3))))

plt.subplot(1,4,1)
plt.plot(lut1)
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.subplot(1,4,2)
plt.plot(lut2)
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.subplot(1,4,3)
plt.plot(lut3)
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.subplot(1,4,4)
plt.plot(lut4)
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.show()