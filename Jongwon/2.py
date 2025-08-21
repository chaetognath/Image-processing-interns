from skimage import io
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np


m = np.loadtxt('images/lena.txt')

n = np.float16(m)

w = plt.imread('images/lena.bmp')
w2 = np.uint8(w/128)*1.0

plt.figure()
plt.subplot(1,2,1)
plt.imshow(w2)
plt.subplot(1,2,2)
plt.imshow(n)
plt.show()