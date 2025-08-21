from skimage import io
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as tr

im = plt.imread('images/lena.bmp')
bps = [(im >> i)%2 for i in range(8)]
'''
for i in range(8):
    plt.subplot(4,2,i+1)
    plt.imshow(bps[i], cmap = 'gray')
    plt.axis('off')
'''
msb2 = bps[6]+(bps[7]<<1)
plt.subplot(1,3,1)
plt.imshow(msb2, cmap = 'gray')
plt.title('2 MSB combined')

msb4 = 0
for i in range(4) :
    msb4 += bps[7-i]
    msb4 = msb4<<1
plt.subplot(1,3,2)
plt.title('4 msb combined')
plt.imshow(msb4, cmap = 'gray')

lsb4 = 0
for i in range(4) :
    lsb4 += bps[3-i]
    lsb4 = lsb4<<1
plt.subplot(1,3,3)
plt.title('4 lsb combined')
plt.imshow(lsb4, cmap = 'gray')

plt.show()