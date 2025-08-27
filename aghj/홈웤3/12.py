import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi

i = io.imread('images/roundImage.png')
af1 = ndi.uniform_filter(i,[15,15], mode='constant')

'''
plt.subplot(1,2,1)
plt.title('output image')
plt.imshow(af1,cmap='gray')

plt.subplot(1,2,2)
plt.title('input image')
plt.imshow(i,cmap='gray')
plt.show()
'''

'''
ih = i.shape[0]//2
qoduf = i[ih]
af1h = af1.shape[0]//2
qodufdldy = af1[af1h]

plt.figure()
plt.plot(qoduf)
plt.plot(qodufdldy)
plt.show()
'''