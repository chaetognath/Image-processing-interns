import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
w=io.imread('images/lena.bmp')


plt.figure()
io.imshow(w)
plt.show()

plt.figure()
plt.imshow(w)
plt.imshow(w,cmap='gray')
plt.axis('off')
plt.show()