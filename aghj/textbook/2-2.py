import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
a = io.imread('images/lena.bmp')
plt.figure()
io.imshow(a)
plt.axis('off')
plt.show()

print(a.shape)
print(a.dtype)
print(a.ndim)
print(a[100,200,2])
print(a[100,200,:])
print(a[100,200,0:2])