import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex

arr = np.arange(256)

X_ray2 = np.uint8(plt.imread('images/x_ray2.png')*255)
x_ray_HE = ex.equalize_hist(X_ray2)


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.hist(x_ray_HE.ravel()*255, bins=256)

plt.subplot(1,2,2)
plt.imshow(X_ray2, cmap = 'gray')
plt.show()
