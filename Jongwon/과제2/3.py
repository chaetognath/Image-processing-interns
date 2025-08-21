import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex

x_ray2 = plt.imread('images/x_ray2.png')

x_ray_HE = ex.equalize_hist(x_ray2)

plt.subplot(1,2,1)
plt.imshow(x_ray_HE, cmap='grey')
plt.subplot(1,2,2)
plt.hist(x_ray_HE.ravel()*255, bins = 256)

hist_desired = 