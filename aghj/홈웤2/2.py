import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex

arr = np.arange(256)

i = np.uint8(plt.imread('images/x-ray.png')*255)
lut2 = np.uint8(np.clip(arr*0.5,0,255))
p = lut2[i]

ph =ex.equalize_hist(p)


plt.figure(figsize=(10,5))
plt.subplot(1,2,2)
plt.title('image')
plt.hist(ph.ravel()*255, bins=256)
plt.subplot(1,2,1)
plt.title('Histogram')
plt.imshow(ph, cmap='gray')
plt.show()