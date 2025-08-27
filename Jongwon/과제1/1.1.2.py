from skimage import io
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

w = plt.imread('images/koreaUniversity.jpg')
# plt.imshow(w)

m = np.array(w)
r = m * [1,0,0]
g = m * [0,1,0]
b = m * [0,0,1]

plt.figure()
plt.subplot(2,2,1)
plt.imshow(w)
plt.subplot(2,2,2)
plt.imshow(r)
plt.subplot(2,2,3)
plt.imshow(g)
plt.subplot(2,2,4)
plt.imshow(b)

plt.show()