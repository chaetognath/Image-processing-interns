from skimage import io
import skimage.io as io

import numpy as np
import matplotlib.pyplot as plt

"""
# Read in image file
w = io.imread('images/lena.bmp')

# Display image
plt.figure()
io.imshow(w)
plt.show()

plt.figure()
plt.imshow(w)
plt.imshow(w, cmap='gray')
plt.axis('off')
plt.show()
"""

# Read in RGB image
a = io.imread('images/koreaUniversity.jpg')
plt.figure()
io.imshow(a)
plt.axis('off')
plt.show()