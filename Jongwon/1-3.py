from skimage import io, color
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

w = plt.imread('images/koreaUniversity.jpg')
r, g, b = w[:, :, 0], w[:, :, 1], w[:, :, 2]
m = 0.299 * r + 0.587 * g + 0.114 * b
plt.imshow(m, cmap='gray')
plt.show()