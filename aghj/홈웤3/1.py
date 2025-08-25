import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi

i = io.imread('images/roundImage.png')
af1 = ndi.uniform_filter(i,[15,15], mode='constant')
plt.imshow(af1,cmap='gray')
plt.savefig("output.png")