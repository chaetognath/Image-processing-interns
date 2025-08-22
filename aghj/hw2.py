import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt


i = np.uint8(plt.imread('images/x-ray.png')*255)

arr = np.arange(256) # [0, ..., 255]

lut1 = np.uint8(np.clip(arr*2,0,255))
lut2 = np.uint8(np.clip(arr*0.5,0,255))
lut3 = np.uint8(np.clip(arr*-1 +255,0,255))

t1 = 0.33335*np.arange(97)
t2 = ((192-32)*(np.arange(97,201)-96)/(200-96))+32
t3 = ((255-192)*(np.arange(201,256)-200)/(255-200))+192
lut4 = np.uint8(np.floor(np.concatenate((t1,t2,t3))))

plt.figure()
plt.subplot(1,4,1)
plt.imshow(lut1[i], cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(lut2[i], cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(lut3[i], cmap='gray')
plt.axis('off')
#t1 = 0.33335*np.arange(97)
#t2 = ((192-32)*(np.arange(97,201)-96)/(200-96))+32
#t3 = ((255-192)*(np.arange(201,256)-200)/(255-200))+192
#lut4 = np.uint8(np.floor(np.concatenate((t1,t2,t3))))

plt.subplot(1,4,4)
plt.imshow(lut4[i], cmap='gray')
plt.axis('off')

plt.show()