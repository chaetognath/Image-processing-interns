import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex

arr = np.arange(256)

X_ray2 = np.uint8(plt.imread('images/x_ray2.png')*255)
x_ray_HE = ex.equalize_hist(X_ray2)


# x축 (0~255 픽셀 값)
x = np.arange(0, 256)

# 정규분포 PDF
pdf = 1/(20*np.sqrt(2*np.pi)) * np.exp(-((x - 128)**2)/(2*20**2))

# histogram처럼 만들기 위해 정규화 (합이 1이 되도록)
hist_desired = pdf / np.sum(pdf)
hist_desired = hist_desired.reshape(1,256)

cdf_inputs, cdf_bins = ex.cumulative_distribution(X_ray2)
cdf_inputs = cdf_inputs.reshape(1,256)

cdf_disired = hist_desired.cumsum()
cdf_disired = cdf_disired.reshape(1,256)

Lut = np.zeros(256, dtype=np.uint8)

for i in range(256):
    j = np.argmin(np.abs(cdf_inputs[0, i] - cdf_disired))
    Lut[i] = j

print(Lut)

x_ray_HS = Lut[np.uint(X_ray2*255)]
print(x_ray_HS)

plt.subplot(2,1,1)
plt.hist(x_ray_HS.ravel()*255, bins=256)
plt.subplot(2,1,2)
plt.imshow(np.uint8(x_ray_HS*255), cmap='grey', vmin=0, vmax=255)
plt.show()
