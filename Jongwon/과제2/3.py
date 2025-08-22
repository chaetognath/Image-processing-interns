import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex

x_ray2 = plt.imread('images/x_ray2.png')
print(x_ray2)
print(x_ray2.shape) # (214, 300)

x_ray_HE = ex.equalize_hist(x_ray2)
cdf_values, bin_centers = ex.cumulative_distribution(x_ray_HE)

'''
plt.figure(figsize=(12, 5))
plt.subplot(1,3,1)
plt.title('Histogram Equalized Image')
plt.imshow(x_ray_HE, cmap='grey')
plt.subplot(1,3,2)
plt.title('Histogram of Equalized Image')
plt.hist(x_ray_HE.ravel()*255, bins = 256)
plt.subplot(1,3,3)
plt.title('Cumulative Distribution Function')
#plt.hist(x_ray_HE.ravel()*255, bins=256, cumulative=True)
plt.plot(bin_centers, cdf_values, color='blue')

plt.show()
'''

u = np.arange(256)
# Create a Gaussian-like distribution
# You can adjust the mean (loc) and standard deviation (scale) to change the shape
hist_desired = np.exp(-((u - 128)**2) / (2 * 60**2))
# Normalize the histogram so that the sum of its elements is 1
hist_desired = hist_desired / hist_desired.sum()
hist_desired = hist_desired.reshape(1, 256)

hist_x_ray2, desired_bins = ex.histogram(x_ray2)
hist_x_ray2 = np.float64(hist_x_ray2 / hist_x_ray2.sum())

cdf_input, cdf_bins = ex.cumulative_distribution(x_ray2)
cdf_input = cdf_input.reshape(1, 256) # Ensure cdf_input is a 2D array

cdf_desired = hist_desired.cumsum()
cdf_desired = cdf_desired.reshape(1,256)




print(cdf_desired.shape)
print(cdf_input.shape)

plt.subplot(1,2,1)
plt.title('CDF of Input Image')
plt.plot(cdf_input[0], label='Input CDF')
plt.subplot(1,2,2)
plt.title('Custom CDF')
plt.plot(cdf_desired[0], label='Desired CDF')
plt.show()




# Create emety list of size 256
LUT = np.zeros(256, dtype=np.uint8)


input = np.uint8(cdf_input[0]*255)
desired = np.uint8(cdf_desired[0]*255)


print(input)
print(desired)


# Create a lookup table (LUT) to map the input histogram to the desired histogram
for i in range(256):
    j = np.argmin(np.abs(cdf_input[0, i] - cdf_desired))
    LUT[i] = j

print(LUT)

x_ray_HS = LUT[np.uint(x_ray2*255)]
print(x_ray_HS)



plt.figure(figsize=(13, 6))
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(np.uint8(x_ray2*255), cmap='grey', vmin=0, vmax=255)
plt.subplot(1,3,2)
plt.title('Histogram Equalized Image')
plt.imshow(x_ray_HE, cmap='grey')
plt.subplot(1,3,3)
plt.title('Specialized Image with custom LUT')
plt.imshow(np.uint8(x_ray_HS*255), cmap='grey', vmin=0, vmax=255)

plt.show()

