import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi

i = np.float64(io.imread('images/roundImage.png'))


prewitt_filter = np.array([
            [-1,  0,  1],
            [-1,  0,  1],
            [-1,  0,  1]
        ], dtype=np.float32)

sobel_filter = np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ], dtype=np.float32)


apply_h_prewitt = ndi.convolve(i, prewitt_filter, mode='reflect')
apply_h_sobel = ndi.convolve(i, sobel_filter, mode='reflect')


center_h_p =apply_h_prewitt.shape[0]//2
center_h_s =apply_h_sobel.shape[1]//2
center_v_p =apply_h_prewitt.shape[0]//2
center_v_s = apply_h_sobel.shape[1]//2
'''
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.imshow(apply_h_prewitt, cmap='gray')

plt.subplot(2,2,2)
plt.imshow(apply_h_sobel, cmap='gray')

plt.subplot(2,2,3)
plt.plot(apply_h_prewitt[center_h_p])


plt.subplot(2,2,4)
plt.plot(apply_h_sobel[center_v_p])

'''

#plt.show()

prewitt_transpose = np.transpose(apply_h_prewitt)
sobel_transpose = np.transpose(apply_h_sobel)
'''
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.imshow(prewitt_transpose, cmap='gray')

plt.subplot(2,2,2)
plt.imshow(sobel_transpose, cmap='gray')

plt.show()
center_index = i.shape[1]//2
vertical_center = [item[center_index] for item in prewitt_transpose]
plt.plot(vertical_center)

'''

#plt.show()


perfect = np.abs(apply_h_prewitt) + np.abs(prewitt_transpose)
perfect_2 = np.abs(apply_h_sobel) + np.abs(sobel_transpose)

plt.subplot(1,2,1)
plt.imshow(perfect, cmap='viridis')

plt.subplot(1,2,2)
plt.imshow(perfect_2, cmap='gray')
plt.show()