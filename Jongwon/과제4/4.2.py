import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

image = plt.imread('images/sadimg.bmp') # uint8

def sort_filter(image, x, y, filter_size):
    arr = []
    # Fixed range calculations
    for i in range(x-filter_size//2, x+filter_size//2+1):
        for j in range(y-filter_size//2, y+filter_size//2+1):
            arr.append(image[i][j])
    arr = sorted(arr)
    return arr[(filter_size**2)//2]


def MedianFilter(input_image, filter_size):
    x, y = input_image.shape
    # Fix padding creation syntax
    image_padded = np.pad(input_image, (filter_size//2, filter_size//2), mode='constant')
    
    # Fix output image creation syntax
    output_image = np.zeros((x, y), dtype=np.uint8)
    
    for i in range(x):
        for j in range(y):
            output_image[i][j] = sort_filter(image_padded, 
                                           i+filter_size//2, 
                                           j+filter_size//2, 
                                           filter_size)
    return output_image

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1,2,2)
plt.title('Median Filtered Image (filter size = 5)')
plt.imshow(MedianFilter(image, 5), cmap='gray')
plt.show()

