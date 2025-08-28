import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as tf
import skimage.io as io

noisy_image = io.imread('images/sadimg2.bmp')
original_image = io.imread('images/img.jpg')

original_w, original_h = original_image.shape[:2]

# print(original_image.shape) # 350 x 350

def Resize(image, scale):
    resized_image = tf.rescale(image, scale, order = 3)
    return resized_image

def Rotate(image, degree):
    rotated_image = tf.rotate(image, degree, order = 3, resize=True)

    restored_h, restored_w = rotated_image.shape[:2]
    center_x, center_y = restored_w // 2, restored_h // 2

    # 원본 크기만큼 중앙에서 잘라내기
    half_h, half_w = original_h // 2, original_w // 2

    # 잘라낼 영역 계산 (정수형으로 변환)
    start_y = int(center_y - half_h)
    end_y = int(center_y + half_h)
    start_x = int(center_x - half_w)
    end_x = int(center_x + half_w)

    # 최종 복원 이미지
    final_restored_image = rotated_image[start_y:end_y, start_x:end_x]

    print(rotated_image.shape)
    return final_restored_image

def Gaussian(image):
    return ndi.gaussian_filter(image, sigma = 3.0)

result = noisy_image
result = Resize(result, (478/256, 478/64))
result = Gaussian(result)
result = Rotate(result, -30)



# print(tf.rotate(original_image, 30, resize = True).shape)


plt.figure(figsize = (12,8))
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(original_image, cmap = 'gray')
plt.subplot(1,3,2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap = 'gray')
plt.subplot(1,3,3)
plt.title('Processed Image')
plt.imshow(result, cmap = 'gray')
plt.show()