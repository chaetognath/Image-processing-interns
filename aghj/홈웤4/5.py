#median filter
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi
import gaussian as gau
import skimage.transform as tr

original_image = io.imread('images/img.jpg')
i = io.imread('images/sadimg2.bmp')

print(tr.rotate(original_image, 30, resize=True).shape)

gaussian_mask = gau.make_gaussian_mask2d(5,1)

lego_1_original = ndi.convolve(i,gaussian_mask)
lego_1_ori = tr.resize(lego_1_original, (478, 478), order =3)
lego_2_original = tr.rotate(lego_1_ori, -30)

# 원본 이미지 크기
h, w = original_image.shape[:2]
print(original_image.shape)

# 회전된 이미지 크기
h2, w2 = lego_2_original.shape[:2]

# 중앙 좌표
center_y, center_x = h2 // 2, w2 // 2

# 자를 영역 시작, 끝 좌표
start_y = center_y - h // 2
end_y   = center_y + h // 2
start_x = center_x - w // 2
end_x   = center_x + w // 2

# crop (원본 크기만큼 잘라내기)
lego_2_cropped = lego_2_original[start_y:end_y, start_x:end_x]

plt.subplot(1,4,1)
plt.imshow(i, cmap = 'gray')

plt.subplot(1,4,2)
plt.imshow(lego_1_original, cmap='gray')


plt.subplot(1,4,3)
plt.imshow(lego_1_ori, cmap='gray')

plt.subplot(1,4,4)
plt.imshow(lego_2_cropped,cmap='gray')


plt.show()