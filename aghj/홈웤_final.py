###################################
#                                 #
#          Final Project          #
#                                 #
###################################

# Complete the segmentation of of lung in the CT imageusing thresholding, edge detection, and morphological operations.


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.filters as filters
import scipy.ndimage as ndi

#필요한 libraries들을 import해오는 코드

image = io.imread('images/lungCT.PNG', as_gray=True)
print(image)

# Apply binary thresholding
# image의 픽셀 값 히스토그램을 분석해서 '최적의 임계값'을 계산하는 단계이다
thresh = filters.threshold_otsu(image)
# thresholding이란, 앞에서 계산한 밝기값 T를 기준으로 나누어 검은색으로 표현되는, 즉 T보다 낮은 폐를 골라내는 필터이다.
binary_image = (image < thresh).astype(np.float64)

# Morphological operation filters
# Morphological mask를 사용하게 되면, 잡음은 보다 줄어들고, 객체를 분리시킬 수 있다. 즉 경계값에도 동일한 1을 할당해, 이웃픽셀의 형태적인 패턴을 바탕으로 사진을 보정한다.
B_sq = np.ones((3,3), dtype = np.uint8)


# Apply morphological opening : remove small objects
#작은 밝은 하얀색 얼룩을 지운다.
binary_image = morph.binary_opening(binary_image, mode = 'min')


# Apply morphological closing : fill small holes
# 작은 검은 구멍을 메운다.
binary_image = morph.binary_closing(binary_image, np.ones((7,7)), mode = 'max')

# Apply Region Filling
binary_image = ndi.binary_fill_holes(binary_image, structure = np.ones((3,3))).astype(np.uint8) * 255

# Find connected components
#앞에서 잡음을 제거하고, 형태를 보정한 사진에서 흰색(1)픽셀들이 어떻게 연결되어있는지를 찾아내어 그룹에 라벨링을 한다.
lung = morph.label(binary_image, connectivity=1)
# 그중 우폐, 좌폐에 해당하는 라벨인, 3과 4를 추출한 필터 마스크를 만든다.
lung_mask = ((lung == 4) | (lung == 3)).astype(np.uint8)


# Find edges using morphological gradient
dilated = morph.binary_dilation(lung_mask, footprint = B_sq, mode = 'min').astype(np.uint8) * 255
eroded = morph.binary_erosion(lung_mask, footprint = B_sq, mode = 'min').astype(np.uint8) * 255
edge_image = dilated - eroded


image_rgb = plt.cm.gray(image)[:, :, :3] # Take only RGB channels, discard alpha if present

# Overlay the red edges
# Where edge_image is 255 (white), set the red channel to 1 (full red) and others to 0
# Where edge_image is 0 (black), leave the original pixel color
image_with_edges = image_rgb.copy()
image_with_edges[edge_image == 255, 0] = 1.0  # Set Red channel to max
image_with_edges[edge_image == 255, 1] = 0.0  # Set Green channel to min
image_with_edges[edge_image == 255, 2] = 0.0  # Set Blue channel to min



plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title('CT Image')
plt.imshow(image, cmap='grey')
plt.subplot(1,3,2)
plt.title('Lung Segmentated')
plt.imshow(image_with_edges, cmap='grey')
plt.subplot(1,3,3)
plt.title('binary image')
plt.imshow(binary_image, cmap='gray')

plt.show()