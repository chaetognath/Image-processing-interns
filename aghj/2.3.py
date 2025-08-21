import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

def imdisp(imarr):
    plt.figure()
    io.imshow(imarr)
    #plt.axis('off')
    plt.show()

def imdisp2(imarr,color_map=None):
    plt.figure()
    if color_map ==None:
        plt.imshow(imarr)
    elif color_map =='gray':
        plt.imshow(imarr, cmap = color_map)
    else:
        print("에러가 남")
    #plt.axis('off')
    plt.show()

t1 = io.imread('images/trees.tif')
print(f"{t1.ndim}, {t1.shape}, {t1.dtype}")
imdisp(t1)
imdisp2(t1)

from PIL import Image
t2index = Image.open('trees.tif')
t2rgb =t2index.convert('RGB')
t2 =np.array(t2rgb)
imdisp(t2)

##subplot : 하나의 화면 안에 여러개의 그래프를 작성하는 것인데, 그렇다면, 원하시는 문제에 입각하여, 총3개의 plane을 각각 뽑아내어 그리면 되겠다.