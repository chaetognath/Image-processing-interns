import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
import scipy.ndimage as ndi


def make_grid2d(size):
    grid = np.linspace(-(size//2), (size//2), size)
    xx,yy = np.meshgrid(grid,grid)
    return xx,yy

def make_gaussian_mask2d(size,sigmma):
    xx,yy = make_grid2d(size)
    mask2d = np.exp(-(xx**2+yy**2)/(2*sigmma**2))
    mask2d /= np.sum(mask2d)
    return mask2d

from mpl_toolkits.mplot3d import Axes3D
def surface_plot(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    surf = ax.plot_surface(x,y,z, cmap ='gray')
    ax.plot_wireframe(x,y,z, color = 'c', linewidth = 0.5)
    plt.show()

size = 3
xx,yy = make_grid2d(size)
sigmas = [0.5]
for sigma in sigmas:
    m2= make_gaussian_mask2d(size, sigma)
    surface_plot(xx,yy,m2)


laplacian_mask2d = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)

prewitt_filter = np.array([
            [-1,  0,  1],
            [-1,  0,  1],
            [-1,  0,  1]
        ], dtype=np.float32)


prewitt_filter_horizontal = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float32)


sobel_filter = np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ], dtype=np.float32)


sobel_filter_horizontal = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float32)


size = 3
xx,yy = make_grid2d(size)

# surface_plot(xx,yy,laplacian_mask2d)

surface_plot(xx,yy,prewitt_filter)

surface_plot(xx,yy,prewitt_filter_horizontal)

# surface_plot(xx,yy,sobel_filter)

# surface_plot(xx,yy,sobel_filter_horizontal)