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