import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_filter(ax, filter_matrix, title):
    """
    Helper function to display a 3x3 filter as a 3D surface plot.
    """
    # Create a 3x3 grid of coordinates for the plot
    x = np.arange(-1, 2)
    y = np.arange(-1, 2)
    X, Y = np.meshgrid(x, y)

    # Create the surface plot
    ax.plot_surface(X, Y, filter_matrix, cmap='grey')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Weight')
    # Set ticks to correspond to matrix indices
    ax.set_xticks(np.arange(-1, 2))
    ax.set_yticks(np.arange(-1, 2))


# --- a) 3x3 Gaussian filter with standard deviation 0.5 ---
sigma = 0.5
size = 3
# Create a grid of coordinates from -(size//2) to size//2
grid = np.linspace(-(size // 2), size // 2, size)
x, y = np.meshgrid(grid, grid)

# Calculate the Gaussian function
gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
# Normalize the filter so that the sum of all elements is 1
gaussian_filter /= np.sum(gaussian_filter)


# --- b) 3x3 Laplacian filter with center value = 4 ---
# This is the negative of the standard Laplacian operator
laplacian_filter = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])


# --- c) 3x3 Prewitt filter for vertical edges ---
# This filter detects horizontal gradients (vertical lines)
prewitt_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


# --- d) 3x3 Sobel filter for vertical edges ---
# Similar to Prewitt, but with more weight on the center row
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])


# --- Print the generated filters ---
print("a) 3x3 Gaussian Filter (sigma=0.5):\n", np.round(gaussian_filter, 3), "\n")
print("b) 3x3 Laplacian Filter (center=4):\n", laplacian_filter, "\n")
print("c) 3x3 Prewitt Filter (vertical edges):\n", prewitt_vertical, "\n")
print("d) 3x3 Sobel Filter (vertical edges):\n", sobel_vertical, "\n")


# --- Plotting all filters ---
fig = plt.figure(figsize=(11, 8))

# Plot Gaussian filter
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
plot_filter(ax1, gaussian_filter, 'a) 3x3 Gaussian Filter (σ=0.5)')

# Plot Laplacian filter
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
plot_filter(ax2, laplacian_filter, 'b) 3x3 Laplacian Filter')

# Plot Prewitt filter
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
plot_filter(ax3, prewitt_vertical, 'c) 3x3 Prewitt Filter (Vertical)')

# Plot Sobel filter
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
plot_filter(ax4, sobel_vertical, 'd) 3x3 Sobel Filter (Vertical)')

plt.tight_layout()
plt.show()
