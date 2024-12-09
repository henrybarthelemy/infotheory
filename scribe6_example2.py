import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def shannon_entropy(p1, p2):
    """Calculate Shannon entropy for a distribution with three probabilities."""
    p3 = 1 - p1 - p2
    if p3 < 0 or p1 < 0 or p2 < 0:
        return np.nan
    return -np.sum([p * np.log2(p) for p in [p1, p2, p3] if p > 0])


# Generate a triangular grid of probabilities
resolution = 300  # Increase for smoother visualization
p1 = np.linspace(0, 1, resolution)
p2 = np.linspace(0, 1, resolution)
p1_grid, p2_grid = np.meshgrid(p1, p2)

# Calculate p3 and Shannon entropy values
p3_grid = 1 - p1_grid - p2_grid  # p3 from p1 and p2
entropy_grid = np.zeros_like(p1_grid)
for i in range(resolution):
    for j in range(resolution):
        entropy_grid[i, j] = shannon_entropy(p1_grid[i, j], p2_grid[i, j])

# Mask invalid regions where p3 < 0
mask = (p1_grid + p2_grid > 1)
p3_grid = np.ma.array(p3_grid, mask=mask)
entropy_grid = np.ma.array(entropy_grid, mask=mask)

# Create 3D plot for Shannon entropy with p3 as color map
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface where color reflects Shannon entropy
surf = ax.plot_surface(p1_grid, p2_grid, p3_grid, facecolors=plt.cm.viridis(entropy_grid / np.nanmax(entropy_grid)), edgecolor='none')

# Labels and title
ax.set_xlabel('$p_1$')
ax.set_ylabel('$p_2$')
ax.set_zlabel('$p_3$')
ax.set_title("3D visualization of Shannon entropy over probability simplex")

# Add color bar for Shannon Entropy
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Shannon Entropy')

# Show plot
plt.show()
