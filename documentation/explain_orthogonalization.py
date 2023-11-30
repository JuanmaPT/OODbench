import numpy as np
import matplotlib.pyplot as plt

# Create vectors a and b
a = np.array([3, 4])  # Example vector a
b = np.array([2, 1])  # Example vector b

# Normalize vector a
a_norm = np.linalg.norm(a)
a_normalized = a / a_norm

# Compute the first coefficient
first_coef = np.dot(a_normalized, b)

# Calculate the projection of b onto a
b_projection = first_coef * a_normalized

# Calculate the orthogonalized vector b_orthog
b_orthog = b - b_projection

# Plot the vectors
plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector a')
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector b')
plt.quiver(0, 0, b_projection[0], b_projection[1], angles='xy', scale_units='xy', scale=1, color='g', label='Projection of b onto a')
plt.quiver(0, 0, b_orthog[0], b_orthog[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Orthogonalized b')

# Set axis limits
plt.xlim(0, 4)
plt.ylim(0, 4)

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()

# Show the plot
plt.show()
