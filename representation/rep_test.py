import matplotlib.pyplot as plt
import pickle

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

path_results = '/net/travail/jpenatrapero/results/results_3_3_imagenet_original.pkl'
with open(path_results, 'rb') as file:
    data = pickle.load(file)

# Read the class labels from the file
with open('imagenet1klabels.txt', 'r') as file:
    class_labels_dict = eval(file.read())
# List the keys in the dictionary
keys = data.keys()




# Create an image (replace this with your matrix)
matrix = data['Matrix_0']  # Your 50x50 matrix of class labels

def on_click(event):
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        value = matrix[y, x]
        ax.set_title(f'Value at ({x}, {y}): {value}')
# Create a sample 50x50 matrix (you can use your own data)

# Define a colormap for different colors
cmap = plt.get_cmap('viridis')  # You can choose a different colormap

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the matrix as an image with colors
img = ax.imshow(matrix, cmap=cmap)

# Add a colorbar for reference
cbar = plt.colorbar(img)




# Connect the click event to the function
fig.canvas.mpl_connect('button_press_event', on_click)

# Display the figure with interactivity
plt.show()


