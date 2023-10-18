import matplotlib.pyplot as plt
import pickle

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

path_results = './results/results_3_3_imagenet_original.pkl'
with open(path_results, 'rb') as file:
    data = pickle.load(file)

# Read the class labels from the file
with open('imagenet1klabels.txt', 'r') as file:
    class_labels_dict = eval(file.read())
# List the keys in the dictionary
keys = data.keys()

for i in range(20):
    print(data[f'Combi_{i}'])




# Create an image (replace this with your matrix)
matrix = data['Matrix_1']  # Your 50x50 matrix of class labels


cmap = plt.get_cmap('viridis') 

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the matrix as an image with colors
img = ax.imshow(matrix, cmap=cmap)

# Add a colorbar for reference
cbar = plt.colorbar(img)


plt.show()


