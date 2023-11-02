import matplotlib.pyplot as plt
import pickle
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches

def print_matrix(matrix):
    for row in matrix:
        for value in row:
            print(value, end='\t')  # Separate values with tabs for a matrix-like appearance
        print()  # Move to the next row

def count_and_map_values_2d(arr):
    # Step 1: Count the number of different values
    unique_values = np.unique(arr)
    NdifferentValues = unique_values.size

    # Step 2: Map the values to the new range
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    mapped_array = np.vectorize(lambda x: value_to_index[x])(arr)

    return mapped_array, NdifferentValues


path_results = './results/results_3_3_imagenet_original.pkl'
with open(path_results, 'rb') as file:
    data = pickle.load(file)

# Read the class labels from the file
with open('imagenet1klabels.txt', 'r') as file:
    class_labels_dict = eval(file.read())
# List the keys in the dictionary

print(class_labels_dict[1])

keys = data.keys()

i_triplet = 0




# Create an image (replace this with your matrix)
matrix = data[f'Matrix_{i_triplet}']  # Your 50x50 matrix of class labels


mapped_array, NdifferentValues = count_and_map_values_2d(matrix)



# Create a color map for the number of different values
cmap = plt.get_cmap('rainbow', NdifferentValues)

# Plot the mapped array with color labels
fig, ax = plt.subplots()
cax = ax.matshow(mapped_array, cmap=cmap)

# Create a color bar with labels from class_labels_dict
cbar = plt.colorbar(cax, ticks=np.arange(NdifferentValues))
unique_values = np.unique(matrix)
cbar.ax.set_yticklabels([class_labels_dict[value] for value in unique_values])


triangle = patches.Polygon([[4, 4], [4, 5], [5, 4]], closed=True, fill=True, color='black')
circle = patches.Circle((44, 4), radius=1, fill=True, color='black')
square = patches.Rectangle((20, 44), width=2, height=2, fill=True, color='black')

ax.add_patch(triangle)
ax.add_patch(circle)
ax.add_patch(square)

# Add text labels to the shapes
ax.annotate("German Shepherd", (4, 4), xytext=(4, 2), textcoords='offset points', color='black')
ax.annotate("Fountain", (32, 3), xytext=(44, 2), textcoords='offset points', color='black')
ax.annotate("Jeep", (20, 44), xytext=(20, 2), textcoords='offset points', color='black')


plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the paths to the images
combi_path = data[f'Combi_{i_triplet}']

# Create a 1x3 subplot for the three images
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

labels = ['German_shepherd','fountain','jeep']

# Loop through the paths and display the images
for i, path in enumerate(combi_path):
    img = mpimg.imread(path)
    axs[i].imshow(img)
    axs[i].set_title(labels[i])  # Set a title for each image
    axs[i].axis('off')  # Turn off the axes

# Set the aspect of images to 'auto' to display them without distortion
for ax in axs:
    ax.set_aspect('equal')

# Display the plot
plt.show()

