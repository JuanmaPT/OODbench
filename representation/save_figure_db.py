import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches

def count_and_map_values_2d(arr):
    # Step 1: Count the number of different values
    unique_values = np.unique(arr)
    NdifferentValues = unique_values.size

    # Step 2: Map the values to the new range
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    mapped_array = np.vectorize(lambda x: value_to_index[x])(arr)

    return mapped_array, NdifferentValues

def generate_and_save_combined_figure(i_triplet, pkl_path, output_path):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    # Read the class labels from the file
    with open('imagenet1klabels.txt', 'r') as file:
        class_labels_dict = eval(file.read())

    # Create an image (replace this with your matrix)
    matrix = data[f'Matrix_{i_triplet}']

    mapped_array, NdifferentValues = count_and_map_values_2d(matrix)

    # Create a color map for the number of different values
    cmap = plt.get_cmap('rainbow', NdifferentValues)

    # Plot the mapped array with color labels
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.tight_layout(pad=5.0)

    cax = axs[0].matshow(mapped_array, cmap=cmap)

    # Create a color bar with labels from class_labels_dict
    unique_values = np.unique(matrix)
    axs[0].set_title("Mapped Array")
    cbar = plt.colorbar(cax, ax=axs[0], ticks=np.arange(NdifferentValues))
    cbar.ax.set_yticklabels([class_labels_dict[value] for value in unique_values])

    triangle = patches.Polygon([[4, 4], [4, 5], [5, 4]], closed=True, fill=True, color='black')
    circle = patches.Circle((44, 4), radius=1, fill=True, color='black')
    square = patches.Rectangle((20, 44), width=2, height=2, fill=True, color='black')

    axs[0].add_patch(triangle)
    axs[0].add_patch(circle)
    axs[0].add_patch(square)

    # Add text labels to the shapes
    axs[0].annotate("German Shepherd", (4, 4), xytext=(4, 2), textcoords='offset points', color='black')
    axs[0].annotate("Fountain", (32, 3), xytext=(44, 2), textcoords='offset points', color='black')
    axs[0].annotate("Jeep", (20, 44), xytext=(20, 2), textcoords='offset points', color='black')

    # Display the combination of mapped array and shapes
    combi_path = data[f'Combi_{i_triplet}']
    labels = ['German_shepherd', 'fountain', 'jeep']

    # Loop through the paths and display the images
    for i, path in enumerate(combi_path):
        img = mpimg.imread(path)
        axs[i+1].imshow(img)
        axs[i+1].set_title(labels[i])  # Set a title for each image
        axs[i+1].axis('off')  # Turn off the axes


    # Set the aspect of images to 'auto' to display them without distortion
  
    # Save the combined figure as a JPEG
    plt.savefig(output_path, format='jpeg')
    plt.close()


for i_triplet in range(125):
    pkl_path = './results/results_3_3_imagenet_original.pkl'
    output_path = f'./combined_figures_db/combined_figure{i_triplet}.jpg'
    generate_and_save_combined_figure(i_triplet, pkl_path, output_path)
