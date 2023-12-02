import os
import numpy as np
import matplotlib.pyplot as plt



# Save the plot to an image
results_folder = "results"
model_name = 'resnet18'
dataset_name = 'generated'
dataset_folder = os.path.join(results_folder, model_name,dataset_name)
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

accuracy_file_path = os.path.join(dataset_folder, "accuracy_triplet.npy")
margin_file_path = os.path.join(dataset_folder, "margin_triplet.npy")
margin_triplet = np.load(margin_file_path)
accuracy_triplet = np.load(accuracy_file_path)

dataset_folder = os.path.join(dataset_folder, 'figures')
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)



import numpy as np
import matplotlib.pyplot as plt



############################### FIGURE WITH ACCURACIES ##########################33

# Assuming accuracy_array is already defined with the data

class_names = ['German_shepherd', 'fountain', 'jeep']

# Calculate the percentages of True values for each class
accuracy_array = accuracy_triplet
percentages = []
for i in range(3):
    column_data = accuracy_array[:, i]
    true_count = np.count_nonzero(column_data)
    total_count = len(column_data)
    percentage = true_count / total_count * 100
    percentages.append(percentage)

# Create a bar plot of the percentages
plt.figure(figsize=(8, 6))
plt.bar(class_names, percentages)
plt.title('Accuracy Percentages for Each Class')
plt.xlabel('Class')
plt.ylabel('Accuracy Percentage')

# Add the percentage values as text on top of the bars
for i, percentage in enumerate(percentages):
    plt.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')


plot_file_path = os.path.join(dataset_folder, "accuracies_total.png")
plt.savefig(plot_file_path)
plt.close()


################### FIGURE WITH MARGIN HISTOGRAM ###########################


# Create a mask for NaN values in margin_triplet
nan_mask = np.isnan(margin_triplet)

# Plot joint histograms for each column of margin_triplet
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
max_value = 0  # Variable to store the maximum value among all histograms

for i in range(3):
    column_data = margin_triplet[:, i]
    column_data_without_nan = column_data[~nan_mask[:, i]]

    # Plot the joint histogram ignoring NaN values
    counts, bins, patches = ax[i].hist(column_data_without_nan, bins='auto', color='#6aa84f')

    # Normalize the counts to obtain the PMF
    total_points = len(column_data_without_nan)
    pmf = counts / total_points

    bin_edges = np.linspace(0, 4, 11)

    ax[i].hist(pmf, bins=bin_edges, align='left', density=True, rwidth=0.8, alpha=0.7)


    # Update the maximum value if necessary
    max_value = max(max_value, pmf.max())

    # Set the title with class name and accuracy
    title = f'{class_names[i]}, Accuracy: {percentages[i]}%'
    ax[i].set_title(title)

# Set the y-axis limit to the maximum PMF value for all histograms
for i in range(3):
    ax[i].set_ylim(0, max_value)

# Optionally, you can also set the y-axis label to 'PMF'
ax[0].set_ylabel('PMF')


# Set the x-axis limit
max_value = np.nanmax(margin_triplet)
ax[0].set_xlim(0, max_value)
ax[1].set_xlim(0, max_value)
ax[2].set_xlim(0, max_value)

# Set the axis labels
ax[1].set_xlabel('Margin')
ax[1].set_ylabel('Occurrences')
ax[2].set_xlabel('Margin')
ax[2].set_ylabel('Occurrences')
ax[0].set_xlabel('Margin')
ax[0].set_ylabel('Occurrences')
plt.tight_layout()



plot_file_path = os.path.join(dataset_folder, "Margin_" + dataset_name +"_" + model_name+ ".png")
plt.savefig(plot_file_path)