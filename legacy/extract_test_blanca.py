# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:45:51 2023

@author: Blanca
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import cv2

from PIL import Image as im, ImageFilter

#%%
np.random.seed(42)
# Create a 10x10 matrix filled with zeros
matrix = np.zeros((10, 10), dtype=int)

# Define the anchors and their corresponding values
anchors = {
    (2, 5): 1,
    (8, 3): 2,
    (8, 8): 3
}



# Iterate through the anchors and set their corresponding values
for anchor, value in anchors.items():
    row, col = anchor
    matrix[row, col] = value

    # Define the positions around the anchor where the value should be the same
    positions_around_anchor = [
        (row - 1, col - 1),
        (row - 1, col),
        (row - 1, col + 1),
        (row, col - 1),
        (row, col + 1),
        (row + 1, col - 1),
        (row + 1, col),
        (row + 1, col + 1),
        (row - 2, col - 2),
        (row - 2, col),
        (row - 2, col + 2),
        (row, col - 2),
        (row, col + 2),
        (row + 2, col - 2),
        (row + 2, col),
        (row + 2, col + 2)
    ]

    # Set the same value around the anchor
    for position in positions_around_anchor:
        x, y = position
        if 0 <= x < 10 and 0 <= y < 10:
            matrix[x, y] = value

# Fill the remaining places with random numbers 1, 2, or 3
remaining_positions = np.where(matrix == 0)
random_values = np.random.choice([1, 2, 3], size=len(remaining_positions[0]))
matrix[remaining_positions] = random_values

# Create your original 10x10 matrix (you can replace this with your actual data)
original_matrix = matrix

# Define the new size (double the size)
new_size = (20, 20)

# Create a new empty matrix with the new size
rescaled_matrix = np.zeros(new_size)

# Calculate the scaling factors for row and column indices
row_scale = new_size[0] / original_matrix.shape[0]
col_scale = new_size[1] / original_matrix.shape[1]

# Populate the new matrix using bilinear interpolation
for i in range(new_size[0]):
    for j in range(new_size[1]):
        # Calculate the corresponding position in the original matrix
        original_row = i / row_scale
        original_col = j / col_scale

        # Calculate the indices of the four neighboring points in the original matrix
        top_left = (int(original_row), int(original_col))
        top_right = (int(original_row), min(int(original_col) + 1, original_matrix.shape[1] - 1))
        bottom_left = (min(int(original_row) + 1, original_matrix.shape[0] - 1), int(original_col))
        bottom_right = (min(int(original_row) + 1, original_matrix.shape[0] - 1), min(int(original_col) + 1, original_matrix.shape[1] - 1))

        # Perform bilinear interpolation to calculate the new value
        alpha = original_row - top_left[0]
        beta = original_col - top_left[1]
        rescaled_matrix[i, j] = (1 - alpha) * ((1 - beta) * original_matrix[top_left] + beta * original_matrix[top_right]) + alpha * ((1 - beta) * original_matrix[bottom_left] + beta * original_matrix[bottom_right])

# Now, rescaled_matrix contains the 20x20 matrix with elements rescaled from the original 10x10 matrix
matrix = rescaled_matrix

map = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])

# Display the matrix with color-coded classes
plt.imshow(matrix, cmap=map, interpolation='nearest')
plt.colorbar(ticks=[1, 2, 3], label='Classes')
plt.title('Space with Different Classes')
plt.show()

































#%%
class TripletInfoExtractor:
    
    def __init__(self, label_img, class_dict):
        self.label_img = label_img
        self.class_dict = class_dict
        self.distance_transforms = self.calculate_distance_transforms()
        self.connected_components = self.calculate_connected_componets()

    def calculate_distance_transforms(self): 
        distance_transforms= []
        for target_class, anchor in self.class_dict.items():
            class_mask = np.zeros_like(self.label_img)
            class_mask[self.label_img == target_class] = 1

            distance_transform = cv2.distanceTransform((class_mask* 255).astype(np.uint8), cv2.DIST_L2, 3) #neighborhood size 3x3
            distance_transforms.append(distance_transform)
            
        return distance_transforms
    
    def get_max_distance_transforms(self):
         max_distances = [np.max(dt) for dt in self.distance_transforms]
         max_positions = [np.unravel_index(np.argmax(dt, axis=None), dt.shape) for dt in self.distance_transforms]
         return [(i,j) for i,j in zip(max_distances, max_positions)]

    def get_distance_from_anchor_to_border(self):
        # distance from the anchor to the region border 
        # using distance transform approach
        distances = [] 
        for i, (target_class, anchor) in enumerate(self.class_dict.items()):
            row, col = anchor
            distances.append(self.distance_transforms[i][row, col])  
        return distances
            
    def calculate_connected_componets(self):
        conn = [] 
        for i, (target_class, anchor) in enumerate(self.class_dict.items()):
            row, col = anchor
            class_mask = np.zeros_like(self.label_img)
            class_mask[self.label_img == target_class] = 1     
            
            # Label connected components in the binary mask
            labeled_image, num_labels = measure.label(class_mask, connectivity=2, return_num=True)
            
            # Find the label containing the anchor
            anchor_label = labeled_image[row, col]
            
            # Create a mask for the component containing the anchor
            component_mask = (labeled_image == anchor_label).astype(np.uint8)
            conn.append(component_mask)
            
        return conn
    
    def get_distances_and_orientations(self):
        out = [] 
        # get the contour of the component
        for i, (target_class, anchor) in enumerate(self.class_dict.items()):
            row, col = anchor
            contours, _ = cv2.findContours(np.uint8(self.connected_components[i]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            # compute distance and angle from anchor to point in contour
            distances_and_orientations = {}
            for contour in contours:
                for point in contour:
                    x,y = point[0]
                    # euclidean distance
                    distance = np.linalg.norm(np.array([col, row]) - np.array([x, y]))
                    # angle
                    angle = np.arctan2(y - row, x - col)
                    # Check if the angle is already in the dictionary
                    if angle in distances_and_orientations:
                        # If the distance for this angle is greater, update it
                        if distance > distances_and_orientations[angle][0]:
                            distances_and_orientations[angle] = (distance, angle)
                    else:
                        distances_and_orientations[angle] = (distance, angle)
                   
            out.append(list(distances_and_orientations.values()))    
        return  out
    
    def get_margins(self):
        # the minimum distance btw the anchor and all the points in the border 
        margin= []
        distances = self.get_distances_and_orientations()
        for class_list in distances:
            min_distance = min(class_list, key=lambda x:x[0])
            margin.append(min_distance[0])
        return margin 

    def get_RegionProps(self):
        region_props_dict = {}
        for i, (target_class, anchor) in enumerate(self.class_dict.items()):
            component_mask = self.connected_components[i]
            properties = regionprops(component_mask)

            # Convert region properties to a dictionary
            region_dict = {
                'area': properties[0].area,
                'centroid': properties[0].centroid,
                'orientation': properties[0].orientation,
                'major_axis': properties[0].major_axis_length,
                'minor_axis': properties[0].minor_axis_length,
            }

            region_props_dict[target_class] = region_dict

        return region_props_dict
    
#%%
anchors = {1:(4,10), 2:(17, 3), 3:(17, 15)}
tie = TripletInfoExtractor(matrix, anchors)
region_max_distances = tie.get_max_distance_transforms()
anchor_distances= tie.get_distance_from_anchor_to_border()
distances_class3= tie.get_distances_and_orientations()[2]
plt.imshow(tie.connected_components[2])
regProp= tie.get_RegionProps()
margins= tie.get_margins()

