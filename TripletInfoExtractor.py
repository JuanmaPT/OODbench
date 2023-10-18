import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import cv2

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