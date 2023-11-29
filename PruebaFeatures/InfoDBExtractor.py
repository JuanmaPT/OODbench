from skimage import measure
from skimage.measure import regionprops
import numpy as np
import cv2
from utils import *

class PlanesetInfoExtractor:    
     def __init__(self, planeset, config):
         
        self.planeset= planeset
        self.config = config
        self.classMasks = self.get_class_masks()
        self.margin = self.extractMargin()
        #self.regionProps = self.get_RegionProps()
         
     
     def get_class_masks(self):
         class_masks = []
         for target_class in self.planeset.predictedClasses:
             class_mask = np.zeros_like(self.planeset.prediction )
             class_mask[self.planeset.prediction  == target_class] = 1
             class_masks.append(class_mask)
             
         return class_masks
     
     def calculate_distance_transforms(self): 
         distance_transforms= []
         for class_mask in self.classMasks:
             # using euclidean distance in distance transform 
             distance_transform = cv2.distanceTransform((class_mask* 255).astype(np.uint8), cv2.DIST_L2, 3) #neighborhood size 3x3
             distance_transforms.append(distance_transform)        
         return distance_transforms  
     
     def get_max_distance_transforms(self):
          max_distances = [np.max(dt) for dt in self.distanceTransforms]
          #max_positions = [np.unravel_index(np.argmax(dt, axis=None), dt.shape) for dt in self.distanceTransforms]
          #return [(i,j) for i,j in zip(max_distances, max_positions)]
          return max_distances
        
    
     def extractMargin(self):
        "extract the margin as the min distance between the anchor and a different class prediction"        
        margin = []
        coords = self.planeset.planeset.coords
        x = self.planeset.planeset.coefs1.cpu().numpy()
        y = self.planeset.planeset.coefs2.cpu().numpy()

        # Loop over anchors
        for i, (target_class, anchor) in enumerate(self.planeset.anchors.items()):
            coord = coords[i]  # Coordinate of current anchor

            # Get x, y positions and predicted classes
            x_positions = torch.from_numpy(x.flatten())
            y_positions = torch.from_numpy(y.flatten())
            predicted_classes = self.planeset.prediction.flatten()

            # Find positions where predicted class is different from anchor's target class
            different_class_positions = torch.nonzero(predicted_classes != target_class)

            # Calculate distances between anchor and positions with different class
            distances = torch.sqrt((x_positions[different_class_positions] - coord[0]) ** 2 +
                                (y_positions[different_class_positions] - coord[1]) ** 2)

            # Find the minimum distance
            min_distance = torch.min(distances)
            margin.append(min_distance)

        return margin
        
     
     def get_RegionProps(self):
         region_props_dict = {}
         for i, (target_class, anchor) in enumerate(self.planeset.anchors.items()):
             component_mask = self.classMasks[i]
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
