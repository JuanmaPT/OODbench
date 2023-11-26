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
        for i, (target_class, anchor) in enumerate(self.planeset.anchors.items()): 
            loc = np.where(self.planeset.predictedClasses == target_class)[0][0] #index containg the anchor mask in classMasks lists
            anchorMask= self.classMasks[loc]
            restClassesMask = np.sum([self.classMasks[i] for i in range(len(self.classMasks)) if i != loc], axis=0)
            
            #get the contour of restClassesMask
            distance_transform = cv2.distanceTransform((restClassesMask* 255).astype(np.uint8), cv2.DIST_L2, 3)
            min_val_DT = np.unique(distance_transform)[1]
            coords = np.argwhere(distance_transform == min_val_DT)  #contains the coordenates of the border of  restClassesMask
            
            #compute difference between anchor feature vector and feature vector corresponding to each coord in the grid
            #get the feature vectors corresponding to the coordenates in the border as row * num_colums + column
            coords_idx=[i*self.config.resolution+j for i,j in coords]
            distances = [euclidean_distance(self.planeset.triplet.features[i],self.planeset.planeset[idx]) for idx in coords_idx]
        
            #compute the margin as the minimum distance btw the anchor and the points belonging to the other clases
            margin.append(np.min(distances) )
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
