import torch
import torch.nn as nn
from torchvision import models, transforms
import json 

import numpy as np 
import cv2
from skimage import measure
from skimage.measure import regionprops

from SomepalliFunctions import get_plane, plane_dataset


class Configuration:
    def __init__(self, model, N, id_classes, resolution):
     
        if model == 'ResNet18':
            path_to_weights = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/resnet18-5c106cde.pth"
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # model to make predictions over images
            self.model = models.resnet18(pretrained= True)
            
            # base model as feature extractor
            self.base_model= nn.Sequential(*list(self.model.children())[:-1]) 
            self.base_model.eval()
            
            # classification head to make predictions over features
            self.head_model = head_model = nn.Sequential(                                                   
                nn.Flatten(),
                nn.Linear(512, 1000)
            )
            weights = torch.load(path_to_weights)
            self.head_model[-1].load_state_dict({
                'weight': weights['fc.weight'],
                'bias': weights['fc.bias']
            })
            self.head_model.eval()
            
        if model == 'ResNet50':
            pass
        
      
        self.N = N
        
        # return the class for the given id
        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
        
        self.labels = []
        for search_id in id_classes:
            for key, value in data.items():
                if search_id in value:
                    self.labels.append(int(key))
                    
        self.resolution = resolution
                    
                    
        
class Triplet:
    def __init__(self, pathImgs, config):
        self.pathImgs = pathImgs
        self.config= config
        self.imges = self.getImages()
        self.features = self.extractFeatures()
        self.prediction, self.score = self.predict()

    def getImages(self):
        return [Image.open(path[0]]), Image.open(path[1]), Image.open(path[2]) for path in self.pathImgs]
    
    def predict(self):
        preds_img = []
        scores_imgs = []
        for image in self.images:
            # Preprocess the image to match the input requirements of the ResNet model preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_image = preprocess(image)
            input_batch = input_image.unsqueeze(0)  # Add a batch dimension
            
            # Make predictions using the model
            with torch.no_grad():
                output = config.model(input_batch)
                _,pred_class= output.max(1)
                preds_img.append(pred_class.item())
                scores_imgs.append(output.softmax(dim=-1).max().item())
                
        return preds_img, scores_imgs
    
    def extractFeatures(self):
        feature_triplet = []
        for image in self.images::
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Define data transformations and apply to the image
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image= preprocess(image)
            
            # Expand dimensions to simulate batch size of 1
            image = image.unsqueeze(0)
            
            # Extract features with the base model
            with torch.no_grad():
                features = config.base_model(image)
            
            #print(features.shape)
            feature_triplet.append(features)

        return feature_triplet
    

class Planeset:
    def __init__(self, triplet, config):
        self.triplet = triplet
        self.config = config
        self.planeset = self.computePlaneset()
        self.anchors = self.getAnchors()
        self.prediction, self.score = self.predict()
    
       
    def computePlaneset(self):
        # calculate the plane spanned by the three images 
        a, b_orthog, b, coords = get_plane(triplet.features[0], triplet.features[1], triplet.features[2])
        
        # get the dataset of images on a 2D plane by the combination of the features
        return plane_dataset(featTri[0], a, b_orthog, coords, resolution= self.config.resolution )
        
    
    def predict(self):
        "returns 2D labelled image"
        self.config.model.eval()
        r = self.config.resolution
        preds = []
        scores = []
        for idx in range(len(self.planeset)):
            mixfeat = self.planeset[idx]
            with torch.no_grad():
                pred = self.model(mixfeat)
                pred_class = pred.argmax().item()
                preds.append(pred_class)
                scores.append(pred.softmax(dim=-1).max().item())
        
        planeset_pred = np.array(preds).reshape(r,r)
        planeset_score = np.array(scores).reshape(r,r)
                
        return planeset_pred, planeset_score
        
    def get_Anchors(self):
        anchor_dict = {}
        
        distances = []  # List to store the distances
        triplet_index = []
        for image_idx in range(3):
            for batch_idx, inputs in enumerate(planeset):
                distance = torch.dist(inputs, images[image_idx])
                distances.append(distance.item()) 
            min_distance_index = distances.index(min(distances))           
            distances = [] 
            triplet_index.append(min_distance_index)
        
        
        for i,idx in enumerate(triplet_idx):
            # convert 1d idx to x,y coords in a 2d grid
            x = idx % planeset.resolution
            y = idx // planeset.resolution
            # create the dictionary 
            anchor_dict[preds[i]] = (x,y)
        
        return anchor_dict 
        
    
class PlanesetInfoExtractor:    
    def __init__(self, planeset, config):
        
        self.label_img = planeset.planeset
        self.class_dict = planeset.anchors
        self.distance_transforms = self.calculate_distance_transforms()
        self.connected_components = self.calculate_connected_componets()
        self.max_distance_transform = self.get_max_distance_transforms()
        self.dist_angle_from_anchor = self.get_distances_and_orientations()
        self.margin = self.get_margins()
        self.regionProps = self.get_RegionProps()

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
        # the minimum distance between  the anchor and all the points in the border      
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
        
    
        
          
            
        
        
        
            