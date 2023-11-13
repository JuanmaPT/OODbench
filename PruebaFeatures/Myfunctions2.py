import torch
import torch.nn as nn
from torchvision import models, transforms
import json 

import os
import numpy as np 
import cv2
from skimage import measure
from skimage.measure import regionprops

from PIL import Image
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
            self.head_model = nn.Sequential(                                                   
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
        self.id_classes = id_classes
        
        # return the class for the given id
        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
        
        self.labels = []
        for search_id in id_classes:
            for key, value in data.items():
                if search_id in value:
                    self.labels.append(int(key))
                    print(key,value)
                    
        self.resolution = resolution


def get_folders(path):
  """Returns a list of folders in the given path."""
  folders = []
  for item in os.listdir(path):
    if os.path.isdir(os.path.join(path, item)):
      folders.append(path + item)
  return folders


def getCombiFromDBoptimal(config, db_path):
    filenames_combinations = []

    # Get the file paths of the images in each folder
    class1_folder = db_path + "/" + config.id_classes[0] + "/"
    class2_folder = db_path + "/" + config.id_classes[1] + "/"
    class3_folder = db_path + "/" + config.id_classes[2] + "/"

    class1 = [os.path.join(class1_folder, filename) for filename in os.listdir(class1_folder)[:config.N]]
    class2 = [os.path.join(class2_folder, filename) for filename in os.listdir(class2_folder)[:config.N]]
    class3 = [os.path.join(class3_folder, filename) for filename in os.listdir(class3_folder)[:config.N]]

    # Generate combinations while ensuring unique rotations
    filenames_set = set()  # Use a set to store unique combinations

    for combi in itertools.product(class1, class2, class3):
        combi_sorted = sorted(combi)  # Sort the paths within each combination
        combi_tuple = tuple(combi_sorted)

        # Check if the combination is unique
        if combi_tuple not in filenames_set:
            filenames_set.add(combi_tuple)
            filenames_combinations.append(combi_tuple)

    print(f"Total number of unique combinations: {len(filenames_combinations)}")
    return filenames_combinations



def get_custom_colors(num):
    custom_colors = [
        (0, 1, 1),  #Cyan
        (1, 0, 0),  # Red
        (0.8, 0.6, 1),  # Lavender
        (0, 1, 0.5), # Lime Green
        (1, 1, 0),  # Yellow
        (1, 0.5, 1),  # Pink
        (0, 0.5, 1),    # Sky Blue
        (0.5, 0, 0.5),  # Purple      
       ]  
    
    return custom_colors[:num]



class Triplet:
    def __init__(self, pathImgs, config):
        self.pathImgs = pathImgs
        self.config= config
        self.images = self.getImages()
        self.features = self.extractFeatures()
        self.prediction, self.score = self.predict()
        self.isImgPredCorrect = self.checkPred()

    def getImages(self):
        return [Image.open(self.pathImgs[0]), Image.open(self.pathImgs[1]), Image.open(self.pathImgs[2])]
    
    def extractFeatures(self):
        feature_triplet = []
        for image in self.images:
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
                features = self.config.base_model(image)
            
            #print(features.shape)
            feature_triplet.append(features)

        return feature_triplet
    
    def predict(self):
        pred_imgs = []
        score_imgs = []
        for image in self.images:
            # Preprocess the image to match the input requirements of the ResNet model 
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_image = preprocess(image)
            input_batch = input_image.unsqueeze(0)  # Add a batch dimension
            
            # Make predictions using the model
            with torch.no_grad():
                output = self.config.model(input_batch)
                _,pred_class= output.max(1)
                pred_imgs.append(pred_class.item())
                score_imgs.append(output.softmax(dim=-1).max().item())
               
        return pred_imgs, score_imgs
        
    def checkPred(self):
        return [pred == true_label for pred, true_label in zip(self.prediction, self.config.labels)]
    

class Planeset:
    def __init__(self, triplet, config):
        self.triplet = triplet
        self.config = config
        self.planeset = self.computePlaneset()
        self.prediction, self.score = self.predict()
        self.anchors = self.getAnchors()
    
       
    def computePlaneset(self):
        # calculate the plane spanned by the three images 
        a, b_orthog, b, coords = get_plane(self.triplet.features[0], self.triplet.features[1], self.triplet.features[2])
        
        # get the dataset of images on a 2D plane by the combination of the features
        return plane_dataset(self.triplet.features[0], a, b_orthog, coords, resolution= self.config.resolution )
        
    
    def predict(self):
        "returns 2D labelled image"
        self.config.model.eval()
        r = self.config.resolution
        preds = []
        scores = []
        for idx in range(len(self.planeset)):
            mixfeat = self.planeset[idx]
            with torch.no_grad():
                pred = self.config.head_model(mixfeat)
                pred_class = pred.argmax().item()
                preds.append(pred_class)
                scores.append(pred.softmax(dim=-1).max().item())
        
        planeset_pred = np.array(preds).reshape(r,r)
        planeset_score = np.array(scores).reshape(r,r)
                
        return planeset_pred, planeset_score
        
    def getAnchors(self):
        anchor_dict = {}
        
        distances = []  # List to store the distances
        triplet_index = []
        for image_idx in range(3):
            for batch_idx, inputs in enumerate(self.planeset):
                distance = torch.dist(inputs, self.triplet.features[image_idx])
                distances.append(distance.item()) 
            min_distance_index = distances.index(min(distances))           
            distances = [] 
            triplet_index.append(min_distance_index)
        
        for i,idx in enumerate(triplet_index):
            # convert 1d idx to x,y coords in a 2d grid
            x = idx % self.planeset.resolution
            y = idx // self.planeset.resolution
            # create the dictionary 
            anchor_dict[self.triplet.prediction[i]] = (x,y)
        
        return anchor_dict
    
    """def getAnchors(self):
        anchor_dict = {}
        anchor_coords = []
        # find the position by the one matching original prediciton
        for y in range(self.config.resolution):
            for x in range(self.config.resolution):
                for i,pred in enumerate(self.triplet.prediction):
                    if self.prediction[y,x] == pred and self.score[y,x] == self.triplet.score[i]:
                        print(self.score[y,x])
                        print(self.triplet.score[i])
                        anchor_coords.append([x,y]) 
       
        for i,coords in enumerate(anchor_coords):
            # create the dictionary 
            anchor_dict[self.triplet.prediction[i]] = coords
            
            return anchor_dict"""
        
        
    def show(self, title=None):
        unique_classes = np.unique(self.prediction)   
        num_classes = len(unique_classes)

        # Create a color map with black background
        color_map = np.zeros((self.prediction.shape[0], self.prediction.shape[1], 3))

        # Generate colors for each class
        #cmap = plt.get_cmap('rainbow')
        #colors = [to_rgba(cmap(i))[:3] for i in np.linspace(0, 1, num_classes)]
        custom_colors = get_custom_colors(num_classes)

        # Assign colors based on prediction scores
        for class_label, color in zip(unique_classes, custom_colors):
            class_indices = np.where(self.prediction == class_label)
            class_scores = self.score[class_indices]
            
            #normalized_scores = (class_scores - np.min(class_scores)) / (np.max(class_scores) - np.min(class_scores))
            
            for idx, score in zip(zip(*class_indices), class_scores):
                color_map[idx] = np.array(color) * score # Adjust color intensity based on the prediction score

        
        fig, (ax1, ax2, ax3_1, ax3_2, ax3_3) = plt.subplots(1, 5, figsize=(20, 5))

        # Display the color map
        im = ax1.imshow(color_map)
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Planset prediction')
        ax1.axis('off')

        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
        
        
        # Create a horizontal color bar for each class (going from less bright to more bright)
        bar_height = 0.05  # Adjust the height based on your preference
        space_between_bars = 0.02  # Adjust the space between bars
        total_height = num_classes * (bar_height + space_between_bars) - space_between_bars
        start_y = (1 - total_height) / 2

        for i, (class_label, color) in enumerate(zip(unique_classes, custom_colors)):
            color_bar = np.ones((1, 100, 3)) * np.array(color)
            color_bar[0, :, :] *= np.linspace(0, 1, 100)[:, np.newaxis]  # Adjust color intensity (reversed)
            ax2.imshow(color_bar, extent=[0, 0.5, start_y + i * (bar_height + space_between_bars), start_y + (i + 1) * bar_height + i * space_between_bars], aspect='auto')

            # Label indicating the corresponding class, centered
            ax2.text(0.55, start_y + i * (bar_height + space_between_bars) + bar_height / 2, f'{class_label}: {data[str(class_label)][1]}', ha='left', va='center', rotation=0, fontsize=10)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        # Add text annotations for 0 and 1 below the bars
        #ax2.text(0, 0.25, '0', ha='center', va='center', fontsize=7)
        #ax2.text(0.5, 0.25, '1', ha='center', va='center', fontsize= 7) 
        ax2.text(0.45, 0.85, 'Prediction Scores colorbar', ha='center', va='center', fontsize= 9)
      
        
        # visualize triplet of images:
        for i, path in enumerate(self.triplet.pathImgs):
            img = mpimg.imread(path)
            ax3 = plt.subplot(1, 5, i + 3)
            ax3.imshow(img)
            ax3.axis('off')
            ax3.set_title(f"True class: {self.config.labels[i]}, {data[str(self.config.labels[i])][1]}\nPrediction:  {self.triplet.prediction[i]}, {data[str(self.triplet.prediction[i])][1]} \nScore:  {self.triplet.score[i]}")

        plt.show()
        
        
class PlanesetInfoExtractor:    
    def __init__(self, planeset, config):

        self.label_img = planeset.prediction
        self.class_dict = planeset.anchors
        self.classMasks = self.get_class_masks()
        
        # for the distance transform approach 
        self.distanceTransforms = self.calculate_distance_transforms()
        #self.distanceFromAnchorToBorder = self.get_distance_from_anchor_to_border()
        #self.margin1 = self.get_max_distance_transforms
        
    
        #self.connected_components = self.calculate_connected_componets()
        
        #self.dist_angle_from_anchor = self.get_distances_and_orientations()
        #self.margin = self.get_margins()
        #self.regionProps = self.get_RegionProps()

    
    def get_class_masks(self):
        class_masks = []
        for target_class, anchor in self.class_dict.items():
            class_mask = np.zeros_like(self.label_img)
            class_mask[self.label_img == target_class] = 1
        return class_masks
       
    def calculate_distance_transforms(self): 
        distance_transforms= []
        for class_mask in self.classMasks:
            distance_transform = cv2.distanceTransform((class_mask* 255).astype(np.uint8), cv2.DIST_L2, 3) #neighborhood size 3x3
            distance_transforms.append(distance_transform)        
        return distance_transforms
    
    """def get_max_distance_transforms(self):
         max_distances = [np.max(dt) for dt in self.distanceTransforms]
         max_positions = [np.unravel_index(np.argmax(dt, axis=None), dt.shape) for dt in self.distance_transforms]
         return [(i,j) for i,j in zip(max_distances, max_positions)]

    def get_distance_from_anchor_to_border(self):
        # distance from the anchor to the region border 
        # using distance transform approach
        distances = [] 
        for i, (target_class, anchor) in enumerate(self.class_dict.items()):
            print(i)
            plt.imshow(self.distanceTransforms[i])
            row, col = anchor
            distances.append(self.distanceTransforms[i][row, col])  
        return distances
            
  
    def get_distances_and_orientations(self):
        out = [] 
        # get the contour of the component
        for i, (target_class, anchor) in enumerate(self.class_dict.items()):
            row, col = anchor
            contours, _ = cv2.findContours(np.uint8(self.classMasks[i]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
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

        return region_props_dict"""