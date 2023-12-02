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
import matplotlib.patches as patches

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from SomepalliFunctions import get_plane, plane_dataset

class Configuration:
    def __init__(self, model, N, id_classes, resolution, margin_th):
        self.modelType = model
     
        if  self.modelType == 'ResNet18':
           
            # model to make predictions over images
            self.model = models.resnet18(pretrained= True)
            resnet_weights = self.model.state_dict()
            
            # base model as feature extractor
            self.base_model= nn.Sequential(*list(self.model.children())[:-1]) 
            self.base_model.eval()
            
            # classification head to make predictions over features
            self.head_model = nn.Sequential(                                                   
                nn.Flatten(),
                nn.Linear(512, 1000)
            )
           
            # weighs for head model 
            self.head_model[-1].weight.data = resnet_weights['fc.weight'].view(self.head_model[-1].weight.size())
            self.head_model[-1].bias.data = resnet_weights['fc.bias'].view(self.head_model[-1].bias.size())
            self.head_model.eval()
            
            
            
        if  self.modelType == 'ViT': 
            # model to make predictions over images
            self.model = models.vit_b_16(weights='IMAGENET1K_V1')
            vit_weights= self.model.state_dict()
            
            # base model as feature extractor
            self.base_model = nn.Sequential(*list(self.model.children())[:-2])  # Remove the last two layers (head and classifier)
            self.base_model.eval()
    
            # classification head to make predictions over features
            self.head_model = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(768, 1000)  
            )
            
            
            #print(self.model.state_dict().keys())
            
            #weights for head model
            self.head_model[-1].weight.data = vit_weights['heads.head.weight'].view(self.head_model[-1].weight.size())
            self.head_model[-1].bias.data = vit_weights['heads.head.bias'].view(self.head_model[-1].bias.size())
            self.head_model.eval()
      
       
                  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.N = N
        self.id_classes = id_classes
        
        # return the class for the given id
        with open( "PruebaFeatures\imagenet_class_index.json", 'r') as json_file:
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
        self.config = config
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
                #print(features.size())
            
           
            feature_triplet.append(features)

        return feature_triplet
    
    def predict(self):
        pred_imgs = []
        score_imgs = []
        for image in self.images:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image= preprocess(image)
            input_batch = image.unsqueeze(0)  # Add a batch dimension
            
            # Make predictions using the model
            with torch.no_grad():
                output = self.config.model(input_batch)
                _,pred_class= output.max(1)
                pred_imgs.append(pred_class.item())
                score_imgs.append(output.softmax(dim=-1).max().item())
            #print(pred_class.item())
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
        self.predictedClasses = np.unique(self.prediction)
    
       
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

        #########################################################################
        classes = self.prediction.flatten()
        # Get unique classes
        unique_classes = np.unique(classes)

        # Compute a colormap based on the number of unique classes
        colormap = plt.cm.get_cmap('viridis', len(unique_classes))

        # Normalize the classes to be between 0 and 1 for colormap indexing
        norm = plt.Normalize(classes.min(), classes.max())

        # Map the colors to the original prediction matrix
        colors = colormap(norm(classes))

        coords = self.planeset.coords
        x = self.planeset.coefs1.cpu().numpy()
        y = self.planeset.coefs2.cpu().numpy()

        markerd = {
            0: 'o',
            1 : '^',
            2 : 'X'
        }
        
        ############################################

        
        fig, (ax1, ax2, ax3_1, ax3_2, ax3_3) = plt.subplots(1, 5, figsize=(20, 5))

        # Display the color map
        detailed_fig = True
        if detailed_fig:
            plt.figure()
            # Create the scatter plot
            im = plt.scatter(x.flatten(), y.flatten(), c=colors, alpha=0.5, s=15)
            # Display the colorbar for reference
            cbar = plt.colorbar(im, ax=ax1)
            cbar.set_label('Class')
            for i in range(3):
                coord = coords[i]
                ax1.scatter(coord[0], coord[1], s=150, c='black', marker=markerd[i])
            
             # Loop over anchors
            for i, (target_class, anchor) in enumerate(self.anchors):
                coord = coords[i]  # Coordinate of the current anchor

                # Find positions where predicted class is different from anchor's target class
                different_class_positions = torch.nonzero(predicted_classes != target_class)

                if len(different_class_positions) > 0:
                    # Calculate distances between anchor and positions with different class
                    distances = torch.sqrt((x_positions[different_class_positions] - coord[0]) ** 2 +
                                        (y_positions[different_class_positions] - coord[1]) ** 2)

                    # Find the position with the minimum distance
                    min_distance_position = different_class_positions[torch.argmin(distances)]

                    # Get the corresponding coordinates
                    min_distance_coord = (x_positions[min_distance_position].item(), y_positions[min_distance_position].item())

                    # Plot vector connecting anchor to the element with minimum distance
                    plt.arrow(coord[0], coord[1], min_distance_coord[0] - coord[0], min_distance_coord[1] - coord[1],
                            head_width=0.2, head_length=0.2, fc='black', ec='black')

                # Set labels and legend
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.legend()

        im = ax1.imshow(color_map)
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Planeset prediction')
        ax1.axis('off')
        
        #annotate anchors---------> modificar lo del tamaño
        keys = list(self.anchors.keys())
        square1 = patches.Rectangle(self.anchors[keys[0]], width=0.1, height=0.1, fill=True, color='black') 
        square2 = patches.Rectangle(self.anchors[keys[1]], width=0.1, height=0.1, fill=True, color='black')
        square3 = patches.Rectangle(self.anchors[keys[2]], width=0.1, height=0.1, fill=True, color='black')
    
        ax1.add_patch(square1)
        ax1.add_patch(square2)
        ax1.add_patch(square3)

        with open( "PruebaFeatures\imagenet_class_index.json", 'r') as json_file:
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


def euclidean_distance(vector1, vector2):
    flattened_vector1 = vector1.view(-1) # flatten
    flattened_vector2 = vector2.view(-1)
    return torch.sqrt(torch.sum((flattened_vector1 - flattened_vector2)**2))

        
class PlanesetInfoExtractor:    
     def __init__(self, planeset, config):
         
        self.planeset= planeset
        self.config = config
        self.classMasks = self.get_class_masks()
        self.margin = self.extractMargin()
        #self.classDTs = self.calculate_distance_transforms()
        #self.classDTs = self.calculate_distance_transforms()
        #self.marginGrid = self.get_marginGrid()
        #self.marginFeat = self.get_marginFeat()
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

        #I think this is much easier taking directly the points
        print('?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????')

        coords = self.planeset.planeset.coords
        x = self.planeset.planeset.coefs1.cpu().numpy()
        y = self.planeset.planeset.coefs2.cpu().numpy()

        # Loop over anchors
        for i, (target_class, anchor) in enumerate(self.planeset.anchors.items()):
            coord = coords[i]  # Coordinate of current anchor

            # Get x, y positions and predicted classes
            x_positions = torch.from_numpy(x.flatten())
            y_positions = torch.from_numpy(y.flatten())
            predicted_classes = torch.from_numpy(self.planeset.predictedClasses.flatten())

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


def min_max_normalize(distances):
    min_value = min(distances)
    max_value = max(distances)
    normalized_distances = [(distance - min_value) / (max_value - min_value) for distance in distances]
    return normalized_distances

def calculate_pmf(distances):
    normalized_distances = min_max_normalize(distances)
    unique_values, count = np.unique(normalized_distances, return_counts=True)
    #probabilities = count / len(count)
    return unique_values, count

def showClassPMF(marginClass, config):
    with open("imagenet_class_index.json", 'r') as json_file:
        data = json.load(json_file)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, margin in enumerate(marginClass):
        unique_values, probabilities = calculate_pmf(margin)
        axs[i].bar(unique_values, probabilities,width=.02, alpha=0.8)
        axs[i].set_title(f'{config.labels[i]}: {data[str(config.labels[i])][1]}')
        axs[i].set_xlabel('Normalized Margin')
        axs[i].set_ylabel('Count')

    plt.suptitle('Probability Mass Functions (PMFs) for Margin Classes')
    plt.show()
                
                























    