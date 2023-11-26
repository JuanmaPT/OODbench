import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np 
import json 
import timm
import os

from PIL import Image
import itertools
import matplotlib.pyplot as plt



class Configuration:
    def __init__(self, model, N, id_classes, resolution):
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
            
            
            
        if self.modelType == 'ViT':
            # model to make predictions over images
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            vit_weights = self.model.state_dict()
      
            
            #for key in vit_weights:
                #print(key)
            
            # base model as feature extractor
            base_model_layers = [
                self.model.patch_embed,
                self.model.pos_drop,
                self.model.norm_pre,
                self.model.blocks,
                self.model.norm,
                self.model.fc_norm
            ]
            
            self.base_model = nn.Sequential(*base_model_layers)
            self.base_model.eval()

            # Classification head for making predictions over features
            self.head_model = nn.Sequential(
                nn.Linear(768, 1000)
            )

            # Set weights for the head model
            self.head_model[-1].weight.data = vit_weights['head.weight'].view(self.head_model[-1].weight.size())
            self.head_model[-1].bias.data = vit_weights['head.bias'].view(self.head_model[-1].bias.size())
            self.head_model.eval()
      
       
                  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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


def euclidean_distance(vector1, vector2):
    flattened_vector1 = vector1.view(-1) # flatten
    flattened_vector2 = vector2.view(-1)
    return torch.sqrt(torch.sum((flattened_vector1 - flattened_vector2)**2))


def min_max_normalize(distances):
    min_value = min(distances)
    max_value = max(distances)
    normalized_distances = [(distance - min_value) / (max_value - min_value) for distance in distances]
    return normalized_distances

def plot_pmf(data,bin_width,config, class_):
    # Calculate the PMF
    data = min_max_normalize(data)
    values, counts = np.unique(data, return_counts=True)
    pmf = counts / len(data)
    
    with open( "imagenet_class_index.json", 'r') as json_file:
        data = json.load(json_file)
    title = data[str(config.labels[class_])][1]
    
    # Plot the PMF
    plt.bar(values, pmf, width= bin_width, alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Probability')
    #plt.ylim([0, 1])
    plt.title(f"Probability Mass Function (PMF)\n {title}")
    plt.show()























    