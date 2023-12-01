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

from scipy.stats import norm
from scipy.optimize import curve_fit

class Configuration:
    def __init__(self, model, N, id_classes, resolution, dataset):
        self.modelType = model
        self.N = N
        self.id_classes = id_classes
        self.resolution = resolution
        self.dataset = dataset
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
      
        with open( "imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)
        
        self.labels = []
        for search_id in id_classes:
            for key, value in data.items():
                if search_id in value:
                    self.labels.append(int(key))
                    print(key,value)
     
       
        if  self.modelType == 'ResNet18':
           
            # model to make predictions over images
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            resnet_weights = self.model.state_dict()
            
            # base model as feature extractor: remove classification head
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
            self.model = models.vit_b_16(pretrained= True)
            vit_weights = self.model.state_dict()
      
            
            # base model as feature extractor: remove classification head
            self.base_model = models.vit_b_16(weights='IMAGENET1K_V1')
            self.base_model.heads.head = nn.Identity()
            self.base_model.eval()

            # Classification head for making predictions over features
            self.head_model = nn.Sequential(
                nn.Linear(768, 1000)
            ) 
           
            # Set weights for the head model
            self.head_model[-1].weight.data = vit_weights['heads.head.weight'].view(self.head_model[-1].weight.size())
            self.head_model[-1].bias.data = vit_weights['heads.head.bias'].view(self.head_model[-1].bias.size())
            self.head_model.eval()
      
        
    
        

def get_folders(path):
  """Returns a list of folders in the given path."""
  folders = []
  for item in os.listdir(path):
    if os.path.isdir(os.path.join(path, item)):
      folders.append(path + item)
  return folders


def getCombiFromDBoptimal(config):
    import os
    if os.getlogin() == 'Blanca':
        rootDir = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/smallDatasets/"
    if os.getlogin() == 'juanm':
        rootDir = "C:/Users/juanm/Documents/IPCV_3/TRDP/smallDatasets/"
    db_path = rootDir + config.dataset
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


#def euclidean_distance(vector1, vector2):
    #flattened_vector1 = vector1.view(-1) # flatten
    #flattened_vector2 = vector2.view(-1)
    #return torch.sqrt(torch.sum((flattened_vector1 - flattened_vector2)**2))


def min_max_normalize(distances):
    min_value = min(distances)
    max_value = max(distances)
    normalized_distances = [(distance - min_value) / (max_value - min_value) for distance in distances]
    return normalized_distances

def plot_pmf(marginList, num_bins, config, class_, min_val, max_val):
    with open("imagenet_class_index.json", 'r') as json_file:
        dataDict = json.load(json_file)
    title = dataDict[str(config.labels[class_])][1]

    counts, bins = np.histogram(marginList, bins=num_bins, density=True)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2
    probs = counts / float(counts.sum())

    plt.bar(bins, probs, width=(bins[1] - bins[0]))
    plt.xticks(np.arange(np.ceil(min_val), np.ceil(max_val) + 1))

    # Plot the PMF
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.title(f"PMF - {config.dataset}\n{title} | {config.modelType} | N= {config.N} ")
    
    # Save the figure before showing
    plt.savefig(f"results\pmf_results_exp2_30\PMF_{config.dataset}_{title}_{config.modelType}_N_{config.N}.png")
    plt.close()























    