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
import csv

class Configuration:
    def __init__(self, model, N, id_classes, resolution, dataset):
        self.modelType = model
        self.N = N
        self.id_classes = id_classes
        self.resolution = resolution
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = self.get_labels()

        if self.modelType == 'ResNet18':
            self.model, self.base_model, self.head_model = self.load_resnet18()
        elif self.modelType == 'ViT':
            self.model, self.base_model, self.head_model = self.load_vit()

    def get_labels(self):
        with open("imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)

        labels = []
        for search_id in self.id_classes:
            for key, value in data.items():
                if search_id in value:
                    labels.append(int(key))
                    print(key, value)

        return labels

    def load_resnet18(self):
        # Model to make predictions over images
        model = models.resnet18(weights='IMAGENET1K_V1')
        resnet_weights = model.state_dict()

        # Base model as feature extractor: remove classification head
        base_model = nn.Sequential(*list(model.children())[:-1])
        base_model.eval()

        # Classification head to make predictions over features
        head_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1000)
        )

        # Weights for head model
        head_model[-1].weight.data = resnet_weights['fc.weight'].view(head_model[-1].weight.size())
        head_model[-1].bias.data = resnet_weights['fc.bias'].view(head_model[-1].bias.size())
        head_model.eval()

        return model, base_model, head_model

    def load_vit(self):
        # Model to make predictions over images
        model = models.vit_b_16(pretrained=True)
        vit_weights = model.state_dict()

        # Base model as feature extractor: remove classification head
        base_model = models.vit_b_16(weights='IMAGENET1K_V1')
        base_model.heads.head = nn.Identity()
        base_model.eval()

        # Classification head for making predictions over features
        head_model = nn.Sequential(
            nn.Linear(768, 1000)
        )

        # Set weights for the head model
        head_model[-1].weight.data = vit_weights['heads.head.weight'].view(head_model[-1].weight.size())
        head_model[-1].bias.data = vit_weights['heads.head.bias'].view(head_model[-1].bias.size())
        head_model.eval()

        return model, base_model, head_model
        
    
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
    
    # JuanManueeeel creo que es mejor que lo pongamos los dos aquí:
    #rootDir = "smallDatasets/"
    #db_path = os.path.join(rootDir, config.dataset)
    #filenames_combinations = []

    # Get the file paths of the images in each folder
    class_folders = [os.path.join(db_path, class_id) for class_id in config.id_classes]

    # Generate the a list with the path to the images for each class 
    class_lists = []
    for class_folder in class_folders: 
        class_list = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)[:config.N]]
        class_lists.append(class_list)
     
    # combinations at class level (k_clases over 3)
    combinations_3 = list(itertools.combinations(class_lists, 3))
    #combinations at image level N^3
    for combo in combinations_3:
        for values in itertools.product(*combo):
            filenames_combinations.append(values)
    
    return filenames_combinations


#def euclidean_distance(vector1, vector2):
    #flattened_vector1 = vector1.view(-1) # flatten
    #flattened_vector2 = vector2.view(-1)
    #return torch.sqrt(torch.sum((flattened_vector1 - flattened_vector2)**2))


def min_max_normalize(distances):
    min_value = min(distances)
    max_value = max(distances)
    normalized_distances = [(distance - min_value) / (max_value - min_value) for distance in distances]
    return normalized_distances


def plot_pmf(marginList, class_, num_bins,config, min_val, max_val,result_folder_name):
    
    with open( "imagenet_class_index.json", 'r') as json_file:
        dataDict = json.load(json_file)
    title = dataDict[str(class_)][1]
    
    counts, bins = np.histogram(marginList, bins=num_bins, density=True)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2
    probs = counts / float(counts.sum())

    plt.bar(bins, probs, width=(bins[1] - bins[0]))
    #plt.plot(bins, probs, linestyle='-')
    plt.xticks(np.arange(np.ceil(min_val), np.ceil(max_val) + 1))

    # Plot the PMF
    plt.xlabel('Values')
    plt.ylabel('Probability')
    #plt.xlim=(np.ceil(min_val), np.ceil(max_val))
    # Uncomment the following line if you want to set y-axis limit between 0 and 1
    # plt.ylim([0, 1])
    plt.title(f"PMF - {config.dataset}\n{title} | {config.modelType} | N= {config.N} ") 
    plt.savefig(f"results/{config.model}/plots/PMF_{config.dataset}_{title}_{config.modelType}_N_{config.N}.png")
    #plt.close()
    plt.show()


def create_result_folder(result_folder_name):
    # Check if the folder exists
    folder_exists = os.path.exists(f"results/{result_folder_name}")
    # If the folder already exists, find a new folder name
    if folder_exists:
        counter = 1
        while folder_exists:
            new_folder_name = f"{result_folder_name}_{counter}"
            folder_exists = os.path.exists(f"results/{new_folder_name}")
            counter += 1
        os.makedirs(f"results/{new_folder_name}")
        return new_folder_name
    else:
        os.makedirs(f"results/{result_folder_name}")
        return result_folder_name


def save_to_csv(model_dict, output_folder):
    for dataset, classes in model_dict.items():
        file_path = os.path.join(output_folder, f"{dataset}.csv")
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write data
            for class_name, values in classes.items():
                row_data = [class_name] + values
                writer.writerow(row_data)

















    