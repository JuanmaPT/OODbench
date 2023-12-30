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
from matplotlib.colors import ListedColormap

import gdown
import zipfile

import plotly.graph_objects as go

class Configuration:
    def __init__(self, model, N, useFilteredPaths,id_classes, resolution, dataset):
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
            
        if useFilteredPaths == "True":
            self.useFilteredPaths = True
        else:
            self.useFilteredPaths = False
            

    def get_labels(self):
        with open("imagenet_class_index.json", 'r') as json_file:
            data = json.load(json_file)

        labels = []
        for search_id in self.id_classes:
            for key, value in data.items():
                if search_id in value:
                    labels.append(int(key))
                    #print(key, value)

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
            #nn.Flatten(),
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


def filterPaths(config, image_paths,i):
    #filter paths in folder belonging to class i 
    class_label = torch.ones(1, len(image_paths))*config.labels[i]
    
    # Create a batch tensor from the list of preprocessed images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = [transform(Image.open(image_path).convert('RGB')) for image_path in image_paths]
    batch_tensor = torch.stack(images)
    #print(batch_tensor.size())
    
    # Make predictions
    with torch.no_grad():
        outputs = config.model(batch_tensor)
    
    predicted_class = torch.argmax(outputs,dim =1)
    
    # get only the correct predictions
    filteredPaths = [path for path, label, pred in zip(image_paths,  class_label.squeeze().tolist(), predicted_class.tolist()) if label == pred]
    
    #if the number correct predictions is less that N, add incorrect 
    remaining_count = config.N - len(filteredPaths)
    if remaining_count > 0:
        incorrect_paths = [path for path, label, pred in zip(image_paths,  class_label.squeeze().tolist(),predicted_class.tolist()) if label !=pred]
        filteredPaths.extend(incorrect_paths[:remaining_count])
    
    return filteredPaths


def getCombiFromDBoptimal(config):
    import os
    if os.getlogin() == 'Blanca':
        rootDir = "smallDatasets/"
        db_path = os.path.join(rootDir, config.dataset)
    
    if os.getlogin() == 'juanm':
        rootDir = "C:/Users/juanm/Documents/IPCV_3/TRDP/smallDatasets/"
        db_path = rootDir + config.dataset
    
    filenames_combinations = []
    
    # Get the file paths of the images in each folder
    class_folders_paths = [os.path.join(db_path, class_id) for class_id in config.id_classes]

    path_class_lists = [] 
    for i, folder_path in enumerate(class_folders_paths): 
        print(folder_path)
        folder_images_path = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
        
        if config.useFilteredPaths:
            folder_images_path = filterPaths(config, folder_images_path, i )
 
        path_class_lists.append(folder_images_path[:config.N])
    
    
    # combinations at class level (k_clases over 3)
    combinations_3 = list(itertools.combinations(path_class_lists, 3))
    #combinations at image level N^3
    for combo in combinations_3:
        for values in itertools.product(*combo):
            filenames_combinations.append(values)
    
    return filenames_combinations


def unzip_file(zip_path, destination_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(destination, 'downloaded_file.zip')
    gdown.download(url, output, quiet=False)

def getCombiFromDBoptimalGoogleDrive(config):
    # Set your Google Drive file ID and destination folder
    dataset_folder = os.path.join(os.getcwd(),'smallDatasets')
  
    if not os.path.exists(dataset_folder):

        file_id = '1Z38dLQJFeqV-JPR0F2RsHDzfAuEP54SD'
            # Create a folder called 'dataset' in the current working directory
        print(file_id)
        create_folder(dataset_folder)
        # Use the 'dataset' folder as the destination folder
        destination_folder = os.getcwd()
        # Download the file
        download_file_from_google_drive(file_id, destination_folder)
        # Unzip the downloaded file
        zip_path = os.path.join(destination_folder, 'downloaded_file.zip')
        unzip_file(zip_path, destination_folder)
        # Optionally, remove the zip file after extracting its contents
        os.remove(zip_path)
  
    db_path = os.path.join(dataset_folder, config.dataset)
    filenames_combinations = []

    # Get the file paths of the images in each folder
    class_folders_paths = [os.path.join(db_path, class_id) for class_id in config.id_classes]
 
    # Generate the a list with the path to the images for each class 
    path_class_lists = [] 
    nMax=50
    for i, folder_path in enumerate(class_folders_paths): 
        folder_images_path = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)][:nMax]
        
        if config.useFilteredPaths:
           
            folder_images_path = filterPaths(config, folder_images_path, i )
 
        path_class_lists.append(folder_images_path[:config.N])
    
    #### Path combinations #####
    # combinations at class level (k_clases over 3)
    combinations_3 = list(itertools.combinations(path_class_lists, 3))
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

"""def create_result_folder(result_folder_name):
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
        return result_folder_name"""
    
def create_result_folder(models):
    root_folder = "results"
    #models = ["ResNet18", "ViT"]
    subfolders = ["plots", "margin_values", "classPredictions"]
    
    # Create the root folder if it doesn't exist
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    # Create subfolders for each model
    for model in models:
        model_folder = os.path.join(root_folder, model)
        # Create the model folder if it doesn't exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
    
        # Create subfolders inside the model folder
        for subfolder in subfolders:
            subfolder_path = os.path.join(model_folder, subfolder)
            
            # Create the subfolder if it doesn't exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
    
    #print("Folder structure created successfully.")

    
def save_to_csv(model_dict, output_folder):
    for dataset, classes in model_dict.items():
        file_path = os.path.join(output_folder, f"{dataset}.csv")
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write data
            for class_name, values in classes.items():
                row_data = [class_name] + values
                writer.writerow(row_data)

def get_diff_color(planesets):
    unique_classes_all = np.unique(np.concatenate([planeset.predictedClasses for planeset in planesets]))
    cmap = plt.cm.get_cmap('tab20', len(unique_classes_all))

    # Generate equally spaced values between 0 and 1
    color_positions = np.linspace(0, 1, len(unique_classes_all))

    # Extract colors from the colormap at the specified positions
    distinct_colors = [cmap(pos) for pos in color_positions]
    

    # Create a ListedColormap from the distinct colors
    custom_cmap = ListedColormap(distinct_colors)
    # Map each unique class to a color
    class_to_color = {cls: custom_cmap(i) for i, cls in enumerate(unique_classes_all)}

    return class_to_color


def show_scores_3D(planeset, class_to_color):
    keys = list(planeset.anchors.keys())
    unique_classes = np.unique(planeset.prediction)

    # Create an empty figure
    fig = go.Figure()

    with open("imagenet_class_index.json", 'r') as json_file:
        class_index = json.load(json_file)

    for c, class_label in enumerate(unique_classes):
        class_indices = np.where(planeset.prediction == class_label)
        # Create a boolean matrix where True corresponds to the class_label
        class_mask = (planeset.prediction == class_label)
        # Use the boolean matrix to extract scores for the current class
        class_score_matrix = planeset.score.copy()
        class_score_matrix[~class_mask] = 0
        color = class_to_color.get(class_label, [0, 0, 0])[:3]
        color = [round(value * 255) for value in color]

        # Create a surface plot for the current class
        X, Y = np.meshgrid(np.arange(planeset.score.shape[0]), np.arange(planeset.score.shape[1]-1, -1, -1))
        surface = go.Surface(z=class_score_matrix, x=X, y=Y,
                            colorscale=[[0, f'rgb(0,0,0)'],
                                        [1, f'rgb({color[0]}, {color[1]}, {color[2]})']],
                            showscale=False,
                            name=f"Class {class_label}"  # Add the name attribute
                            )

        # Add the surface to the figure
        fig.add_trace(surface)

    # Update trace and layout settings
    fig.update_traces(
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )
    fig.update_layout(
        title='3D DB',
        width=1000,
        height=1000,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            xaxis_title='xaxis decision space',
            yaxis_title='yaxis decision space',
            zaxis_title='Score'
        )
    )

    for c, class_label in enumerate(unique_classes):
        fig.add_trace(go.Scatter3d(
            x=[(0.3)],
            y=[(0.3)],
            z=[(0.3)],
            mode='markers',
            marker=dict(size=12, color=f'rgb({color[0]}, {color[1]}, {color[2]})'),
            name=f"{class_label}: {class_index[str(class_label)][1]}"
        ))

    for i, key in enumerate(keys):
        x_anchor, y_anchor = planeset.anchors[key]
        text_anchor = ['bottom', 'top', 'middle'][i]

        # Create a scatter plot for the anchor point with a circle marker
        fig.add_trace(go.Scatter3d(
            x=[x_anchor],
            y=[planeset.score.shape[1] - 1 - y_anchor],
            z=[planeset.score[y_anchor, x_anchor] + 0.05],
            mode='markers',
            marker=dict(
                size=10,
                color=['black', 'black', 'black'][i],
                symbol=['diamond', 'diamond', 'diamond'][i],
            ),
            text=f'Anchor_{i}',
            textposition=f'{text_anchor} center',
        ))

        # Adding a line trace to create a dotted line in the z-direction
        fig.add_trace(go.Scatter3d(
            x=[x_anchor, x_anchor],
            y=[planeset.score.shape[1] - 1 - y_anchor, planeset.score.shape[1] - 1 - y_anchor],
            z=[planeset.score[y_anchor, x_anchor] + 0.05, 0],
            mode='lines',
            line=dict(color=['black', 'black', 'black'][i], dash='solid', width=2),
        ))

    # Show the plot
    fig.show()
    print('Figure shown')
















    