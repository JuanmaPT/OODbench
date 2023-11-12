import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import itertools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

def get_folders(path):
  """Returns a list of folders in the given path."""
  folders = []
  for item in os.listdir(path):
    if os.path.isdir(os.path.join(path, item)):
      folders.append(path + item)
  return folders


def get_triplets(c1, c2, c3, db, N):
    ''' return a list with N list of image combinations, 
        where each combination is a list of three images
    '''
    
    filenames_combinations = []

    # Get the file paths of the images in each folder
    class1_folder = db + "/"+ c1 + "/"
    class2_folder = db + "/"+ c2 + "/"
    class3_folder = db + "/"+ c3 + "/"

    class1 = [os.path.join(class1_folder, filename) for filename in os.listdir(class1_folder)[:N]]
    class2 = [os.path.join(class2_folder, filename) for filename in os.listdir(class2_folder)[:N]]
    class3 = [os.path.join(class3_folder, filename) for filename in os.listdir(class3_folder)[:N]]

    # Generate combinations while ensuring unique rotations
    imgCombinations = []
    filenames_set = set()  # Use a set to store unique combinations

    for combi in itertools.product(class1, class2, class3):
        combi_sorted = sorted(combi)  # Sort the paths within each combination
        combi_tuple = tuple(combi_sorted)

        # Check if the combination is unique
        if combi_tuple not in filenames_set:
            filenames_set.add(combi_tuple)
            filenames_combinations.append(combi_tuple)

            img1 = Image.open(combi_tuple[0])
            img2 = Image.open(combi_tuple[1])
            img3 = Image.open(combi_tuple[2])

            imgCombinations.append([img1, img2, img3])
            
    print(f"N= {N}")
    print(f"Total number of unique combinations: {len(filenames_combinations)}")
    return imgCombinations,  filenames_combinations


def make_predictions_tripletImg(resnet_model, triplet):
    preds_img = []
    for image in triplet:
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
            output = resnet_model(input_batch)
            _,pred_class= output.max(1)
            preds_img.append(pred_class.item())
            
    return preds_img

        

def extract_features_from_triplet(base_model, triplet):
    ''' takes a Ttripet of images and return a tripet of feature vectors'
    '''
    feature_triplet = []
    for image in triplet:
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
            features = base_model(image)
        
        #print(features.shape)
        feature_triplet.append(features)

    return feature_triplet




def make_preds_from_planeset(model, planeset):
    "returns 2D labelled image"
    model.eval()
    r =int(np.sqrt(len(planeset)))
    preds = []
    scores = []
    for idx in range(len(planeset)):
        mixfeat = planeset[idx]
        with torch.no_grad():
            pred = model(mixfeat)
            pred_class = pred.argmax().item()
            preds.append(pred_class)
            scores.append(pred.softmax(dim=-1).max().item())
    
    planeset_pred = np.array(preds).reshape(r,r)
    planeset_score = np.array(scores).reshape(r,r)
            
    return planeset_pred, planeset_score

"""
def mse(image1, image2):
    return ((image1 - image2) ** 2).mean()

def find_index_by_image(dataset, generated_image, tolerance=1e-2):
    for idx in range(len(dataset)):
        image = dataset[idx]
        if mse(image, generated_image) < tolerance:
            return idx

def get_anchor_dict(triplet, planeset,preds):
    anchor_dict = {}
    for i in range(3):
        # get the anchor for each triplet
        idx = find_index_by_image(planeset, triplet[i])
        # map 1 dimensional index to two dimensional x,y coord on a 2d grid 
        x = idx % planeset.resolution
        y = idx // planeset.resolution
        anchor_dict[preds[i]] = (x,y)
    return anchor_dict 

"""

def extract_idx_triplets_from_tensor(images,planeset):
        distances = []  # List to store the distances
        triplet_index = []
        for image_idx in range(3):
            for batch_idx, inputs in enumerate(planeset):
                distance = torch.dist(inputs, images[image_idx])
                distances.append(distance.item()) 
            min_distance_index = distances.index(min(distances))           
            distances = [] 
            triplet_index.append(min_distance_index)
        return triplet_index


def get_anchor_dict(triplet, planeset,preds):
    anchor_dict = {}
    triplet_idx = extract_idx_triplets_from_tensor(triplet, planeset)
    
    for i,idx in enumerate(triplet_idx):
        # convert 1d idx to x,y coords in a 2d grid
        x = idx % planeset.resolution
        y = idx // planeset.resolution
        # create the dictionary 
        anchor_dict[preds[i]] = (x,y)
    return anchor_dict 
  


def get_custom_colors(N):
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
    
    return custom_colors[:N]
    
def showTripletResult(planeset_scores, planeset_pred, tripletImg, triplet_pred, true_labels, title= None):  
    
    unique_classes = np.unique(planeset_pred)   
    num_classes = len(unique_classes)

    # Create a color map with black background
    color_map = np.zeros((planeset_scores.shape[0], planeset_scores.shape[1], 3))

    # Generate colors for each class
    #cmap = plt.get_cmap('rainbow')
    #colors = [to_rgba(cmap(i))[:3] for i in np.linspace(0, 1, num_classes)]
    colors = get_custom_colors(num_classes)

    # Assign colors based on prediction scores
    for class_label, color in zip(unique_classes, colors):
        class_indices = np.where(planeset_pred == class_label)
        class_scores = planeset_scores[class_indices]
        
        # Use square root scaling for color intensity adjustment
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

    # Create legend for each class
    #legend_elements = [Line2D([0], [0], marker='o', color='w',
                              #label=f'{class_label}',
                              #markerfacecolor=color, markersize=10) for class_label, color in zip(unique_classes, colors)]

    #ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    with open( "imagenet_class_index.json", 'r') as json_file:
        data = json.load(json_file)
    
    
    # Create a horizontal color bar for each class (going from less bright to more bright)
    bar_height = 0.05  # Adjust the height based on your preference
    space_between_bars = 0.02  # Adjust the space between bars
    total_height = num_classes * (bar_height + space_between_bars) - space_between_bars
    start_y = (1 - total_height) / 2

    for i, (class_label, color) in enumerate(zip(unique_classes, colors)):
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
    
    #poner labels en genral
    print(triplet_pred)
    
    # visualize triplet of images:
    for i, path in enumerate(tripletImg):
        img = mpimg.imread(path)
        ax3 = plt.subplot(1, 5, i + 3)
        ax3.imshow(img)
        ax3.axis('off')
        ax3.set_title(f"True class: {true_labels[i]}, {data[str(true_labels[i])][1]} \nPrediction:  {triplet_pred[i]}, {data[str(triplet_pred[i])][1]} ")

    plt.show()













            