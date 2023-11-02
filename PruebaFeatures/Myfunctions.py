import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import itertools


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


def get_triplets(c1, c2, c3, db, N):
    ''' return a list with N list of image combinations, 
        where each combination is a list of three images
    '''
    
    filenames_combinations = []

    # Get the file paths of the images in each folder
    class1_folder = db + c1 + "/"
    class2_folder = db + c2 + "/"
    class3_folder = db + c3 + "/"

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


def make_preds_from_planeset(model, planeset):
    "returns 2D labelled image"
    model.eval()
    r =int(np.sqrt(len(planeset)))
    preds = []
    for idx in range(len(planeset)):
        mixfeat = planeset[idx]
        with torch.no_grad():
            pred = model(mixfeat)
            pred_class = pred.argmax().item()
            preds.append(pred_class)
            
    return np.array(preds).reshape(r,r)

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
  












            