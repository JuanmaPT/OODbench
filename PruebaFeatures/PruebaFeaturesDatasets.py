# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 08:54:19 2023

@author: Blanca
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from Myfunctions import *
from SomepalliFunctions import *

from InfoDB import TripletInfoExtractor
#%%
dir_datasets = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/smallDatasets/"
datasets = get_folders(dir_datasets)
path_to_weights = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/resnet18-5c106cde.pth"

############################# CLASS SELECTION #################################
c1 = "n01498041"  # stingray - mantaraya
c2 = "n01687978"  # agama - lagarto
c3 = "n01534433"  # junco - pajaro

true_labels = [6, 13, 42]

############################# MODEL SELECTION #################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet_model = models.resnet18(pretrained=True)                                # to make predictions over images

# BASE MODEL AS FEATURE EXTRACTOR
base_model = nn.Sequential(*list(resnet_model.children())[:-1])                # to extract features 
base_model.eval()


# CLASSIFICATION HEAD TO MAKE PREDICTIONS
head_model = nn.Sequential(                                                    # to make oredictions over features 
    nn.Flatten(),
    nn.Linear(512, 1000)
)

# Load the weights for the last layer from the weights dictionary
weights = torch.load(path_to_weights)
head_model[-1].load_state_dict({
    'weight': weights['fc.weight'],
    'bias': weights['fc.bias']
})
head_model.eval()


###################################################################################################################################
# 1.generate all the posible triplet combinations of the images belonging to each class --> [img_class_1, img_class_2, img_class_3]
# 2.for each tripet we are goint to:
#       1. extract features using base model --> [feat_img_class_1, feat_img_class2_feat_img_class3]
#       2. predict the extracted fetures using head model --> [class_1, class_2, class_3]
#       3. generate a plane dataset (planeset) containing the combinations of the 3 fearures vectors [N,N] where N=resolution
#       4. make the predictions of the planeset and store the prediction scores as well in planeset_pred and planeset_score
#       5. decision boundary/region info extraction using 
# 3. Class analysis based on the info extracted; using TripletInfoExtractor(planeset, anchor)
####################################################################################################################################

N= 10 # number of samples
resolution = 10
all_planesetPreds= []
all_planesetScores= []
all_tripletPreds= []
all_dicts= []
filesTriplets= []


for dataset in datasets:
    print(f"{dataset}")
    planesetPreds = []
    planesetScores = []
    tripletPreds = []
    dataset_dicts = []
    
    # get triplet combination
    imgCombis, filenamesCombi = get_triplets(c1, c2, c3, dataset, N)
    filesTriplets.append(filenamesCombi)

    for i,triplet in enumerate(imgCombis):
        # extract features from images in the triplet
        featTri = extract_features_from_triplet(base_model, triplet)
        
        # make predictions of features 
        triplet_pred = make_predictions_tripletImg(resnet_model, triplet)
        # get true_triplet
        
        # generate planeset (dataset of images on a 2D plane by combining a base feature vector with two vectors)
        print(f"Generating planeset for triplet {i+1}/{len(imgCombis)} ")
        a, b_orthog, b, coords = get_plane(featTri[0], featTri[1], featTri[2])
        planeset = plane_dataset(featTri[0], a, b_orthog, coords, resolution= resolution )
        
        # get the anchor position 
        anchor = get_anchor_dict(featTri, planeset, triplet_pred)
        
        print(f"Predicting planeset {i+1}/{len(imgCombis)}")
        planeset_pred, planeset_score = make_preds_from_planeset(head_model, planeset)
     
        # create DB info extractor object
        tripletInfo = TripletInfoExtractor(planeset_pred, anchor)
        
        triplet_dict = {
            'margin': tripletInfo.margin,
            'max_DT': tripletInfo.max_distance_transform,
            'regProp': tripletInfo.regionProps,
            # Add other keys and values as needed
        }

    
        dataset_dicts.append(triplet_dict)
        planesetPreds.append(planeset_pred)
        planesetScores.append(planeset_score)
        tripletPreds.append(triplet_pred)
        
        
    # Append the dataset_dicts for this dataset
    all_dicts.append(dataset_dicts)
    
    # Append planeset predictions for this dataset
    all_planesetPreds.append(planesetPreds)    
    all_planesetScores.append(planesetScores)
    all_tripletPreds.append(tripletPreds)
    
    # store results per class 
        

#%% visualization example of a triplet
triplet_id= 255 
for dataset_id in range(4):
    showTripletResult(all_planesetScores[dataset_id][triplet_id], 
                       all_planesetPreds[dataset_id][triplet_id], 
                       filesTriplets[dataset_id][triplet_id],
                       all_tripletPreds[dataset_id][triplet_id],
                       true_labels,
                       f"Tiplet {triplet_id}")   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    