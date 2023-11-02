import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from Myfunctions import *
from SomepalliFunctions import *

from InfoDB import TripletInfoExtractor

#import matplotlib.pyplot as plt
#%% load the base model as feature extractor  and the clasification head
device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet_model = models.resnet18(pretrained=True)                                # to make predictions over images

base_model = nn.Sequential(*list(resnet_model.children())[:-1])                # to extract features 
base_model.eval()


head_model = nn.Sequential(                                                    # to make oredictions over features 
    nn.Flatten(),
    nn.Linear(512, 1000)
)

# Load the weights for the last layer from the weights dictionary
weights = torch.load('resnet18-5c106cde.pth')
head_model[-1].load_state_dict({
    'weight': weights['fc.weight'],
    'bias': weights['fc.bias']
})
head_model.eval()

#%% TIPLET PREDICTIONS & FEATURE EXTRACTION

# select classes
c1 = "n01498041"
c2 = "n01534433"
c3 = "n01558993"

true_labels =[6, 13, 15 ]

# select dataset
path_to_db = "smallDatasets/ImageNetVal_small/"

# generate triplets
N=3
imgCombis, filenamesCombi = get_triplets(c1, c2, c3,path_to_db, N)

# extract features
feature_triplets = []
print(f"Extracting features for tripet images..")
for i,triplet in enumerate(imgCombis):
    print(f"{i+1}/{len(imgCombis)}")
    feature_triplets.append(extract_features_from_triplet(base_model, triplet))
print('Done')

#%% PLANESET PREDICTIONS
planesetPreds = []
anchors = []
resolution = 50
for i,featTri in enumerate(feature_triplets):
    # make the prediction over triplet
    pred_triplet = make_predictions_tripletImg(resnet_model, triplet)
    
    print(f"Generating planeset for triplet {i+1}/{len(feature_triplets)} ")
    # calculation of two orthogonal basis vectors for the plane spanned by the feature vectors 
    a, b_orthog, b, coords = get_plane(featTri[0], featTri[1], featTri[2])
    #print(f"Basis Vectors:{a}{b_orthog}{b}")
    #print(f"Coords: {coords}")
    
    # generate a dataset of images on a 2D plane by combining a base feature vector with two vectors
    planeset = plane_dataset(featTri[0], a, b_orthog, coords, resolution= resolution )
     
    # get anchors (*)
    anchors.append(get_anchor_dict(featTri, planeset, pred_triplet))
    
    # get the predictions over planeset
    print(f"Predicting planeset {i+1}/{len(feature_triplets)}")
    planeset_pred = make_preds_from_planeset(head_model, planeset)
    planesetPreds.append(planeset_pred)
    print('Done')

   
#%% EXTRACTING DR INFO  FROM PLANESET PREDICTIONS = LABELLED IMAGE

tripletsInfo = [TripletInfoExtractor(planesetPreds[i], anchors[i]) for i in range(len(planesetPreds))]

# info to extract 
margin_c1= []
margin_c2= []
margin_c3= []

for i,triplet in enumerate(tripletsInfo):
    for idx in range(3):
        if list(anchors[i].keys())[idx] == true_labels[idx]:
            if idx == 0:
                margin_list = margin_c1
            if idx == 1:
                margin_list = margin_c2
            if idx == 2:
                margin_list = margin_c3
                
        margin_list.append(triplet.margin[idx])
        #max_dt_list.append(triplet.)
   
            



































