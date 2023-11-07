# -*- coding: utf-8 -*-

import os 
import numpy as np

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image

def extract_features_from_images_in_dataset(base_model, dataset_path, folder_names):
    '''feature_dict.keys()                                                            -> folders
       feature_dict['n01498041'].keys()                                               -> images names: key (image_name): value (feature vector)
       feature_dict_ImageNet['n01694178']['ILSVRC2012_val_00048675_n01694178.JPEG']   -> the feature vector
    '''
    feature_dict = {}
    
    for folder_name in folder_names:
        print("Folder:", folder_name)
        folder_path = os.path.join(dataset_path, folder_name)
        image_names = os.listdir(folder_path)
        
        folder_features = {}
        
        for image_name in image_names:
            print("Image:", image_name)
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
           
            
            # Preprocess the image and extract features using the base model
            image = image.resize((224, 224)) 
            x = preprocess_input(np.expand_dims(np.array(image), axis=0))
            print(x.shape)
            
            # Extract features with base_model
            features = base_model.predict(x)
            print(features.shape)
            
            # Store the feature vector in the dictionary
            folder_features[image_name] = features

        # Store the folder's feature dictionary in the main dictionary
        feature_dict[folder_name] = folder_features
        
    
    return feature_dict



# feature extractor to use
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False
x= GlobalAveragePooling2D()(base_model.output)
base_model = Model(inputs=base_model.input, outputs=x) 


#%% usage::
#dataset_path = "./smallDatasets/ImageNetVal_small/"
#select the folder names of the classes we are goin to use or folder_names = os.listdir(dataset_path)
#feature_dict_ImageNet = extract_features_from_images_in_dataset(base_model, dataset_path, folder_names)






























