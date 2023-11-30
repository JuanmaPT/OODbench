# -*- coding: utf-8 -*-
from PIL import Image
from torchvision import transforms
import torch 
from utils import *
import numpy as np

class Triplet:
    def __init__(self, pathImgs, config):
        self.pathImgs = pathImgs
        self.config= config
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
            image = preprocess(image)
            #print(image.size())
            
            # Expand dimensions to simulate batch size of 1
            input_batch= image.unsqueeze(0)
            #print(image.size())
            # Extract features with the base model
            with torch.no_grad():
                #print("here")
                features = self.config.base_model(input_batch)
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
            image = preprocess(image)
            input_batch = image.unsqueeze(0)  # Add a batch dimension
            
            # Make predictions using the model
            with torch.no_grad():
                output = self.config.model(input_batch)
                #print(output.size())
                _,pred_class= output.max(1)
                pred_imgs.append(np.uint16(pred_class.item()))
                score_imgs.append(output.softmax(dim=-1).max().item())
            #print(pred_class.item())
            #print(pred_imgs)
        return pred_imgs, score_imgs
        
    def checkPred(self):
        return [pred == true_label for pred, true_label in zip(self.prediction, self.config.labels)]