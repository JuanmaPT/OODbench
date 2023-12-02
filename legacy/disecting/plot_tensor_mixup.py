import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import sys

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time
from myFunctions import *
from utils import produce_plot_sepleg_IMAGENET #TODO : fix representation of images
import numpy as np

#Important Note! -> Change the model loading in the intialization
load_net = '/net/travail/jpenatrapero/dbViz/pretrained_models/resnet18-5c106cde.pth'
net_name = 'resnet' 
set_seed = 777
set_data_seed = 1
save_net = 'saves'
imgs = 600,4000,1600
epochs = 2
lr = 0.01
resolution = 50 #Default is 500 and it takes 3 mins
batch_size_planeloader = 1
saveplot = False
num_classes = 3
num_images_experiment = 3
idx_pred_im = [11,81,68]#Fixed values with the indexes corresponding 
#to our original images in the format of the vector pred
# the class_pred[idx_pred_im[1]] is the predicted class for the first imae of the triplet

c1= "n02106662" #German shepard
c2= "n03388043" #Fountain
c3= "n03594945" #Jeep
#Labesl of the imagenet
labels = ['German_shepherd','fountain','jeep']
ground_truth_im = [235,562,609]

#Saving the results
results_folder = "results"
model_name = 'resnet50' #Name of the model for saved data

# Log of the results
active_log = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print("CUDA IS AVALIABLE $.$ ")
else:
    print("No cuda avaliable :´( ")