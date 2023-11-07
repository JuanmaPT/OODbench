import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

#Added imports TRDP
import sys

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time

args = options().parse_args()
print(args)

#Fixing the arguments

#Important Note! -> Change the model loading in the intialization
args.load_net = '/media/jpenatrapero/TAU/TRDP/OODrepo/dbViz/pretrained_models/resnet18-5c106cde.pth'
args.net = 'resnet'
args.set_seed = '777'
args.save_net = 'saves'
args.imgs = 600,4000,1600
args.epochs = 2
args.lr = 0.01
args.resolution = 10 #Default is 500 and it takes 3 mins
args.batch_size_planeloader = 1

# Log of the results
args.active_log = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = args.save_net
if args.active_log:
    import wandb
    idt = '_'.join(list(map(str,args.imgs)))
    wandb.init(project="decision_boundaries", name = '_'.join([args.net,args.train_mode,idt,'seed'+str(args.set_seed)]) )
    wandb.config.update(args)

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
net = get_model(args, device)


criterion = get_loss_function(args)
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(args, optimizer)

elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)



# Train or load base network
print("Training the network or loading the network")

start = time.time()
best_acc = 0  # best test accuracy
best_epoch = 0


if args.load_net is None:
    print("args.load_net is None -> You need to provide the path to the weights!")
#########################################################
#   LOADING THE NETWORK
#########################################################
else:
    net.load_state_dict(torch.load(args.load_net))
    

# test_acc, predicted = test(args, net, testloader, device)
# print(test_acc)
end = time.time()
simple_lapsed_time("Time taken to load the model", end-start)
saveplot = False


##############  DATASET   #################
dataset_study_name = 'test10images'
args.imgs = dataset_study_name


###### CLASSES ##########
num_classes = 3
num_images_experiment = 10
c1= "n02106662" #German shepard
c2= "n03388043" #Fountain
c3= "n03594945" #Jeep
#Labesl of the imagenet
labels = ['German_shepherd','fountain','jeep']
ground_truth_im = [235,562,609]
############################################################


from myFunctions import *
from utils import produce_plot_sepleg_IMAGENET #TODO : fix representation of images
import numpy as np


start = time.time()
if args.imgs is None:
    print("args.imgs is None -> You need to provide the images to load")

elif args.imgs == 'handcrafted':
    path_to_db= "OODatasets/handcrafted/"
    images_triplets = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'signal':
    path_to_db= "OODatasets/signal/"
    images_triplets = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'generated':
    path_to_db= "OODatasets/generated/"
    images_triplets = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'imagenet':
    path_to_db= "OODatasets/imagenet_val_resized/"
    images_triplets = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'test10images':
    path_to_db= "/net/cremi/jpenatrapero/DATASETS/10images/"
    images_triplets = getCombiFromDB(c1, c2, c3,path_to_db)
    m = np.array(images_triplets)
    print(m.shape)
else:
    print('UNRECOGNICED image dataset')
  

sampleids = '_'.join(list(map(str,labels)))



n_combis = num_images_experiment**num_classes

accuracy_triplet = []
margin_triplet = []

print('==> Starting loop through all triple combinations..')
for i_triplet in range(n_combis):

    progress = (i_triplet + 1) / n_combis * 100
    print(f"Progress: {progress:.2f}% complete", end="\r", flush=True)

    images = images_triplets[i_triplet]

    #Creating planeloader for the image space
    planeloader = make_planeloader(images, args)
    #Using the model to predict all the plane
    preds = decision_boundary(args, net, planeloader, device)

    net_name = args.net
    if saveplot:
        os.makedirs(f'images/{net_name}/{args.train_mode}/{sampleids}/{str(args.set_seed)}', exist_ok=True)
        #lot_path = os.path.join(args.plot_path,f'{net_name}_{sampleids}_{args.set_seed}cifar10')
        plot_path = os.path.join('plots',f'{net_name}_{sampleids}_{args.set_seed}testing')
        os.makedirs(f'{plot_path}', exist_ok=True)
        produce_plot_sepleg_IMAGENET(plot_path, preds, planeloader, images, labels, trainloader, title = 'best', temp=1.0,true_labels = None)
        #produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader)

        # produce_plot_x(plot_path, preds, planeloader, images, labels, trainloader, title=title, temp=1.0,true_labels = None)




    #Getting the labels of the predictions
    preds = torch.stack((preds))
    temp=0.01#Not sure what this does
    preds = nn.Softmax(dim=1)(preds / temp)
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()

    #Converting vector to matrix
    pred_matrix  = np.reshape(class_pred, (args.resolution, args.resolution))

    idx_pred_im = [808,9108,4791]#Fixed values with the indexes corresponding 
    #to our original images in the format of the vector pred
    # the class_pred[idx_pred_im[1]] is the predicted class for the first imae of the triplet

    #adjust to 10 by 10
    idx_pred_im = [11,81,68]

    accuracy_row = []
    margin_row = []
    for i in range(3):
        accuracy_row.append(class_pred[idx_pred_im[i]] == ground_truth_im[i])
        margin_row.append(margin_of_image(idx2label_mat(idx_pred_im[i]), pred_matrix, ground_truth_im[i]))
    print(accuracy_row)
    print(margin_row)
    accuracy_triplet.append(accuracy_row)
    margin_triplet.append(margin_row)


############# END OF FOR LOOP TRHOUGH ALL THE TRIPLETS

end = time.time()
simple_lapsed_time("Time taken for all combinatios of triplets", end-start)
# Calculate average margins for accurate predictions

import numpy as np
import matplotlib.pyplot as plt


results_folder = "results"
model_name = 'resnet50'
dataset_name = dataset_study_name
dataset_folder = os.path.join(results_folder, model_name,dataset_name)
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
# Save accuracy_triplet and margin_triplet to numpy files
accuracy_file_path = os.path.join(dataset_folder, "accuracy_triplet.npy")
margin_file_path = os.path.join(dataset_folder, "margin_triplet.npy")
np.save(accuracy_file_path, accuracy_triplet)
np.save(margin_file_path, margin_triplet)


















