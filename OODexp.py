import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import time

'''
import sys
import random
import pickle
import torchvision
import argparse
'''

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD, plot_tensor
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv, decision_boundary_original
from options import options
from utils import simple_lapsed_time
from myFunctions import *
from utils import produce_plot_sepleg_IMAGENET #TODO : fix representation of images
import numpy as np

def plot_space(path, preds, planeloader, images, labels, trainloader=None, title='best', temp=0.01,true_labels = None):
    import seaborn as sns
    sns.set_style("whitegrid")
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 15,}                  
    sns.set_context("paper", rc = paper_rc,font_scale=1.5)  
    plt.rc("font", family="Times New Roman")
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap('gist_rainbow')
    cmaplist = [col_map(i) for i in range(col_map.N)]
    classes = ['AIRPL', 'AUTO', 'BIRD', 'CAT', 'DEER',
                   'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']
    cmaplist = [cmaplist[45],cmaplist[30],cmaplist[170],cmaplist[150],cmaplist[65],cmaplist[245],cmaplist[0],cmaplist[220],cmaplist[180],cmaplist[90]]
    cmaplist[2] = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)
    cmaplist[4] = (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0)

    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig, ax1  = plt.subplots()

    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(10)], cmaplist))

    # Mapping the class predictions to a range of 0 to 9
    class_pred = (class_pred % 10).astype(int)


    print(class_pred)

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=0.5, s=0.1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]

    coords = planeloader.dataset.coords


    markerd = {
        0: 'o',
        1 : '^',
        2 : 'X'
    }
    for i, image in enumerate(images):
        coord = coords[i]
        plt.scatter(coord[0], coord[1], s=150, c='black', marker=markerd[i])


    # JMPT part
    plt.scatter(x, y, s=150, c='pink', marker= 'o')
    # Load and plot images at each point

    image_folder='results/inputs'
    for i in range(len(x)):
        image_path = os.path.join(image_folder, f'tensor_{i}.png')
        img = Image.open(image_path)
        
        # Calculate the position to place the image
        img_x = x[i]
        img_y = y[i]
        
        # Plot the image at the calculated position

        gap = 50

        ax1.imshow(img, extent=(img_x - gap, img_x + gap, img_y - gap, img_y + gap), alpha=0.7)



    # plt.title(f'{title}',fontsize=20)
    ax1.spines['right'].set_visible(True)
    ax1.spines['top'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(True)    

    plt.margins(0,0)
    if path is not None:
        plt.savefig(f'{path}/test_x.png', bbox_inches='tight')


    plt.close(fig)
    return

args = options().parse_args()

#Important Note! -> Change the model loading in the intialization
args.load_net = 'pretrained_models/resnet18-5c106cde.pth'
args.net = 'resnet' 
args.set_seed = '777'
args.save_net = 'saves'
args.imgs = 600,4000,1600
args.epochs = 2
args.lr = 0.01
args.resolution = 5 #Default is 500 and it takes 3 mins
args.batch_size_planeloader = 1
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
#Indices of img1 in the grid: [4, 4]
#Indices of img2 in the grid: [44, 4]
#Indices of img3 in the grid: [20, 44]
ground_truth_im = [235,562,609]

#Saving the results
results_folder = "results"
model_name = 'resnet50' #Name of the model for saved data

# Log of the results
args.active_log = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print("CUDA IS AVALIABLE $.$ ")
else:
    print("No cuda avaliable :´( ")

save_path = args.save_net

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
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


#########################################################
#   LOADING THE NETWORK
#########################################################
if args.load_net is None:
    print("args.load_net is None -> You need to provide the path to the weights!")
else:
    net.load_state_dict(torch.load(args.load_net))
    
# data_loader -> testloader
# test_acc, predicted = test(args, net, testloader, device)
# print(test_acc)
end = time.time()
simple_lapsed_time("Time taken to load the model", end-start)



##############  DATASET   #################
args.imgs = 'imagenet'



start = time.time()
if args.imgs is None:
    print("args.imgs is None -> You need to provide the images to load")

elif args.imgs == 'handcrafted':
    path_to_db= "OODatasets/handcrafted/"
    imgCombinationsTensor, filenames_combinations = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'signal':
    path_to_db= "OODatasets/signal/"
    imgCombinationsTensor, filenames_combinations = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'generated':
    path_to_db= "OODatasets/generated/"
    imgCombinationsTensor, filenames_combinations = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'imagenet':
    path_to_db= "OODatasets/imagenet_val_resized/"
    imgCombinationsTensor, filenames_combinations = getCombiFromDB(c1, c2, c3,path_to_db)

elif args.imgs == 'test10images':
    path_to_db= "/net/cremi/jpenatrapero/DATASETS/10images/"
    imgCombinationsTensor, filenames_combinations = getCombiFromDB(c1, c2, c3,path_to_db)
else:
    print('UNRECOGNICED image dataset')
  

sampleids = '_'.join(list(map(str,labels)))

n_combis = num_images_experiment**num_classes

accuracy_triplet = []
margin_triplet = []
results_all_pred = {}  # Initialize an empty dictionary to store pred matrices

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to find the closest index in pred_matrix for each anchor position
def closest_index(pred_matrix, x_array, y_array, anchor):
    distances = np.zeros_like(pred_matrix, dtype=np.float32)
    index_flat = 0
    index_matrix = np.zeros(pred_matrix.shape)
    for i in range(pred_matrix.shape[0]):
        for j in range(pred_matrix.shape[1]):
            distances[i, j] = euclidean_distance([x_array[index_flat], y_array[index_flat]], anchor)
            index_matrix[i, j] = index_flat
            index_flat = index_flat + 1
    min_distance = np.min(distances)
    min_indices = np.argwhere(distances == min_distance)
    return min_distance, min_indices,index_matrix


print('==> Starting loop through all triplet combinations..')
for i_triplet in range(n_combis):

    progress = (i_triplet + 1) / n_combis * 100
    print(f"Progress: {progress:.2f}% complete", end="\r", flush=True)

    images = imgCombinationsTensor[i_triplet]

    #Creating planeloader for the image space
    plot_tensor(images[0],'Image tensor first image','test.png')
    planeloader = make_planeloader(images, args)

    x_coefs = planeloader.dataset.coefs1.cpu().numpy()
    y_coefs = planeloader.dataset.coefs2.cpu().numpy()
    coords = planeloader.dataset.coords


    print(coords)
    #Using the model to predict all the plane
    preds = decision_boundary_original(args, net, planeloader, device)

    plot_space('results', preds, planeloader, images, labels)



    #Getting the labels of the predictions
    preds = torch.stack((preds))
    temp  = 0.01 #Not sure what this does
    preds = nn.Softmax(dim=1)(preds / temp)
    class_vect = torch.argmax(preds, dim=1).cpu().numpy()

    #Converting vector to matrix
    pred_matrix  = np.reshape(class_vect, (args.resolution, args.resolution))

    results_all_pred[f"Matrix_{i_triplet}"] = pred_matrix
    results_all_pred[f"Combi_{i_triplet}"] = filenames_combinations[i_triplet]

    #accuracy_triplet, margin_triplet = margin_TRDP_I (class_pred,pred_matrix,idx_pred_im,ground_truth_im,accuracy_triplet,margin_triplet)

    # Find the closest indices for each anchor
    closest_indices = [closest_index(pred_matrix, x_coefs, y_coefs, anchor) for anchor in coords]

    # Print the results
    print("Closest indices:")
    for i, (distance, indices,index_matrix) in enumerate(closest_indices):
        print(f"Anchor {i+1} - Distance: {distance}, Indices: {indices}")

    break


############# END OF FOR LOOP TRHOUGH ALL THE TRIPLETS

end = time.time()
simple_lapsed_time("Time taken for all combinatios of triplets", end-start)
# Calculate average margins for accurate predictions


save_results('/net/travail/jpenatrapero/results',results_all_pred,"results_3_3_imagenet_original.pkl")

















