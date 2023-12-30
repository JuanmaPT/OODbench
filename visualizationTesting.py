from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np
import random

# Class Selection 
class2id = {
    "stingray": ["n01498041", "6"] ,
    "junco": ["n01534433", "13"],
    "robin": ["n01558993", "15"],
    "jay": ["n01580077", "17"],
    "bad_eagle": ["n01614925", "22"] ,
    "bullfrog": ["n01641577", "30"],
    "agama": ["n01687978", "42"]
}

my_class_selection= ["stingray", "junco", "bullfrog", "agama", "bad_eagle"]
class_selection = sorted([ class2id[class_][0] for class_ in my_class_selection])
print(class_selection)

# Generate the planesets

configResNet = Configuration(model= "ResNet18", 
                       N = 1,
                       dataset = "ImageNetVal_small",
                       id_classes= class_selection,
                       resolution= 15,
                       useFilteredPaths = "False",
                       )

configViT =  Configuration(model= "ViT", 
                       N = 1,
                       dataset = "ImageNetVal_small",
                       id_classes= class_selection,
                       resolution= 15,
                       useFilteredPaths = "False",
                       )

filenamesCombis =  getCombiFromDBoptimalGoogleDrive(configResNet)

planesetsResnet = []
planesetsViT = []

for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
    planesetsResnet.append(Planeset(Triplet(pathImgs, configResNet), configResNet))
    planesetsViT.append((Planeset(Triplet(pathImgs, configViT), configViT)))


# 2D Visualization
# Generate a unique colormap 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class_to_color = get_diff_color(planesetsResnet)

for i,planeset in enumerate(planesetsResnet):
    planeset.show(class_to_color,f"planesets/resNet_val_15_{i}.png")

#%%for i,planeset in enumerate(planesetsViT):
    planeset.show(class_to_color,f"planesets/vit__val_15_{i}.png")
    
#%% 3D visualization
# manually select id
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
idx= 2
show_scores_3D(planesetsResnet[idx], class_to_color)



















