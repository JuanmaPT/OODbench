from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm

c1, c2, c3  = "n01498041","n01534433","n01687978",    #  note:escribir en orden alfabetico 
config = Configuration(model= 'ResNet18', 
                       N = 1, 
                       id_classes= [c1,c2,c3],
                       resolution= 10,
                       dataset = "ImageNetVal_small" 
                       )

filenamesCombis =  getCombiFromDBoptimal(config)
planesets = []
for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
    triplet_object = Triplet(pathImgs, config)
    planeset_object = Planeset(triplet_object, config)
    planesets.append(planeset_object)
    
#%% visualization example 
#for the first image visualize all the triplet combination
path_img_to_show = filenamesCombis[0][0]
for planeset in planesets:
    #if planeset.triplet.pathImgs[0] == path_img_to_show:
    planeset.show()

#%%
print("Extracting DB descriptors...")
descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]   
print("Done.")

#%%
marginClass = [[] for _ in range(len(config.id_classes))]
for i in range(len(planesets)):
    for j in range(len(config.id_classes)):
        if planesets[i].triplet.isImgPredCorrect[j] == True:
            marginClass[j].append(descriptors[i].margin[j])


#%% PMF
for i in range(len(config.labels)):
    plot_pmf(marginClass[i], 0.01, config, i)
           















