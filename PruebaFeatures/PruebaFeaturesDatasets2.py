# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:50:44 2023

@author: Blanca
"""

from Myfunctions2 import *
#%%
dir_datasets = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/smallDatasets/"
datasets = get_folders(dir_datasets)

c1, c2, c3  = "n01498041", "n01687978", "n01534433"

config = Configuration( model= 'ResNet18', 
                       N = 5, 
                       id_classes= [c1,c2,c3],
                       resolution= 10
                       )

# prueba con val test firts luego meterlo en el loop
dataset = datasets[2]
filenamesCombis =  getCombiFromDBoptimal(config, dataset)
print("Genarating planesets...")   
planesets = [Planeset(Triplet(pathImgs, config), config) for pathImgs in filenamesCombis] 
#%%
print("Extracting DB descriptors...")
descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]   
#%% visualization example 
# for the first image visualize all the triplet combination
path_img_to_show = filenamesCombis[0][0]
custom_colors= get_custom_colors(config)
for planeset in planesets:
    if planeset.triplet.pathImgs[0] == path_img_to_show:
        planeset.show(custom_colors)
    





































