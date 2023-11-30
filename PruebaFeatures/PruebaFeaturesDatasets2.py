from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
import os
if os.getlogin() == 'Blanca':
    dir_datasets = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/smallDatasets/"
if os.getlogin() == 'juanm':
    dir_datasets = "C:/Users/juanm/Documents/IPCV_3/TRDP/smallDatasets/"
datasets = get_folders(dir_datasets)


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

c1, c2, c3  = "n01498041","n01534433","n01687978",    #  note:escribir en orden alfabetico 
<<<<<<< Updated upstream
config = Configuration(model= 'ResNet18', 
                       N = 3, 
=======
config = Configuration( model= 'ResNet18', 
                       N = 1, 
>>>>>>> Stashed changes
                       id_classes= [c1,c2,c3],
                       resolution= 10,
                       )

#%%
# prueba con val test first luego meterlo en el loop conlos otros datsets
id_dataset = 2
dataset = datasets[id_dataset]
filenamesCombis =  getCombiFromDBoptimal(config, dataset)
print("Genarating planesets...")   
planesets = [Planeset(Triplet(pathImgs, config), config) for pathImgs in filenamesCombis] 

#%% visualization example 
#for the first image visualize all the triplet combination

first = True

path_img_to_show = filenamesCombis[0][0]
for planeset in planesets:
    if planeset.triplet.pathImgs[0] == path_img_to_show:
        unique_classes = np.unique(planeset.prediction) 
        if first:
            cmap, color_dict = init_colormap(unique_classes)
            first = False
        else:
            cmap, color_dict = add_new_classes_to_colormap(cmap, color_dict, unique_classes)
        planeset.show_simple(cmap,color_dict)

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
           















