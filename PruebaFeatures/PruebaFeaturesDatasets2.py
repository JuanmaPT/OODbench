from Myfunctions2 import *

dir_datasets = "C:/Users/Blanca/Documents/IPCV/TRDP/TRDP2/smallDatasets/"
datasets = get_folders(dir_datasets)

c1, c2, c3  = "n01498041","n01534433","n01687978",    #  note:escribir en orden alfabetico 
config = Configuration( model= 'ResNet18', 
                       N = 5, 
                       id_classes= [c1,c2,c3],
                       resolution= 50,
                       margin_th = 0.9,
                       )


# prueba con val test first luego meterlo en el loop conlos otros datsets
dataset = datasets[2]
filenamesCombis =  getCombiFromDBoptimal(config, dataset)
print("Genarating planesets...")   
planesets = [Planeset(Triplet(pathImgs, config), config) for pathImgs in filenamesCombis] 

#%% visualization example 
# for the first image visualize all the triplet combination
path_img_to_show = filenamesCombis[0][0]
for planeset in planesets:
    if planeset.triplet.pathImgs[0] == path_img_to_show:
        planeset.show()
#%%
print("Extracting DB descriptors...")
descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]   
print("Done.")

marginClass = [[] for _ in range(len(config.id_classes))]
for i in range(len(planesets)):
    for j in range(len(config.id_classes)):
        if planesets[i].triplet.isImgPredCorrect[j] == True:
            marginClass[j].append(descriptors[i].marginFeat[j])
            



















