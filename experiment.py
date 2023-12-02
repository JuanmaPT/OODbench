from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np

datasets = ["ImageNetA_small", "ImageNetSD_small", "ImageNetVal_small",  "SignalBlur_small"]
models = ["ResNet18", "ViT"]
models = ["ResNet18"]
c1, c2, c3  = "n01498041","n01534433","n01687978"

result_folder_name = 'CambioMargin'
result_folder_name = create_result_folder(result_folder_name)

save_info = True

if save_info:

    margin_ResNet18 = []
    margin_ViT = []

marging_save_list = []

for dataset in datasets:
    for model in models:
        config = Configuration(model= model, 
                               N = 2, 
                               id_classes= [c1,c2,c3],
                               resolution= 15,
                               dataset = dataset
                               )
    
        filenamesCombis =  getCombiFromDBoptimal(config)
        margins_save = [] #list for storing the marging values
        margins_save.append(model)
        margins_save.append(dataset)
        planesets = []
        for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
            triplet_object = Triplet(pathImgs, config)
            planeset_object = Planeset(triplet_object, config)
            planesets.append(planeset_object)
           
        descriptors = [PlanesetInfoExtractor(planeset,config) for planeset in planesets]
        
        marginClass = [[] for _ in range(len(config.id_classes))]
        for i in range(len(planesets)):
            for j in range(len(config.id_classes)):
                if planesets[i].triplet.isImgPredCorrect[j] == True:
                    marginClass[j].append(descriptors[i].margin[j])
                    margins_save.append(round(descriptors[i].margin[j], 4))
       
        if save_info:
            if model == "ResNet18":
                margin_ResNet18.append(marginClass)

            if model == "ViT":
                margin_ViT.append(marginClass)
            
            
        combined_array = np.concatenate(marginClass)
        try: 
            min_val = int(np.ceil(np.min(combined_array)))
            max_val = int(np.ceil(np.max(combined_array)))
            
            for i in range(len(config.labels)):
                plot_pmf(marginClass[i], 15, config, i, min_val, max_val,result_folder_name)
        except:
            print("pass")
        marging_save_list.append(margins_save)
    

save_to_csv(marging_save_list, f"results\{result_folder_name}\margin_ResNet18.csv")
print('All done')
#%%













