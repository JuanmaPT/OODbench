from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
from tqdm import tqdm
import numpy as np
import argparse

from memory_profiler import profile

# -Guardar las unique clases en cada triplet.
# -Añadir opción de filtrar por predicción 

# Availabe classes for the study
class2id = {
    "stingray":  ["n01498041", "6"] ,
    "junco":     ["n01534433", "13"],
    "robin":     ["n01558993", "15"],
    "jay":       ["n01580077", "17"],
    "bad_eagle": ["n01614925", "22"] ,
    "bullfrog":  ["n01641577", "30"],
    "agama":     ["n01687978", "42"]
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration selection for OOD study")
    parser.add_argument("--dataset", 
                        type = str, 
                        nargs="+",
                        choices=["ImageNetA_small", "ImageNetSD_small", "ImageNetVal_small",  "SignalBlur_small"], 
                        default=["ImageNetVal_small"], 
                        help="Dataset selection")
    
    parser.add_argument("--model", 
                        type= str,
                        nargs="+",
                        choices=["ResNet18", "ViT"],
                        default=["ResNet18"], 
                        help="Model selection")
    
    parser.add_argument("--classes",
                        type =str, 
                        nargs="+", 
                        choices=class2id.keys(), 
                        default=["stingray", "junco", "bullfrog", "agama", "bad_eagle"], 
                        help="Class selection")
    
    parser.add_argument("--N", 
                        type= int,
                        default= 1,
                        help= "Number of images per class selection" )
  
    parser.add_argument("--resolution",
                        type = int,
                        default = 15,
                        help = "Resolution to generate a plane dataset given a triplet of images")
    
    parser.add_argument("--loadDataset",
                        type= str,
                        default = "False",
                        help= "Boolean parameter to download the dataset from Google Drive")
    
    args = parser.parse_args()
    
    
    if len(args.classes)<3:
        raise ValueError("Number of classes should be at least 3") 
    return args

@profile
def main():
    # Parse command line arguments
    args = parse_arguments()
    class_selection = []
    for class_ in args.classes:
        class_selection.append(class2id[class_][0])
        print(class_, class2id[class_])
        
    # Create folder to store results
    create_result_folder(args.model) 
    
    # run experiment for the given configuration
    for dataset in args.dataset:
        for model in args.model:
            print(dataset)
            print(model)
            config = Configuration(model= model, 
                                   N = args.N, 
                                   id_classes= class_selection,
                                   resolution= args.resolution,
                                   dataset = dataset
                                   )
            
    
            # generate all the possible combinations between clasess and triplets of images
            if args.loadDataset == "True":
                filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
                
            else:
                filenamesCombis =  getCombiFromDBoptimal(config)
            
            # Create a csv file to write results 
            with open(f"results/{model}/margin_values/{model}_{dataset}_values.csv", "w") as csvfile:
                for i, pathImgs in tqdm(enumerate(filenamesCombis), total=len(filenamesCombis), desc="Processing"):
                    #1. generate triplet object
                    triplet_obj= Triplet(pathImgs, config)
                    #2. create plane set 
                    planeset_obj = Planeset(triplet_obj, config)
                    #3. extract the margin for those images in the triplet correctly predicted
                    for j in range(3):
                        if triplet_obj.true_label[j] == triplet_obj.prediction[j]:
                            margin = round(PlanesetInfoExtractor(planeset_obj,config).margin[j],4)
                            #write results
                            csvfile.write(f"{triplet_obj.true_label[j]},{margin}\n")
                    
                    # delete variables to free up memory
                    del triplet_obj
                    del planeset_obj
        
            
    print('\n All done. Results saved at results/')

if __name__ == "__main__":
    main()          
















