import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt
import os
import argparse
import time
import sys
import itertools

c1= "n02106662" #German shepard
c2= "n03388043" #Fountain
c3= "n03594945" #Jeep

def getCombiFromDB(c1, c2, c3,db):

    filenames_combinations = []

    # Get the file paths of the images in each folder
    class1_folder= db+ c1 +"/"
    class2_folder= db+ c2 + "/"
    class3_folder= db+ c3 + "/"
   
    class1 = [os.path.join(class1_folder, filename) for filename in os.listdir(class1_folder)]
    class2 = [os.path.join(class2_folder, filename) for filename in os.listdir(class2_folder)]
    class3 = [os.path.join(class3_folder, filename) for filename in os.listdir(class3_folder)]
   
    #get all possible combinations
    combinations = list(itertools.product(class1, class2, class3))
   
    imgCombinationsTensor =[]    
    for combi in combinations:
        filenames_combinations.append(combi)
   
    return imgCombinationsTensor, filenames_combinations

def getCombiFromDBoptimal(c1, c2, c3, db):
    filenames_combinations = []

    # Get the file paths of the images in each folder
    class1_folder = db + c1 + "/"
    class2_folder = db + c2 + "/"
    class3_folder = db + c3 + "/"

    class1 = [os.path.join(class1_folder, filename) for filename in os.listdir(class1_folder)]
    class2 = [os.path.join(class2_folder, filename) for filename in os.listdir(class2_folder)]
    class3 = [os.path.join(class3_folder, filename) for filename in os.listdir(class3_folder)]

    # Generate combinations while ensuring unique rotations
    imgCombinationsTensor = []
    filenames_set = set()  # Use a set to store unique combinations

    for combi in itertools.product(class1, class2, class3):
        combi_sorted = sorted(combi)  # Sort the paths within each combination
        combi_tuple = tuple(combi_sorted)

        # Check if the combination is unique
        if combi_tuple not in filenames_set:
            filenames_set.add(combi_tuple)
            filenames_combinations.append(combi_tuple)

    return imgCombinationsTensor, filenames_combinations


path_to_db= "OODatasets/imagenet_val_resized/"
imgCombinationsTensor, filenames_combinations = getCombiFromDB(c1, c2, c3,path_to_db)
imgCombinationsTensor_op, filenames_combinations_op = getCombiFromDBoptimal(c1, c2, c3,path_to_db)


print(len(filenames_combinations))
print(len(filenames_combinations_op))


  















