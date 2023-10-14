from myFunctions import *

## class and BD selection 
path_to_db= "OODatasets/Handcrafted/"

c1= "star"
c2="chip_poker_chip"
c3= "triangle_trigon_trilateral"

images = getCombiFromDB(c1, c2, c3,path_to_db)
print('buenas')
