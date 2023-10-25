import requests
from bs4 import BeautifulSoup
import reactome2py 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#convert from pdb to reactome, find prot in reactme

#look thru list of pathways to find the  chain 
path = r"C:\Users\prone\OneDrive\Desktop\drug_proj\classification\reactome_ReactomePathwaysRelation.txt"

parent_list = []
child_list = []
with open(path, 'r') as file:
    
    lines = file.readlines()

    for line in lines:
        line = line.split()
        parent = line[0]
        child = line[1]
        parent_list.append(parent)
        child_list.append(child)

parent_list = np.array(parent_list)
child_list = np.array(child_list)

list_of_source_paths = ['R-HSA-9612973','R-HSA-1640170','R-HSA-1500931', 'R-HSA-8953897', 'R-HSA-4839726', 'R-HSA-400253','R-HSA-1266738','R-HSA-8963743','R-HSA-1643685','R-HSA-73894','R-HSA-69306','R-HSA-9748784','R-HSA-1474244','R-HSA-74160','R-HSA-109582','R-HSA-168256','R-HSA-1430728','R-HSA-392499','R-HSA-8953854','R-HSA-397014','R-HSA-112316','R-HSA-1852241','R-HSA-5357801','R-HSA-9609507','R-HSA-1474165','R-HSA-9709957','R-HSA-162582','R-HSA-382551','R-HSA-5653656']
        
for path in list_of_source_paths:

    current_source_node = path
    immediate_children = []
    for parent_path in parent_list:
        if path == parent_path:
            immediate_children.append(parent_path)