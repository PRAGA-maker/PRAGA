import requests
from bs4 import BeautifulSoup
import reactome2py 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import warnings
from bs4 import BeautifulSoup
print(sys.getrecursionlimit())

#convert from pdb to reactome, find prot in reactme

#look thru list of pathways to find the  chain 
def create_pathway_dictionary(parents, children):
    pathway_dict = {}
    
    for i, child in enumerate(children):
        parent = parents[i]
        
        if parent not in pathway_dict:
            pathway_dict[parent] = set()
        
        pathway_dict[parent].add(child)
    
    return pathway_dict

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
parents = parent_list
children = child_list
pathway_dictionary = create_pathway_dictionary(parents, children)

original_dict = pathway_dictionary
unique_values = []
result_dict = {}

for value in original_dict.values():
    if value not in unique_values:
        unique_values.append(value)

for key, value in original_dict.items():
    if value in unique_values:
        result_dict[key] = value

pathway_dictionary = result_dict

n = 0
for pathway, parents in pathway_dictionary.items():
    #print(f"{pathway}: {parents}")
    n+=1
print("N: " + str(n))

# Get the full set of pathway labels from parents and children
all_pathways = set()
for parent, children in pathway_dictionary.items():
    all_pathways.add(parent)
    all_pathways.update(children)
mlb = MultiLabelBinarizer(classes=sorted(all_pathways))
data = [(parent, child) for parent, children in pathway_dictionary.items() for child in children]
parents, children = zip(*data)
parent_encoded = mlb.fit_transform([[parent] for parent in parents])
child_encoded = mlb.transform([[child] for child in children])
df = pd.DataFrame(parent_encoded, columns=mlb.classes_)
df['Child'] = child_encoded.tolist()

#print(df)

#now we need to loook thru pathways to get identifiers for pathways

#then webscrape to get pathways that PDB ID is in
url = r"https://reactome.org/content/detail/R-HSA-167428"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    print(soup)

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")