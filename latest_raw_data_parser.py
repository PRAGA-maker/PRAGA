import requests

# i need some bitches :(

path = r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data.csv"
with open(path, 'r') as file:
    
    lines = file.readlines()

    for line in lines:
        prot = line[0:4]
        pdb_id.append(prot)
coords = []
atoms = []
total = []