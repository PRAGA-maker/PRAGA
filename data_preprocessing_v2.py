import csv
import os
import matplotlib.pyplot as plt
import pubchempy
import requests
import tqdm
from bs4 import BeautifulSoup
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from Bio import PDB
from Bio.SeqUtils import seq1

list_of_prots = []
list_of_ligands = []
list_of_affinity = [] #to-do
list_of_units = [] 
moad = 0
bind = 0
binding = 0

def read_output_file(file_path, moad, bind, binding): 
    with open(file_path, "r",  encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(",")
            list_of_prots.append(line[0])
            ligand_interm = line[1].split("'")
            list_of_ligands.append(ligand_interm[1])
            
            for part in line:
                if "assay" in part:
                    if " 'Kd" == part.split(":")[0]:
                        list_of_units.append("Kd")
                        list_of_affinity.append(((part.split("\\xa0")[0]).split(":&nbsp"))[1])
                    elif " 'Ki" == part.split(":")[0]:
                        list_of_units.append("Ki")
                        list_of_affinity.append(((part.split("\\xa0")[0]).split(":&nbsp"))[1])
                    elif " 'IC50" == part.split(":")[0]:
                        list_of_units.append("IC50")
                        list_of_affinity.append(((part.split("\\xa0")[0]).split(":&nbsp"))[1])
                    elif " 'EC50" == part.split(":")[0]:
                        list_of_units.append("EC50")
                        list_of_affinity.append(((part.split("\\xa0")[0]).split(":&nbsp"))[1])
                    elif " 'Ka" == part.split(":")[0]:
                        list_of_units.append("Ka")
                        list_of_affinity.append(((part.split("\\xa0")[0]).split(":&nbsp"))[1])
                    else:
                        if " max" == part.split(":")[0]:
                            prev_part = (prev_part.split(":")[-1:])[-1:]
                            prev_part = prev_part[0]
                            min_val = float(prev_part)
                            part = part.split("\\x")
                            part = part[0]
                            try:
                                max_val = float(part[4:])
                            except ValueError:                          
                                max_val = float(part[6:])
                            list_of_affinity.append((min_val+max_val)/2)
                        else:
                            print(part.split(":")[0])
                elif "Binding MOAD" in part:
                    moad +=1 
                elif "PDBBind" in part:
                    bind +=1
                elif "BindingDB" in part:
                    binding +=1
                prev_part = part
        file.close()
    return moad, bind, binding

file_path=r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data.csv"
moad, bind, binding = read_output_file(file_path, moad, bind, binding)
file_path=r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy.csv"
moad, bind, binding = read_output_file(file_path, moad, bind, binding)
file_path=r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy 2.csv"
moad, bind, binding = read_output_file(file_path, moad, bind, binding)
file_path=r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy 3.csv"
moad, bind, binding = read_output_file(file_path, moad, bind, binding)
file_path=r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy 4.csv"
moad, bind, binding = read_output_file(file_path, moad, bind, binding)

print("Num of Prots: " + str(len(list_of_prots)))
print("Num of Unique Prots: " + str(len(list(set(list_of_prots)))))
print("Num of Ligands: " + str(len(list_of_ligands)))
print("Num of Unique Ligands: " + str(len(list(set(list_of_ligands)))))

print("Num MOAD: " + str(moad))
print("Num PDBBind: " + str(bind))
print("Num BindingDB: " + str(binding))

num_kd = 0
num_ki = 0
num_ic50 = 0
num_ec50 = 0
num_ka = 0
for unit in list_of_units:
    if "Kd" in unit:
        num_kd+=1
    elif "Ki" in unit:
        num_ki+=1 
    elif "IC50" in unit:
        num_ic50+=1
    elif "EC50" in unit:
        num_ec50+=1
    elif "Ka" in unit:
        num_ka+=1
print("Num Kd: " + str(num_kd))
print("Num Ki: " + str(num_ki))
print("Num IC50: " + str(num_ic50))
print("Num EC50: " + str(num_ec50))
print("Ka: " + str(num_ka))

# 5G01,"[('SO4', 'BindingDB:\xa0 5G01', 'Ki:&nbspmin: 8.90e+7, max: 3.00e+8\xa0(nM) from 4 assay(s)'), ('OE2', 'BindingDB:\xa0 5G01', 'Ki:&nbsp88\xa0(nM) from 1 assay(s)')]"

#make list of prot seq. 
list_of_sequence = []    
for pdb_id in list_of_prots:
    sequence_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
    
    response = requests.get(sequence_url)
    
    if response.status_code == 200:
        sequence_data = response.text
        sequence = sequence_data.splitlines()[1]  # Extracting the sequence from the response
        list_of_sequence.append(sequence)
    else:
        print(f"Error retrieving sequence for PDB ID {pdb_id}. Status code: {response.status_code}")
        


