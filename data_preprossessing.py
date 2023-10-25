import os
import requests
from bs4 import BeautifulSoup
import tqdm
import csv
import matplotlib.pyplot as plt
import pubchempy

file_path = r"C:\Users\prone\OneDrive\Desktop\drug_proj\regression\terminal_output.txt"

with open(file_path, "r",  encoding="utf8") as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]  # repeats per 4 
    file.close()

#How many Kd vs Ki
n_Kd=0
n_Ki=0
n_IC50=0
n_bind = 0
n_moad=0
n_binding=0 

#linked lists:
kd_ligand = []
kd_protein = []
kd_affinity = []
ki_ligand = []
ki_protein = []
ki_affinity = []

temp_ligand = ""
temp_protein = ""
temp_affinity = ""
if_ki = True

line_num = 1 #empty cache for first line
for line in lines: #NNNNN prots # when multiple vals / PDB, breaks

    if "Affinity" in line:
        if "Ki" in line: #implement cache-like system
            n_Ki+=1
            line = line.split()
            og_line = line
            line = line[1]
            if line == "Ki:&nbspmin:":
                line = og_line
                int_num_1 = line[2]
                num_1 = float(int_num_1[:-1])
                num_2 = float(line[4])
                avg_num = (num_1 + num_2) / 2 #make it so if too big range, not counted
                if_ki = True
                temp_affinity = avg_num
            else:
                n=0
                character = line[n]
                while character != "p":
                    character = line[n]
                    n+=1
                if_ki = True
                temp_affinity = float(line[n:])
        elif "Kd" in line:
            n_Kd+=1
            line = line.split()
            og_line = line
            line = line[1]
            if line == "Kd:&nbspmin:":
                line = og_line
                int_num_1 = line[2]
                num_1 = float(int_num_1[:-1])
                num_2 = float(line[4])
                avg_num = (num_1 + num_2) / 2
                if_ki = False
                temp_affinity = avg_num
            else:
                n=0
                character = line[n]
                while character != "p":
                    character = line[n]
                    n+=1
                if_ki = False
                temp_affinity = float(line[n:])
        elif "IC50" in line:
            n_IC50+=1
    elif "Source" in line:
        if "PDBBind" in line:
            n_bind+=1
        elif "MOAD" in line:
            n_moad+=1
        elif "BindingDB" in line:
            n_binding+=1
        line = line.split()
        line = line[2]
        temp_protein = line
    elif "Ligand" in line:
        line = line.split()
        line = line[1]
        temp_ligand = line
    else:
        if "it/s" in line:
            if line_num != 1:
                print(temp_protein)
                print(temp_affinity)
                print(temp_ligand)

                if if_ki == True: 
                    ki_affinity.append(temp_affinity)
                    ki_ligand.append(temp_ligand)
                    ki_protein.append(temp_protein)
                else:   
                    kd_affinity.append(temp_affinity)
                    kd_ligand.append(temp_ligand)
                    kd_protein.append(temp_protein)

    line_num +=1



print("Num Kd: " + str(n_Kd))
print("Num Ki: " + str(n_Ki))
print("Num IC50: " + str(n_IC50))

print("Num from PDBBind: " + str(n_bind))
print("Num from Binding MOAD: " + str(n_moad))
print("Num from BindingDB: " + str(n_binding))

# print("Num Ki Affinity Values: " + str(len(ki_affinity)))
# print("Num Kd Affinity Values: " + str(len(kd_affinity)))

# all_affinity_vals = []
# for val in ki_affinity:
#     all_affinity_vals.append(val)
# for val in kd_affinity:
#     all_affinity_vals.append(val)
# plt.boxplot(all_affinity_vals)
# plt.title("All Kd / Ki: Boxplot Distribution")
# plt.xlabel("Data")
# plt.ylabel("Values")
# plt.show()

# plt.boxplot(ki_affinity)
# plt.title("Ki Affinity Values: Boxplot Distribution")
# plt.xlabel("Data")
# plt.ylabel("Values")
# plt.show()

# plt.boxplot(kd_affinity)
# plt.title("Kd Affinity Values: Boxplot Distribution")
# plt.xlabel("Data")
# plt.ylabel("Values")
# plt.show()

ligand_list = ki_ligand
print(ligand_list)
for ligand in ligand_list:
    response = requests.get(f'https://www.rcsb.org/ligand/{ligand}')

    soup = BeautifulSoup(response.content, 'html.parser')
    
    isomeric_smiles_element = soup.find(id='chemicalIsomeric')
    if isomeric_smiles_element:
        isomeric_smiles = isomeric_smiles_element.td.get_text()
        print("Isomeric SMILES:", isomeric_smiles)
    else:
        print("Isomeric SMILES information not found in the HTML.")

smile_list = []
for pdb_id in ligand_list:
    try:
        compound = pubchempy.get_compounds(pdb_id, 'name', record_type='3d')
        smile_list.append(compound.canonical_smiles)
    except pubchempy.PubChemHTTPError:
        smile_list.append(None)
        print("None Error, No Can. SMILE")

