import os
import requests
from bs4 import BeautifulSoup
import tqdm
import csv

#make list of PDB IDs in PocketDB

#txt_path = r'C:\Users\prone\OneDrive\Desktop\drug_proj\pocketdb_files.txt' 
pocketdb_pdb = []

# if os.path.isfile(txt_path):
    
#     with open(txt_path, 'r') as file:

#         lines = file.readlines()

#         for line in range(0,len(lines)):

#             eye_line = lines[line]

#             eye_line = eye_line[:-1]

#             eye_line = eye_line.lower()

#             pocketdb_pdb.append(eye_line)

txt_path = r'C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy.csv'
if os.path.isfile(txt_path):
    with open(txt_path , 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.split(',')
            line = line[:-1]
            
            for id in line:
                pocketdb_pdb.append(id)

print("Len of PocketDB PDB IDs: " + str(len(pocketdb_pdb)))
print("PocketDB PDB ID Examples: " + str(pocketdb_pdb[0:10]))

#for each PDB ID, check PDB to see if affinity value for drug - ligand pair. if yes, log into database. 

in_pocketdb_and_binding_data_pdb_ids = []
paired_affinity_data_list = []

for pdb_id in tqdm.tqdm(pocketdb_pdb):
    pdb_id = pdb_id.upper()
    response = requests.get(f'https://www.rcsb.org/structure/{pdb_id}')

    soup = BeautifulSoup(response.content, 'html.parser')

    binding_affinity_table = soup.find('table', id='binding-affinity-table')
    if binding_affinity_table == None:
        pass
    else:
        table_body = binding_affinity_table.find('tbody')

        binding_affinity_data = []

        for row in table_body.find_all('tr'):
            columns = row.find_all('td')
            num_columns = len(columns) #1,2,3 --> index 0,1,2 
            if columns:
                if num_columns == 1:
                    ligand = columns[0].text.strip()
                    #binding_affinity_data.append((ligand))
                elif num_columns == 2:
                    ligand = columns[0].text.strip()
                    source = columns[1].text.strip()
                    #binding_affinity_data.append((ligand, source))
                elif num_columns == 3:
                    ligand = columns[0].text.strip()
                    source = columns[1].text.strip()       
                    affinity = columns[2].text.strip()
                    binding_affinity_data.append((ligand, source, affinity))
                    print(ligand)
                    print(source)
                    print(affinity)
                else:
                    print("Num Columns > 3: " + str(num_columns))

        for data in binding_affinity_data:
            n = len(data) #n = 1,2,3 --> index 0,1,2
            if n == 1:
                #print("Ligand: ", data[0])
                pass
            if n == 2:
                #print("Ligand: ", data[0])
                #print("Source: ", data[1])
                pass
            if n == 3:
                #print("Ligand:", data[0])
                #print("Source:", data[1])
                #print("Affinity:", data[2])    

                in_pocketdb_and_binding_data_pdb_ids.append(pdb_id)
                paired_affinity_data_list.append(data)

print("N: " + str(len(in_pocketdb_and_binding_data_pdb_ids)))

#paired affinity data list , in pocketdb and binding data pdb ids

csv_filename = r'C:\Users\prone\OneDrive\Desktop\drug_proj\regression\pdb_ids\2.txt'
t=0
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    while t <= 1322:
        try:
            csv_writer.writerow([in_pocketdb_and_binding_data_pdb_ids[t], paired_affinity_data_list[t]])
        except UnicodeEncodeError as e:
            print("Unicode Error: " + str([in_pocketdb_and_binding_data_pdb_ids[t], paired_affinity_data_list[t]]))
        t+=1
    
#regression model to predict the both. 
