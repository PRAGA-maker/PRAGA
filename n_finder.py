import os
import tqdm
import requests 

def moab_pocketdb_n_check():

    # -- N Checker |  MOAB - Pocket DB -- 

    folder_path = r'c:\Users\prone\OneDrive\Desktop\test' 

    pdb = []

    for filename in tqdm.tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
        
            with open(file_path, 'r') as file:
    
                # lines = file.readlines()
                # if len(lines) >= 2:
                #     second_line = lines[1].strip()


                #     words = second_line.split()
                #     if "TITLE" in words:
                #         index = words.index("TITLE")
                #         title_words = words[index+1:]
                    
                #     third_line = lines[2].strip()

                #     words = third_line.split() 
                #     if "TITLE" in words:
                #         index = words.index("TITLE")
                #         title_words = title_words + words[index+1:]
                        
                #     title_words = ' '.join(title_words)
                #     pdb.append(title_words)

                file_name = os.path.splitext(os.path.basename(file_path))[0]
            
                pdb.append(file_name)

                file.close()
                
    print("N in MOAB: " + str(len(pdb)))

    print("MOAB-sourced PDB examples: " + str(pdb[0:10]))

    n = 0

    txt_path = r'C:\Users\prone\OneDrive\Desktop\drug_proj\pocketdb_files.txt' 

    pocketdb_pdb = []

    if os.path.isfile(txt_path):
        
        with open(txt_path, 'r') as file:

            lines = file.readlines()

            for line in range(0,len(lines)):

                eye_line = lines[line]

                eye_line = eye_line[:-1]

                eye_line = eye_line.lower()

                pocketdb_pdb.append(eye_line)

    print("Len of PocketDB PDB IDs: " + str(len(pocketdb_pdb)))
    print("PocketDB PDB ID Examples: " + str(pocketdb_pdb[0:10]))

    same_pdb = []
    for pdb_id in tqdm.tqdm(pdb):

        for pocketdb_pdb_id in pocketdb_pdb:

            if pdb_id == pocketdb_pdb_id:

                n += 1

                same_pdb.append(pdb_id)

    print(" PocketDB | MOAB N: " + str(n))

    print("Same ID Ex: " + str(same_pdb[0:10]))

    unmatched_pocketdb_pdbid = []
    for id in pocketdb_pdb:
        if id in same_pdb:
            pass
        else:
            unmatched_pocketdb_pdbid.append(str(id.lower()))

    return unmatched_pocketdb_pdbid


def bindingdb_pocketdb_n_check(unmatched_pocketdb_pdbid):

    # -- N Checker |  BindingDB - Pocket DB -- 

    txt_path = r'C:\Users\prone\OneDrive\Desktop\drug_proj\bindingdb_pdbid.txt' 

    matched_pocketdb_pdbid = []
    still_unmatched_pocketdb_pdbid = []

    if os.path.isfile(txt_path):
        with open(txt_path, 'r') as file:

            lines = file.readlines()

            for line in tqdm.tqdm(lines):
                line = line.lower()
                line = line[:-1]

                if (str(line) in unmatched_pocketdb_pdbid) == True:
                                                           
                    matched_pocketdb_pdbid.append(line)

                else:
                    still_unmatched_pocketdb_pdbid.append(line)

            file.close()

    print(" PocketDB | BindingDB N: " + str(len(matched_pocketdb_pdbid)))
    print(" Same ID Ex: " + str(matched_pocketdb_pdbid[0:10]))

    return still_unmatched_pocketdb_pdbid



def main():
    unmatched_pocketdb_pdbid = moab_pocketdb_n_check()
    print("N still unmatched afer MOAB: " + str(len(unmatched_pocketdb_pdbid)))
    og_unmatched = unmatched_pocketdb_pdbid
    unmatched_pocketdb_pdbid = bindingdb_pocketdb_n_check(unmatched_pocketdb_pdbid)
    print("N still unmatched afer BindingDB: " + str(len(unmatched_pocketdb_pdbid)))


main()


# -- PDB DATA API --

# from Bio.PDB import PDBList

# for id in same_pdb:
#     if str(id) != "3eki":
#         pdb_list = PDBList()

#         pdb_id = str(id)

#         pdb_filename = pdb_list.retrieve_pdb_file(pdb_id, pdir = "data/PDB_files", file_format="mmCif")

# folder_path = r'C:\Users\prone\OneDrive\Desktop\drug_proj\data\PDB_files' 

# human_ids = []

# for filename in tqdm.tqdm(os.listdir(folder_path)):
#     file_path = os.path.join(folder_path, filename)
    
#     if os.path.isfile(file_path):
     
#         with open(file_path, 'r') as file:
#             try:
#                 lines = file.readlines() 
                
#                 for line in lines:
#                     word = "homo sapien"
#                     line = line.lower()
#                     if word in line:
#                         human_ids.append(filename)

#                 file.close()
#             except UnicodeDecodeError:
#                 pass

# print("N Human: " + str(len(human_ids)))
# print("Ex. of N-open PDB IDs: " + str(human_ids[0:3]))


import requests

# data = requests.get("https://data.rcsb.org/rest/v1/core/entry/" + str(pdb_id))

# if data.status_code != 200:
#     print(data.status_code)

# info_pdb = data.json()

# print(str(info_pdb.keys()))

# print(str(info_pdb["cell"]))

# print(str(info_pdb["exptl"]))

# print(str(info_pdb["struct_keywords"]))




# interface = requests.get("https://data.rcsb.org/rest/vl/core/interface/" + str(pdb_id) + "/1/2")

# interface.status_code

# interface_info = interface.json()
# interface_info["rcsb_interface_info"]


# -- PDB SEARCH API -- 

# import json

# file_path = r"C:\Users\prone\OneDrive\Desktop\drug_proj\Kidb_names.txt"

# ki_list = []
    
# with open(file_path, 'r') as file:
#     lines = file.readlines()
#     for line in lines:
        
#         if line not in ki_list:
#             ki_list.append(line)

#     file.close()
# print(ki_list)
# print("Num unique Ki DB proteins: " + str(len(ki_list)))

# id = "HTR2C"

# my_query = {

#     "query": {

#         "type": "terminal",
#         "service": "full_text",
#         "parameters": {
#             "value": str(id),
#         }

#     },

#     "return_type": "entry"

# }

# my_query = json.dumps(my_query)

# data = requests.get(f"https://search.rcsb.org/rcsbsearch/v2/query?json={my_query}")
# results = data.json()
# print("Num results: " + str(results))

# first_result = results["result_set"][0]["identifier"]




