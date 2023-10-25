import os
import requests

#make list of PDB IDs in PocketDB

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

#for each PDB ID, check PDB to see if affinity value for drug - ligand pair. if yes, log into database. 

def get_binding_data(pdb_id):
    # Query the PDBsum API to get ligand information
    pdbsum_url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"
    response = requests.get(pdbsum_url)
    pdbsum_data = response.json()

    if "ligandInfo" not in pdbsum_data:
        return None

    ligand_data = pdbsum_data["ligandInfo"]
    binding_data = []

    for ligand in ligand_data:
        ligand_id = ligand["chemicalId"]
        ligand_name = ligand["chemicalName"]
        
        # Query the Ligand Expo API to get binding affinity data
        ligand_expo_url = f"https://www.rcsb.org/rest/v1/core/ligand/{ligand_id}/details"
        response = requests.get(ligand_expo_url)
        ligand_expo_data = response.json()

        if "rcsb_chem_comp_descriptor" in ligand_expo_data:
            affinity_data = ligand_expo_data["rcsb_chem_comp_descriptor"]
            for entry in affinity_data:
                if "Ki" in entry or "Kd" in entry:
                    affinity_type = "Ki" if "Ki" in entry else "Kd"
                    affinity_value = entry[affinity_type]
                    protein = entry.get("polymer_entity")
                    if protein:
                        protein = protein[0].get("entity_id")
                        binding_data.append({
                            "protein": protein,
                            "ligand_id": ligand_id,
                            "ligand_name": ligand_name,
                            "affinity_type": affinity_type,
                            "affinity_value": affinity_value
                        })

    return binding_data

pocketdb_pdb = ["1abf"]
for pdb_id in pocketdb_pdb:

    pdb_id = pdb_id.upper()
    binding_data = get_binding_data(pdb_id)

    if binding_data:
        print("Binding Data:")
        for entry in binding_data:
            print(f"Protein: {entry['protein']}")
            print(f"Ligand ID: {entry['ligand_id']}")
            print(f"Ligand Name: {entry['ligand_name']}")
            print(f"Affinity Type: {entry['affinity_type']}")
            print(f"Affinity Value: {entry['affinity_value']}")
            print("=" * 40)
    else:
        print(f"No Binding Affinity Data for {pdb_id}")

#regression model to predict the both. 