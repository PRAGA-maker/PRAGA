import matplotlib.pyplot as plt
import numpy as np
import requests
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
from sklearn.cluster import KMeans
import tqdm
from Bio.SVDSuperimposer import SVDSuperimposer

def instance(id):
    def download_and_format_pdb(pdb_id):

        pdb_url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
        response = requests.get(pdb_url)

        if response.status_code != 200:
            print(f"Failed to download PDB file for {pdb_id}")
            return
        
        pdb_data = response.text.split("\n")

        protein_coords = []
        protein_atoms = []
        ligand_coords = []
        ligand_atoms = []   
        n_ligand = 0
        try:
            for line in pdb_data:
                if line.startswith("ATOM"):
                    line = line.split()
                    if len(line) == 12:
                        x = float(line[6])
                        y = float(line[7])
                        z = float(line[8])
                        atom = str(line[11])
                    elif len(line) == 11:
                        x = float(line[5])
                        y = float(line[6])
                        z = float(line[7])
                        atom = str(line[10])

                    protein_coords.append([x,y,z])
                    protein_atoms.append([atom])
                elif line.startswith("HETATM"):
                    id = str(line[17:20])
                    if id != "HOH": 
                        n_ligand +=1
                        line = line.split()
                        if len(line) == 12:
                            x = float(line[6])
                            y = float(line[7])
                            z = float(line[8])
                            atom = str(line[11])
                        elif len(line) == 11:
                            x = float(line[5])
                            y = float(line[6])
                            z = float(line[7])
                            atom = str(line[10])

                        ligand_coords.append([x,y,z])
                        ligand_atoms.append([atom])
        except ValueError:
            return " ",0,0,0,0

        protein_coords = np.array(protein_coords)
        protein_atoms = np.array(protein_atoms)
        ligand_coords = np.array(ligand_coords)
        ligand_atoms = np.array(ligand_atoms)

        #print(protein_coords.shape)
        #print(protein_atoms.shape)
        #print(ligand_coords.shape)
        #print(ligand_atoms.shape)

        #for COM calc, keep npy array of atoms reg
        atoms = np.vstack((protein_atoms,ligand_atoms))

        unique_prot_atoms = np.unique(protein_atoms)
        unique_ligand_atoms = np.unique(ligand_atoms)

        protein_atom_type_to_id = {atom_type: idx for idx, atom_type in enumerate(unique_prot_atoms)}
        ligand_atom_type_to_id = {atom_type: idx for idx, atom_type in enumerate(unique_ligand_atoms)}

        num_protein_atom_types = len(unique_prot_atoms)
        num_ligand_atom_types = len(unique_ligand_atoms)

        protein_atom_types_encoded = np.zeros((len(protein_atoms), num_protein_atom_types))
        ligand_atom_types_encoded = np.zeros((len(ligand_atoms), num_ligand_atom_types))

        for i, atom_type in enumerate(protein_atoms):
            atom_type_str = atom_type[0]  # Assuming atom_type is a numpy array with a single element
            protein_atom_types_encoded[i, protein_atom_type_to_id[atom_type_str]] = 1

        for i, atom_type in enumerate(ligand_atoms):
            atom_type_str = atom_type[0]  # Assuming atom_type is a numpy array with a single element
            ligand_atom_types_encoded[i, ligand_atom_type_to_id[atom_type_str]] = 1


        protein_atoms = protein_atom_types_encoded
        ligand_atoms = ligand_atom_types_encoded #prot, ligand atoms FIN

        #now process dem coord shits

        min_protein_coords = np.min(protein_coords, axis=0)
        max_protein_coords = np.max(protein_coords, axis=0)

        min_ligand_coords = np.min(ligand_coords, axis=0)
        max_ligand_coords = np.max(ligand_coords, axis=0)

        normalized_protein_coordinates = (protein_coords - min_protein_coords) / (max_protein_coords - min_protein_coords)
        normalized_ligand_coordinates = (ligand_coords - min_ligand_coords) / (max_ligand_coords - min_ligand_coords)

        # print("-----")
        # print("-----")
        # print("-----")
        # print(pdb_id)
        # print(normalized_protein_coordinates)
        # print("-----")
        # print("-----")
        # print("-----")
        # print(normalized_ligand_coordinates)
        return normalized_protein_coordinates, normalized_ligand_coordinates

    # coords1 = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # First set of coordinates
    # coords2 = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # Second set of coordinates
    pdb_id = id
    try:
        prot_coords, ligand_coords = download_and_format_pdb(pdb_id)
        coord = []
        atom = []
        total_list = []

        if type(prot_coords) == str: 
            print("NO DATA")

        coords1 = [] #prot+ligand coords 
        for three_coords in prot_coords:
            three_coords = list(three_coords)
            x = three_coords[0]
            y = three_coords[1]
            z = three_coords[2]
            coords1.append([x,y,z])
        # for three_coords in ligand_coords:
        #     three_coords = list(three_coords)
        #     x = three_coords[0]
        #     y = three_coords[1]
        #     z = three_coords[2]
        #     coords1.append([x,y,z])

        coords2 = ligand_coords

        coords1 = np.array(coords1)
        coords2 = np.array(coords2)

        def pocket_find(distance):
            # Define the maximum distance threshold (in normalized units)
            max_distance = distance  # Adjust this value as needed

            # Initialize an empty list to store coordinates within the threshold
            pocket = []

            # Iterate through coords1 and check the distance to coords2
            for coord1 in coords1:
                distances = np.linalg.norm(coords2 - coord1, axis=1)
                if np.min(distances) * (1 / max_distance) <= 1.0:
                    pocket.append(coord1)

            # Convert the pocket list back to a NumPy array
            pocket = np.array(pocket)

            return len(pocket)

        distance_list = []
        atoms_found_list = []

        distances = [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for distance in distances:
            atoms_found = pocket_find(distance)
            distance_list.append(distance)
            atoms_found_list.append(atoms_found)
        
        max = atoms_found_list[-1]
        
        return distance_list, atoms_found_list,max
    except ValueError:
        return "nope", "nope","nope"

pdb_ids = [] 
path = r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data.csv"
with open(path, 'r') as file:
    
    lines = file.readlines()

    for line in lines:
        prot = line[0:4]
        pdb_ids.append(prot)
pdb_ids = pdb_ids[0:425]

distances = []
atoms = []
maxs = []
trash = 1
for id in tqdm.tqdm(pdb_ids):
    distance_list, atoms_found_list,total = instance(id)
    for num in distance_list:
        if num == "nope" or type(num) == str:
            trash = 0
        else:
            distances.append(num)
            #print(num)
    for num in atoms_found_list:
        if num == "nope" or type(num) == str:
            trash = 0
        else:
            atoms.append(num)
            maxs.append(total)
            #print(num)

distance_list = distances
atoms_found_list = atoms

plt.figure(figsize=(10, 6))
plt.scatter(distance_list, atoms_found_list, marker='o', color='b', label='Scatter Plot')
plt.yscale('log')
degree = 3  
coefficients = np.polyfit(distance_list, atoms_found_list, degree)
poly = np.poly1d(coefficients)
x_values = np.linspace(min(distance_list), max(distance_list), 100)
y_values = poly(x_values)
plt.plot(x_values, y_values, color='r', linestyle='-', label=f'Trend Line (Degree {degree})')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('# Atoms in Pocket')
plt.title('Scatter Plot of Distance vs. Atoms In Pocket')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
#plt.show()


# # Calculate the derivatives
# first_derivative = np.polyder(poly, 1)
# second_derivative = np.polyder(poly, 2)
# third_derivative = np.polyder(poly, 3)

# # Create separate plots for derivatives
# plt.figure(figsize=(10, 6))

# plt.subplot(311)
# plt.plot(x_values, first_derivative(x_values), color='g', linestyle='-', label='1st Derivative')
# plt.xlabel('Distances (Å / 100)')
# plt.ylabel('1st Derivative')
# plt.title('1st Derivative of the Trend Line')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()

# plt.subplot(312)
# plt.plot(x_values, second_derivative(x_values), color='m', linestyle='-', label='2nd Derivative')
# plt.xlabel('Distances (Å / 100)')
# plt.ylabel('2nd Derivative')
# plt.title('2nd Derivative of the Trend Line')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()

# plt.subplot(313)
# plt.plot(x_values, third_derivative(x_values), color='c', linestyle='-', label='3rd Derivative')
# plt.xlabel('Distances (Å / 100)')
# plt.ylabel('3rd Derivative')
# plt.title('3rd Derivative of the Trend Line')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()

# plt.tight_layout()
# plt.show()

#log_scale
plt.figure(figsize=(10, 6))
plt.scatter(distance_list, atoms_found_list, marker='o', color='b', label='Scatter Plot')
plt.xscale('log')
plt.yscale('log')
degree = 3
coefficients = np.polyfit(distance_list, atoms_found_list, degree)
poly = np.poly1d(coefficients)
x_values = np.linspace(min(distance_list), max(distance_list), 100)
y_values = poly(x_values)
plt.plot(x_values, y_values, color='r', linestyle='-', label=f'Trend Line (Degree {degree})')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('# Atoms in Pocket')
plt.title('Scatter Plot of Distance vs. Atoms In Pocket')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
#plt.show()


pocket_len = []
x = []
for num in range(0,len(atoms_found_list)):
    if maxs[num] == 0:
        pass
    else:
        ratio = atoms_found_list[num]/maxs[num]
        x.append(distance_list[num])
        pocket_len.append(ratio)
atoms_found_list = pocket_len
print(len(atoms_found_list))
print(len(x))
plt.figure(figsize=(10, 6))
plt.scatter(x, atoms_found_list, marker='o', color='b', label='Scatter Plot')
plt.xscale('log')
degree = 3  
coefficients = np.polyfit(x, atoms_found_list, degree)
poly = np.poly1d(coefficients)
x_values = np.linspace(min(x), max(x), 100)
y_values = poly(x_values)
plt.plot(x_values, y_values, color='r', linestyle='-', label=f'Trend Line (Degree {degree})')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('Ratio #Atoms in Pocket / #Atoms in Protein')
plt.title('Scatter Plot of Distance vs. Ratio')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x, atoms_found_list, marker='o', color='b', label='Scatter Plot')
degree = 3  
coefficients = np.polyfit(x, atoms_found_list, degree)
poly = np.poly1d(coefficients)
x_values = np.linspace(min(x), max(x), 100)
y_values = poly(x_values)
plt.plot(x_values, y_values, color='r', linestyle='-', label=f'Trend Line (Degree {degree})')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('Ratio #Atoms in Pocket / #Atoms in Protein')
plt.title('Scatter Plot of Distance vs. Ratio')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Calculate the derivatives
first_derivative = np.polyder(poly, 1)
second_derivative = np.polyder(poly, 2)
third_derivative = np.polyder(poly, 3)

# Create separate plots for derivatives
plt.figure(figsize=(10, 6))

plt.subplot(311)
plt.plot(x_values, first_derivative(x_values), color='g', linestyle='-', label='1st Derivative')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('1st Derivative')
plt.title('1st Derivative of the Trend Line')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

plt.subplot(312)
plt.plot(x_values, second_derivative(x_values), color='m', linestyle='-', label='2nd Derivative')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('2nd Derivative')
plt.title('2nd Derivative of the Trend Line')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

plt.subplot(313)
plt.plot(x_values, third_derivative(x_values), color='c', linestyle='-', label='3rd Derivative')
plt.xlabel('Distances (Å / 100)')
plt.ylabel('3rd Derivative')
plt.title('3rd Derivative of the Trend Line')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()

#Sample data (replace with your actual data)
distances = x
ratios = atoms_found_list

#Compute the sum of squared differences for each distance
ssd = []
for i in range(1, len(distances)):
    try:
        slope1 = (ratios[i] - ratios[i-1]) / (distances[i] - distances[i-1])
        slope2 = (1 - ratios[i]) / (1 - distances[i])
        ssd.append(abs(slope1 - slope2))
    except ZeroDivisionError:
        pass

#Identify the "elbow" point
elbow_index = np.argmax(ssd)
elbow_distance = distances[elbow_index]

#Plotting
plt.figure(figsize=(10, 5))
plt.plot(distances, ratios, c='b', marker='o', label='Data')
plt.axvline(x=elbow_distance, color='r', linestyle='--', linewidth=0.1, label=f'Elbow at {elbow_distance}')
plt.title('Elbow Method to Determine Optimal Distance Threshold')
plt.xlabel('Distance (Å/100)')
plt.ylabel('Ratio (#atoms in pocket / #atoms in protein)')
plt.legend()
plt.grid(True)
plt.show()

print(f"The optimal distance threshold using the elbow method is approximately {elbow_distance*100} Å")