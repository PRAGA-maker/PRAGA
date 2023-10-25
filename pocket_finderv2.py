import matplotlib.pyplot as plt
import numpy as np
import requests
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
from sklearn.cluster import KMeans


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

    #print("sze prot: " + str(normalized_protein_coordinates.shape))
    #print("sze lig: " + str(normalized_ligand_coordinates.shape))

    #print("sze atom prot: " + str(protein_atom_types_encoded.shape))
    #print("sze atom lig: " + str(ligand_atom_types_encoded.shape))

    return normalized_protein_coordinates, normalized_ligand_coordinates, protein_atom_types_encoded, ligand_atom_types_encoded, atoms, n_ligand

# coords1 = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # First set of coordinates
# coords2 = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # Second set of coordinates
pdb_id = "830C"
prot_coords, ligand_coords, prot_atoms, ligand_atoms,elements, n_ligand = download_and_format_pdb(pdb_id)
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

# Extract x, y, and z values from the coordinate lists
x1, y1, z1 = zip(*coords1)
x2, y2, z2 = zip(*coords2)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the first set of coordinates in blue
ax.scatter(x1, y1, z1, c='blue', label='Coords 1')

# Plot the second set of coordinates in red
ax.scatter(x2, y2, z2, c='red', label='Coords 2')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Convert lists to NumPy arrays for easier calculations
coords1 = np.array(coords1)
coords2 = np.array(coords2)

# Define the maximum distance threshold (in normalized units)
max_distance = 0.1  # Adjust this value as needed

# Initialize an empty list to store coordinates within the threshold
pocket = []

# Iterate through coords1 and check the distance to coords2
for coord1 in coords1:
    distances = np.linalg.norm(coords2 - coord1, axis=1)
    if np.min(distances) * (1 / max_distance) <= 1.0:
        pocket.append(coord1)

# Convert the pocket list back to a NumPy array
pocket = np.array(pocket)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the coords2 in red
ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c='red', label='Coords 2')

# Plot the pocket points in blue
if len(pocket) > 0:
    ax.scatter(pocket[:, 0], pocket[:, 1], pocket[:, 2], c='green', label='Pocket')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a legend
ax.legend()

# Show the plot
plt.show()
