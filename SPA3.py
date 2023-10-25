import matplotlib.pyplot as plt
import numpy as np
import requests
from Bio import PDB
from Bio.SVDSuperimposer import SVDSuperimposer
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
                #if "RS1" in line: 
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

    # unique_prot_atoms = np.unique(protein_atoms)
    # unique_ligand_atoms = np.unique(ligand_atoms)

    # protein_atom_type_to_id = {atom_type: idx for idx, atom_type in enumerate(unique_prot_atoms)}
    # ligand_atom_type_to_id = {atom_type: idx for idx, atom_type in enumerate(unique_ligand_atoms)}

    # num_protein_atom_types = len(unique_prot_atoms)
    # num_ligand_atom_types = len(unique_ligand_atoms)

    # protein_atom_types_encoded = np.zeros((len(protein_atoms), num_protein_atom_types))
    # ligand_atom_types_encoded = np.zeros((len(ligand_atoms), num_ligand_atom_types))

    # for i, atom_type in enumerate(protein_atoms):
    #     atom_type_str = atom_type[0]  # Assuming atom_type is a numpy array with a single element
    #     protein_atom_types_encoded[i, protein_atom_type_to_id[atom_type_str]] = 1

    # for i, atom_type in enumerate(ligand_atoms):
    #     atom_type_str = atom_type[0]  # Assuming atom_type is a numpy array with a single element
    #     ligand_atom_types_encoded[i, ligand_atom_type_to_id[atom_type_str]] = 1


    #protein_atoms = protein_atom_types_encoded
    #ligand_atoms = ligand_atom_types_encoded #prot, ligand atoms FIN

    #now process dem coord shits

    min_protein_coords = np.min(protein_coords, axis=0)
    max_protein_coords = np.max(protein_coords, axis=0)

    min_ligand_coords = np.min(ligand_coords, axis=0)
    max_ligand_coords = np.max(ligand_coords, axis=0)

    normalized_protein_coordinates = (protein_coords - min_protein_coords) / (max_protein_coords - min_protein_coords)
    normalized_ligand_coordinates = (ligand_coords - min_ligand_coords) / (max_ligand_coords - min_ligand_coords)

    return normalized_protein_coordinates, normalized_ligand_coordinates, protein_atoms, ligand_atoms, n_ligand

pdb_id = "830C" #2BVS
prot_coords, ligand_coords, prot_atoms, ligand_atoms, n_ligand = download_and_format_pdb(pdb_id)
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
# x1, y1, z1 = zip(*coords1)
# x2, y2, z2 = zip(*coords2)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1, y1, z1, c='blue', label='Coords 1')
# ax.scatter(x2, y2, z2, c='red', label='Coords 2')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()

# plt.show()

coords1 = np.array(coords1)
coords2 = np.array(coords2)
max_distance = 0.2
pocket = []
pocket_atoms = []
pocket_atom_index = []
n = 0
for coord1 in coords1:
    n+=1
    distances = np.linalg.norm(coords2 - coord1, axis=1)
    if np.min(distances) * (1 / max_distance) <= 1.0:
        pocket.append(coord1)
        pocket_atoms.append(list(prot_atoms[n]))
        pocket_atom_index.append(int(n))
pocket = np.array(pocket)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c='red', label='Coords 2')
# if len(pocket) > 0:
#     ax.scatter(pocket[:, 0], pocket[:, 1], pocket[:, 2], c='blue', label='Pocket')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()
# plt.show()


protein_coords = coords1 
protein_elements = prot_atoms
pocket_coords = pocket
pocket_elements = pocket_atoms
print(pocket_elements)
pocket_elements = list(pocket_elements)
p = 0
common_elements = []
surviving_pocket_index = []
surviving_protien_index = []
for index in pocket_atom_index:
    if protein_elements[index] == pocket_atoms[p]:
        common_elements.append(pocket_atoms[p])
        surviving_pocket_index.append(p)
        surviving_protien_index.append(index)

if len(common_elements) == 0:
    print('No common elements')
else:
    print("Num Common Elements: " + str(len(common_elements)))
# Extract coordinates of common atoms
common_protein_coords =[]
common_pocket_coords = []
for index in surviving_protien_index:
    common_protein_coords.append(protein_coords[index])
for index in surviving_pocket_index:
    common_pocket_coords.append(pocket_coords[index])

common_protein_coords = np.array(common_protein_coords)
common_pocket_coords = np.array(common_pocket_coords)
covariance_matrix = np.dot(common_protein_coords.T, common_pocket_coords)
U, S, Vt = np.linalg.svd(covariance_matrix)
rotation_matrix = np.dot(U, Vt)
#3x3 rotation matrix
aligned_pocket_coords = np.dot(common_pocket_coords, rotation_matrix)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
theta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
theta_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X-axis Rotation')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y-axis Rotation')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z-axis Rotation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.legend()
plt.title('Rotation Matrix Visualization')
plt.show()

#downselect prot coords only that align w pocket
protein_coords = common_protein_coords
squared_differences = np.sum((aligned_pocket_coords - protein_coords)**2, axis=1)

mean_squared_difference = np.mean(squared_differences)

rmsd = np.sqrt(mean_squared_difference)

print("RMSD:", rmsd)
print("Length of Overlap (Lower Values Indicate More RMSD Variability): " + str(len(common_elements)))



