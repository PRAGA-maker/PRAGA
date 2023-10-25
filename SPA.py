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
                if "RS1" in line: 
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

def visualize_alignment(coords1, coords2, aligned_coords2):
    # Convert lists to NumPy arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    aligned_coords2 = np.array(aligned_coords2)

    # Create a 3D scatter plot for the original coordinates1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], label='Original coords1', c='b')

    # Plot the centroids of both structures
    ax.scatter(coords1.mean(axis=0)[0], coords1.mean(axis=0)[1], coords1.mean(axis=0)[2], marker='o', s=100, label='Centroid1', c='r')
    ax.scatter(coords2.mean(axis=0)[0], coords2.mean(axis=0)[1], coords2.mean(axis=0)[2], marker='o', s=100, label='Centroid2', c='g')

    # Create a 3D scatter plot for the original coordinates2
    ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], label='Original coords2', c='y')

    # Create a 3D scatter plot for the aligned coordinates2
    ax.scatter(aligned_coords2[:, 0], aligned_coords2[:, 1], aligned_coords2[:, 2], label='Aligned coords2', c='m')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

# Function to apply the Kabsch algorithm for structural alignment
def kabsch_alignment(coords1, coords2): #fix later
    # Calculate centroids of both structures
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)

    # Center the coordinates by subtracting the centroids
    centered_coords1 = coords1 - centroid1
    centered_coords2 = coords2 - centroid2
    # Transpose one of the matrices to make them compatible
    centered_coords1 = np.array(centered_coords1)
    centered_coords2 = np.array(centered_coords2)
    #compatible_centered_coords2 = centered_coords2.T
    centered_coords1 = centered_coords1.T
    # Calculate the covariance matrix
    print(len(centered_coords1))
    print(len(centered_coords1[0]))
    print(len(centered_coords2))
    print(len(centered_coords2[0]))
    covariance_matrix = np.dot(centered_coords1, centered_coords2)
    print(len(covariance_matrix))
    print(len(covariance_matrix[0]))
    # Compute the SVD   
    U, _, Vt = np.linalg.svd(covariance_matrix)
    #print(U)
    #print(Vt)
    # Ensure that U and Vt have compatible shapes for multiplication
    U_rows, U_cols = U.shape
    Vt_rows, Vt_cols = Vt.shape

    # Resize U and Vt as needed to match dimensions
    if U_cols != Vt_rows:
        min_dim = min(U_cols, Vt_rows)
        U = U[:, :min_dim]
        Vt = Vt[:min_dim, :]

    # Compute the rotation matrix
    rotation_matrix = np.dot(U, Vt)

    # Apply the rotation and translation to coords2
    centered_coords2 = centered_coords2.T
    rotation_matrix = rotation_matrix.T
    #print(centered_coords2, rotation_matrix)
    dot_product = np.dot(centered_coords2, rotation_matrix)
    #print(dot_product)
    centroid1 = centroid1[:, np.newaxis]
    aligned_coords2 = dot_product + centroid1
    #visualize_alignment(coords1, coords2, aligned_coords2)
    return aligned_coords2

# Function to calculate RMSD between two sets of coordinates
def calculate_rmsd(coords1, coords2):
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def plot_k_cluster_map_3d(coords, cluster_labels, num_clusters):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot of the clustered points
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))  # Assign colors to clusters
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        coords = np.array(coords)
        cluster_coords = coords[cluster_indices]
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2], label=f'Cluster {cluster_id}', c=colors[cluster_id])

    ax.set_title(f'K-Means Clustering (K={num_clusters})')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_zlabel('Z-coordinate')
    ax.legend()
    plt.show()

# Function to perform spatial clustering of atoms and calculate local RMSD
def calculate_local_rmsd_with_clustering(coords1, coords2, num_clusters):
    # Combine the coordinates from both structures
    combined_coords = np.vstack((coords1, coords2))

    # Perform K-Means clustering to identify clusters of atoms
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(combined_coords)
    #plot_k_cluster_map_3d(np.vstack((coords1, coords2)), cluster_labels, num_clusters)

    local_rmsd_values = []

    # Calculate local RMSD for each cluster
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_coords1 = []
        cluster_coords2 = []
        if len(cluster_indices) > 0:
            #cluster_coords1 = np.take(coords1, cluster_indices, axis=0)
            #cluster_coords2 = np.take(coords2, cluster_indices, axis=0)
            total_cluster_coords = np.take(combined_coords, cluster_indices, axis=0)
            for coords in total_cluster_coords:
                x_t = coords[0]
                y_t = coords[1]
                z_t = coords[2]
                ye = False
                for coord in coords2:
                    x = coord[0]
                    y = coord[1]
                    z = coord[2]
                    if [x_t,y_t,z_t] == [x,y,z]:
                        #print(np.array_str(coords) + str("2 FR FFR"))
                        cluster_coords2.append(coords)
                        ye = True
                for coord in coords1:
                    x = coord[0]
                    y = coord[1]
                    z = coord[2]
                    if [x_t,y_t,z_t] == [x,y,z] and ye == False:
                        #print(coords)
                        cluster_coords1.append(coords)
            
            if cluster_coords2 == []:
                pass
            else:
                aligned_cluster_coords2 = kabsch_alignment(cluster_coords1, cluster_coords2)

                local_rmsd = calculate_rmsd(cluster_coords1, aligned_cluster_coords2)
                local_rmsd_values.append(local_rmsd)

    return local_rmsd_values


# coords1 = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # First set of coordinates
# coords2 = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # Second set of coordinates
pdb_id = "830C"
prot_coords, ligand_coords, prot_atoms, ligand_atoms,elements, n_ligand = download_and_format_pdb(pdb_id)
coord = []
atom = []
total_list = []

if type(prot_coords) == str: 
    pass
else:
    stacked_array = np.vstack((prot_coords, ligand_coords)) #prot on top, then ligands
    #print("stacked coord array: " + str(stacked_array.shape))
    #coord = np.append(coord, stacked_array)
    coord.append(stacked_array)

    coords = stacked_array

    try:
        stacked_array = np.vstack((prot_atoms, ligand_atoms)) #prot on top, then ligands
    except ValueError:
        prot_colmn = (prot_atoms.shape)[1]
        ligand_colmn = (ligand_atoms.shape)[1]

        if prot_colmn > ligand_colmn:
            shape = (((ligand_atoms.shape)[0]),prot_colmn-ligand_colmn)
            #print("prot clmn: " + str(prot_colmn))
            #print("ligand clmn: " + str(ligand_colmn))
            #print("shape: " + str(shape))
            #print("lig atoms shape: " + str(ligand_atoms.shape))    
            empty = np.zeros(shape)
            #print(empty)
            #print(ligand_atoms)
            expanded_smol = np.hstack((ligand_atoms,empty))
            stacked_array = np.vstack((prot_atoms,expanded_smol))
        else:
            shape = (((prot_atoms.shape)[0]),ligand_colmn-prot_colmn)
            empty = np.zeros(shape)
            expanded_smol = np.hstack((prot_atoms,empty))
            stacked_array = np.vstack((expanded_smol,ligand_atoms))

    #print("stacked atom array: " + str(stacked_array.shape))
    #atom = np.append(atom,stacked_array)
    atom.append(stacked_array)

    atoms = stacked_array

    try: 
        stacked_array = np.vstack((coords,atoms))
    except ValueError:
        coords_colmn = (coords.shape)[1]
        atoms_colmn = (atoms.shape)[1]

        if coords_colmn > atoms_colmn:
            shape = ((atoms.shape)[0],coords_colmn-atoms_colmn)
            empty = np.zeros(shape)
            expanded_smol = np.hstack((atoms,empty))
            stacked_array = np.vstack((coords,expanded_smol))
        else:
            shape = ((coords.shape)[0],atoms_colmn-coords_colmn)
            empty = np.zeros(shape)
            expanded_smol = np.hstack((coords,empty))
            stacked_array = np.vstack((expanded_smol,atoms))
    #total = np.append(total,stacked_array)
    total_list.append(stacked_array)
    #print("total size: " + str(stacked_array.shape))

    total = stacked_array

coords1 = [] #prot+ligand coords 
for three_coords in prot_coords:
    three_coords = list(three_coords)
    x = three_coords[0]
    y = three_coords[1]
    z = three_coords[2]
    coords1.append([x,y,z])
for three_coords in ligand_coords:
    three_coords = list(three_coords)
    x = three_coords[0]
    y = three_coords[1]
    z = three_coords[2]
    coords1.append([x,y,z])

coords2 = ligand_coords

# Specify the number of clusters for spatial clustering
#print(str(len(prot_coords)))
#print(str(n_ligand))
num_clusters = int(int(len(prot_coords)) / int(n_ligand ))

# Calculate local RMSD values for each cluster
local_rmsd_values = calculate_local_rmsd_with_clustering(coords1, coords2, num_clusters)
print(local_rmsd_values)
