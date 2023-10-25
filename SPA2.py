import matplotlib.pyplot as plt
import numpy as np
import requests
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
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

    return normalized_protein_coordinates, normalized_ligand_coordinates, protein_atom_types_encoded, ligand_atom_types_encoded, n_ligand

pdb_id = "2BVS"
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
max_distance = 0.05
pocket = []
for coord1 in coords1:
    distances = np.linalg.norm(coords2 - coord1, axis=1)
    if np.min(distances) * (1 / max_distance) <= 1.0:
        pocket.append(coord1)
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
    
def calculate_local_rmsd_with_clustering(coords1, coords2,num):
    combined_coords = np.vstack((coords1, coords2))

    #find num clusters --> elbow method
    wcss = []
    clusters_range = range(1, int(num))  

        
    for n_clusters in clusters_range:
        # Fit K-means clustering model
        kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='warn')
        kmeans.fit(combined_coords)
        
        # Calculate the WCSS for this number of clusters
        wcss.append(kmeans.inertia_)

    wcss_diff = np.diff(wcss)
    wcss_diff2 = np.diff(wcss_diff)

    # Find the elbow point by looking for a sign change in the second derivative
    elbow_point = np.where(wcss_diff2 > 0)[0][0] + 2  # Add 2 because of differences

    # Plot the WCSS values
    plt.figure(figsize=(10, 6))
    plt.plot(clusters_range, wcss, marker='o', linestyle='-', label='WCSS')
    plt.plot(clusters_range[1:], wcss_diff, marker='x', linestyle='-', label='1st Derivative')
    plt.plot(clusters_range[2:], wcss_diff2, marker='*', linestyle='-', label='2nd Derivative')

    plt.xlabel('Number of Clusters')
    plt.ylabel('Value')
    plt.title('Elbow Method with Derivatives to Determine Optimal Clusters')
    plt.grid(True)

    # Mark the elbow point on the plot
    plt.scatter(elbow_point, wcss[elbow_point - 1], c='red', marker='x', s=100, label='Elbow Point')

    plt.legend()
    plt.show()

    # The 'elbow_point' variable now contains the automatically determined optimal number of clusters
    print("Optimal number of clusters:", elbow_point)

    num_clusters = elbow_point

    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(combined_coords)
    plot_k_cluster_map_3d(np.vstack((coords1, coords2)), cluster_labels, num_clusters)

    local_rmsd_values = []
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
                local_rmsd = calculate_rmsd(cluster_coords1, cluster_coords2)
                local_rmsd_values.append(local_rmsd)

    return local_rmsd_values

num = int(int(len(prot_coords)) / int(n_ligand ))
localrmsd = calculate_local_rmsd_with_clustering(coords1,pocket,num)
print(localrmsd)