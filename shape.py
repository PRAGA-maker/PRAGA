import requests
import numpy as np

def download_and_format_pdb(pdb_id):
    # Step 1: Download the PDB file using requests
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(pdb_url)

    if response.status_code != 200:
        print(f"Failed to download the PDB file for {pdb_id}.")
        return

    pdb_data = response.text.split('\n')

    # Initialize lists to store atom coordinates and atom types for both protein and ligand
    protein_coordinates = []
    protein_atom_types = []
    ligand_coordinates = []
    ligand_atom_types = []

    # Determine if the current line corresponds to the protein or ligand by checking chain ID or residue name
    current_coordinates = protein_coordinates
    current_atom_types = protein_atom_types

    # Step 2: Parse the PDB data
    for line in pdb_data:
        if line.startswith("ATOM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_type = line[76:78].strip()  # Extract the atom type (e.g., 'C', 'N', 'O', 'H', etc.)

            # Check if the line corresponds to the ligand (you may need to adjust the condition)
            if line[21:22].strip() != "":
                current_coordinates = ligand_coordinates
                current_atom_types = ligand_atom_types

            current_coordinates.append([x, y, z])
            current_atom_types.append(atom_type)

    # Convert the lists to NumPy arrays for both protein and ligand
    protein_coordinates_array = np.array(protein_coordinates)
    ligand_coordinates_array = np.array(ligand_coordinates)

    print(protein_coordinates_array)
    print(ligand_coordinates_array)

    # Create a dictionary to map atom types to unique integer IDs for both protein and ligand
    unique_protein_atom_types = np.unique(protein_atom_types)
    unique_ligand_atom_types = np.unique(ligand_atom_types)

    protein_atom_type_to_id = {atom_type: idx for idx, atom_type in enumerate(unique_protein_atom_types)}
    ligand_atom_type_to_id = {atom_type: idx for idx, atom_type in enumerate(unique_ligand_atom_types)}

    # One-hot encode the atom types for both protein and ligand
    num_protein_atom_types = len(unique_protein_atom_types)
    num_ligand_atom_types = len(unique_ligand_atom_types)

    protein_atom_types_encoded = np.zeros((len(protein_atom_types), num_protein_atom_types))
    ligand_atom_types_encoded = np.zeros((len(ligand_atom_types), num_ligand_atom_types))

    for i, atom_type in enumerate(protein_atom_types):
        protein_atom_types_encoded[i, protein_atom_type_to_id[atom_type]] = 1

    for i, atom_type in enumerate(ligand_atom_types):
        ligand_atom_types_encoded[i, ligand_atom_type_to_id[atom_type]] = 1

    # Step 3: Format the coordinates and one-hot encoded atom types for PointNet Model
    # The format depends on the specific requirements of your model.
    # You may need to normalize, rescale, or preprocess the data as necessary.

    # Example: Normalizing coordinates to the range [0, 1] for both protein and ligand
    min_protein_coords = np.min(protein_coordinates_array, axis=0)
    max_protein_coords = np.max(protein_coordinates_array, axis=0)

    min_ligand_coords = np.min(ligand_coordinates_array, axis=0)
    max_ligand_coords = np.max(ligand_coordinates_array, axis=0)

    normalized_protein_coordinates = (protein_coordinates_array - min_protein_coords) / (max_protein_coords - min_protein_coords)
    normalized_ligand_coordinates = (ligand_coordinates_array - min_ligand_coords) / (max_ligand_coords - min_ligand_coords)

    # Now, normalized_protein_coordinates contains the 3D coordinates of the protein,
    # normalized_ligand_coordinates contains the 3D coordinates of the ligand,
    # protein_atom_types_encoded contains the one-hot encoded atom types for the protein,
    # and ligand_atom_types_encoded contains the one-hot encoded atom types for the ligand.

    # You can use these arrays as input to your PointNet model.

    # np.savetxt(f"{pdb_id}_protein_coordinates.txt", normalized_protein_coordinates, fmt='%.6f')
    # np.savetxt(f"{pdb_id}_protein_atom_types_encoded.txt", protein_atom_types_encoded, fmt='%d')
    # np.savetxt(f"{pdb_id}_ligand_coordinates.txt", normalized_ligand_coordinates, fmt='%.6f')
    # np.savetxt(f"{pdb_id}_ligand_atom_types_encoded.txt", ligand_atom_types_encoded, fmt='%d')

    print("sze prot: " + str(normalized_protein_coordinates.shape))
    print("sze lig: " + str(normalized_ligand_coordinates.shape))

    print("sze atom prot: " + str(protein_atom_types_encoded.shape))
    print("sze atom lig: " + str(ligand_atom_types_encoded.shape))

# Usage example:
pdb_id = "16PK"  # Replace with the PDB ID of your choice
download_and_format_pdb(pdb_id)
