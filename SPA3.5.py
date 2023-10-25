import requests
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
#now DB maker to incorporate PDB, HGV, PDBFlex, AlphaMissense
def prot_extract(pdb_id):
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

        return normalized_protein_coordinates, normalized_ligand_coordinates, protein_atom_types_encoded, ligand_atom_types_encoded, atoms

    coord = []
    atom = []
    total_list = []
    com = []
    torsion_angle_list = []

    element_masses = {
        "H": 1.00794,
        "He": 4.0026,
        "Li": 6.941,
        "Be": 9.01218,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Ne": 20.180,
        "Na": 22.990,
        "Mg": 24.305,
        "Al": 26.982,
        "Si": 28.086,
        "P": 30.974,
        "S": 32.065,
        "Cl": 35.453,
        "K": 39.098,
        "Ar": 39.948,
        "Ca": 40.078,
        "Sc": 44.956,
        "Ti": 47.867,
        "V": 50.942,
        "Cr": 51.996,
        "Mn": 54.938,
        "Fe": 55.845,
        "Ni": 58.693,
        "Co": 58.933,
        "Cu": 63.546,
        "Zn": 65.38,
        "Ga": 69.723,
        "Ge": 72.631,
        "As": 74.922,
        "Se": 78.971,
        "Br": 79.904,
        "Kr": 83.798,
        "Rb": 85.468,
        "Sr": 87.62,
        "Y": 88.906,
        "Zr": 91.224,
        "Nb": 92.906,
        "Mo": 95.95,
        "Tc": 98.0,
        "Ru": 101.07,
        "Rh": 102.91,
        "Pd": 106.42,
        "Ag": 107.87,
        "Cd": 112.41,
        "In": 114.82,
        "Sn": 118.71,
        "Sb": 121.76,
        "I": 126.90,
        "Te": 127.60,
        "Xe": 131.29,
        "Cs": 132.91,
        "Ba": 137.33,
        "La": 138.91,
        "Ce": 140.12,
        "Pr": 140.91,
        "Nd": 144.24,
        "Pm": 145.0,
        "Sm": 150.36,
        "Eu": 152.0,
        "Gd": 157.25,
        "Tb": 158.93,
        "Dy": 162.50,
        "Ho": 164.93,
        "Er": 167.26,
        "Tm": 168.93,
        "Yb": 173.05,
        "Lu": 175.00,
        "Hf": 178.49,
        "Ta": 180.95,
        "W": 183.84,
        "Re": 186.21,
        "Os": 190.23,
        "Ir": 192.22,
        "Pt": 195.08,
        "Au": 196.97,
        "Hg": 200.59,
        "Tl": 204.38,
        "Pb": 207.2,
        "Bi": 208.98,
        "Th": 232.04,
        "Pa": 231.04,
        "U": 238.03,
        "Np": 237.0,
        "Pu": 244.0,
        "Am": 243.0,
        "Cm": 247.0,
        "Bk": 247.0,
        "Cf": 251.0,
        "Es": 252.0,
        "Fm": 257.0,
        "Md": 258.0,
        "No": 259.0,
        "Lr": 262.0,
        "Rf": 267.0,
        "Db": 270.0,
        "Sg": 271.0,
        "Bh": 270.0,
        "Hs": 277.0,
        "Mt": 276.0,
        "Ds": 281.0,
        "Rg": 280.0,
        "Cn": 285.0,
        "Nh": 284.0,
        "Fl": 289.0,
        "Mc": 288.0,
        "Lv": 293.0,
        "Ts": 294.0,
        "Og": 294.0,
    }

    # Define a function to get the atom mass based on its element symbol
    def get_atom_mass(element_symbol):
        return element_masses.get(element_symbol, 0.0)  # Default to 0.0 if the symbol is not found

    for id in tqdm.tqdm(pdb_id):
        #print("CURRENT PDB ID BEING ANALYZED: " + str(id))
        prot_coords, ligand_coords, prot_atoms, ligand_atoms,elements = download_and_format_pdb(id)

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

            # plt.figure(figsize=(20, 20))  # Adjust the figure size as needed
            # plt.imshow(atoms, cmap='plasma', interpolation='nearest')
            # plt.colorbar()  # Optionally add a colorbar
            # plt.title("Large NumPy Array")
            # plt.show()

            #center of mass calc
            coordinates = coords
            elements = elements

            total_mass = 0.0
            com_coords = np.array([0.0, 0.0, 0.0])

            for i in range(len(coordinates)):
                atom_coord = coordinates[i]
                element = elements[i]
                element = str(element[0])
                atom_mass = get_atom_mass(element)  

                total_mass += atom_mass
                com_coords += atom_mass * atom_coord

            com_coords /= total_mass
            #print(f"Center of Mass (XYZ coordinates): {com_coords}")
            #com = np.append(com, com_coords)
            com.append(com_coords)


            #torsion angles
            def calc_torsion_angle(atom1,atom2,atom3,atom4):
                vector1 = atom1 - atom2
                vector2 = atom3 - atom2
                vector3 = atom4 - atom3
                normal1 = np.cross(vector1,vector2)
                normal2 = np.cross(vector2,vector3)

                dot_product = np.dot(normal1,normal2)
                cross_product = np.cross(normal1,normal2)

                torsion_angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
                return torsion_angle
            
            def calc_torsion_angles(atom_coords):
                torsion_angles = []

                for i in range(1, len(atom_coords) - 2):
                    atom1 = atom_coords[i-1]
                    atom2 = atom_coords[i]
                    atom3 = atom_coords[i+1]
                    atom4 = atom_coords[i+2]

                    torsion_angle = calc_torsion_angle(atom1,atom2,atom3,atom4)
                    torsion_angles.append(torsion_angle)

                return torsion_angles
            
            atom_coords = coordinates

            torsion_angles = calc_torsion_angles(atom_coords)
            torsion_angle_list.append(torsion_angles)

            for i,angle in enumerate(torsion_angles):
                #print(f"T angle {i+1}: {np.degrees(angle):.2f} degrees")
                pass

    pca_2d = PCA(n_components=2)
    pcas = []
    for i in range(0,len(total_list)):
        split_data_atoms = atom[i]
        split_data_coords = coord[i]
        #2d, split
        atoms_pca = pca_2d.fit(split_data_atoms)
        coords_pca = pca_2d.fit(split_data_coords)

        #print(atoms_pca.singular_values_)
        #print(coords_pca.singular_values_)
        total_pca = np.vstack((atoms_pca.singular_values_,coords_pca.singular_values_))
        pcas.append(total_pca)

    return coord, atom, total_list, com, torsion_angle_list, pcas, prot_coords, ligand_coords, prot_atoms, ligand_atoms

#prot, ligand, features of em, db source,, affinity type
prots = []
ligands = []
com = []
pca = []
torsion_angles = []
db_source = []
kd_or_ki = []

def extract_data(lines): #list of all
    n_pdb_bind = 0
    n_moad = 0
    n_binding_db = 0
    n_kd = 0
    n_ki = 0
    n_other = 0
    pdb_ids = []
    for line in lines:
        line = line.split(',')
        pdb_ids.append(line[0])

        #now db source 
        meat = line[1:]
        #ex.
        #['3IN4', '"(\'Binding MOAD:\\xa0 3IN4\'', ' \'IC50:&nbsp40\\xa0(nM) from 1 assay(s)\')"\n']
        #['3IN4', '"(\'BindingDB:\\xa0 3IN4\'', " 'IC50:&nbspmin: 30", ' max: 40\\xa0(nM) from 2 assay(s)\')"\n']

        for section in meat:
            if "PDBBind" in section:
                n_pdb_bind +=1
            elif "Binding MOAD" in section:
                n_moad +=1
            if "BindingDB" in section:
                n_binding_db +=1 

        #now affinity type
        meat_copy = meat
        for section in meat: 
            if "Kd" in section:
                n_kd +=1 
            elif "Ki" in section:
                n_ki +=1 
            elif "nM" in section:
                n_other +=1

    coord, atom, total_list, com, torsion_angle_list, pcas, prot_coords, ligand_coords, prot_atoms, ligand_atoms = prot_extract(pdb_ids[0:3])

    # print("coord: " + str(len(coord)))
    # print("atom: " + str(len(atom)))
    # print("total list: " + str(len(total_list)))
    # print("com: " + str(len(com)))
    # print("torsion_angle_list: " + str(len(torsion_angle_list)))
    # print("pcas: " + str(len(pcas)))
    # print("prot_coords: " + str(len(prot_coords)))
    # print("ligand_coords: " + str(len(ligand_coords)))
    # print("prot_atoms: " + str(len(prot_atoms)))
    # print("ligand_atoms: " + str(len(ligand_atoms)))

    return n_pdb_bind, n_moad, n_binding_db, n_kd, n_ki, n_other,prot_coords, ligand_coords, prot_atoms, ligand_atoms

t_n_pdbs = 0
t_n_moad = 0
t_m_bind = 0
t_n_kd = 0
t_n_ki = 0
t_other = 0
all_prot_coords = [] 
all_ligand_coords = []
all_prot_atoms = []
all_ligand_atoms = []
list_path = [r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data.csv",r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy.csv",r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy 2.csv",r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy 3.csv",r"C:\Users\prone\OneDrive\Desktop\drug_proj\n_data copy 4.csv"]
for path in list_path:

    with open(path, "r") as file:
        lines = file.readlines()

    n_pdb_bind, n_moad, n_binding_db, n_kd,n_ki, n_other,prot_coords, ligand_coords, prot_atoms, ligand_atoms = extract_data(lines)

    t_n_pdbs += n_pdb_bind
    t_n_moad += n_moad
    t_m_bind += n_binding_db
    t_n_kd += n_kd 
    t_n_ki += n_ki
    t_other += n_other

    for prot in prot_coords:
        all_prot_coords.append(prot)
    for ligand in ligand_coords:
        all_ligand_coords.append(ligand)
    for prot in prot_atoms:
        all_prot_atoms.append(prot)
    for ligand in ligand_atoms:
        all_ligand_atoms.append(ligand)
        
print("Num PDBBind: " + str(t_n_pdbs))
print("Num MOAD: " + str(t_n_moad))
print("Num BindingDB: " + str(t_m_bind))
