#convert PDB ligand ID to 3d SMILES
from bs4 import BeautifulSoup
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import PyMol


ligand_list = ['PP1', 'LP1','Q16', 'PD2','AEF', 'Q1A', '2KC', '9CR', 'Y1Z', '2NG', 'COA']
smiles = []
for ligand in ligand_list:
    response = requests.get(f'https://www.rcsb.org/ligand/{ligand}')

    soup = BeautifulSoup(response.content, 'html.parser')
    
    isomeric_smiles_element = soup.find(id='chemicalIsomeric')
    if isomeric_smiles_element:
        isomeric_smiles = isomeric_smiles_element.td.get_text()
        print("Isomeric SMILES:", isomeric_smiles)
        smiles.append(isomeric_smiles)
    else:
        print("Isomeric SMILES information not found in the HTML.")

def convert_smiles_to_pdb(smiles, output_filename):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        print("Invalid SMILES string.")
        return
    
    molecule = Chem.AddHs(molecule)  # Add hydrogens
    AllChem.EmbedMolecule(molecule)  # Generate 3D coordinates
    
    writer = Chem.PDBWriter(output_filename)
    writer.write(molecule)
    writer.close()

if __name__ == "__main__":
    smiles_input = "CC[C@@H](C(=O)N[C@@H](Cc1cccc2c1cccc2)C(=O)N)NC(=O)C[C@@H]([C@H](CC(C)C)NC(=O)[C@H](C(C)C)NC(=O)[C@H](Cc3cccc4c3cccc4)NC(=O)C)O"
    pdb_output_filename = r"C:\Users\prone\OneDrive\Desktop\drug_proj\file.pdb"
    
    convert_smiles_to_pdb(smiles_input, pdb_output_filename)
    print("Conversion completed.")
