#import PDB
from Bio.PDB import PDBParser
import numpy as np

def find_pocket(pdb_id):
  """Finds the pocket of the given PDB ID protein-ligand pair.

  Args:
    pdb_id: The PDB ID of the protein-ligand pair.

  Returns:
    A list of atoms in the pocket.
  """
  pdb = PDB.PDBFile(pdb_id)
  protein = pdb.get_structure("protein")
  ligand = pdb.get_structure("ligand")

  
  # Find the atoms that are within 5 angstroms of the ligand atoms.
  pocket_atoms = []
  for atom in protein.iter_atoms():
    for ligand_atom in ligand.iter_atoms():
      if atom.distance(ligand_atom) < 5:
        pocket_atoms.append(atom)

  return pocket_atoms

def make_pdb_file(pocket_atoms, output_file):
  """Makes a PDB file of the given pocket.

  Args:
    pocket_atoms: A list of atoms in the pocket.
    output_file: The output PDB file name.
  """
  with open(output_file, "w") as f:
    f.write("HEADER\n")
    f.write("MODEL 0\n")
    for atom in pocket_atoms:
      f.write(atom.as_pdb_line())
    f.write("ENDMDL\n")

def main():
  """The main function."""
  pdb_id = "1A3X"
  output_file = "file.pdb"

  pocket_atoms = find_pocket(pdb_id)
  make_pdb_file(pocket_atoms, output_file)

if __name__ == "__main__":
  main()
