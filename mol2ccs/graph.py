"""Graph construction for molecular data."""

import logging

import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from spektral.data import Dataset, Graph

from mol2ccs.constants import ATOM_MASSES, ATOM_RADII
from mol2ccs.utils import min_max_scaling, one_hot_encode

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(formatter)
logger.setLevel(logging.INFO)


def generate_coordinates(
    smiles,
    adduct,
    ccs,
    all_atoms,
    ps=None,
):
    """
    * Using ETKDG to generate 3D coordinates of molecules
    *
    * Attributes
    * ----------
    * smiles    : The SMILES string of the molecule
    * adduct    : Adduct of molecules
    * ccs       : CCS of molecules
    * all_atoms : Element set (The type of element provided must cover all elements contained
                    in the molecule)
    * ps        : ETKDG algorithm provided by RDkit
    *
    * Returns
    * -------
    * succ_smiles : SMILES of The molecules with 3D conformation can be successfully generated
    * succ_adduct : Adduct of The molecules with 3D conformation can be successfully generated
    * succ_ccs    : CCS of The molecules with 3D conformation can be successfully generated
    * Coordinate  : 3D coordinates of molecules
    """
    ps = AllChem.ETKDGv3() if ps is None else ps
    succ_smiles = []
    succ_adduct = []
    succ_ccs = []
    coordinates = []

    idx = -1
    for smi in smiles:
        idx += 1
        try:
            iMol = Chem.MolFromSmiles(smi)
            iMol = Chem.RemoveHs(iMol)
        except Exception as e:
            logger.error(f"Error in molecule {idx}: {e}")
            continue

        if idx % 2500 == 0:
            logger.info(f"on molecule number {idx}")

        atoms = [atom.GetSymbol() for atom in iMol.GetAtoms()]
        bonds = [bond for bond in iMol.GetBonds()]
        # Is the number of atoms greater than 1
        if len(atoms) == 1 and len(bonds) <= 1:
            continue
        # Determine whether the element is in all_atoms
        Elements_not_included = 0
        for atom in atoms:
            if atom not in all_atoms:
                Elements_not_included = 1
        if Elements_not_included == 1:
            continue
        # Adding H to a molecular object
        mol3d = Chem.AddHs(iMol)

        # The 3D conformation of the generating molecule
        ps.randomSeed = -1
        ps.maxAttempts = 1
        ps.numThreads = 0
        ps.useRandomCoords = True
        re = AllChem.EmbedMultipleConfs(mol3d, numConfs=1, params=ps)
        # Whether the conformation is successful or not
        if len(re) == 0:
            logger.error(
                f"Error in molecule {idx}: \
                Could not generate 3D coordinates. Continuing..."
            )
            continue

        # MMFF94
        re = AllChem.MMFFOptimizeMoleculeConfs(mol3d, numThreads=0)

        this_mol_coordinate = []
        for atom in mol3d.GetAtoms():
            coords = list(mol3d.GetConformer().GetAtomPosition(atom.GetIdx()))
            this_mol_coordinate.append(coords)
        coordinates.append(this_mol_coordinate)

        succ_smiles.append(smi)
        succ_adduct.append(adduct[idx])
        succ_ccs.append(ccs[idx])

    return succ_smiles, succ_adduct, succ_ccs, coordinates


def get_smiles_atom_set(smiles):
    """
    * Gets the collection of all elements in the dataset
    *
    * Attributes
    * ----------
    * smiles    : The SMILES string of the molecule
    *
    * Returns
    * -------
    * all_atoms : Element set
    """
    all_atoms = []
    for i in range(len(smiles)):
        try:
            mol = Chem.MolFromSmiles(smiles[i])
        except Exception as e:
            logger.error(f"Error in molecule {i}: {e}")
            continue
        try:
            all_atoms += [atom.GetSymbol() for atom in mol.GetAtoms()]
            all_atoms = list(set(all_atoms))
        except Exception as e:
            logger.error(f"Error in molecule {i}: {e}")
            continue
    all_atoms.sort()
    return all_atoms


def convert_to_graph(
    smi_list, coordinates, all_atoms, atom_radii=ATOM_RADII, atom_masses=ATOM_MASSES
):
    """
    * Construct a graph dataset for the input molecular dataset
    *
    * Attributes
    * ----------
    * smi_lst    : The SMILES string list of the molecule
    * Coordinate : The coordinate data of each molecule
    * all_atoms  : A Set of all elements in a SMILES dataset
    *
    * Returns
    * -------
    * adj           : Adjacency matrix
    * features      : The feature vector of each node(Atom) in the graph
    * edge_features : The feature vector(one-hot encode) of each edge(Bond) in the graph
    """
    ##################################################
    # !!!Note!!!: you need to record the radius and mass of the atoms that exist in the atomic set
    ##################################################
    # The atomic radius and atomic mass are normalized
    atom_radii = min_max_scaling(atom_radii)
    atom_masses = min_max_scaling(atom_masses)

    adj, features, edge_features = [], [], []
    idx = -1

    # Traverses all the SMILES strings
    for smi in smi_list:
        idx += 1
        mol = Chem.MolFromSmiles(smi)  # Converts a SMILES string to a MOL object

        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(mol)  # The adjacency matrix of MOL is obtained

        contribution_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
        asa_contribution = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
        tpsa_contribition = rdMolDescriptors._CalcTPSAContribs(mol)

        # Characteristics of structural chemical bonds(Edge)
        one_edge_features = edge_feature(mol)
        edge_features.append(one_edge_features)

        # Construct vectors for each atom of the molecule
        iFeatureTmp = []
        for atom in mol.GetAtoms():
            iFeatureTmp.append(
                atom_feature(
                    atom=atom,
                    index=idx,
                    coordinates=coordinates,
                    all_atoms=all_atoms,
                    atom_radius=atom_radii,
                    atom_mass=atom_masses,
                    tpsa_contribition=tpsa_contribition,
                    asa_contribution=asa_contribution,
                    crippen_contribution=contribution_crippen,
                )
            )
        features.append(np.array(iFeatureTmp))
        adj.append(iAdjTmp)

    features = np.asarray(features, dtype=object)
    edge_features = np.asarray(edge_features, dtype=object)
    return adj, features, edge_features


def atom_feature(
    atom,
    index,
    coordinates,
    all_atoms,
    atom_radius,
    atom_mass,
    tpsa_contribition,
    asa_contribution,
    crippen_contribution,
):
    """
    * Component atom vector
    *
    * Attributes
    * ----------
    * atom        : Atom object
    * index       : The molecule to which the atom belongs and the index of the molecule in
                    SMILES list
    * coordinates  : The 3D coordinates of all atoms of each molecule
    * all_atoms   : A Set of all elements in a SMILES dataset
    * atom_radius : Atomic radius dictionary
    * atom_mass   : Atomic mass dictionary
    * mol   : RDKit molecule object
    *
    * Returns
    * -------
    * adj           : Adjacency matrix
    * features      : The feature vector of each node(Atom) in the graph
    * edge_features : The feature vector(one-hot encode) of each edge(Bond) in the graph
    """
    return np.array(
        # Atomic Type (One-Hot)
        one_hot_encode(atom.GetSymbol(), all_atoms)
        # Atomic Degree (One-Hot)
        + one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4])
        # Atomic radius  Atomic mass (float)
        + [atom_radius[atom.GetSymbol()], atom_mass[atom.GetSymbol()]]
        # Atomic is in Ring ? (One-Hot)
        + one_hot_encode(atom.IsInRing(), [0, 1])
        # Coordinate (float)
        + list(coordinates[index][atom.GetIdx()])
        # Atomic charge (float)
        + [float(atom.GetFormalCharge())]
        # atom chiral (float)
        + [float(atom.HasProp("_ChiralityPossible"))]
        # Atomic mass (float)
        + [float(atom.GetMass() / 100)]
        # Get LabuteASA and TPSA contribution
        + [tpsa_contribition[atom.GetIdx()]]
        + [asa_contribution[atom.GetIdx()]]
        # Get Crippen contribution
        + [crippen_contribution[atom.GetIdx()][0]]
        + [crippen_contribution[atom.GetIdx()][1]]
    )


def edge_feature(iMol):
    """
    * Constructing edge feature matrix
    *
    * Attributes
    * ----------
    * iMol : Molecular objects
    *
    * Returns
    * -------
    * Edge_feature : Edge feature matrix of molecules
    """

    Edge_feature = []
    count = 0
    for bond in iMol.GetBonds():
        count += 1
        bond_feature = np.array(
            # One-hot 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE
            one_hot_encode(bond.GetBondTypeAsDouble(), [1, 1.5, 2, 3])
            # One-hot 1.0 for False, 1.0 for True for the following
            + one_hot_encode(bond.GetIsConjugated(), [False, True])
            + one_hot_encode(bond.IsInRing(), [False, True])
            + one_hot_encode(
                bond.GetStereo(),
                [
                    Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOANY,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS,
                    Chem.rdchem.BondStereo.STEREOTRANS,
                ],
            )
            + one_hot_encode(
                bond.GetBondDir(),
                [
                    Chem.rdchem.BondDir.NONE,
                    Chem.rdchem.BondDir.ENDUPRIGHT,
                    Chem.rdchem.BondDir.ENDDOWNRIGHT,
                ],
            )
        )
        # Edge features is added twice because the graph is bidirectional (from
        # node A > node B and node B -> node A)
        Edge_feature.append(bond_feature)
        Edge_feature.append(bond_feature)
    Edge_feature = np.array(Edge_feature)
    Edge_feature = Edge_feature.astype(float)
    return Edge_feature


class MyDataset(Dataset):
    """
    * Constructing edge feature matrix
    *
    * Attributes
    * ----------
    * features      : Node feature matrix
    * adj           : Adjacency matrix
    * edge_features : Edge feature matrix
    * ccs           : CCS of molecules
    """

    def __init__(self, features, adj, edge_features, ccs, **kwargs):
        self.features = features
        self.adj = adj
        self.edge_features = edge_features
        self.ccs = ccs
        super().__init__(**kwargs)

    def read(self):
        return [
            Graph(
                x=self.features[i],
                a=self.adj[i],
                e=self.edge_features[i],
                y=float(self.ccs[i]),
            )
            for i in range(len(self.adj))
        ]
