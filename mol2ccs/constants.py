from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

TENSORBOARD_LOG_DIR = Path("tensorboard_logs")


"""Set of parameters for each atom."""

ALLOWED_ADDUCTS = [
    "[M+H]+",
    "[2M+H]+",
    "[M+Na]+",
    "[2M+Na]+",
    "[M-H]-",
    "[2M-H]-",
    "[M+K]+",
    "[M+H-H2O]+",
    "[M+NH4]+",
]

ALLOWED_CCS_TYPES = ["TIMS", "DT", "TW"]

ALLOWED_MOL_TYPES = ["small molecule", "lipid", "peptide", "carbohydrate"]

ALLOWED_DIMER_TYPES = ["Monomer", "Dimer"]


FEATURES_HIDDEN_LAYERS = [256, 512, 512, 256, 128]


ECC_OUTPUT_LAYER_SIZE = 16


ATOM_RADII = {
    "N": 71,
    "Se": 116,
    "F": 64,
    "Co": 111,
    "O": 63,
    "As": 121,
    "Br": 114,
    "Cl": 99,
    "S": 103,
    "C": 75,
    "P": 111,
    "I": 133,
    "H": 32,
    "Na": 155,
    "K": 196,
}
ATOM_MASSES = {
    "N": 14.00674,
    "Se": 78.96,
    "F": 18.9984032,
    "Co": 58.933195,
    "As": 74.92160,
    "O": 15.9994,
    "Br": 79.904,
    "Cl": 35.453,
    "S": 32.065,
    "C": 12.0107,
    "P": 30.973762,
    "I": 126.90447,
    "Na": 22.989769,
    "K": 39.0983,
    "H": 1.00794,
}
ALL_ATOMS = sorted(ATOM_RADII.keys())


@dataclass
class Parameter:
    """
    * Constants used within the code
    *
    * Attributes
    * ----------
    * max_coor   : The maximum value in all coordinate data
    * min_coor   : The minimum value in all coordinate data
    """

    max_coor: int
    min_coor: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_coor": self.max_coor,
            "min_coor": self.min_coor,
        }


sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
