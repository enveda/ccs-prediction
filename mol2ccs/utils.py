"""Utility functions for the project."""

import logging
from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mol2ccs.constants import (
    ALLOWED_ADDUCTS,
    ALLOWED_CCS_TYPES,
    ALLOWED_DIMER_TYPES,
    ALLOWED_MOL_TYPES,
)

# Always print floating point numbers using fixed point
# notation, in which case numbers equal to zero in the
# current precision will print as zero.
np.set_printoptions(suppress=True)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(formatter)
logger.setLevel(logging.INFO)


def read_data(
    filename,
):
    """
    * Attributes
    * ----------
    * filename : input file
    *
    * Returns
    * -------
    * smiles : SMILES
    * adduct : Adduct ([M+H]+,[M+Na]+,[M-H]]-, etc.)
    * ccs    : CCS
    """
    data = pd.read_csv(filename)

    logger.info(f"Read data from {filename}")

    smiles = np.array(data["SMILES"])
    adduct = np.array(data["Adduct"])
    ccs = np.array(data["True CCS"])
    return smiles, adduct, ccs


def min_max_scaling(data):
    """
    * Attributes
    * ----------
    * data : Data that need to be normalized
    *
    * Returns
    * -------
    * data : Normalized data
    """
    data_list = [data[i] for i in data]
    max_data, min_data = np.max(data_list), np.min(data_list)
    for i in data:
        data[i] = (data[i] - min_data) / (max_data - min_data)
    return data


def calculate_adduct_descriptors(
    smiles_list,
    adducts,
):
    descriptors_list = []

    for smiles, adduct in zip(smiles_list, adducts):
        mol = generate_mol_from_smiles(smiles)

        # get molecular weight
        mw, original_mw = ExactMolWt(mol), ExactMolWt(mol)

        # remove the adduct weight from the molecular weight if adduct is [M]-
        # or add the adduct is [M]+
        if adduct == "[M+H]+":
            mw += 1.007276
        elif adduct == "[M-H]-":
            mw -= 1.007276
        elif adduct == "[M+Na]+":
            mw += 22.989218
        elif adduct == "[M+K]+":
            mw += 38.9632
        elif adduct == "[M+NH4]+":
            mw += 18.033823
        elif adduct == "[2M-H]-":
            mw = (mw * 2) - 1.007276
        elif adduct == "[2M+H]+":
            mw = (mw * 2) + 1.007276
        elif adduct == "[2M+Na]+":
            mw = (mw * 2) + 22.989218
        elif adduct == "[M+H-H2O]+":
            mw += 1.007276 - 18.01528
        else:
            raise ValueError(f"Adduct {adduct} not recognized")

        descriptors_list.append([mw, original_mw])

    return descriptors_list


def prepare_data(
    batch,
    ltd_index,
    adduct,
    descriptors,
    fingerprint,
    ccs_type,
    mol_type,
    dimer,
    drugtax,
):
    # Calculate one hot encoding for adduct, ccs_type and mol_type
    batch_adduct_one_hot = np.array(
        [
            one_hot_encode(adduct[ltd_index + ltd_index_i], ALLOWED_ADDUCTS)
            for ltd_index_i in range(len(batch[1]))
        ]
    )

    batch_ccs_type_one_hot = np.array(
        [
            one_hot_encode(ccs_type[ltd_index + ltd_index_i], ALLOWED_CCS_TYPES)
            for ltd_index_i in range(len(batch[1]))
        ]
    )

    batch_mol_type_one_hot = np.array(
        [
            one_hot_encode(mol_type[ltd_index + ltd_index_i], ALLOWED_MOL_TYPES)
            for ltd_index_i in range(len(batch[1]))
        ]
    )

    batch_dimer_one_hot = np.array(
        [
            one_hot_encode(dimer[ltd_index + ltd_index_i], ALLOWED_DIMER_TYPES)
            for ltd_index_i in range(len(batch[1]))
        ]
    )

    # Calculate one hot encoding for adduct, ccs_type and mol_type
    batch_descriptors = np.array(
        [descriptors[ltd_index + ltd_index_i] for ltd_index_i in range(len(batch[1]))]
    )

    # Get the fingerprints for the batch
    batch_fingerprint = np.array(
        [fingerprint[ltd_index + ltd_index_i] for ltd_index_i in range(len(batch[1]))]
    )

    # Get the fingerprints for the batch
    batch_drugtax = np.array(
        [drugtax[ltd_index + ltd_index_i] for ltd_index_i in range(len(batch[1]))]
    )

    # Concatenate all the input
    return (
        batch_fingerprint,
        batch_ccs_type_one_hot,
        batch_mol_type_one_hot,
        batch_dimer_one_hot,
        batch_descriptors,
        batch_drugtax,
        # baseline features
        batch_adduct_one_hot,
        batch[0][0],  # Node features
        batch[0][1],  # Adjacency matrix
        batch[0][2],  # Edge features
    )


def one_hot_encode(x, class_order):
    """One-hot encode a list of samples.

    Args:
        x (list): List of samples.
        n_classes (int): Number of classes.

    Returns:
        np.array: One-hot encoded samples.
    """
    if x not in class_order:
        raise ValueError(f"Unknown class: {x}")

    empty_array = [0 for _ in range(len(set(class_order)))]

    empty_array[class_order.index(x)] = 1

    return empty_array


@cache
def calculate_fingeprint(smiles: str):
    """
    * Calculate the fingerprint of a molecule
    *
    * Attributes
    * ----------
    * mol : Molecule
    *
    * Returns
    * -------
    * fp : Fingerprint
    """
    mol = generate_mol_from_smiles(smiles)

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=256)
        return np.array(fp)
    except Exception as e:
        raise e


@cache
def generate_mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)

    except Exception as e:
        raise ValueError(f"Could not generate mol from SMILES: {smiles}") from e


def calculate_metrics(y, r):
    """
    * The gap between the predicted result and the real value of the evaluation model
    *
    * Attributes
    * ----------
    * y : y_true
    * r : y_pred
    """
    RelativeError = [abs(y[i] - r[i]) / y[i] for i in range(len(y))]
    R2_Score = linregress(y, r)
    return R2_Score, np.median(RelativeError) * 100


def performance_evaluation(input_true, input_pred, verbose=True, output_type="dataframe"):
    """
    Evaluate the performance of the CCS prediction model. The evaluated metrics include:
    MAE: mean absolute error
    MSE: mean squared error
    RMSE: root mean squared error
    R2: R squared
    Mean Percent RSD: mean relative standard CCS deviation
    Median Percent RSD: median relative standard CCS deviation
    Max Percent RSD: maximum relative standard CCS deviation
    Min Percent RSD: minimum relative standard  CCS deviation
    Pearson's correlation: Pearson's correlation coefficient
    Spearman's correlation: Spearman's correlation coefficient

    :input_true: list or numpy array of true CCS values
    :input_pred: list or numpy array of predicted CCS values
    :verbose: boolean, whether to print the performance metrics

    :return: dictionary of performance metrics and numpy array of relative standard CCS deviation
    """

    results_dict = {}
    try:
        results_dict["MAE"] = mean_absolute_error(input_true, input_pred)
    except ValueError:
        results_dict["MAE"] = np.nan

    try:
        results_dict["MSE"] = mean_squared_error(input_true, input_pred)
    except ValueError:
        results_dict["MSE"] = np.nan

    try:
        results_dict["RMSE"] = np.sqrt(results_dict["MSE"])
    except ValueError:
        results_dict["RMSE"] = np.nan

    try:
        results_dict["R2"] = linregress(input_true, input_pred).rvalue ** 2
    except ValueError:
        results_dict["R2"] = np.nan

    try:
        percent_RSD = 100 * np.abs((input_true - input_pred) / input_true)
        results_dict["Mean Percent RSD"] = np.mean(percent_RSD)
        results_dict["Median Percent RSD"] = np.median(percent_RSD)
        results_dict["Max Percent RSD"] = np.max(percent_RSD)
        results_dict["Min Percent RSD"] = np.min(percent_RSD)
    except ValueError:
        percent_RSD = np.nan
        results_dict["Mean Percent RSD"] = np.nan
        results_dict["Median Percent RSD"] = np.nan
        results_dict["Max Percent RSD"] = np.nan
        results_dict["Min Percent RSD"] = np.nan

    try:
        results_dict["Pearson's correlation"] = pearsonr(input_true, input_pred)[0]
    except ValueError:
        results_dict["Pearson's correlation"] = np.nan

    try:
        results_dict["Spearman's correlation"] = spearmanr(input_true, input_pred)[0]
    except ValueError:
        results_dict["Spearman's correlation"] = np.nan

    if verbose is True:
        for key, value in results_dict.items():
            print(key, ":", round(float(value.item()), 2))

    if output_type == "dataframe":
        return (
            pd.DataFrame.from_dict(results_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "Metric", 0: "Value"})
        ), percent_RSD
    return results_dict, percent_RSD
