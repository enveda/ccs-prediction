import json
import logging

import numpy as np
from pandas import DataFrame, read_parquet

from mol2ccs.constants import ALL_ATOMS, ALLOWED_ADDUCTS, Parameter
from mol2ccs.graph import (
    MyDataset,
    convert_to_graph,
    generate_coordinates,
    get_smiles_atom_set,
)
from mol2ccs.model import (
    load_model_from_file,
    mol2ccs_model,
    predict,
    train,
)
from mol2ccs.utils import (
    calculate_adduct_descriptors,
    calculate_fingeprint,
    calculate_metrics,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def wrapper_predict(
    input_file,
    parameter_path,
    model_file_h5,
    output_file,
    is_evaluate=1,
    coordinates_present=False,
    coordinates_col_name="coordinates",
    smiles_col_name="smiles",
    adduct_col_name="adduct",
    ccs_col_name="ccs",
    ccs_type_col_name="ccs_type",
    mol_type_col_name="mol_type",
    dimer_col_name="dimer",
    drugtax_col_name="drugtax",
):
    file_data = read_parquet(input_file)
    smiles_list, adduct, ccs, ccs_type, mol_type, dimer, drugtax = (
        file_data[smiles_col_name].values,
        file_data[adduct_col_name].values,
        file_data[ccs_col_name].values,
        file_data[ccs_type_col_name].values,
        file_data[mol_type_col_name].values,
        file_data[dimer_col_name].values,
        file_data[drugtax_col_name].tolist(),
    )

    if coordinates_present:
        coordinates = file_data[coordinates_col_name].values
    logger.info(f"Read data: {len(smiles_list)}")

    param = None

    with open(parameter_path, "r") as file:
        parameter_dict = json.load(file)
        max_coor = parameter_dict["max_coor"]
        min_coor = parameter_dict["min_coor"]
        param = Parameter(
            max_coor=max_coor,
            min_coor=min_coor,
        )

    if not coordinates_present:
        smiles_list, adduct, ccs, Coordinate = generate_coordinates(
            smiles=smiles_list, adduct=adduct, ccs=ccs, all_atoms=ALL_ATOMS
        )
        logger.info("3D coordinates generated successfully ")
    else:
        Coordinate = coordinates

    for i in range(len(Coordinate)):
        Coordinate[i] = (np.array(Coordinate[i]) - param.min_coor) / (
            param.max_coor - param.min_coor
        )

    adj, features, edge_features = convert_to_graph(
        smi_list=smiles_list,
        all_atoms=ALL_ATOMS,
        coordinates=Coordinate,
    )
    dataset = MyDataset(features, adj, edge_features, ccs)
    logger.info("Graph & Adduct dataset completed")

    ECC_Model = load_model_from_file(model_file_h5)
    logger.info("Model loading completed")

    fingerprint_pred = [calculate_fingeprint(smiles=smi) for smi in smiles_list]

    descriptors_pred = calculate_adduct_descriptors(smiles_list, adduct)

    results = predict(
        model=ECC_Model,
        dataset=dataset,
        descriptors=descriptors_pred,
        adduct_pred=adduct,
        ccs_type_pred=ccs_type,
        mol_type_pred=mol_type,
        dimer_pred=dimer,
        fingerprint_pred=fingerprint_pred,
        drugtax_pred=drugtax,
    )

    data = {
        "smiles": smiles_list,
        "adduct": adduct,
        "ccs": ccs,
        "pred_ccs": results,
    }

    df = DataFrame(data)

    df.to_parquet(output_file, index=False)
    logger.info("CCS predicted completed")

    if is_evaluate == 1:
        re_Metrics = calculate_metrics(ccs, results)
        return re_Metrics


def wrapper_train(
    ifile,
    parameter_path,
    ofile,
    epochs,
    batch_size,
    verbose,
    all_atoms=ALL_ATOMS,
    adduct_set=ALLOWED_ADDUCTS,
    dropout_rate=0.0,
    coordinates_present=False,
    coordinates_col_name="coordinates",
    smiles_col_name="smiles",
    adduct_col_name="adduct",
    ccs_col_name="ccs",
    ccs_type_col_name="ccs_type",
    mol_type_col_name="mol_type",
    dimer_col_name="dimer",
    drugtax_col_name="drugtax",
):
    """
    * Train
    *
    * Attributes
    * ----------
    * ifile         : File path for storing the data of smiles and adduct
    * ParameterPath : Save path of related data parameters
    * ofile         : File path where the model is stored
    """
    # this was being initialized as an empty list if not provided
    # but it shouldn't be done in the definition
    all_atoms = [] if all_atoms is None else ALL_ATOMS
    adduct_set = [] if adduct_set is None else ALLOWED_ADDUCTS

    # Read the smiles adduct CCS in the file
    file_data = read_parquet(ifile)

    smiles_list, adduct, ccs, ccs_type, mol_type, dimer, drugtax = (
        file_data[smiles_col_name].values,
        file_data[adduct_col_name].values,
        file_data[ccs_col_name].values,
        file_data[ccs_type_col_name].values,
        file_data[mol_type_col_name].values,
        file_data[dimer_col_name].values,
        file_data[drugtax_col_name].tolist(),
    )

    logger.info(f"Read data: {len(smiles_list)}")
    if coordinates_present:
        Coordinate = file_data[coordinates_col_name].values

    # If the user does not enter the number of elements, then the default is the set of
    # all elements in the training set
    if len(all_atoms) == 0:
        all_atoms = get_smiles_atom_set(
            smiles_list
        )  # Calculate the set of elements used in the training set

    """1. Graph data generation"""
    if not coordinates_present:
        # 3D conformation of the input SMILES
        smiles_list, adduct, ccs, Coordinate = generate_coordinates(
            smiles_list, adduct, ccs, all_atoms
        )
        logger.info("3D coordinates generated successfully ")
    else:
        logger.info("3D coordinates read from file")

    # Data normalization of the generated coordinate data
    ALL_DATA = []
    for i in Coordinate:
        for ii in i:
            ALL_DATA.append(ii[0])
            ALL_DATA.append(ii[1])
            ALL_DATA.append(ii[2])
    max_coor, min_coor = np.max(ALL_DATA), np.min(ALL_DATA)

    for i in range(len(Coordinate)):
        Coordinate[i] = (np.array(Coordinate[i]) - min_coor) / (max_coor - min_coor)

    # Adduct set
    if len(adduct_set) == 0:
        adduct_set = list(set(list(adduct)))
        adduct_set.sort()

    logger.info(f"All element types : {all_atoms}")
    logger.info(f"All adduct types : {adduct_set}")

    # Construct Graph from the input data
    adj, features, edge_features = convert_to_graph(
        smi_list=smiles_list,
        all_atoms=all_atoms,
        coordinates=Coordinate,
    )

    DataSet = MyDataset(features, adj, edge_features, ccs)
    logger.info(f"Build graph data successfully. Dataset: {DataSet} len({len(DataSet)})")

    """2: Fingerprints"""
    logger.info("Calculating fingerprints")

    fingerprint_list = [calculate_fingeprint(smi) for smi in smiles_list]

    """3: Descriptors"""
    logger.info("Calculating descriptors")
    descriptors = calculate_adduct_descriptors(smiles_list, adduct)

    # Storing parameters in objects
    rw = Parameter(
        max_coor=max_coor,
        min_coor=min_coor,
    )

    # export json with parameters
    with open(parameter_path, "w") as file:
        json.dump(rw.__dict__, file)

    """4: Model training"""
    # Production of models for training
    ECC_Model = mol2ccs_model(dataset=DataSet, dropout_rate=dropout_rate)

    # Training Model
    ECC_Model = train(
        Model=ECC_Model,
        dataset=DataSet,
        descriptors_train=descriptors,
        fingerprint_train=fingerprint_list,
        ccs_type_train=ccs_type,
        mol_type_train=mol_type,
        dimer_train=dimer,
        drugtax_train=drugtax,
        adduct_train=adduct,
        epochs=epochs,
        ofile=ofile,
        batch_size=batch_size,
        verbose=verbose,
    )
    return ECC_Model
