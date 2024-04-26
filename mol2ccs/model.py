"""mol2ccs model."""

import logging

import numpy as np
from keras import Model, callbacks, initializers
from keras.layers import Dense, Dropout, Input, concatenate
from keras.models import load_model
from keras.optimizers import Adam
from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalSumPool

from mol2ccs.constants import (
    ALLOWED_ADDUCTS,
    ALLOWED_CCS_TYPES,
    ALLOWED_DIMER_TYPES,
    ALLOWED_MOL_TYPES,
    ECC_OUTPUT_LAYER_SIZE,
    FEATURES_HIDDEN_LAYERS,
    TENSORBOARD_LOG_DIR,
)
from mol2ccs.utils import prepare_data

np.set_printoptions(suppress=True)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.handlers[0].setFormatter(formatter)
logger.setLevel(logging.INFO)

tensorboard_callback = callbacks.TensorBoard(
    log_dir=TENSORBOARD_LOG_DIR.resolve().as_posix(),
    histogram_freq=1,
    write_graph=True,
)

early_stopping_callback = callbacks.EarlyStopping(
    monitor="loss", min_delta=0.1, patience=5, verbose=1
)

# Create a new instance of the GlorotUniform initializer with a seed
initializer = initializers.GlorotUniform(seed=42)


def load_model_from_file(ModelfilePath):
    Model = load_model(
        ModelfilePath,
        custom_objects={
            "ECCConv": ECCConv,
            "GlobalSumPool": GlobalSumPool,
        },
    )
    return Model


def mol2ccs_model(
    dataset,
    adduct_set=ALLOWED_ADDUCTS,
    ccs_type_set=ALLOWED_CCS_TYPES,
    mol_type_set=ALLOWED_MOL_TYPES,
    dimer_set=ALLOWED_DIMER_TYPES,
    fingerprint_size=256,
    drugtax_size=35,
    descriptors_size=2,
    hidden_layers_features=FEATURES_HIDDEN_LAYERS,
    dropout_rate=0.2,
    dropout_type="Dropout",
):
    """
    * Constructing mol2ccs model
    *
    * Attributes
    * ----------
    * dataset    : Input Graph data of the model
    *
    * Returns
    * -------
    * model : The constructed mol2ccs model
    """
    # Ensure that the dropout rate is a float
    if isinstance(dropout_rate, str):
        dropout_rate = float(dropout_rate)

    Kernel_Network = [64, 64, 64, 64]
    F = dataset.n_node_features
    E = dataset.n_edge_features
    X_in = Input(shape=(None, F))
    A_in = Input(shape=(None, None))
    E_in = Input(shape=(None, None, E))

    # Concatenated features to the ones from the graph (GNN)
    adduct_in = Input(shape=(len(adduct_set),))

    fingerprint_in = Input(shape=(fingerprint_size,))  # the size of the fingerprint (see utils.py)

    ccs_type_in = Input(shape=(len(ccs_type_set),))

    mol_type_in = Input(shape=(len(mol_type_set),))

    dimer_in = Input(shape=(len(dimer_set),))

    descriptor_in = Input(shape=(descriptors_size,))

    drugtax_in = Input(shape=(drugtax_size,))

    """Edge-conditioned convolutional layer (ECC)"""
    gnn_output = ECCConv(
        ECC_OUTPUT_LAYER_SIZE,
        Kernel_Network,
        activation="relu",
        kernel_regularizer="l2",
    )([X_in, A_in, E_in])
    gnn_output = ECCConv(
        ECC_OUTPUT_LAYER_SIZE,
        Kernel_Network,
        activation="relu",
        kernel_regularizer="l2",
    )([gnn_output, A_in, E_in])
    gnn_output = ECCConv(
        128,
        Kernel_Network,
        activation="relu",
        kernel_regularizer="l2",
    )([gnn_output, A_in, E_in])

    # A global sum pooling layer. Pools a graph by computing the sum of its node features.
    gnn_output = GlobalSumPool()(gnn_output)

    """"Feature input and hidden layers"""

    logger.info(
        f"""
        Adding ({len(hidden_layers_features)}) hidden layers to the
        features: {hidden_layers_features}\n
        Dropout rate applied on features' layers: {dropout_rate}\n
        Dropout type: {dropout_type}
        """
    )  # noqa

    feature_input = concatenate(
        [
            # additional features
            fingerprint_in,
            ccs_type_in,
            mol_type_in,
            dimer_in,
            descriptor_in,
            drugtax_in,
            adduct_in,
        ]
    )

    # Hidden layers combining the features (adduct, instrument, etc.) with the GNN output
    layers_dict = {}

    for i, neuron_n in enumerate(hidden_layers_features):
        # Add a dense layer with the specified number of neurons and a ReLU activation function
        layers_dict[f"Dense{i}"] = Dense(neuron_n, activation="relu", kernel_regularizer="l2")
        # Add a dropout layer with the specified dropout rate
        layers_dict[f"Dropout{i}"] = Dropout(dropout_rate)

    # Initialize the first layer
    mlp_vector = layers_dict["Dense0"](feature_input)
    mlp_vector = layers_dict["Dropout0"](mlp_vector)

    for i in range(1, len(hidden_layers_features)):
        mlp_vector = layers_dict[f"Dense{i}"](mlp_vector)
        mlp_vector = layers_dict[f"Dropout{i}"](mlp_vector)

    # Concatenate the output of the features layers with the GNN output
    graph_and_features = concatenate(
        [
            mlp_vector,
            gnn_output,
            adduct_in,
        ],
    )

    # Create the final layers with hidden_layers_final with dropout similar as before

    """Same architecture as SigmaCCS joining features and output of the GNN"""
    Dense1 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense2 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense3 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense4 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense5 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense6 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense7 = Dense(384, activation="relu", kernel_regularizer="l2")
    Dense8 = Dense(384, activation="relu", kernel_regularizer="l2")
    output_layer = Dense(
        1,
        activation="relu",
    )

    stack_output = output_layer(
        Dense8(Dense7(Dense6(Dense5(Dense4(Dense3(Dense2(Dense1(graph_and_features))))))))
    )

    model = Model(
        inputs=[
            # additional features
            fingerprint_in,
            ccs_type_in,
            mol_type_in,
            dimer_in,
            descriptor_in,
            drugtax_in,
            # baseline feature
            adduct_in,
            # 3D graph data
            X_in,
            A_in,
            E_in,
        ],
        outputs=stack_output,
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0001,
        ),
        loss="mse",
        metrics=["mse"],
    )

    logger.info(model.summary())

    return model


def train(
    Model,
    dataset,
    descriptors_train,
    adduct_train,
    fingerprint_train,
    ccs_type_train,
    mol_type_train,
    dimer_train,
    drugtax_train,
    ofile,
    epochs,
    batch_size,
    verbose,
):
    """
    * Training model
    *
    * Attributes
    * ----------
    * Model      : The constructed mol2ccs model
    * dataset_train : Input Graph  data for training
    * adduct_tr  : Input Adduct data for training
    * adduct_set : Adduct Set
    * descriptors: Descriptors
    * fingerprint_list : fingerprints
    * ccs_type : instrument type
    * mol_type : Molecule type
    * drugtax : DrugTax fingerprints
    * EPOCHS     : Number of epochs to iterate over the dataset. By
                    default (None) iterates indefinitely
    * BATCHS     : Size of the mini-batches
    * Vis        : Trained model
    *
    * Returns
    * -------
    * Model : The constructed mol2ccs model
    """
    best_loss = float("inf")
    improvement_threshold = 0.05
    patience = 20
    impatience = 0

    for epoch in range(epochs):
        # Random scrambling of data

        # set a different seed for each epoch
        np.random.seed(epoch)
        shuffled_idx = np.random.permutation(len(dataset))

        """Apply the same permutation to all the data"""
        dataset = dataset[shuffled_idx]
        adduct_train = list(np.array(adduct_train)[shuffled_idx])
        descriptors_train = list(np.array(descriptors_train)[shuffled_idx])
        fingerprint_train = list(np.array(fingerprint_train)[shuffled_idx])
        ccs_type_train = list(np.array(ccs_type_train)[shuffled_idx])
        mol_type_train = list(np.array(mol_type_train)[shuffled_idx])
        dimer_train = list(np.array(dimer_train)[shuffled_idx])
        drugtax_train = list(np.array(drugtax_train)[shuffled_idx])

        # Loading data
        loader_tr = BatchLoader(
            dataset,
            batch_size=batch_size,
            # It's set to one epoch because the goal is to iterate
            # over the entire dataset once and modify it
            epochs=1,
            shuffle=False,
        )

        loader_train_data = ()
        current_batch_index = 0

        # The Graph data of each molecule is spliced with adduct data
        for batch in loader_tr.load():
            input_batch = (
                prepare_data(
                    adduct=adduct_train,
                    batch=batch,
                    descriptors=descriptors_train,
                    ccs_type=ccs_type_train,
                    fingerprint=fingerprint_train,
                    dimer=dimer_train,
                    drugtax=drugtax_train,
                    ltd_index=current_batch_index,
                    mol_type=mol_type_train,
                ),
                # Target (CCS values) [batch_size, 1]
                batch[1],
            )

            loader_train_data += (input_batch,)  # Input data

            # Update the batch index and step counter
            current_batch_index += len(batch[1])

        # Pack all the batches for all the epochs into a generator
        loader_train_data = (i for i in loader_train_data)

        # Train
        Model.fit(
            loader_train_data,
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=1,
            # Verbose=2 (one line per epoch), Verbose=1 (progress bar), Verbose=0 (silent)
            verbose=verbose,
            callbacks=[tensorboard_callback, early_stopping_callback],
        )

        loss = float(Model.history.history["loss"][-1])
        logger.info(f"Epoch {epoch}/{epochs}. Loss: {round(loss, 4)}")

        # Early stopping does not apply for the first 20 epochs
        if epoch < 150:
            continue

        if loss < (best_loss - improvement_threshold):
            best_loss = loss
            impatience = 0
            Model.save(ofile)

        else:
            impatience += 1
            if impatience > patience:
                logger.warning(
                    f"Early stopping: Loss has not improved for {patience} epochs \
                        by less than {improvement_threshold}. Stopping training."
                )
                Model.save(ofile)
                return Model

    return Model


def predict(
    model,
    dataset,
    descriptors,
    adduct_pred,
    fingerprint_pred,
    ccs_type_pred,
    mol_type_pred,
    dimer_pred,
    drugtax_pred,
):
    """Predicting values CCS with a trained model."""
    loader = BatchLoader(dataset, batch_size=1, epochs=1, shuffle=False)
    loader_data = ()
    ltd_index = 0

    # The Graph data of each molecule is spliced with adduct data
    for batch in loader.load():
        batch_input = (
            prepare_data(
                adduct=adduct_pred,
                batch=batch,
                descriptors=descriptors,
                ccs_type=ccs_type_pred,
                dimer=dimer_pred,
                drugtax=drugtax_pred,
                fingerprint=fingerprint_pred,
                ltd_index=ltd_index,
                mol_type=mol_type_pred,
            ),
            batch[1],
        )

        loader_data += (batch_input,)
        ltd_index += len(batch[1])

    loader_data = (batch for batch in loader_data)

    y_true = []
    y_pred = []
    for batch in loader_data:
        inputs, target = batch

        predictions = model(inputs, training=False)  # predict
        pred = np.array(predictions[0])
        y_pred.append(pred[0])
        y_true.append(target[0])

        if len(y_pred) % 1000 == 0:
            logger.info(f"Predicted: {len(y_pred)} molecules")
    return y_pred
