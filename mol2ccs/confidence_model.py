import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
)
from sklearn.model_selection import ParameterSampler

from mol2ccs.constants import (
    ALLOWED_ADDUCTS,
    ALLOWED_CCS_TYPES,
    ALLOWED_DIMER_TYPES,
    ALLOWED_MOL_TYPES,
)
from mol2ccs.utils import calculate_fingeprint, one_hot_encode


def smiles_featurizer(
    data,
    smiles_col="smiles",
):
    fingerprints = [calculate_fingeprint(s) for s in data[smiles_col]]
    return pd.DataFrame(
        [*fingerprints], index=data.index, columns=[f"fp_{i}" for i in range(len(fingerprints[0]))]
    )


def optional_featurizer(
    data,
    feature_col,
    allowed_values=ALLOWED_ADDUCTS,
):
    one_hot = [one_hot_encode(a, allowed_values) for a in data[feature_col]]
    return pd.DataFrame([*one_hot], index=data.index, columns=allowed_values)


def prediction_featurizer(
    data,
    pred_ccs_col="pred_ccs",
):
    return data[pred_ccs_col]


def target_featurizer(
    df: pd.DataFrame,
    true_ccs_col: Optional[str] = "ccs",
    pred_ccs_col: Optional[str] = "pred_ccs",
    target_type: str = "accurate_ccs",
    error_threshold: Optional[float] = None,
) -> pd.Series:
    """Create target for confidence model training."""
    if target_type == "binary_error":
        y = df[true_ccs_col] - df[pred_ccs_col]
        y = y.abs()
        # if y is less than 5% of the true CCS, set to True, else False
        y = y < error_threshold * df[true_ccs_col]

        # Print the number of True and False values
        print(
            f"""
            Number of labels: {y.value_counts()}
            % of True labels: {y.mean() * 100:.2f}%
            % of False labels: {(1 - y.mean()) * 100:.2f}%
            """
        )

    elif target_type == "error":
        y = df[true_ccs_col] - df[pred_ccs_col]
        y = y.abs()
    else:
        raise ValueError(f"unknown target_type: {target_type}")

    return y


class RFStructureConfidenceModel:
    def __init__(
        self,
        estimator_type: str = "classifier",
        target_type: str = "binary_error",
        structure_field: str = "smiles",
        adduct_field: str = "adduct",
        true_ccs_field: str = "ccs",
        pred_ccs_field: str = "pred_ccs",
        ccs_type_field: str = "ccs_type",
        mol_type_field: str = "mol_type",
        dimer_field: str = "dimer",
    ):
        self.estimator_type = estimator_type
        self.target_type = target_type
        self.structure_field = structure_field
        self.adduct_field = adduct_field
        self.true_ccs_field = true_ccs_field
        self.pred_ccs_field = pred_ccs_field
        self.ccs_type_field = ccs_type_field
        self.mol_type_field = mol_type_field
        self.dimer_field = dimer_field

    def save(self, save_dir: str):
        """Save model and attributes to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=False)

        init_params = {
            "estimator_type": self.estimator_type,
            "target_type": self.target_type,
            "structure_field": self.structure_field,
            "adduct_field": self.adduct_field,
            "true_ccs_field": self.true_ccs_field,
            "pred_ccs_field": self.pred_ccs_field,
        }

        with open(save_path / "init_params.json", "w") as f:
            f.write(json.dumps(init_params))

        if hasattr(self, "model"):
            with open(str(save_path / "model.pkl"), "wb") as f:
                pickle.dump(self.model, f)

    @classmethod
    def load(cls, load_dir: str):
        """Load model from saved artifacts."""

        load_path = Path(load_dir)

        with open(load_path / "init_params.json", "r") as f:
            init_params = json.load(f)

        obj = cls(**init_params)

        if (load_path / "model.pkl").exists():
            with open(str(load_path / "model.pkl"), "rb") as f:
                mod = pickle.load(f)
            obj.model = mod

        return obj

    def fit(
        self,
        data: pd.DataFrame,
        estimator_params: Optional[Union[None, dict]] = None,
        hpo_params: Optional[Union[None, dict]] = None,
        error_threshold: Optional[float] = None,
    ):
        """Fit the random forest model given the data.
        Instantiate the estimator using `estimator_params` if provided.
        Perform HPO via random grid sampling using `hpo_params` if provided.
        Set the fitted estimator as `self.model`.

        If `hpo_params` is provided, it may have the following keys
        'n_hpo_iter': number of hpo iterations (will default to 50)
        'val_frac': fraction of structures to split off for val set (default .1)
        estimator init params mapped to lists of values defining the hpo space
        """
        feature_df = self.featurize(data)
        targets = target_featurizer(
            data,
            true_ccs_col=self.true_ccs_field,
            pred_ccs_col=self.pred_ccs_field,
            target_type=self.target_type,
            error_threshold=error_threshold,
        )

        nan_feature_rows = feature_df.isna().any(axis=1)
        logging.info(f"dropping {nan_feature_rows.sum()} rows due to NaNs")
        feature_df = feature_df.loc[~nan_feature_rows]
        targets = targets.loc[~nan_feature_rows]

        if hpo_params is not None:
            n_hpo_iter = hpo_params.pop("n_hpo_iter", 50)
            val_frac = hpo_params.pop("val_frac", 0.1)
            structures = list(data[self.true_ccs_field].unique())
            n_val_structures = int(val_frac * len(structures))
            val_structures = set(np.random.choice(structures, size=n_val_structures, replace=False))

            val_mask = data[self.true_ccs_field].isin(val_structures)
            train_features = feature_df.loc[~val_mask]
            train_targets = targets.loc[~val_mask]
            val_features = feature_df.loc[val_mask]
            val_targets = targets.loc[val_mask]

            param_list = list(ParameterSampler(hpo_params, n_iter=n_hpo_iter, random_state=42))
        elif estimator_params is not None:
            param_list = []
            best_params = estimator_params
        else:
            raise ValueError("one of `estimator_params` and `hpo_params` must not be None")

        min_metric = np.inf
        best_model = None

        for i, s in enumerate(param_list):
            model = self._init_sklearn_model(self.estimator_type, s)
            model.fit(train_features, train_targets)
            val_preds = self._predict_sklearn_model(model, val_features)
            val_metric = self._eval_predictions(self.estimator_type, val_preds, val_targets)
            if val_metric < min_metric:
                logging.info(f"new best params: sample={i}, metric={val_metric}, params={s}")
                min_metric = val_metric
                best_params = s

        best_model = self._init_sklearn_model(self.estimator_type, best_params)
        best_model.fit(feature_df, targets)

        self.model = best_model

    def predict_confidence(self, data: pd.DataFrame) -> pd.Series:
        confs = pd.Series(data=np.zeros(len(data)), index=data.index)
        feature_df = self.featurize(data)
        nan_feature_rows = feature_df.isna().any(axis=1)
        confs[~nan_feature_rows] = self._predict_sklearn_model(
            self.model, feature_df.loc[~nan_feature_rows]
        )
        return confs

    def predict_label(self, data: pd.DataFrame) -> pd.Series:
        labels = pd.Series(data=np.zeros(len(data)), index=data.index)
        feature_df = self.featurize(data)
        nan_feature_rows = feature_df.isna().any(axis=1)
        labels[~nan_feature_rows] = self._predict_class_sklearn_model(
            self.model, feature_df.loc[~nan_feature_rows]
        )
        return labels

    def featurize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Featurize input data for the model."""
        smiles_features = smiles_featurizer(
            data,
            smiles_col=self.structure_field,
        )
        adduct_features = optional_featurizer(
            data,
            feature_col=self.adduct_field,
            allowed_values=ALLOWED_ADDUCTS,
        )

        ccs_type_features = optional_featurizer(
            data,
            feature_col=self.ccs_type_field,
            allowed_values=ALLOWED_CCS_TYPES,
        )

        mol_type_features = optional_featurizer(
            data,
            feature_col=self.mol_type_field,
            allowed_values=ALLOWED_MOL_TYPES,
        )

        dimer_features = optional_featurizer(
            data,
            feature_col=self.dimer_field,
            allowed_values=ALLOWED_DIMER_TYPES,
        )

        prediction_features = prediction_featurizer(
            data,
            pred_ccs_col=self.pred_ccs_field,
        )

        feature_df = pd.concat([smiles_features, adduct_features, prediction_features, ccs_type_features, mol_type_features, dimer_features], axis=1)

        # Ensure there are no NaNs in the feature matrix
        assert not feature_df.isna().any().any(), "NaNs found in feature matrix"

        return feature_df

    def _init_sklearn_model(self, estimator_type, params):
        if estimator_type == "regressor":
            model_cls = RandomForestRegressor
        elif estimator_type == "classifier":
            model_cls = RandomForestClassifier
        else:
            raise ValueError(f"unknown estimator_type: {estimator_type}")

        return model_cls(**params, n_jobs=-1)

    def _predict_sklearn_model(self, model, data):
        if isinstance(model, RandomForestRegressor):
            return model.predict_confidence(data).clip(0, 1)
        elif isinstance(model, RandomForestClassifier):
            return model.predict_proba(data)[:, 1]
        else:
            raise ValueError(f"unknown model type: {type(model)}")

    def _predict_class_sklearn_model(self, model, data):
        if isinstance(model, RandomForestRegressor):
            return model.predict(data).clip(0, 1)
        elif isinstance(model, RandomForestClassifier):
            return model.predict(data)
        else:
            raise ValueError(f"unknown model type: {type(model)}")

    def _eval_predictions(self, estimator_type, preds, targets):
        if estimator_type == "regressor":
            metric_fn = mean_absolute_error
        elif estimator_type == "classifier":
            metric_fn = log_loss
        else:
            raise ValueError(f"unknown estimator_type: {estimator_type}")

        return metric_fn(targets, preds)
