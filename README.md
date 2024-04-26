# Evaluating the generalizability of graph neural networks for predicting collision cross section

This repository contains code and data described in detail in our paper (Engler *et al.*, 2024).

## Table of Contents

* [Citation](#citation)
* [Reproducibility](#reproducibility)
* [Data](#data)

### Citation
If you have found our manuscript useful in your work, please consider citing:

>  Engler Hart*, C., Preto*, A. J., Chanana*, S., Healey, D., Kind, T., Domingo-Fernandez, D. (2024).
Evaluating the generalizability of graph neural networks for predicting collision cross section. *bioRxiv*.

## Reproducibility

## Installation for development

```shell
poetry install
poetry run pre-commit install
```

### Train the models

See the commands in the `Makefile` to train the models. Run them as `make train-metlin-test-metlin`

### Reproduce results

Run the notebooks located in the `notebooks` corresponding to each analysis.
There are two folders:
- data_processing: notebooks to process the METLIN-CCS and CCSBase and make the data splits
- reproduce_figures: the name of the notebooks indicates which notebook can reproduce which figures of the manuscript manuscript.
- exploring_predictions: notebooks to explore the predictions of the models in detail

## Data

Datasets and predictions are available and can be directly downloaded from [![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/). The files should be unzipped and placed in the `data` directory.


## Predicting CCS
Train the model based on your own training dataset with [wrapper_train] and predict with [wrapper_predict](mol2ccs/train_and_predict.py#L23) function.

The baseline original repositories are:
- SigmaCCS: https://github.com/zmzhang/SigmaCCS
- GraphCCS: https://github.com/tingxiecsu/GraphCCS
