# Evaluating the generalizability of graph neural networks for predicting collision cross section

This repository contains code and data described in detail in our paper (Engler *et al.*, 2024).

## Table of Contents

* [Citation](#citation)
* [Reproducibility](#reproducibility)
* [Data](#data)

### Citation
If you have found our manuscript useful in your work, please consider citing:

>  Engler Hart*, C., Preto*, A. J., Chanana*, S., Healey, D., Kind, T., Domingo-Fernandez, D. (2024).
Evaluating the generalizability of graph neural networks for predicting collision cross section. *Chemrxiv*.

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

Predictions are available and can be directly downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11199061.svg)](https://doi.org/10.5281/zenodo.11199061). The files should be unzipped and placed in the `data` directory.

The original datasets are available here:
- CCSBase: https://ccsbase.net/query
- Metlin: https://metlin.scripps.edu/


## Predicting CCS
Train the model based on your own training dataset with [wrapper_train] and predict with [wrapper_predict](mol2ccs/train_and_predict.py#L23) function.

The baseline original repositories are:
- SigmaCCS: https://github.com/zmzhang/SigmaCCS
- GraphCCS: https://github.com/tingxiecsu/GraphCCS

## References
>  Ross, D. H., Cho, J. H., and Xu, L. (2020). Breaking down structural diversity for comprehensive prediction of ion-neutral collision cross sections. Analytical chemistry, 92(6), 4548-4557. https://doi.org/10.1021/acs.analchem.9b05772

> Baker, E. S., Hoang, C., Uritboonthai, W., Heyman, H. M., Pratt, B., MacCoss, M., et al. (2023). METLIN-CCS: an ion mobility spectrometry collision cross section database. Nature methods, 20(12), 1836-1837. https://doi.org/10.1038/s41592-023-02078-5

> Guo, R., Zhang, Y., Liao, Y., Yang, Q., Xie, T., Fan, X., et al. (2023). Highly accurate and large-scale collision cross sections prediction with graph neural networks. Communications Chemistry, 6(1), 139. https://doi.org/10.1038/s42004-023-00939-w

> Xie, T., Yang, Q., Sun, J., Zhang, H., Wang, Y., and Lu, H. Large-Scale Prediction of Collision Cross-Section with Graph Convolutional Network for Compound Identification.


