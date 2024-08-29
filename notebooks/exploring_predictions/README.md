# Predictions analysis

Predictions are available and can be directly downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11199061.svg)](https://doi.org/10.5281/zenodo.11199061). The files should be unzipped and placed in the `data` directory.

Then, the predictions can be loaded and analyzed using the notebooks in this directory. The notebooks are ordered in the following way:

├── ccsbase_both.ipynb
├── ccsbase_train_test_metlin.ipynb
├── combined.ipynb
├── metlin_both.ipynb
├── metlin_train_test_ccsbase.ipynb
└── utils.py

The first two notebooks starting by the "ccsbase" prefix analyze the predictions for the ccsbase database. The next two notebooks starting by the "metlin" prefix analyze the predictions for the metlin database. The last notebook "combined.ipynb" analyzes the predictions for the combined database. The "utils.py" file contains common functions that are used in the notebooks.