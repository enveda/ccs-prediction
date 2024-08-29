# Data prepraration

This directory contains notebooks that are used to prepare the data for the machine learning models. The notebooks are ordered in the following way:

├── 1_ccsbase.ipynb
├── 1_database_similarity.ipynb
├── 1_metlin.ipynb
├── 2_data_splits.ipynb
├── 3_3dfile_ccs_value_editing.ipynb
├── 3_threed-stuff-and-split-combined.ipynb

The first notebooks starting by the "1_" prefix prepare the ccsbase and metlin database.
Since we cannot provide the raw data for licensing reasons, we include the links on how to download the data in the notebooks. After the data is processed in the same format (columns), the database similarity is calculated in the notebook "1_database_similarity.ipynb". Lastly, the data is split into training, validation, and test sets in the notebook "2_data_splits.ipynb". The last two notebooks are used to prepare the 3D files and the CCS values for the machine learning models.