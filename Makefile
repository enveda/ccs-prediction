add-3d-coords:
	poetry run python scripts/generate-3d-coords.py \
		--input-path "ccs-prediction/combined_metlin_ccsbase_train_test_split.parquet" \
		> add-3d-coords.out 2>&1

train-metlin-test-metlin:
	poetry run python scripts/train-test.py \
	--prefix "train-metlin-test-metlin" \
	--train-input-file "ccs-prediction/metlin_train_3d.parquet" \
	--test-input-file "ccs-prediction/metlin_test_3d.parquet" \
	--coordinates-column-name "coordinates" \
	--coordinates-present \
    --smiles-column-name "smiles" \
	--adduct-column-name "adduct" \
	--ccs-column-name "ccs" \
	--epochs 400 \
	--dropout-rate 0.1 \
	--train \
	> train-metlin-test-metlin.out 2>&1

# train on metlin-train and test on all ccsbase
# so we have to pass the previous model output file and
# the parameter file
train-metlin-test-ccsbase:
	poetry run python scripts/train-test.py \
	--prefix "train-metlin-test-ccsbase" \
	--train-input-file "ccs-prediction/metlin_train_3d.parquet" \
	--test-input-file "ccs-prediction/ccsbase_3d.parquet" \
	--parameter-path "parameter/parameter-train-metlin-test-metlin.json" \
	--model-output-file "model/train-metlin-test-metlin.h5" \
	--coordinates-column-name "coordinates" \
	--coordinates-present \
	--smiles-column-name "smiles" \
	--adduct-column-name "adduct" \
	--ccs-column-name "ccs" \
	--dropout-rate 0.1 \
	--epochs 400 \
	> train-metlin-test-ccsbase.out 2>&1


train-ccsbase-test-ccsbase:
	poetry run python scripts/train-test.py \
	--prefix "train-ccsbase-test-ccsbase" \
	--train-input-file "ccs-prediction/ccsbase_train_3d.parquet" \
	--test-input-file "ccs-prediction/ccsbase_test_3d.parquet" \
	--coordinates-column-name "coordinates" \
	--coordinates-present \
	--smiles-column-name "smiles" \
	--adduct-column-name "adduct" \
	--ccs-column-name "ccs" \
	--dropout-rate 0.1 \
	--epochs 400 \
	--train \
	> train-ccsbase-test-ccsbase.out 2>&1

# train on ccsbase-train and test on all metlin
train-ccsbase-test-metlin:
	poetry run python scripts/train-test.py \
	--prefix "train-ccsbase-test-metlin" \
	--train-input-file "ccs-prediction/ccsbase_train_3d.parquet" \
	--test-input-file "ccs-prediction/metlin_3d.parquet" \
	--parameter-path "parameter/parameter-train-ccsbase-test-ccsbase.json" \
	--model-output-file "model/train-ccsbase-test-ccsbase.h5" \
	--coordinates-column-name "coordinates" \
	--coordinates-present \
	--smiles-column-name "smiles" \
	--adduct-column-name "adduct" \
	--ccs-column-name "ccs" \
	--dropout-rate 0.1 \
	--epochs 400 \
	> train-ccsbase-test-metlin.out 2>&1


train-combined-test-combined:
	poetry run python scripts/train-test.py \
	--prefix "train-combined-test-combined" \
	--train-input-file "ccs-prediction/metlin_ccsbase_train_3d.parquet" \
	--test-input-file "ccs-prediction/metlin_ccsbase_test_3d.parquet" \
	--coordinates-column-name "coordinates" \
	--smiles-column-name "smiles" \
	--adduct-column-name "adduct" \
	--ccs-column-name "ccs" \
	--epochs 400 \
	--dropout-rate 0.1 \
	--train \
	> train-combined-test-combined.out 2>&1

# grid-search:
# 	poetry run python scripts/grid-search.py \
# 	--prefix "grid-search" \
# 	--train-input-file "ccs-prediction/combined_metlin_ccsbase_train_3d_spilt_0.parquet" \
# 	--test-input-file "ccs-prediction/combined_metlin_ccsbase_test_3d.parquet" \
# 	--epochs 300
