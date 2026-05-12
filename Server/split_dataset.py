import scanpy as sc
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ================= Configuration =================
SOURCE_FILE = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad"
TRAIN_FILE = "train.h5ad"
TEST_FILE = "test.h5ad"
TEST_SIZE = 0.2 # 20% as the test set (rigorous scientific research usually uses 20% or 10%)
RANDOM_STATE = 42 # Fixed random seed to ensure the same result every time
# =======================================

def split_data():
    print("[Data set division] Start dividing the data set into training set and test set...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_dir, SOURCE_FILE)
    train_path = os.path.join(base_dir, TRAIN_FILE)
    test_path = os.path.join(base_dir, TEST_FILE)

    if not os.path.exists(source_path):
        print(f"Source file not found: {source_path}")
        return

    # 1. Load original data
    print(f"Reading {SOURCE_FILE}...")
    try:
        adata = sc.read_h5ad(source_path)
    except Exception as e:
        print(f"Reading failed: {e}")
        return
        
    n_cells = adata.n_obs
    print(f"Original data: {n_cells} cells x {adata.n_vars} genes")

    # 2. Execute partitioning
    print(f"Dividing according to the ratio of {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%}...")
    indices = np.arange(n_cells)
    # Use sklearn for random partitioning to ensure uniform distribution
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 3. Create subset object
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()

    print(f" - Training set (Train): {train_adata.n_obs} cells")
    print(f" - Test set (Test): {test_adata.n_obs} cells")

    # 4. Save file
    print(f"Saving {TRAIN_FILE}...")
    train_adata.write(train_path)
    
    print(f"Saving {TEST_FILE}...")
    test_adata.write(test_path)
    
    print("Data set division completed!")
    print("Suggestions for subsequent operations:")
    print("1. Run train_imputation.py (using train.h5ad)")
    print("2. Run evaluate_imputation.py (using test.h5ad)")

if __name__ == "__main__":
    split_data()