import scanpy as sc
import numpy as np
import os

H5AD_FILE = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad"

if not os.path.exists(H5AD_FILE):
    print("Not found train.h5ad")
    exit()

print(f"Reading {H5AD_FILE}...")
adata = sc.read_h5ad(H5AD_FILE)
genes = adata.var_names.tolist()

# 保存
np.save("gene_vocab.npy", genes)
print(f"Saved gene_vocab.npy (contains {len(genes)} genes)")