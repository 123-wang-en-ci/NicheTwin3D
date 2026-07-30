# Nicheformer Training & Fine-Tuning Guide with Custom H5AD Data

If you want to use your own spatial transcriptomics dataset in `.h5ad` format to re-train the foundation model and fine-tune downstream classifiers, please follow the detailed steps below.

---

## Step 1: Prepare Your Custom H5AD Dataset

Ensure your custom `.h5ad` file adheres to the following specifications:

### 1. Expression Data

Nicheformer relies on aggregating raw expression counts within local spatial niches.

- Your `adata.X` or `adata.layers['counts']` must contain **raw expression counts**, rather than highly variable gene-filtered or over-normalized values.
- The script automatically checks for a `counts` layer. If not present, it will copy `adata.X` to `counts`:

  ```python
  if 'counts' not in self.adata.layers:
      self.adata.layers['counts'] = self.adata.X.copy()
  ```

### 2. Spatial Coordinates

The model constructs KNN spatial neighborhood graphs using 2D or 3D cell coordinates.

- Ensure cell spatial coordinates are saved in `adata.obsm['spatial']` or `adata.obsm['X_spatial']`.
- The matrix shape should be `(n_cells, n_dimensions)`, where dimensions are typically 2 (X, Y) or 3 (X, Y, Z).

### 3. Ground Truth Labels

Downstream fine-tuning tasks (cell type annotation and tissue region segmentation) require supervision signals. Therefore, your cell metadata `adata.obs` must include ground-truth label columns.

- **Cell Type Column**: e.g., column names such as `cell_type`, `cell_type_annotation`, or `cell_ontology_class`.
- **Region Column**: e.g., tissue layer or anatomical region annotations like `clust_annot` or `ccf_region_name`.

---

## Step 2: Nicheformer Foundation Model Pre-Training

### 1. Configure `train_nicheformer.py`

Open `train_nicheformer.py`, locate the configuration section inside the `train()` function, and modify the following parameters:

```python
# 1. Replace with your custom H5AD dataset path (recommended in the same directory)
H5AD_PATH = os.path.join(current_dir, "your_custom_dataset.h5ad") 

# 2. Output weights filename (keep default or change if desired)
OUTPUT_PATH = os.path.join(current_dir, "nicheformer_weights.pth")

# 3. Hardware & Hyperparameter tuning (adjust according to your GPU VRAM)
BATCH_SIZE = 8          # Reduce to 4 or 2 if GPU memory is limited (< 12GB)
MAX_EPOCHS = 100        # Training epochs. 50-100 epochs usually converge for large datasets
LR = 3e-4               # Learning rate
N_NEIGHBORS = 20        # Number of neighbors for KNN spatial graph construction (20-30 recommended)
```

### 2. Run Pre-Training Command

Execute in your terminal:

```bash
python train_nicheformer.py
```

### 3. Key Outputs from This Step

* **`gene_vocab.npy`**: The script automatically extracts all gene names from `adata.var_names` in your custom H5AD file and saves them in a fixed order dictionary.

  > [!IMPORTANT]
  > This step is critical! It ensures that `server.py` and the Unity client align gene token indexing and embedding dimensions perfectly with the pre-trained weights during runtime queries.

* **`nicheformer_weights.pth`**: Output weights file for the pre-trained foundation model.

---

## Step 3: Downstream Multi-Task Classifier Fine-Tuning

After obtaining the foundation model weights, we use the foundation model as a feature extractor to train dedicated residual classification heads on custom cell type and region labels.

### 1. Configure `train_downstream.py`

Open `train_downstream.py`, locate the top configuration area and `main()` function, and update the parameters:

```python
# ================= Top Configuration Area =================
# 1. Replace with the column names in your adata.obs
CELL_TYPE_COL = "your_cell_type_column"   # e.g., "cell_type"
REGION_COL    = "your_region_column"      # e.g., "clust_annot"

# 2. Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001

# ================= main() Function Area =================
def main():
    # 3. Ensure loading your custom dataset and the generated foundation weights
    h5ad_path  = os.path.join(current_dir, "your_custom_dataset.h5ad")
    model_path = os.path.join(current_dir, "nicheformer_weights.pth")
```

### 2. Run Fine-Tuning Command

Execute in your terminal:

```bash
python train_downstream.py
```

### 3. Key Outputs from This Step

* **`embeddings_cache.npy`**: Cached high-dimensional spatial niche embeddings to accelerate training iterations.
* **`cell_type_model.pth` & `cell_type_model_labels.pkl`**: Trained classification head weights and category label mappings for cell types.
* **`region_model.pth` & `region_model_labels.pkl`**: Trained classification head weights and category label mappings for tissue region segmentation.

### 4. Force Clearing Old Feature Cache

> [!WARNING]
> The backend server checks whether `embeddings_cache.npy` exists before running gene imputation and zero-shot clustering. If you switch to a new dataset (with different cell counts or gene counts) without **manually deleting** the old cache file, the server will load obsolete embeddings, causing severe data misalignment.
>
> **Action**: Before running `server.py` with a new dataset, **always delete `embeddings_cache.npy` manually**!

### 5. Update Dataset Filename in `server.py`

Open `server.py` and modify the `H5AD_FILENAME` variable to your custom file:

```python
H5AD_FILENAME = "your_custom_dataset.h5ad"  # Replace with your dataset filename
```

---

## Step 4: Deployment & Unity Co-Simulation

After completing the steps above, your directory will contain custom-trained weights and data. You can now deploy and interact with them in Unity:

1. **Start Communication Server**:
   Execute in terminal:

   ```bash
   python server.py
   ```

   `server.py` will load `gene_vocab.npy`, foundation weights, and downstream classifier weights, and automatically generate `unity_cell_data.csv` for the frontend.

2. **Interact in Unity**:
   Launch Play mode in Unity (or run the executable). Click feature buttons to stream spatial data between Unity and Python, invoking your fine-tuned models for real-time inference and 3D visualization!
