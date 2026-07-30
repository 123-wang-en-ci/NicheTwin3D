# Guide to Nicheformer Training and Fine-Tuning with Custom H5AD Data

If you want to use your own spatial transcriptomics dataset (`.h5ad` format) to re-train the foundation model and fine-tune downstream classifiers, please follow the detailed steps below.

---

## Step 1: Prepare Custom H5AD Dataset

Ensure that your custom `.h5ad` file complies with the following specifications:

### 1. Expression Data

Nicheformer relies on raw expression counts aggregated across local niches.

- Your `adata.X` or `adata.layers['counts']` should contain **raw expression counts (Raw Counts)**, rather than highly variable gene-filtered or over-normalized data.
- The script automatically checks this. If there is no `counts` layer, it will automatically copy `adata.X` to `counts`:

  ```python
  if 'counts' not in self.adata.layers:
      self.adata.layers['counts'] = self.adata.X.copy()

### 2. Spatial Coordinates

The model requires 2D or 3D spatial coordinates of cells to construct KNN neighborhood graphs.

- Ensure that cell coordinates are stored in `adata.obsm['spatial']` or `adata.obsm['X_spatial']`.
- The coordinate matrix shape should be `(number_of_cells, dimensions)`, where dimensions are typically 2 (X, Y) or 3 (X, Y, Z).

### 3. Ground Truth Labels

Downstream fine-tuning tasks (cell type annotation and region tissue semantic segmentation) require supervision signals. Therefore, your cell metadata `adata.obs` must include label columns for classification:

- **Cell Type Label Column**: e.g., column names such as `cell_type`, `cell_type_annotation`, or `cell_ontology_class`.
- **Region Label Column**: e.g., brain region annotations or tissue structural layer columns such as `clust_annot` or `ccf_region_name`.

---

## Step 2: Nicheformer Foundation Model Self-Supervised Pre-Training

### 1. Modify Configuration in `train_nicheformer.py`

Open `train_nicheformer.py`, locate the configuration section in the `train()` function, and make the following modifications:

```python
# 1. Replace with your custom H5AD dataset path (recommended to place under the Server directory)
H5AD_PATH = os.path.join(current_dir, "your_custom_dataset.h5ad") 

# 2. Output weight filename (no need to change)
OUTPUT_PATH = os.path.join(current_dir, "nicheformer_weights.pth")

# 3. Hardware and hyperparameter fine-tuning (adjust according to your VRAM size)
BATCH_SIZE = 8          # If GPU VRAM is small (e.g., < 12GB), reduce to 4 or 2
MAX_EPOCHS = 100        # Training epochs. For large datasets, 50-100 epochs are sufficient to converge
LR = 3e-4               # Learning rate
N_NEIGHBORS = 20        # Number of neighbors for KNN spatial graph construction, typically 20-30
```

### 2. Run Pre-Training Command

Execute the following in your terminal:

```bash
python train_nicheformer.py
```

### 3. Key Outputs Generated in This Step

* **`gene_vocab.npy`**: The script automatically extracts all genes corresponding to `adata.var_names` in your custom H5AD and saves them as a local dictionary file in a fixed order.
  > [!IMPORTANT]
  > This step is critical! It ensures that `server.py` and the Unity client align token dictionary indices and dimensions perfectly with pre-trained weights when encoding input genes.

* **`nicheformer_weights.pth`**: The foundation model pre-trained weight file output upon completion.

---

## Step 3: Downstream Multi-Task Classifier Fine-Tuning

After obtaining custom foundation model weights, we use the pre-trained model as a feature extractor to train dedicated fully-connected residual network heads on your custom cell type and region labels.

### 1. Modify Configuration in `train_downstream.py`

Open `train_downstream.py`, locate the top configuration section and the `main()` function, and update them with your custom data parameters:

```python
# ================= Top Configuration Section =================
# 1. Replace with the actual label column names in your h5ad file's adata.obs
CELL_TYPE_COL = "your_cell_type_column"   # e.g., "cell_type"
REGION_COL    = "your_region_column"      # e.g., "clust_annot"

# 2. Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001

# ================= main() Function Section =================
def main():
    # 3. Ensure loading your custom dataset and the newly generated foundation weights
    h5ad_path  = os.path.join(current_dir, "your_custom_dataset.h5ad")
    model_path = os.path.join(current_dir, "nicheformer_weights.pth")
```

### 2. Run Fine-Tuning Command

Execute the following in your terminal:

```bash
python train_downstream.py
```

### 3. Key Outputs Generated in This Step

* **`embeddings_cache.npy`**: Cached high-dimensional spatial niche embeddings of cells to accelerate subsequent multi-epoch iterations.
* **`cell_type_model.pth` & `cell_type_model_labels.pkl`**: Trained custom cell type classification head weights and corresponding text label mappings.
* **`region_model.pth` & `region_model_labels.pkl`**: Trained custom tissue region segmentation classification head weights and label mappings.

### 4. Force Clear Legacy Embeddings Cache (Critical)

Before running gene imputation or zero-shot clustering, the backend AI engine checks if a local `embeddings_cache.npy` file exists. If you switch to a new dataset (where cell counts or gene counts have changed) without **manually deleting** the old cache file, the backend will directly load and use stale feature embeddings from the previous dataset, causing severe misalignment in clustering and imputation results.

> [!CAUTION]
> **Action: Before switching to new data and running `server.py`, be sure to manually delete `embeddings_cache.npy` in the folder!**

### 5. Modify Dataset and Column Configurations in `server.py`

Open `server.py` and modify the configuration variables at the top of the file:

```python
H5AD_FILENAME = "your_custom_dataset.h5ad"   # Replace with your new dataset filename
CELL_TYPE_COLUMN = "your_cell_type_column"  # Replace with your cell type column name (must match train_downstream.py)
```

---

## Step 4: Deployment and Unity Integration

After completing the steps above, your local directory will contain a complete set of trained weights tailored to your custom dataset. You can now deploy the new dataset and models to Unity for 3D visualization:

1. **Start Communication Server**:
   Execute the following in your terminal:
   ```bash
   python server.py
   ```
   `server.py` will automatically load your newly trained `gene_vocab.npy`, foundation weights, and classifier weights, and automatically generate and transmit `unity_cell_data.csv` to the frontend.

2. **Experience in Unity**:
   Launch Play mode in Unity or run the standalone application. Clicking feature buttons will stream custom data to Python in real time, where the backend calls your fine-tuned model for inference and returns live 3D rendering updates!

