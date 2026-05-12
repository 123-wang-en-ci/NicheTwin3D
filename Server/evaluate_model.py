import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import sys
import os
import warnings

# Ignore some warnings from Scanpy
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Reference your model_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_engine import NicheformerEngine

class NicheformerEvaluator:
    def __init__(self, h5ad_path, model_path, cell_type_col="cell_type", region_col="region"):
        self.engine = NicheformerEngine()
        self.engine.load_data(h5ad_path)
        self.engine.load_model(model_path)
        
        #Load downstream classifier
        self.engine.load_downstream_models()
        
        self.adata = self.engine.adata
        self.cell_type_col = cell_type_col
        self.region_col = region_col
        
        # Precompute Embeddings
        if self.engine.embeddings_cache is None:
            self.engine._precompute_embeddings()

    def evaluate_cell_type_annotation(self):
        print("\n" + "="*40)
        print("Assessment Task 1: Cell Type Annotation")
        
        if self.cell_type_col not in self.adata.obs:
            print(f"Error: Column '{self.cell_type_col}'" not found in adata.obs)
            print(f"Available column names: {list(self.adata.obs.columns)}")
            return

        try:
            pred_ids, legend = self.engine.predict_cell_types()
            if len(pred_ids) == 0: 
                print("The prediction result is empty, skip evaluation")
                return
                
            # Extract real tags (including n/a)
            true_labels_all = self.adata.obs[self.cell_type_col].astype(str).values
            
            #The ID output by the mapping model is a name string
            id_to_name = {item['id']: str(item['name']) for item in legend}
            pred_names_all = np.array([id_to_name.get(pid, "Unknown") for pid in pred_ids])
            
            # Filter samples without real labels
            valid_mask = self.adata.obs[self.cell_type_col].notna()
            true_valid = true_labels_all[valid_mask]
            pred_valid = pred_names_all[valid_mask]
            
            # Use the same random seed to divide 20% of the test set from the valid samples
            _, y_test_true, _, y_test_pred = train_test_split(
                true_valid, pred_valid, test_size=0.2, random_state=42
            )
            
            # Calculate accuracy on test set only
            acc = accuracy_score(y_test_true, y_test_pred)
            f1 = f1_score(y_test_true, y_test_pred, average='weighted')
            
            print(f"Accuracy : {acc:.4f}")
            print(f"F1-Score : {f1:.4f}")
        except Exception as e:
            print(f"Evaluation error: {e}")

    def evaluate_tissue_segmentation(self):
        print("\n" + "="*40)
        print("Assessment task 2: Tissue Segmentation")
        
        if self.region_col not in self.adata.obs:
            print(f"Error: Column '{self.region_col}'" not found in adata.obs)
            print(f"Please check the available column names below and modify the GT_REGION variable in the code:")
            print(f"{list(self.adata.obs.columns)}")
            return

        try:
            pred_ids, region_names_list = self.engine.segment_tissue_regions()
            
            #Convert the elements in region_names_list to str to prevent type mismatch
            region_names_list = [str(x) for x in region_names_list]
            pred_names_all = np.array([region_names_list[rid] for rid in pred_ids])
            
            #Extract real tags
            true_labels_all = self.adata.obs[self.region_col].astype(str).values
            
            # Filter out n/a invalid samples
            valid_mask = ~self.adata.obs[self.region_col].astype(str).str.lower().isin(['n/a', 'nan', 'none'])
            
            true_valid = true_labels_all[valid_mask]
            pred_valid = pred_names_all[valid_mask]
            
            # Use the same random seed to divide 20% of the test set from the valid samples
            _, y_test_true, _, y_test_pred = train_test_split(
                true_valid, pred_valid, test_size=0.2, random_state=42
            )
            
            # Calculate accuracy on test set only
            acc = accuracy_score(y_test_true, y_test_pred)
            f1 = f1_score(y_test_true, y_test_pred, average='weighted')
            
            print(f"Accuracy : {acc:.4f}")
            print(f"F1-Score : {f1:.4f}")
        except Exception as e:
            print(f"Evaluation error: {e}")

    def evaluate_zero_shot_clustering(self, n_clusters=10, alt_ref_col=None):
        print("\n" + "="*40)
        print(f"Evaluation Task 3: Zero-shot Clustering (K={n_clusters})")
        
        if self.cell_type_col not in self.adata.obs:
            print(f"Error: Reference column '{self.cell_type_col}'" not found)
            return
        
        try:
            cluster_labels, _ = self.engine.run_zero_shot_clustering(n_clusters=n_clusters)
            
            # Compare cell type labels
            true_labels = self.adata.obs[self.cell_type_col].values
            ari  = adjusted_rand_score(true_labels, cluster_labels)
            nmi  = normalized_mutual_info_score(true_labels, cluster_labels)
            print(f"[vs cell type] ARI : {ari:.4f}")
            print(f"[vs cell type] NMI : {nmi:.4f}")
            
            # Compare spatial area labels (better reflects Nicheformer’s spatial awareness capabilities)
            if alt_ref_col and alt_ref_col in self.adata.obs:
                region_labels = self.adata.obs[alt_ref_col].astype(str)
                valid = region_labels != 'n/a'
                if valid.sum() > 0:
                    ari_r = adjusted_rand_score(region_labels[valid], cluster_labels[valid])
                    nmi_r = normalized_mutual_info_score(region_labels[valid], cluster_labels[valid])
                    print(f"[vs spatial brain area] ARI : {ari_r:.4f}")
                    print(f"[vs spatial brain area] NMI: {nmi_r:.4f}")
        except Exception as e:
            print(f"❌ Clustering evaluation error: {e}")

    def evaluate_gene_imputation(self, n_test_genes=50):
        print("\n" + "="*40)
        print(f"Evaluation Task 4: Gene Imputation (Gene Imputation, Top {n_test_genes} Genes)")
        
        # --- [Repair Core] More robust gene selection logic ---
        hv_genes = []
        try:
            # 1. Try using Scanpy hypervariable genes (may crash)
            print("Try to select hypervariable genes...")
            #Create temporary objects to avoid modifying the original data
            temp_adata = self.adata.copy()
            # If it is an integer, it may be Raw Counts, using seurat_v3 flavor
            if np.issubdtype(temp_adata.X.dtype, np.integer):
                sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_test_genes, flavor='seurat_v3')
            else:
# Otherwise log first and then calculate
                sc.pp.log1p(temp_adata)
                sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_test_genes)
            
            hv_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
            del temp_adata # Release memory
            
        except Exception as e:
            print(f"Scanpy failed to calculate hypervariable genes ({str(e)}), switch to backup plan...")
            
        # 2. Backup plan: If no gene is selected (or crashes), select the gene with the highest average expression
        if len(hv_genes) < n_test_genes:
            print("Use the genes with the highest average expression level as the test set (Fallback Strategy)")
            # Calculate average expression
            if hasattr(self.adata.X, 'toarray'):
                means = np.array(self.adata.X.mean(axis=0)).flatten()
            else:
                means = np.array(self.adata.X.mean(axis=0)).flatten()
                
            # Get Top N index
            top_indices = np.argsort(means)[-n_test_genes:]
            hv_genes = self.adata.var_names[top_indices].tolist()
        
        hv_genes = hv_genes[:n_test_genes]
        print(f"Selected test genes: {hv_genes[:5]} ...")
            
        print(f"Selected test genes: {hv_genes[:5]} ...")
            
        pearson_list = []
        rmse_list = []
        mae_list = []
        
        # 【Start modification】Copy and replace the loop block below
        for gene in tqdm(hv_genes):
            try:
                # 1. Get predicted values (Softplus output)
                pred_vals = self.engine.predict_gene_expression(gene)

                # 2. Get the real value
                if isinstance(self.adata[:, gene].X, np.ndarray):
                    true_vals = self.adata[:, gene].X.flatten()
                else:
                    true_vals = self.adata[:, gene].X.toarray().flatten()

                # 3. Logarithmic (Log1p) for fair comparison
                true_vals_log = np.log1p(true_vals)
                pred_vals_log = np.log1p(pred_vals) 
                
                # 4. Calculate Pearson correlation coefficient
                corr, _ = pearsonr(true_vals_log, pred_vals_log)
                
                # 【Key Fix】This line was missing before, causing the list to be empty!
                if not np.isnan(corr):
                    pearson_list.append(corr)
                
                # 5. Calculate RMSE & MAE (it is recommended to use Log value comparison)
                rmse = np.sqrt(mean_squared_error(true_vals_log, pred_vals_log))
                mae = mean_absolute_error(true_vals_log, pred_vals_log)

                rmse_list.append(rmse)
                mae_list.append(mae)

            except Exception as e_inner:
                #【Debugging enhancement】Print specific errors, no longer be "dumb"
                print(f"Gene {gene} calculation error: {e_inner}")
                continue 
        
        # Aggregate results
            
        # Aggregate results
        if len(pearson_list) > 0:
            avg_pearson = np.mean(pearson_list)
            avg_rmse = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            
            print(f"Pearson Correlation : {avg_pearson:.4f} (the higher, the better)")
            print(f"RMSE (Normalized) : {avg_rmse:.4f} (the lower, the better)")
            print(f"MAE (Normalized) : {avg_mae:.4f} (the lower, the better)")
        else:
            print("Unable to calculate valid index (all genes returned NaN)")

if __name__ == "__main__":
    # --- Configuration area ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    H5AD_FILE = os.path.join(base_dir, "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad")
    MODEL_PATH = os.path.join(base_dir, "nicheformer_weights.pth")
    
    # If you don't know, run the script first and see that the available column names are listed in the error message of "Evaluation Task 2"
    GT_CELL_TYPE = "cell_type" # Cell type column name
    GT_REGION = "ccf_region_name" # Allen CCF brain region list
    # ----------------
    
    evaluator = NicheformerEvaluator(H5AD_FILE, MODEL_PATH, GT_CELL_TYPE, GT_REGION)
    
    evaluator.evaluate_cell_type_annotation()
    evaluator.evaluate_tissue_segmentation()
    # Simultaneously compare the two dimensions of cell type and spatial brain area to make the assessment more complete
    evaluator.evaluate_zero_shot_clustering(n_clusters=20, alt_ref_col="ccf_region_name")
    evaluator.evaluate_gene_imputation(n_test_genes=50)