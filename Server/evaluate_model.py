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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_engine import NicheformerEngine

class NicheformerEvaluator:
    def __init__(self, h5ad_path, model_path, cell_type_col="cell_type", region_col="region"):
        self.engine = NicheformerEngine()
        self.engine.load_data(h5ad_path)
        self.engine.load_model(model_path)

        self.engine.load_downstream_models()
        
        self.adata = self.engine.adata
        self.cell_type_col = cell_type_col
        self.region_col = region_col
        
        if self.engine.embeddings_cache is None:
            self.engine._precompute_embeddings()

    def evaluate_cell_type_annotation(self):
        print("\n" + "="*40)
        print("Assessment Task 1: Cell Type Annotation")
        
        if self.cell_type_col not in self.adata.obs:
            print(f"Error: Column '{self.cell_type_col}' not found in adata.obs")
            print(f"Available columns: {list(self.adata.obs.columns)}")
            return

        try:
            pred_ids, legend = self.engine.predict_cell_types()
            if len(pred_ids) == 0: 
                print("Error: Prediction results are empty, skipping evaluation")
                return
                
            true_labels_all = self.adata.obs[self.cell_type_col].astype(str).values
            
            id_to_name = {item['id']: str(item['name']) for item in legend}
            pred_names_all = np.array([id_to_name.get(pid, "Unknown") for pid in pred_ids])
            
            valid_mask = self.adata.obs[self.cell_type_col].notna()
            true_valid = true_labels_all[valid_mask]
            pred_valid = pred_names_all[valid_mask]
            
            _, y_test_true, _, y_test_pred = train_test_split(
                true_valid, pred_valid, test_size=0.2, random_state=42
            )
            
            acc = accuracy_score(y_test_true, y_test_pred)
            f1 = f1_score(y_test_true, y_test_pred, average='weighted')
            
            print(f"Accuracy : {acc:.4f}")
            print(f"F1-Score : {f1:.4f}")
        except Exception as e:
            print(f"Error: {e}")

    def evaluate_tissue_segmentation(self):
        print("\n" + "="*40)
        print("Assessment Task 2: Tissue Segmentation")
        
        if self.region_col not in self.adata.obs:
            print(f"Error: Column '{self.region_col}' not found in adata.obs")
            print(f"Please check the available column names below and modify the GT_REGION variable in the code:")
            print(f"{list(self.adata.obs.columns)}")
            return

        try:
            pred_ids, region_names_list = self.engine.segment_tissue_regions()

            region_names_list = [str(x) for x in region_names_list]
            pred_names_all = np.array([region_names_list[rid] for rid in pred_ids])

            true_labels_all = self.adata.obs[self.region_col].astype(str).values

            valid_mask = ~self.adata.obs[self.region_col].astype(str).str.lower().isin(['n/a', 'nan', 'none'])
            
            true_valid = true_labels_all[valid_mask]
            pred_valid = pred_names_all[valid_mask]
 
            _, y_test_true, _, y_test_pred = train_test_split(
                true_valid, pred_valid, test_size=0.2, random_state=42
            )

            acc = accuracy_score(y_test_true, y_test_pred)
            f1 = f1_score(y_test_true, y_test_pred, average='weighted')
            
            print(f"Accuracy : {acc:.4f}")
            print(f"F1-Score : {f1:.4f}")
        except Exception as e:
            print(f"Error: {e}")

    def evaluate_zero_shot_clustering(self, n_clusters=10, alt_ref_col=None):
        print("\n" + "="*40)
        print(f"Assessment Task 3: Zero-shot Clustering")
        
        if self.cell_type_col not in self.adata.obs:
            print(f"Error: Column '{self.cell_type_col}' not found in adata.obs")
            return
        
        try:
            cluster_labels, _ = self.engine.run_zero_shot_clustering(n_clusters=n_clusters)

            true_labels = self.adata.obs[self.cell_type_col].values
            ari  = adjusted_rand_score(true_labels, cluster_labels)
            nmi  = normalized_mutual_info_score(true_labels, cluster_labels)
            print(f"[vs Cell Types] ARI : {ari:.4f}")
            print(f"[vs Cell Types] NMI : {nmi:.4f}")
            
            if alt_ref_col and alt_ref_col in self.adata.obs:
                region_labels = self.adata.obs[alt_ref_col].astype(str)
                valid = region_labels != 'n/a'
                if valid.sum() > 0:
                    ari_r = adjusted_rand_score(region_labels[valid], cluster_labels[valid])
                    nmi_r = normalized_mutual_info_score(region_labels[valid], cluster_labels[valid])
                    print(f"[vs Spatial Regions] ARI : {ari_r:.4f}")
                    print(f"[vs Spatial Regions] NMI : {nmi_r:.4f}")
        except Exception as e:
            print(f"Error: {e}")

    def evaluate_gene_imputation(self, n_test_genes=50):
        print("\n" + "="*40)
        print(f"Assessment Task 4: Gene Imputation")
        
        hv_genes = []
        try:
            print("Selecting highly variable genes...")
            temp_adata = self.adata.copy()
            if np.issubdtype(temp_adata.X.dtype, np.integer):
                sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_test_genes, flavor='seurat_v3')
            else:
                sc.pp.log1p(temp_adata)
                sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_test_genes)
            
            hv_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
            del temp_adata 
            
        except Exception as e:
            print(f"Scanpy High Variable Gene calculation failed ({str(e)}), switching to fallback strategy...")
        
        if len(hv_genes) < n_test_genes:
            print("Using highly expressed genes as test set (Fallback Strategy)")
            if hasattr(self.adata.X, 'toarray'):
                means = np.array(self.adata.X.mean(axis=0)).flatten()
            else:
                means = np.array(self.adata.X.mean(axis=0)).flatten()
                
            top_indices = np.argsort(means)[-n_test_genes:]
            hv_genes = self.adata.var_names[top_indices].tolist()
        
        hv_genes = hv_genes[:n_test_genes]
        print(f"Selected test genes: {hv_genes[:5]} ...")
            
        print(f"Selected test genes: {hv_genes[:5]} ...")
            
        pearson_list = []
        rmse_list = []
        mae_list = []
        
        for gene in tqdm(hv_genes):
            try:
                pred_vals = self.engine.predict_gene_expression(gene)

                if isinstance(self.adata[:, gene].X, np.ndarray):
                    true_vals = self.adata[:, gene].X.flatten()
                else:
                    true_vals = self.adata[:, gene].X.toarray().flatten()
                true_vals_log = np.log1p(true_vals)
                pred_vals_log = np.log1p(pred_vals) 
                
                corr, _ = pearsonr(true_vals_log, pred_vals_log)
                
                if not np.isnan(corr):
                    pearson_list.append(corr)
                
                rmse = np.sqrt(mean_squared_error(true_vals_log, pred_vals_log))
                mae = mean_absolute_error(true_vals_log, pred_vals_log)

                rmse_list.append(rmse)
                mae_list.append(mae)

            except Exception as e_inner:
                print(f"Gene {gene} calculation failed: {e_inner}")
                continue 
        
        if len(pearson_list) > 0:
            avg_pearson = np.mean(pearson_list)
            avg_rmse = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            
            print(f"Pearson Correlation : {avg_pearson:.4f}")
            print(f"RMSE (Normalized)   : {avg_rmse:.4f}")
            print(f"MAE (Normalized)    : {avg_mae:.4f}")
        else:
            print("Unable to calculate valid metrics (all genes returned NaN)")

if __name__ == "__main__":
    # --- Configuration ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    H5AD_FILE = os.path.join(base_dir, "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad")
    MODEL_PATH = os.path.join(base_dir, "nicheformer_weights.pth")
    
    # Please modify the column names here!
    # If you don't know, run the script once and check the error message in "Evaluation Task 2" for available column names
    GT_CELL_TYPE = "cell_type"      
    GT_REGION = "ccf_region_name"  
    # ----------------
    
    evaluator = NicheformerEvaluator(H5AD_FILE, MODEL_PATH, GT_CELL_TYPE, GT_REGION)
    
    evaluator.evaluate_cell_type_annotation()
    evaluator.evaluate_tissue_segmentation()
    evaluator.evaluate_zero_shot_clustering(n_clusters=20, alt_ref_col="ccf_region_name")
    evaluator.evaluate_gene_imputation(n_test_genes=50)