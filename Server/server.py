from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD 
import os
import datetime
import sys
from contextlib import asynccontextmanager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_engine import NicheformerEngine 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="", help="Absolute path of Unity packaged StreamingAssets")
args, _ = parser.parse_known_args()


H5AD_FILENAME = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad" 
# H5AD_FILENAME = "train.h5ad" 
# H5AD_FILENAME = "data.h5ad" 
CSV_FILENAME = "unity_cell_data.csv"
CELL_TYPE_COLUMN = "cell_type" 


NICHEFORMER_MODEL_PATH = "nicheformer_weights.pth" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeneRequest(BaseModel):
    gene_name: str
    use_imputation: bool = False 

class PerturbRequest(BaseModel):
    target_id: str
    perturb_type: str = "KO"
    target_gene: str = "ENSMUSG00000037010"

class ClusteringRequest(BaseModel):
    n_clusters: int = 10

# Data management class (logical core)
class DataManager:
    def __init__(self):
        self.adata = None
        self.spatial_tree = None
        self.coords = None
        self.indices_map = None
        self.scaler = MinMaxScaler()
        
        self.cached_total_counts = None
        self.cached_raw_total_counts = None # Cache uncompressed real Total Counts
        self.current_view_gene = "RESET"
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        print("[System] Initializing Nicheformer engine...")
        self.ai_engine = NicheformerEngine() 
        
        # Here model_dir If you need, you can point to Nicheformer's weight folder
        self.model_path = os.path.join(self.base_dir, NICHEFORMER_MODEL_PATH)
        
        # Dynamic path setting: If there is external input, use the external absolute path, otherwise fallback to the relative path of the development environment
        if args.output_dir:
            self.output_dir = args.output_dir
        else:
            self.output_dir = os.path.join(self.base_dir, "..", "..", "StreamingAssets")

    def load_and_sync_data(self):
        print(f"[Backend] Loading data: {H5AD_FILENAME} ...")
        h5ad_path = os.path.join(self.base_dir, H5AD_FILENAME)

        if not os.path.exists(h5ad_path):
            print(f"Error: File not found {h5ad_path}")
            return

        # 1. Load the Scanpy data
        self.adata = sc.read_h5ad(h5ad_path)
        self.h5ad_path = h5ad_path
        print("\n" + "="*40)
        print("[Debugging] Gene index check")
        print(f"Total number of genes: {self.adata.n_vars}")
        
        # 1. Print the first 10 gene names (see if it's Gene Symbol or Ensembl ID)
        top_10_genes = self.adata.var_names[:10].tolist()
        print(f"Sample Gene (Index): {top_10_genes}")
        
        # 2. Check whether the specific ID that caused the error exists
        target_debug_id = "ENSMUSG00000037010"
        if target_debug_id in self.adata.var_names:
            print(f"Target gene {target_debug_id} exists in the index!")
        else:
            print(f"Target gene {target_debug_id} does not exist in the index!")
            
            # 3. Try to find it in other columns (sometimes ID is hidden in a column of var)
            found_in_col = False
            for col in self.adata.var.columns:
                # Check if this column contains the ID
                if self.adata.var[col].astype(str).str.contains(target_debug_id).any():
                    print(f"Found {target_debug_id} in column '{col}', not in index.")
                    print(f"   (Frontend sends ID, but model currently uses index outside '{col}')")
                    found_in_col = True
            
            if not found_in_col:
                print(f"Could not find {target_debug_id} in the entire dataset, it may have been filtered out.")
        print("="*40 + "\n")

        if 'counts' not in self.adata.layers:
            self.adata.layers['counts'] = self.adata.X.copy()

        # --- Synchronize data to Nicheformer engine ---
        print("[Nicheformer] Synchronizing data to AI engine...")
        self.ai_engine.adata = self.adata
        self.ai_engine.gene_list = self.adata.var_names.tolist()
        
        # Variable names must be gene_to_id, and the ID offset must match model_engine (i + 3)
        self.ai_engine.gene_to_id = {name: i + 3 for i, name in enumerate(self.ai_engine.gene_list)}
        
        # Print to verify injection success
        print(f"[Debugging] Engine mapping table built, containing {len(self.ai_engine.gene_to_id)} genes.")

        # 2. Coordinate processing
        if 'spatial' in self.adata.obsm:
            self.coords = self.adata.obsm['spatial']
        else:
            self.coords = self.adata.X[:, :2] if self.adata.X.shape[1] >=2 else np.zeros((self.adata.n_obs, 2))

        if issparse(self.coords): self.coords = self.coords.toarray()
        if not isinstance(self.coords, np.ndarray): self.coords = np.array(self.coords)
        
        # Centralized coordinates 
        self.center = np.mean(self.coords, axis=0)
        self.coords_centered = self.coords - self.center

        # --- Synchronize coordinates to AI engine and build graph ---
        self.ai_engine.coords = self.coords_centered # Use centralized coordinates
        self.ai_engine.center = np.zeros(2) # Engine no longer needs to be offset internally
  
        self.ai_engine.build_spatial_graph()

            
        self.spatial_tree = KDTree(self.coords_centered) # For simple distance queries
        self.indices_map = {idx: i for i, idx in enumerate(self.adata.obs.index)}

        # 3. Cache Total Counts (for RESET view)
        if issparse(self.adata.X):
            raw_counts = np.ravel(self.adata.X.sum(axis=1))
        else:
            raw_counts = np.ravel(self.adata.X.sum(axis=1))
        self.cached_raw_total_counts = raw_counts # Save real values
        self.cached_total_counts = self.scaler.fit_transform(raw_counts.reshape(-1, 1)).flatten()

        # 4. Load Nicheformer model weights
        if os.path.exists(self.model_path):
            try:
                self.ai_engine.load_model(self.model_path)
                print("[System] Nicheformer weights loaded successfully.")
            except Exception as e:
                print(f"[Warning] Failed to load Nicheformer weights: {e}")
        else:
            print(f"[Warning] Weight file not found: {self.model_path}, will use untrained model for testing.")

        print(f"[Backend] Data loaded successfully. Cells: {self.adata.n_obs}")

        # 5. Generate Unity CSV
        self.export_csv_for_unity()
    def update_clusters(self, cluster_ids, legend_info):
        """
        Save the clustering results calculated by AI to adata.obs so that they can be written to the hard disk when you click the "Save" button later.
        """
        if self.adata is None:
            print("[Error] DataManager: adata is None, cannot update clusters.")
            return

        try:
            # 1. Ensure length matches
            if len(cluster_ids) != self.adata.n_obs:
                print(f"[Warning] Cluster count ({len(cluster_ids)}) != Cell count ({self.adata.n_obs})")
                return
            
            # 2. Write results to obs (column name: 'zero_shot_cluster')
            self.adata.obs['zero_shot_cluster'] = cluster_ids
            
            # Force conversion to Categorical type
            self.adata.obs['zero_shot_cluster'] = self.adata.obs['zero_shot_cluster'].astype(str).astype('category')
            
            import json
            self.adata.uns['zero_shot_legend'] = json.dumps(legend_info) 

            
            print("[System] Zero-shot clusters updated in RAM.")
            
        except Exception as e:
            print(f"[Error] Failed to update clusters in DataManager: {e}")
            
        except Exception as e:
            print(f"[Error] Failed to update clusters in DataManager: {e}")
    def export_csv_for_unity(self):
        print("[Sync] Generating CSV for Unity (executing coordinate centralization)...")
        ids = self.adata.obs.index
        
        # Use centralized coordinates calculated in load_and_sync_data
        norm_x = self.coords_centered[:, 0]
        norm_y = self.coords_centered[:, 1]
        
        expression_norm = self.cached_total_counts 
        expression_raw = self.cached_raw_total_counts

        if CELL_TYPE_COLUMN in self.adata.obs:
            cell_type_names = self.adata.obs[CELL_TYPE_COLUMN].values
            cell_type_codes, uniques = pd.factorize(cell_type_names)
        else:
            cell_type_names = ["Unknown"] * len(ids)
            cell_type_codes = [0] * len(ids)

        df_export = pd.DataFrame({
            'id': ids, 
            'x': norm_x,  
            'y': norm_y, 
            'z': 0,
            'expression_level': expression_norm,
            'cell_type_id': cell_type_codes,
            'cell_type_name': cell_type_names,
            'raw_expression': expression_raw 
        })

        unity_csv_path = os.path.join(self.output_dir, CSV_FILENAME)
        os.makedirs(os.path.dirname(unity_csv_path), exist_ok=True)

        try:
            df_export.to_csv(unity_csv_path, index=False)
            print(f"[Successfully] CSV saved to: {unity_csv_path}")
        except Exception as e:
            print(f"[Error] CSV save error: {e}")

    def impute_data(self, gene_values):
        """
        Call Nicheformer to perform gene expression interpolation and align the results to the magnitude range of the original data.            Strategy: Percentile Scaling - The original output of Nicheformer has different dimensions than the Min-Max normalized value returned by get_gene_data - Direct use will lead to a high degree of loss of control after imputation - Solution: Align the P99 quantile of the imputed value to the P99 quantile of the original data so that the overall magnitude of the two matches, but the relative difference within the imputation (who is higher and who is lower) is completely retained
        """
        gene_name = self.current_view_gene
        if gene_name == "RESET": return gene_values

        print(f"[Nicheformer] Interpolating genes: {gene_name}")

        try:
            imputed_vals = self.ai_engine.predict_gene_expression(gene_name)
            imputed_vals = np.clip(imputed_vals, 0, None)  

            raw_p99 = np.percentile(gene_values, 99)
            imp_p99 = np.percentile(imputed_vals, 99)

            if imp_p99 > 1e-8:
                scale_factor = raw_p99 / imp_p99
                imputed_vals_norm = imputed_vals * scale_factor
                print(f"[Nicheformer] P99 Alignment: raw_p99={raw_p99:.4f}  imp_p99={imp_p99:.4f}  scale={scale_factor:.4f}")
            else:
                print(f"[Nicheformer] Imputed results are all 0, reverting to original data")
                return gene_values, imputed_vals
            return np.clip(imputed_vals_norm, 0.0, 5.0), imputed_vals

        except Exception as e:
            print(f"Interpolation error: {e}")
            return gene_values, gene_values  # Fallback to original data

    def get_gene_data(self, gene_name):
        if gene_name.upper() in ["RESET", "TOTAL", "DEFAULT", "HARD_RESET"]:
            base_values = self.cached_total_counts 
            uncompressed = self.cached_raw_total_counts # Returns the true uncompressed total
        else:
            if gene_name not in self.adata.var_names: return None, None
            
            if self.adata.raw is not None:
                try: vals = self.adata.raw[:, gene_name].X
                except: vals = self.adata[:, gene_name].X
            else:
                vals = self.adata[:, gene_name].X
            
            if issparse(vals): vals = vals.toarray()
            
            uncompressed = vals.flatten()
            base_values = self.scaler.fit_transform(vals.reshape(-1, 1)).flatten()

        return np.clip(base_values, 0.0, 5.0), uncompressed

    def save_imputed_data(self, gene_name):
        if gene_name == "RESET": return None, "Cannot save RESET view"
        
        print(f"[Save] Nicheformer Interpolated data {gene_name}...")
        try:
            imputed_values = self.ai_engine.predict_gene_expression(gene_name)
            
            df_result = pd.DataFrame({
                'cell_id': self.adata.obs.index,
                'x': self.coords_centered[:, 0],
                'y': self.coords_centered[:, 1],
                f'{gene_name}_niche_imputed': imputed_values
            })

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"niche_imputed_{gene_name}_{timestamp}.csv"
            save_path = os.path.join(self.output_dir, filename)
            
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            return None, str(e)

    def save_annotation_result(self):
        print("[Save] Saving Nicheformer Annotation Result...")
        try:
            # 1. Get Prediction
            pred_ids, legend = self.ai_engine.predict_cell_types()
            
            id_to_name = {item['id']: item['name'] for item in legend}
            predicted_names = [id_to_name.get(pid, "Unknown") for pid in pred_ids]

            data_dict = {
                'cell_id': self.adata.obs.index,
                'predicted_type_id': pred_ids,
                'predicted_type_name': predicted_names
            }
            
            df_result = pd.DataFrame(data_dict)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"niche_annotation_{timestamp}.csv"
            save_path = os.path.join(self.base_dir, "..", "..", "StreamingAssets", filename)
            
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            return None, str(e)

    def save_region_result(self):
        print("[Save] Saving Tissue Region Segmentation Result...")
        try:
            region_ids, region_names = self.ai_engine.segment_tissue_regions()

            predicted_region_names = [region_names[rid] for rid in region_ids]

            data_dict = {
                'cell_id': self.adata.obs.index,
                'x_coord': self.coords_centered[:, 0],
                'y_coord': self.coords_centered[:, 1],
                'region_id': region_ids,
                'region_name': predicted_region_names
            }

            df_result = pd.DataFrame(data_dict)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"niche_segmentation_{timestamp}.csv"
            save_path = os.path.join(self.output_dir, filename)
            
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            return None, str(e)
    def save_zero_shot_result(self):
        print("[Save] Saving Zero-shot Clustering Result as CSV...")
        
        # 1. Check whether there is clustered data
        if 'zero_shot_cluster' not in self.adata.obs:
            return None, "No clustering results found in memory. Please run clustering first."
            
        try:
            # 2. Obtain basic data
            cluster_ids = self.adata.obs['zero_shot_cluster'].values
 
            import json
            legend_json = self.adata.uns.get('zero_shot_legend', '[]')
            
            cluster_names = []
            cluster_colors = []
            
            try:
               
                legend_list = json.loads(legend_json)

                id_to_name = {str(item['id']): item['name'] for item in legend_list}
                id_to_color = {str(item['id']): item['color'] for item in legend_list}

                for cid in cluster_ids:
                    cid_str = str(cid)
                    cluster_names.append(id_to_name.get(cid_str, f"Cluster {cid}"))
                    cluster_colors.append(id_to_color.get(cid_str, "#ffffff"))
            except Exception as parse_e:
                print(f"[Warning] Failed to parse legend json: {parse_e}")
   
                cluster_names = [f"Cluster {c}" for c in cluster_ids]
                cluster_colors = ["#ffffff"] * len(cluster_ids)

            data_dict = {
                'cell_id': self.adata.obs.index,
                'x_coord': self.coords_centered[:, 0],
                'y_coord': self.coords_centered[:, 1],
                'cluster_id': cluster_ids,
                'cluster_name': cluster_names,
                'cluster_color': cluster_colors
            }
            
            df_result = pd.DataFrame(data_dict)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zero_shot_clustering_{timestamp}.csv"
            
            save_path = os.path.join(self.output_dir, filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            df_result.to_csv(save_path, index=False)
            print(f"[Success] CSV Saved to: {save_path}")
            
            return filename, "Success"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, str(e)



dm = DataManager()

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[LifeSpan] Server starting up...")
    dm.load_and_sync_data()

    dm.ai_engine.load_downstream_models()

    print("[LifeSpan] Pre-calculating downstream tasks (Warming up)...")
    try:
        dm.ai_engine.predict_cell_types()
        dm.ai_engine.segment_tissue_regions()
        print("[LifeSpan] Tasks ready.")
    except Exception as e:
        print(f"[LifeSpan] Warm-up warning: {e}")

    yield
    print("[LifeSpan] Server shutting down...")


app = FastAPI(lifespan=lifespan)

@app.get("/search_genes")
async def search_genes(query: str = ""):
    if dm.adata is None: 
        return {"status": "error", "message": "Data not loaded"}
    
    query_upper = query.upper().strip()
    if not query_upper:
        return {"status": "success", "results": []}

    exact_matches = []
    prefix_matches = []
    contains_matches = []
    
    for g in dm.ai_engine.gene_list:
        g_upper = g.upper()
        if g_upper == query_upper:
            exact_matches.append(g)
        elif g_upper.startswith(query_upper):
            prefix_matches.append(g)
        elif query_upper in g_upper:
            contains_matches.append(g)
    
    all_matches = exact_matches + prefix_matches + contains_matches
    unique_matches = list(dict.fromkeys(all_matches))
    
    return {"status": "success", "results": unique_matches[:50]}

@app.post("/switch_gene")
async def switch_gene(req: GeneRequest):
    if dm.adata is None: raise HTTPException(500, "Data not loaded")
    
    target_gene = req.gene_name
    if target_gene in ["HARD_RESET", "RESET", "TOTAL"]:
        target_gene = "RESET"
    
    dm.current_view_gene = target_gene
    values_norm, values_uncompressed = dm.get_gene_data(target_gene) 
    
    if values_norm is None: return {"status": "error", "message": "Gene not found"}
    

    msg = "View Switched"

  
    if req.use_imputation and target_gene != "RESET":

        display_values_norm, display_values_uncompressed = dm.impute_data(values_norm)
        msg = f"AI Imputation : {target_gene}"
        
        disp_list = display_values_norm.tolist() if isinstance(display_values_norm, np.ndarray) else display_values_norm
        raw_list = display_values_uncompressed.tolist() if isinstance(display_values_uncompressed, np.ndarray) else display_values_uncompressed
    else:
       
        disp_list = values_norm.tolist() if isinstance(values_norm, np.ndarray) else values_norm
        raw_list = values_uncompressed.tolist() if isinstance(values_uncompressed, np.ndarray) else values_uncompressed
    
    updates = []
    ids = dm.adata.obs.index
    
    for i in range(len(ids)):
        updates.append({
            "id": str(ids[i]), 
            "new_expr": round(float(disp_list[i]), 3),
            "raw_expr": round(float(raw_list[i]), 3)
        })
        
    return {"status": "success", "message": msg, "updates": updates}

@app.post("/save_imputation")
async def save_imputation(req: GeneRequest):
    filename, msg = dm.save_imputed_data(req.gene_name)
    if filename:
        return {"status": "success", "message": f"Saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

@app.post("/get_annotation")
async def get_annotation():
    if dm.adata is None: return {"status": "error", "message": "Data not loaded"}
    
    try:
        pred_ids, legend_info = dm.ai_engine.predict_cell_types()

        class_names = [item['name'] for item in legend_info]
        
        updates = []
        ids = dm.adata.obs.index
        for i, pid in enumerate(pred_ids):
            updates.append({
                "id": str(ids[i]),
                "pred_id": int(pid) 
            })
            
        return {
            "status": "success",
            "legend": class_names, 
            "updates": updates
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/save_annotation")
async def save_annotation():
    filename, msg = dm.save_annotation_result()
    if filename:
        return {"status": "success", "message": f"Saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

@app.get("/annotation_legend")
async def get_annotation_legend():
    """
    Get detailed legend information (including colors)
    """
    try:
        _, legend_data = dm.ai_engine.predict_cell_types()
        return {
            "status": "success",
            "legend": legend_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/get_tissue_regions")
async def get_tissue_regions():
    if dm.adata is None: return {"status": "error", "message": "Data not loaded"}
            
    try:
        region_ids, region_names = dm.ai_engine.segment_tissue_regions()
        
        final_regions = region_ids.tolist() if hasattr(region_ids, "tolist") else region_ids
        final_names = region_names.tolist() if hasattr(region_names, "tolist") else list(region_names)

        return {
            "status": "success",
            "regions": final_regions,
            "names": final_names
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/save_tissue_regions")
async def save_tissue_regions():
    filename, msg = dm.save_region_result()
    if filename:
        return {"status": "success", "message": f"Results saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

# Zero-shot clustering

@app.post("/zero_shot_cluster")
async def zero_shot_cluster(req: ClusteringRequest):
    """
    Zero-shot clustering: Return a list of updates with Cell IDs
    """
    try:
        if dm.ai_engine is None:
            return JSONResponse(content={"status": "error", "message": "Model not loaded"}, status_code=500)

        cluster_ids_raw, legend_info = dm.ai_engine.run_zero_shot_clustering(req.n_clusters)

        if dm.adata is None:
            raise Exception("Data not loaded in DataManager")
            
        cell_ids = dm.adata.obs_names.tolist()

        updates_list = []
        for cid, cluster_val in zip(cell_ids, cluster_ids_raw):
            updates_list.append({
                "id": str(cid),
                "cluster_id": int(cluster_val)
            })
        
        dm.update_clusters(cluster_ids_raw, legend_info)

        return {
            "status": "success",
            "message": f"Clustering finished. Found {len(legend_info)} clusters.",
            "legend": legend_info,
            "updates": updates_list 
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
@app.post("/save_zero_shot")
async def save_zero_shot(req: dict):
    """
    Save zero-shot clustering results to a CSV file, with the path consistent with other functions.
    """
    if dm.adata is None:
        raise HTTPException(500, "Data not loaded")

    filename, msg = dm.save_zero_shot_result()
    
    if filename:
        return {"status": "success", "message": f"Clustering saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}
@app.post("/perturb")
async def calculate_perturbation(req: PerturbRequest): return {} 
@app.post("/clear_perturbation")
async def clear_perturbation(): return {} 
@app.post("/save_manual")
async def save_manual(): return {} 
@app.post("/impute_all")
async def impute_all(): return {}
@app.post("/disable_imputation")
async def disable_imputation(): return {}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)