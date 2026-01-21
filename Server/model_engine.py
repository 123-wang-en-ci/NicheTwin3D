import torch
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.cluster import KMeans, MiniBatchKMeans
import sys
import os
import importlib.util
from tqdm import tqdm
import torch.nn as nn
import pickle
from scipy.sparse import issparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
nicheformer_root = os.path.join(current_dir, "Nicheformer")
possible_paths = [
    os.path.join(nicheformer_root, "src"),
    nicheformer_root,
    os.path.join(current_dir, "nicheformer"),
]
found_path = None
for path in possible_paths:
    if os.path.isdir(os.path.join(path, "nicheformer")):
        found_path = path
        break
if found_path and found_path not in sys.path:
    sys.path.append(found_path)

Nicheformer = None
try:
    from nicheformer.models._nicheformer import Nicheformer
    print("Successfully imported Nicheformer class")
except ImportError:
    try:
        from nicheformer.models import Nicheformer
    except ImportError:
        print("Nicheformer cannot be imported, please check the path")

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class NicheformerEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = None
        self.model = None
        self.gene_list = []
        self.gene_to_id = {}
        self.coords = None
        self.kd_tree = None
        self.center = None
        
        self.cls_model = None
        self.cls_labels = []
        self.seg_model = None
        self.seg_labels = []
        
        self.n_neighbors = 20
        self.context_length = 1024
        self.batch_size = 16
        
        self.neighbor_indices = None
        self.embeddings_cache = None 
        self.cell_type_cache = None
        self.region_cache = None

    def load_data(self, h5ad_path):
        print(f"Loading data from {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)

        if issparse(self.adata.X):
            max_val = self.adata.X.data.max() if self.adata.X.nnz > 0 else 0
        else:
            max_val = self.adata.X.max()

        if max_val > 50:
            print(f"[Auto-Fix] Raw count (max={max_val:.1f}) detected, normalization is being performed...")
            if 'counts' not in self.adata.layers:
                self.adata.layers['counts'] = self.adata.X.copy()
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)

        else:
            print(f"The data seems to be already in Log space (Max={max_val:.1f}), skipping preprocessing.")

        print("Building spatial neighbor graph...")
        if 'spatial' in self.adata.obsm:
            coords = self.adata.obsm['spatial']
        elif 'X_spatial' in self.adata.obsm:
            coords = self.adata.obsm['X_spatial']
        else:
            coords = np.zeros((self.adata.n_obs, 2))
            
        if isinstance(coords, pd.DataFrame): coords = coords.values
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        self.neighbor_indices = indices
        self.coords = coords
        print(f"Graph built. Neighbor indices shape: {self.neighbor_indices.shape}")

        if os.path.exists("gene_vocab.npy"):
            print("Found gene_vocab.npy, loading fixed vocabulary...")
            self.gene_list = np.load("gene_vocab.npy", allow_pickle=True).tolist()
        else:
            self.gene_list = self.adata.var_names.tolist()

        start_idx = 8 
        print(f"✅ Using fixed Offset (Start Index): {start_idx}")

        self.gene_to_id = {name: i + start_idx for i, name in enumerate(self.gene_list)}
        
        print(f"Data loaded. Cells: {self.adata.n_obs}, Genes: {self.adata.n_vars}")

    def load_model(self, model_path):
        if Nicheformer is None: return

        print(f"Loading Nicheformer weights from {model_path}...")
        self.model = Nicheformer(
            dim_model=256,
            nheads=8,
            dim_feedforward=1024,
            nlayers=6,
            dropout=0.1,
            batch_first=True,
            masking_p=0.0,
            n_tokens=len(self.gene_list) + 20, 
            context_length=self.context_length,
            lr=1e-4,
            warmup=100,
            batch_size=self.batch_size,
            max_epochs=5,
            learnable_pe=True
        )
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
            self._precompute_embeddings()
            
        except Exception as e:
            print(f"Error loading weights: {e}")

    def _get_batch_tokens(self, cell_indices_batch):
        batch_tokens = []
        batch_masks = []
        batch_neighbor_indices = self.neighbor_indices[cell_indices_batch]

        source_data = self.adata.layers['counts'] if 'counts' in self.adata.layers else self.adata.X

        for i in range(len(cell_indices_batch)):
            neighbors = batch_neighbor_indices[i]

            local_expr = source_data[neighbors].sum(axis=0)
            
            if issparse(source_data):
                local_expr = local_expr.A1
            else:
                local_expr = np.array(local_expr).flatten()

            expressed_indices = np.where(local_expr > 0)[0]
            if len(expressed_indices) > self.context_length:
                top_k_args = np.argsort(local_expr[expressed_indices])[-self.context_length:]
                selected_indices = expressed_indices[top_k_args]
            else:
                selected_indices = expressed_indices

            token_ids = selected_indices + 8 
            
            # Padding
            padding_len = self.context_length - len(token_ids)
            if padding_len > 0:
                padded_tokens = np.pad(token_ids, (0, padding_len), 'constant', constant_values=1) # 1=PAD
                att_mask = np.concatenate([np.zeros(len(token_ids)), np.ones(padding_len)])
            else:
                padded_tokens = token_ids
                att_mask = np.zeros(self.context_length)
                
            batch_tokens.append(padded_tokens)
            batch_masks.append(att_mask)
            
        return (torch.tensor(np.array(batch_tokens), dtype=torch.long).to(self.device),
                torch.tensor(np.array(batch_masks), dtype=torch.bool).to(self.device))

    def _precompute_embeddings(self):
        """Calculate and cache the Embedding of all cells"""
        cache_filename = "embeddings_cache.npy"
        cache_path = os.path.join(current_dir, cache_filename)

        if os.path.exists(cache_path):
            print(f"[Cache] Found cached embeddings, loading...")
            try:
                self.embeddings_cache = np.load(cache_path)
                if self.embeddings_cache.shape[0] == self.adata.n_obs:
                    return
            except: pass

        print("Computing Nicheformer embeddings (First time run)...")
        self.embeddings_cache = []
        n_cells = self.adata.n_obs
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, self.batch_size), desc="Inference"):
                batch_indices = np.arange(i, min(i + self.batch_size, n_cells))
                x, mask = self._get_batch_tokens(batch_indices)
                output = self.model(x, mask)
                feats = output['transformer_output'] # (Batch, Seq, Dim)
                
                mask_expanded = mask.unsqueeze(-1).float()
                feats_sum = (feats * (1 - mask_expanded)).sum(dim=1)
                mask_sum = (1 - mask_expanded).sum(dim=1)
                feats_pooled = feats_sum / (mask_sum + 1e-9)
                
                self.embeddings_cache.append(feats_pooled.cpu().numpy())
                
        self.embeddings_cache = np.concatenate(self.embeddings_cache, axis=0)
        np.save(cache_path, self.embeddings_cache)
        print(f"Embeddings computed and saved. Shape: {self.embeddings_cache.shape}")

    def predict_gene_expression(self, gene_name):
        """
        [Hybrid Imputation] Nicheformer + Spatial Smoothing combines the semantic prediction of AI models with the geometric prior of spatial position to ensure optimal visuals.
        """
        fallback_result = np.zeros(self.adata.n_obs)

        if gene_name not in self.gene_to_id:
            if gene_name in self.adata.var_names:
                raw = self.adata[:, gene_name].X
                if issparse(raw): raw = raw.toarray().flatten()
                else: raw = raw.flatten()
                return self._spatial_smoothing(raw) 
            return fallback_result
            
        target_token_id = self.gene_to_id[gene_name]
        print(f"[Nicheformer] Imputing {gene_name} (Hybrid Mode)...")

        if self.embeddings_cache is None: return fallback_result

        try:
            if isinstance(self.embeddings_cache, np.ndarray):
                embeddings = torch.tensor(self.embeddings_cache).to(self.device)
            else:
                embeddings = self.embeddings_cache.to(self.device)

            decoder_weight = None
            decoder_bias = torch.tensor(0.0).to(self.device)

            if hasattr(self.model, "classifier_head"):
                if target_token_id < self.model.classifier_head.weight.shape[0]:
                    decoder_weight = self.model.classifier_head.weight[target_token_id, :]
                    decoder_bias = self.model.classifier_head.bias[target_token_id]

            if decoder_weight is None and hasattr(self.model, "embeddings"):
                decoder_weight = self.model.embeddings.weight[target_token_id, :]

            if decoder_weight is None:
                print("Decoder not found, falling back to spatial.")
                ai_pred = np.zeros(self.adata.n_obs)
            else:
                if decoder_weight.device != self.device: decoder_weight = decoder_weight.to(self.device)
                
                with torch.no_grad():
                    embeddings = torch.nn.functional.layer_norm(embeddings, embeddings.shape[1:])
                    
                    logits = torch.matmul(embeddings, decoder_weight) + decoder_bias
                    ai_pred = torch.nn.functional.relu(logits).cpu().numpy()

            raw_vals = self.adata[:, gene_name].X
            if issparse(raw_vals): raw_vals = raw_vals.toarray().flatten()
            else: raw_vals = raw_vals.flatten()
            
            spatial_pred = self._spatial_smoothing(raw_vals)
            
            def normalize_safe(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-9)
            
            final_pred = 0.3 * normalize_safe(ai_pred) + 0.7 * normalize_safe(spatial_pred)

            final_pred = final_pred * (raw_vals.max() + 1.0)
            
            return final_pred

        except Exception as e:
            print(f"Error: {e}")
            return fallback_result

    def _spatial_smoothing(self, raw_data):
        if not hasattr(self, 'neighbor_indices'):
            return raw_data

        smoothed = np.zeros_like(raw_data, dtype=np.float32)

        neighbor_vals = raw_data[self.neighbor_indices] # (N, K)
        smoothed = neighbor_vals.mean(axis=1)
        return smoothed
    def build_spatial_graph(self):
        """Building a KDTree for finding neighbors"""
        if self.coords is None: return
        print("Building spatial neighbor graph (KDTree)...")
        self.kd_tree = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree')
        self.kd_tree.fit(self.coords)

        print("Pre-calculating neighbors for all cells...")
        self.distances, self.neighbor_indices = self.kd_tree.kneighbors(self.coords)
        print("Spatial graph ready.")

    def run_zero_shot_clustering(self, n_clusters=10):
        print(f"[Nicheformer] Zero-shot Clustering on Embeddings (K={n_clusters})...")
        
        if self.embeddings_cache is None:
            self._precompute_embeddings()

        X_emb = self.embeddings_cache.copy()

        print("   - Applying L2 Normalization to embeddings...")
        X_emb = normalize(X_emb, norm='l2', axis=1)

        print("   - Running MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=2048,
            n_init=10, 
            random_state=42
        )
        clusters = kmeans.fit_predict(X_emb)

        unique_clusters = np.unique(clusters)
        print(f"   - Finished. Found {len(unique_clusters)} clusters.")
        
        legend = []
        import colorsys
        for i, cid in enumerate(unique_clusters):
            hue = (i * 0.618033988749895) % 1.0 
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95) 
            hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            legend.append({"id": int(cid), "name": f"Cluster {cid}", "color": hex_color})
            
        return clusters, legend

    def load_downstream_models(self):
        try:
            if os.path.exists("cell_type_model_labels.pkl"):
                with open("cell_type_model_labels.pkl", "rb") as f:
                    self.cls_labels = pickle.load(f)
                self.cls_model = ClassifierHead(num_classes=len(self.cls_labels))
                self.cls_model.load_state_dict(torch.load("cell_type_model.pth", map_location=self.device))
                self.cls_model.to(self.device).eval()
                print(f"Cell Type Classifier loaded ({len(self.cls_labels)} classes)")
        except: pass

        try:
            if os.path.exists("region_model_labels.pkl"):
                with open("region_model_labels.pkl", "rb") as f:
                    self.seg_labels = pickle.load(f)
                self.seg_model = ClassifierHead(num_classes=len(self.seg_labels))
                self.seg_model.load_state_dict(torch.load("region_model.pth", map_location=self.device))
                self.seg_model.to(self.device).eval()
                print(f"Region Classifier loaded ({len(self.seg_labels)} regions)")
        except: pass

    def predict_cell_types(self):
        if self.cell_type_cache: return self.cell_type_cache
        if not self.cls_model: self.load_downstream_models()
        if not self.cls_model: return [], []
        
        feats = torch.tensor(self.embeddings_cache).float().to(self.device)
        with torch.no_grad():
            _, preds = torch.max(self.cls_model(feats), 1)
        
        legend = [{"id": i, "name": name, "color": "#ffffff"} for i, name in enumerate(self.cls_labels)]
        return preds.cpu().numpy(), legend

    def segment_tissue_regions(self):
        if self.region_cache: return self.region_cache
        if not self.seg_model: self.load_downstream_models()
        if not self.seg_model:
            # Fallback
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=8).fit(self.embeddings_cache)
            return km.labels_, [f"R{i}" for i in range(8)]
            
        feats = torch.tensor(self.embeddings_cache).float().to(self.device)
        with torch.no_grad():
            _, preds = torch.max(self.seg_model(feats), 1)
        return preds.cpu().numpy(), self.seg_labels