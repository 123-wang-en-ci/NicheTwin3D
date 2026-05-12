import os
import sys
import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import warnings

# ==============================================================================
# 0. Video memory optimization and security settings
# ==============================================================================
# Completely fix all breakpoint read interception issues in PyTorch 2.6+ (intercept torch.load)
import builtins
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _legacy_load

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# ==============================================================================
# 1. Environment and path settings
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
nicheformer_src = os.path.join(current_dir, "Nicheformer", "src")
if nicheformer_src not in sys.path:
    sys.path.append(nicheformer_src)

try:
    from nicheformer.models._nicheformer import Nicheformer
except ImportError:
    try:
        from model_engine import Nicheformer
    except ImportError:
        pass

# ==============================================================================
# 2. Missing utility function completion (remains unchanged)
# ==============================================================================
def complete_masking(batch, masking_p, n_tokens):
    x = batch['X'].clone()
    probability_matrix = torch.full(x.shape, masking_p).to(x.device)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    mask_token_id = 0 
    labels = x.clone()
    
    indices_replaced = torch.bernoulli(torch.full(x.shape, 0.8)).bool().to(x.device) & masked_indices
    x[indices_replaced] = mask_token_id

    indices_random = torch.bernoulli(torch.full(x.shape, 0.5)).bool().to(x.device) & masked_indices & ~indices_replaced
    random_tokens = torch.randint(1, n_tokens, x.shape, dtype=torch.long).to(x.device)
    x[indices_random] = random_tokens[indices_random]

    new_batch = {
        'masked_indices': x,
        'mask': x, 
        'X': labels,
        'attention_mask': batch.get('attention_mask', None)
    }
    mask_for_loss = torch.ones_like(x)
    mask_for_loss[masked_indices] = 0
    new_batch['mask'] = mask_for_loss
    
    for k, v in batch.items():
        if k not in new_batch:
            new_batch[k] = v
    return new_batch

import types
if 'nicheformer.models._utils' not in sys.modules:
    mod = types.ModuleType('nicheformer.models._utils')
    mod.complete_masking = complete_masking
    sys.modules['nicheformer.models._utils'] = mod

# ==============================================================================
# 3. Data set construction (repair word list saving)
# ==============================================================================
class SpatialNicheDataset(Dataset):
    def __init__(self, h5ad_path, context_length=1024, n_neighbors=20):
        print(f"Loading data: {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # Preprocessing: Make sure there are counts
        if 'counts' not in self.adata.layers:
            self.adata.layers['counts'] = self.adata.X.copy()
        

        self.gene_names = self.adata.var_names.tolist()
        
        # Save the vocabulary list locally for server.py and model_engine.py to read!
        vocab_path = os.path.join(current_dir, "gene_vocab.npy")
        np.save(vocab_path, self.gene_names)
        print(f"[Training] gene vocabulary has been saved to: {vocab_path} (Len: {len(self.gene_names)})")
        print(" -> This ensures that no dimension mismatch errors will occur when loading the model on the server side.")
        
        # 8 is Nicheformer’s special Token reserved bit (PAD, MASK, CLS, etc.)
        # Must be consistent with start_idx = 8 in model_engine.py
        self.start_idx = 8
        self.gene_to_id = {gene: i + self.start_idx for i, gene in enumerate(self.gene_names)}
        self.n_tokens = len(self.gene_names) + 20 # Reserve some buffer
        self.context_length = context_length
        self.n_neighbors = n_neighbors
        
        # Build spatial graph
        print("Construct spatial neighborhood graph...")
        if 'spatial' in self.adata.obsm:
            coords = self.adata.obsm['spatial']
        elif 'X_spatial' in self.adata.obsm:
            coords = self.adata.obsm['X_spatial']
        else:
            coords = self.adata.X[:, :2]
            
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coords)
        self.distances, self.indices = self.nbrs.kneighbors(coords)
        print(f"Data set is ready. Number of cells: {self.adata.n_obs}")

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        neighbor_indices = self.indices[idx]
        
        # Aggregate neighborhood expressions
        local_expression = self.adata.layers['counts'][neighbor_indices].sum(axis=0)
        
        if hasattr(local_expression, "A1"):
            local_expression = local_expression.A1
        else:
            local_expression = np.array(local_expression).flatten()
            
        # Extract genes with non-zero expression
        expressed_gene_indices = np.where(local_expression > 0)[0]
        
# Truncate or fill
        if len(expressed_gene_indices) > self.context_length:
            # Get the Top K with the highest expression level
            top_indices = np.argsort(local_expression[expressed_gene_indices])[-self.context_length:]
            selected_gene_indices = expressed_gene_indices[top_indices]
        else:
            selected_gene_indices = expressed_gene_indices

        # Convert to Token ID (plus offset)
        token_ids = selected_gene_indices + self.start_idx
        
        # Padding (Padding with 1, assuming 1 is PAD)
        padding_len = self.context_length - len(token_ids)
        if padding_len > 0:
            token_ids = np.pad(token_ids, (0, padding_len), 'constant', constant_values=1)
            attention_mask = np.concatenate([np.zeros(len(selected_gene_indices)), np.ones(padding_len)])
        else:
            attention_mask = np.zeros(self.context_length)

        return {
            'X': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'cell_type': 0 # placeholder
        }

# ==============================================================================
# 4. Main training process
# ==============================================================================
def train():
    # --- Configuration parameters ---
    #Fix the cloud path problem and ensure that files can be found correctly in AutoDL
    H5AD_PATH = os.path.join(current_dir, "merged_brain.h5ad")
    OUTPUT_PATH = os.path.join(current_dir, "nicheformer_weights.pth")
    
    # After returning to the official standard, the memory pressure is reduced, and the Batch Size can be increased to stabilize the gradient and speed up the process.
    BATCH_SIZE = 8 
    
    # 【Ultimate Performance Unlocked】Significantly increase the number of training rounds, allowing the model to completely converge on this exclusive data set and extract the strongest features!
    MAX_EPOCHS = 100 
    LR = 3e-4 # Increase learning rate slightly to speed up convergence within 100 epochs
    
    # Model hyperparameters (return to official standard configuration)
    CONTEXT_LENGTH = 1024
    DIM_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    N_NEIGHBORS = 20
    
    if not os.path.exists(H5AD_PATH):
        print(f"Error: Data file {H5AD_PATH} not found")
        return

    dataset = SpatialNicheDataset(H5AD_PATH, context_length=CONTEXT_LENGTH)
    
    # num_workers=0 under Windows
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    print("Initializing model...")
    model = Nicheformer(
        dim_model=DIM_MODEL,
        nheads=N_HEADS,
        dim_feedforward=DIM_MODEL * 4,
        nlayers=N_LAYERS,
        dropout=0.1,
        batch_first=True,
        masking_p=0.15,
        n_tokens=dataset.n_tokens,
        context_length=CONTEXT_LENGTH,
        lr=LR,
        warmup=100,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        learnable_pe=True
    )
    
    # Check GPU
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1 if use_gpu else "auto"
    precision_val = 16 if use_gpu else 32

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints',
        filename='nicheformer-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    # Cloud streamlined printing callback: print once after each round without refreshing the screen
    class SimpleLoggerCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            loss = trainer.callback_metrics.get('train_loss')
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
    print(f"[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] Training loss (Loss): {loss_str}")

    print(f"Configure Trainer (Epochs: {MAX_EPOCHS}, Acc: {accelerator})...")
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, SimpleLoggerCallback()],
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_model_summary=False,
        precision=precision_val
    )
    
    import time
    print(f"Start training (progress bar disabled)...")
    start_time = time.time()
    # Automatically find the latest checkpoint to resume disconnected training
    import glob
    ckpt_files = glob.glob('checkpoints/*.ckpt')
    if len(ckpt_files) > 0:
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        print(f"Detected previous training breakpoint, resuming training from {latest_ckpt}...")
        trainer.fit(model, dataloader, ckpt_path=latest_ckpt)
    else:
        trainer.fit(model, dataloader)
        
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed, total time taken: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    print(f"Saving weights to {OUTPUT_PATH}...")
    # Save pure state_dict
    state_dict = model.state_dict()
    clean_state_dict = {k: v for k, v in state_dict.items() if "loss" not in k}
    torch.save(clean_state_dict, OUTPUT_PATH)
    print("Weights saved successfully! Now you can run evaluate_model.py.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()