import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model_engine import NicheformerEngine

current_dir = os.path.dirname(os.path.abspath(__file__))


CELL_TYPE_COL = "cell_type"      
REGION_COL    = "clust_annot"    

BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001


class ClassifierHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        super().__init__()
        hidden_dim = 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.skip = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        residual = self.skip(x)
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + residual
        out = self.act(out)
        return self.out(out)

def train_classifier(features, labels, save_name, device, save_dir):
    print(f"\nStart training the classifier: {save_name}")
    
    le = LabelEncoder()
    targets = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"{num_classes} categories detected:: {le.classes_[:5]}...")
    
    labels_path = os.path.join(save_dir, f"{save_name}_labels.pkl")
    with open(labels_path, "wb") as f:
        pickle.dump(le.classes_.tolist(), f)
    
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    input_dim = features.shape[1]
    model = ClassifierHead(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    model_save_path = os.path.join(save_dir, f"{save_name}.pth")
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, predicted = torch.max(model(X_batch), 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = 100 * correct / total
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val Acc = {acc:.2f}%")
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_save_path)
            
    print(f"Training completed. Best accuracy: {best_acc:.2f}%. The model is saved to  {model_save_path}")

def main():
    h5ad_path  = os.path.join(current_dir, "merged_brain.h5ad")
    model_path = os.path.join(current_dir, "nicheformer_weights.pth")

    engine = NicheformerEngine()
    engine.load_data(h5ad_path)
    engine.build_spatial_graph()
    engine.load_model(model_path)
    engine._precompute_embeddings()
    embeddings = engine.embeddings_cache
    
    if CELL_TYPE_COL in engine.adata.obs:
        print(f"\nProcessing cell type data ({CELL_TYPE_COL})...")
        INVALID_LABELS = ['cell', 'Unknown', 'nan', 'N/A']
        raw_labels = engine.adata.obs[CELL_TYPE_COL].astype(str)
        valid_mask = (engine.adata.obs[CELL_TYPE_COL].notna()) & (~raw_labels.isin(INVALID_LABELS))
        n_total, n_keep = len(raw_labels), valid_mask.sum()
        print(f"Number of blasts: {n_total}, Filtered: {n_keep} (excludes {n_total - n_keep} fuzzy cells)")
        if n_keep > 0:
            train_classifier(embeddings[valid_mask], raw_labels[valid_mask].values,
                             "cell_type_model", engine.device, current_dir)
        else:
            print("Error: There are no remaining cells after filtering, please check the filter conditions!")

    if REGION_COL and REGION_COL in engine.adata.obs:
        print("\nPreparing regional data...")
        region_labels = engine.adata.obs[REGION_COL].astype(str)
        valid_mask = engine.adata.obs[REGION_COL].notna() & (region_labels != 'n/a') & (region_labels != 'N/A')
        n_keep = valid_mask.sum()
        print(f"Regional data: {n_keep}/{engine.adata.n_obs} cells are effective after filtering")
        train_classifier(embeddings[valid_mask],
                         region_labels[valid_mask].values,
                         "region_model", engine.device, current_dir)
    else:
        print(f"Skip region segmentation training (column '{REGION_COL}' does not exist).")

if __name__ == "__main__":
    main()