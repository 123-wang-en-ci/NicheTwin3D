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

CELL_TYPE_COL = "cell_type"  
REGION_COL = "clust_annot"      

BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001

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

    def forward(self, x):
        return self.net(x)

def train_classifier(features, labels, save_name, device):

    le = LabelEncoder()
    targets = le.fit_transform(labels)
    num_classes = len(le.classes_)

    with open(f"{save_name}_labels.pkl", "wb") as f:
        pickle.dump(le.classes_.tolist(), f)

    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ClassifierHead(input_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = 100 * correct / total
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val Acc = {acc:.2f}%")
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{save_name}.pth")


def main():

    engine = NicheformerEngine()
    engine.load_data("Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad") 
    engine.build_spatial_graph()
    engine.load_model("nicheformer_weights.pth") 

    engine._precompute_embeddings() 
    embeddings = engine.embeddings_cache
    
    if CELL_TYPE_COL in engine.adata.obs:
        print(f"\nProcessing cell type data ({CELL_TYPE_COL})...")

        INVALID_LABELS = ['cell', 'Unknown', 'nan', 'N/A']
        
        raw_labels = engine.adata.obs[CELL_TYPE_COL].astype(str)

        valid_mask = (engine.adata.obs[CELL_TYPE_COL].notna()) & \
                     (~raw_labels.isin(INVALID_LABELS))
        
        # Count how many have been filtered
        n_total = len(raw_labels)
        n_keep = valid_mask.sum()

        if n_keep > 0:
            features = embeddings[valid_mask]
            labels = raw_labels[valid_mask].values
            
            train_classifier(features, labels, "cell_type_model", engine.device)
        else:
            print("Error: There are no remaining cells after filtering, please check the filter conditions!")

    if REGION_COL and REGION_COL in engine.adata.obs:

        valid_mask = engine.adata.obs[REGION_COL].notna()
        features = embeddings[valid_mask]
        labels = engine.adata.obs[REGION_COL][valid_mask].values.astype(str)
        
        train_classifier(features, labels, "region_model", engine.device)
    else:
        print(f"Skip region segmentation training (column '{REGION_COL}' does not exist)")

if __name__ == "__main__":
    main()