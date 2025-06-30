import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from rdflib import Graph as RDFGraph, URIRef
from feature_engineering import family_node_features

def run_uncle_classifier(owl_path="data/family.owl", train_file="data/trainingSet.tsv", test_file="data/testSet.tsv"):
    """
    Run the full uncle classification pipeline and return predictions with metrics
    """
    # ---------------------------
    # 1. Functions
    # ---------------------------
    def parse_family_ontology(owl_path):
        g = RDFGraph()
        g.parse(owl_path, format="xml")
        
        family_ns = "http://www.benchmark.org/family#"
        predicates = {
            "hasChild": URIRef(family_ns + "hasChild"),
            "hasParent": URIRef(family_ns + "hasParent"),
            "hasSibling": URIRef(family_ns + "hasSibling"),
            "married": URIRef(family_ns + "married")
        }
        
        relationships = []
        for s, p, o in g:
            if p in predicates.values() and isinstance(s, URIRef) and isinstance(o, URIRef):
                relationships.append((str(s), str(o)))
                if p in [predicates["hasSibling"], predicates["married"]]:
                    relationships.append((str(o), str(s)))
        return relationships

    # ---------------------------
    # 2. Data Preparation
    # ---------------------------
    # Parse ontology
    relationships = parse_family_ontology(owl_path)
    
    # Load data files
    features_df = family_node_features()
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    
    # Create test_uri_to_label mapping BEFORE combining URIs
    test_uri_to_label = {row['uri']: row['label'] for _, row in test_df.iterrows()}
    
    # Combine URIs and create mappings
    all_uris = set(train_df['uri']).union(set(test_df['uri']))
    uri_to_idx = {uri: idx for idx, uri in enumerate(sorted(all_uris))}
    idx_to_uri = {idx: uri for uri, idx in uri_to_idx.items()}
    
    # Prepare features
    feature_cols = ['Male', 'Has_Sibling', 'Sibling_Has_Child', 'Married_to_Sibling_Holder']
    X = np.zeros((len(all_uris), len(feature_cols)))
    
    for uri in all_uris:
        uri_id = uri.split('#')[-1]
        if uri_id in features_df['ID'].values:
            X[uri_to_idx[uri]] = features_df[features_df['ID'] == uri_id][feature_cols].values[0]
        else:
            col_means = features_df[feature_cols].mean().values
            X[uri_to_idx[uri]] = col_means
    
    y = np.full(len(all_uris), -1)
    
    # Set training labels
    for _, row in train_df.iterrows():
        if row['uri'] in uri_to_idx:
            y[uri_to_idx[row['uri']]] = row['label']
    
    # Set test labels
    for uri, label in test_uri_to_label.items():
        if uri in uri_to_idx:
            y[uri_to_idx[uri]] = label
    
    # Create graph edges
    edge_index = []
    for s, o in relationships:
        if s in uri_to_idx and o in uri_to_idx:
            edge_index.append([uri_to_idx[s], uri_to_idx[o]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index \
        else torch.empty((2, 0), dtype=torch.long)
    
    # Create masks
    train_mask = torch.zeros(len(all_uris), dtype=torch.bool)
    test_mask = torch.zeros(len(all_uris), dtype=torch.bool)
    
    # Split training data
    train_indices = [uri_to_idx[uri] for uri in train_df['uri'] if uri in uri_to_idx]
    train_idx, val_idx = train_test_split(
        train_indices, test_size=0.2, random_state=42, stratify=y[train_indices]
    )
    
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(all_uris), dtype=torch.bool)
    val_mask[val_idx] = True
    
    for uri in test_df['uri']:
        if uri in uri_to_idx:
            test_mask[uri_to_idx[uri]] = True
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # ---------------------------
    # 3. Model Definition
    # ---------------------------
    class UncleClassifier(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    # Initialize model
    model = UncleClassifier(
        in_channels=data.x.size(1),
        hidden_channels=64,
        out_channels=2
    ).to(device)
    
    # ---------------------------
    # 4. Training Loop
    # ---------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_true = data.y[data.val_mask]
            
            val_acc = accuracy_score(val_true.cpu(), val_pred.cpu())
            val_f1 = f1_score(val_true.cpu(), val_pred.cpu(), zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
        
        # Print progress
        if epoch % 10 == 0:
            with torch.no_grad():
                train_pred = out[data.train_mask].argmax(dim=1)
                train_true = data.y[data.train_mask]
                
                train_acc = accuracy_score(train_true.cpu(), train_pred.cpu())
                train_f1 = f1_score(train_true.cpu(), train_pred.cpu(), zero_division=0)
                
                print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                      f'Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | '
                      f'Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
    
    model.load_state_dict(best_model_state)
    
    # ---------------------------
    # 5. Evaluation and Predictions
    # ---------------------------
    model.eval()
    with torch.no_grad():
        out = model(data)
        
        # Training metrics
        train_pred = out[data.train_mask].argmax(dim=1)
        train_true = data.y[data.train_mask]
        train_metrics = {
            'accuracy': accuracy_score(train_true.cpu(), train_pred.cpu()),
            'precision': precision_score(train_true.cpu(), train_pred.cpu(), zero_division=0),
            'recall': recall_score(train_true.cpu(), train_pred.cpu(), zero_division=0),
            'f1': f1_score(train_true.cpu(), train_pred.cpu(), zero_division=0)
        }
        
        # Validation metrics
        val_pred = out[data.val_mask].argmax(dim=1)
        val_true = data.y[data.val_mask]
        val_metrics = {
            'accuracy': accuracy_score(val_true.cpu(), val_pred.cpu()),
            'precision': precision_score(val_true.cpu(), val_pred.cpu(), zero_division=0),
            'recall': recall_score(val_true.cpu(), val_pred.cpu(), zero_division=0),
            'f1': f1_score(val_true.cpu(), val_pred.cpu(), zero_division=0)
        }
        
        # Test predictions
        test_pred = out.argmax(dim=1)
        test_indices = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().tolist()
        test_true = data.y[data.test_mask].cpu().numpy()
        
        # Get the original test URIs in order
        test_uris = [idx_to_uri[idx] for idx in test_indices]
        
        # Create predictions DataFrame
        results_df = pd.DataFrame({
            'uri': test_uris,
            'predicted_label': test_pred[data.test_mask].cpu().numpy(),
            'true_label': test_true
        })
        
        # Save predictions to CSV
        results_df.to_csv('predictions/predictions.csv', index=False)

    # ---------------------------
    # 6. Return Results
    # ---------------------------
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'predictions': results_df.to_dict('records')
    }