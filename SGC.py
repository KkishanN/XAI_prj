import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
from sklearn.metrics import accuracy_score, f1_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from feature_engineering import family_node_features
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_sgc():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------
    # 1. Load and Prepare Data
    # --------------------------

    # Load features
    features_df = family_node_features()
    features_df['ID'] = features_df['ID'].apply(lambda x: f"http://www.benchmark.org/family#{x}")

    # Load train/test splits
    train_df = pd.read_csv('data/trainingSet.tsv', sep='\t')
    test_df = pd.read_csv('data/testSet.tsv', sep='\t')

    # Create unified node list and mappings
    all_nodes = pd.concat([train_df['uri'], test_df['uri']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # --------------------------
    # 2. Build Graph Structure
    # --------------------------


    edges = []

    for node in all_nodes:
        node_id = node.split('#')[-1]
        
        # Find this person in features
        person_features = features_df[features_df['ID'] == node]
        
        if len(person_features) == 0:
            continue
            
        person_features = person_features.iloc[0]
        
        if person_features['Married_to_Sibling_Holder'] == 1:
            pass
        
        if person_features['Has_Sibling'] == 1:
            pass

    edges = [
        (node_to_idx["http://www.benchmark.org/family#F10F174"], 
         node_to_idx["http://www.benchmark.org/family#F10F189"])
    ]

    # Convert to edge_index format expected by PyG
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)

    # --------------------------
    # 3. Prepare Features and Labels
    # --------------------------

    # Create feature matrix
    X = np.zeros((len(all_nodes), 4))  # 4 features: Male, Has_Sibling, Sibling_Has_Child, Married_to_Sibling_Holder
    y = np.zeros(len(all_nodes))  # Labels

    for idx, node in enumerate(all_nodes):
        node_id = node.split('#')[-1]
        person_features = features_df[features_df['ID'] == node]
        
        if len(person_features) > 0:
            person_features = person_features.iloc[0]
            X[idx] = person_features[['Male', 'Has_Sibling', 'Sibling_Has_Child', 'Married_to_Sibling_Holder']]
            y[idx] = person_features['Uncle']
        else:
            # For nodes not in features, using zeros
            X[idx] = 0
            y[idx] = 0

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Create train/test masks
    train_mask = torch.zeros(len(all_nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(all_nodes), dtype=torch.bool)

    for _, row in train_df.iterrows():
        train_mask[node_to_idx[row['uri']]] = True
        
    for _, row in test_df.iterrows():
        test_mask[node_to_idx[row['uri']]] = True

    # --------------------------
    # 4. Create PyG Data Object
    # --------------------------

    data = Data(
        x=X,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )

    # --------------------------
    # 5. Define and Train SGC Model
    # --------------------------

    class SGC(torch.nn.Module):
        def __init__(self, num_features, num_classes, K=2):
            super().__init__()
            self.conv = SGConv(num_features, num_classes, K=K)
            
        def forward(self, x, edge_index):
            return self.conv(x, edge_index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SGC(num_features=4, num_classes=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred_probs = F.softmax(out, dim=1).cpu().numpy()
            pred = out.argmax(dim=1)
            acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
            f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        return acc, f1

    # Training loop
    for epoch in range(200):
        loss = train()
        if epoch % 20 == 0:
            acc, f1 = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}')

    # Final evaluation
    acc, f1 = test()
    print(f'\nFinal Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}')

    # --------------------------
    # 6. Predictions
    # --------------------------

    # Get predictions for all nodes
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred_probs = F.softmax(out, dim=1).cpu().numpy()
        pred = out.argmax(dim=1).cpu().numpy()

    # Filter to ONLY test nodes
    test_mask_np = data.test_mask.cpu().numpy()
    test_uris = all_nodes[test_mask_np]
    test_pred = pred[test_mask_np]
    test_true = y[test_mask_np]
    test_probs = pred_probs[test_mask_np]  # Probability scores for test set

    # Create DataFrame with predictions and probabilities
    results = pd.DataFrame({
        'uri': test_uris,
        'predicted_label': test_pred,
        'true_label': test_true,
        #'probability_0': test_probs[:, 0],  # Probability of class 0
        #'probability_1': test_probs[:, 1]   # Probability of class 1
    })

    # Calculate performance metrics
    accuracy = accuracy_score(test_true, test_pred)
    precision = precision_score(test_true, test_pred)
    recall = recall_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred)
    conf_matrix = confusion_matrix(test_true, test_pred)

    results.to_csv("predictions/predictions.csv", index=False)
    # Return all important objects
    return {
        'results': results,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        },
        'model': model,
        'data': data
    }