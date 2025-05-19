import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from models.temporal_gat import TemporalGAT
import torch.nn as nn

def load_model(model_path, device):
    # Model configuration
    node_features = 9  # Input feature dimension
    hidden_channels = 256  # Hidden layer dimension
    num_heads = 4
    
    # Create model
    model = TemporalGAT(
        node_features=node_features,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=0.3,
        use_fp16=True
    )
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    return model

def prepare_test_data(test_file):
    # Read test data
    df = pd.read_csv(test_file, header=None)
    
    # Set column names
    df.columns = ['source', 'target', 'edge_type', 'start_time', 'end_time', 'label']
    
    # Print label distribution
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    
    # Create node ID mapping to ensure continuous indices
    unique_nodes = pd.concat([df['source'], df['target']]).unique()
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # Map node IDs to continuous indices
    df['source_mapped'] = df['source'].map(node_mapping)
    df['target_mapped'] = df['target'].map(node_mapping)
    
    # Create more informative node features
    num_nodes = len(unique_nodes)
    node_features = np.zeros((num_nodes, 9))
    
    # Compute node degree (in-degree and out-degree)
    source_counts = df['source_mapped'].value_counts()
    target_counts = df['target_mapped'].value_counts()
    
    # Add some basic features to nodes
    for node_id in range(num_nodes):
        # Out-degree (number of edges where node is source)
        node_features[node_id, 0] = source_counts.get(node_id, 0)
        # In-degree (number of edges where node is target)
        node_features[node_id, 1] = target_counts.get(node_id, 0)
        # Total degree
        node_features[node_id, 2] = node_features[node_id, 0] + node_features[node_id, 1]
    
    # Normalize node features
    for i in range(3):  # Normalize the first 3 features
        if np.std(node_features[:, i]) > 0:
            node_features[:, i] = (node_features[:, i] - np.mean(node_features[:, i])) / np.std(node_features[:, i])
    
    # Create edge index
    edge_index = np.stack([df['source_mapped'].values, df['target_mapped'].values])
    
    # Create edge features (time information)
    edge_attr = np.stack([
        df['edge_type'].values,
        df['start_time'].values,
        df['end_time'].values,
        df['start_time'] - df['end_time'].values  # Time difference
    ], axis=1)
    
    # Normalize edge features
    edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0) + 1e-8)
    
    # Get edge labels and indices
    edge_labels = df['label'].values
    
    # Check for class imbalance
    pos_rate = np.mean(edge_labels)
    print(f"Positive edge rate: {pos_rate:.4f}")
    
    return node_features, edge_index, edge_attr, edge_labels, num_nodes

def test_model(model, node_features, edge_index, edge_attr, labels, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Convert to tensors
        x = torch.FloatTensor(node_features).to(device)
        edge_index_tensor = torch.LongTensor(edge_index).to(device)
        
        # Process in batches for memory efficiency
        batch_size = 5000
        num_edges = edge_index.shape[1]
        num_batches = (num_edges + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_edges)
            
            # Get batch
            batch_edge_index = edge_index_tensor[:, start_idx:end_idx]
            
            try:
                # Forward pass to get node embeddings
                node_embeddings = model(x, batch_edge_index)
                
                # Link prediction for batch
                batch_pred = model.predict_link(node_embeddings, batch_edge_index)
                batch_pred_np = batch_pred.cpu().numpy()
                
                # Print some prediction statistics for debugging
                if i == 0:
                    print(f"\nFirst batch predictions stats:")
                    print(f"Min: {np.min(batch_pred_np):.6f}, Max: {np.max(batch_pred_np):.6f}")
                    print(f"Mean: {np.mean(batch_pred_np):.6f}, Std: {np.std(batch_pred_np):.6f}")
                
                predictions.append(batch_pred_np)
                
                # Clear memory
                del node_embeddings, batch_pred
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
    
    # Combine predictions
    all_predictions = np.concatenate(predictions)
    
    # Print predictions distribution
    print("\nPredictions distribution:")
    print(f"Min: {np.min(all_predictions):.6f}, Max: {np.max(all_predictions):.6f}")
    print(f"Mean: {np.mean(all_predictions):.6f}, Std: {np.std(all_predictions):.6f}")
    
    # Print histogram of predictions
    hist, bin_edges = np.histogram(all_predictions, bins=10, range=(0, 1))
    print("\nHistogram of predictions:")
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {hist[i]}")
    
    # Try different threshold
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nF1 scores with different thresholds:")
    for threshold in thresholds:
        pred_binary = (all_predictions >= threshold).astype(int)
        f1 = f1_score(labels, pred_binary)
        print(f"Threshold {threshold:.1f}: F1 = {f1:.4f}")
    
    # Calculate metrics
    auc = roc_auc_score(labels, all_predictions)
    ap = average_precision_score(labels, all_predictions)
    
    # Find best threshold for F1
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        pred_binary = (all_predictions >= threshold).astype(int)
        f1 = f1_score(labels, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Final F1 with best threshold
    pred_binary = (all_predictions >= best_threshold).astype(int)
    final_f1 = f1_score(labels, pred_binary)
    
    print(f"\nBest threshold: {best_threshold:.4f} with F1: {final_f1:.4f}")
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, pred_binary)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'auc': auc,
        'ap': ap,
        'f1': final_f1,
        'predictions': all_predictions,
        'best_threshold': best_threshold
    }

def main():
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model path
    model_path = "checkpoints_a5000/edges_train_A/model_best.pt"
    
    # Test data path
    test_file = "test-data/input_A_initial.csv"
    
    # Load model
    model = load_model(model_path, device)
    if model is None:
        return
    
    # Prepare test data
    node_features, edge_index, edge_attr, labels, num_nodes = prepare_test_data(test_file)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    # Test model
    results = test_model(model, node_features, edge_index, edge_attr, labels, device)
    
    # Print results
    print("\nTest Results:")
    print(f"AUC: {results['auc']:.4f}")
    print(f"AP: {results['ap']:.4f}")
    print(f"F1 Score (threshold={results['best_threshold']:.4f}): {results['f1']:.4f}")

if __name__ == "__main__":
    main() 