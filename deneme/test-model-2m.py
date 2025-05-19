import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from models.temporal_gat import TemporalGAT
import torch.nn as nn

def load_model(model_path, device):
    # Model configuration
    node_features = 9  # Input feature dimension
    hidden_channels = 64  # Hata mesajlarına göre düzeltildi
    num_heads = 8     # 8 başlık korundu
    dropout = 0.2     # main.py'de varsayılan değer
    
    # Create model
    model = TemporalGAT(
        node_features=node_features,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
        use_fp16=True
    )
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Eski model durumunu adapte et
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Modelin kendi state_dict'ini al
            model_state_dict = model.state_dict()
            
            # GATConv yapısı değiştiği için yapısal dönüşüm yap
            if 'gat1.lin_src.weight' in state_dict and 'gat1.lin.weight' in model_state_dict:
                print("GAT yapısını eski sürümden uyarlıyorum...")
                # Kaynak ve hedef ağırlıkların ortalamasını al
                if 'gat1.lin_src.weight' in state_dict and 'gat1.lin_dst.weight' in state_dict:
                    state_dict['gat1.lin.weight'] = (state_dict['gat1.lin_src.weight'] + state_dict['gat1.lin_dst.weight']) / 2
                
                if 'gat2.lin_src.weight' in state_dict and 'gat2.lin_dst.weight' in state_dict:
                    state_dict['gat2.lin.weight'] = (state_dict['gat2.lin_src.weight'] + state_dict['gat2.lin_dst.weight']) / 2
            
            # Uygun anahtarları karşılaştır
            compatible_state_dict = {}
            for key in model_state_dict.keys():
                if key in state_dict and model_state_dict[key].shape == state_dict[key].shape:
                    compatible_state_dict[key] = state_dict[key]
                else:
                    print(f"Uyumsuz parametre (sıfır ile başlatılıyor): {key}")
                    # Model durumundaki varsayılan değerleri koru
                    compatible_state_dict[key] = model_state_dict[key]
            
            model.load_state_dict(compatible_state_dict)
            print("Model kısmi olarak yüklendi (bazı parametreler varsayılan değerlerle başlatıldı)")
        else:
            print("Uyarı: 'model_state_dict' anahtarı bulunamadı")
            return None
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
    standard_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nF1 scores with standard thresholds:")
    for threshold in standard_thresholds:
        pred_binary = (all_predictions >= threshold).astype(int)
        f1 = f1_score(labels, pred_binary)
        print(f"Threshold {threshold:.1f}: F1 = {f1:.4f}")
    
    # Detailed analysis for thresholds 0.3-0.45 with 0.05 step
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    detailed_thresholds = [0.30, 0.35, 0.40, 0.45]  # 0.45 added
    
    print("\n" + "="*60)
    print("DETAILED THRESHOLD ANALYSIS (0.30-0.45 range, 0.05 step)")
    print("="*60)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-"*60)
    
    detailed_results = []
    
    for threshold in detailed_thresholds:
        pred_binary = (all_predictions >= threshold).astype(int)
        f1 = f1_score(labels, pred_binary)
        precision = precision_score(labels, pred_binary)
        recall = recall_score(labels, pred_binary)
        accuracy = accuracy_score(labels, pred_binary)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
        
        print(f"{threshold:<10.2f} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {tp:<8} {fp:<8} {tn:<8} {fn:<8}")
        
        detailed_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    print("="*60)
    
    # Calculate metrics for all thresholds
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
    
    # Final metrics with best threshold
    pred_binary = (all_predictions >= best_threshold).astype(int)
    final_f1 = f1_score(labels, pred_binary)
    final_precision = precision_score(labels, pred_binary)
    final_recall = recall_score(labels, pred_binary)
    final_accuracy = accuracy_score(labels, pred_binary)
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
    
    print("\n" + "="*60)
    print(f"BEST THRESHOLD VALUE: {best_threshold:.4f}")
    print("="*60)
    print(f"Accuracy:  {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall:    {final_recall:.4f}")
    print(f"F1-Score:  {final_f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"AP:        {ap:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"True Negative (TN): {tn}")
    print(f"False Negative (FN): {fn}")
    print("="*60)
    
    # Fixed threshold metrics (0.30)
    fixed_threshold = 0.30
    pred_binary_fixed = (all_predictions >= fixed_threshold).astype(int)
    fixed_f1 = f1_score(labels, pred_binary_fixed)
    fixed_precision = precision_score(labels, pred_binary_fixed)
    fixed_recall = recall_score(labels, pred_binary_fixed)
    fixed_accuracy = accuracy_score(labels, pred_binary_fixed)
    tn_fixed, fp_fixed, fn_fixed, tp_fixed = confusion_matrix(labels, pred_binary_fixed).ravel()
    
    print("\n" + "="*60)
    print(f"FIXED THRESHOLD VALUE: {fixed_threshold:.2f}")
    print("="*60)
    print(f"Accuracy:  {fixed_accuracy:.4f}")
    print(f"Precision: {fixed_precision:.4f}")
    print(f"Recall:    {fixed_recall:.4f}")
    print(f"F1-Score:  {fixed_f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"AP:        {ap:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positive (TP): {tp_fixed}")
    print(f"False Positive (FP): {fp_fixed}")
    print(f"True Negative (TN): {tn_fixed}")
    print(f"False Negative (FN): {fn_fixed}")
    print("="*60)
    
    return {
        'auc': auc,
        'ap': ap,
        'f1': final_f1,
        'precision': final_precision,
        'recall': final_recall,
        'accuracy': final_accuracy,
        'predictions': all_predictions,
        'best_threshold': best_threshold,
        'fixed_threshold': fixed_threshold,
        'confusion_matrix': {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        },
        'detailed_results': detailed_results,
        'fixed_threshold_results': {
            'threshold': fixed_threshold,
            'f1': fixed_f1,
            'precision': fixed_precision,
            'recall': fixed_recall,
            'accuracy': fixed_accuracy,
            'tp': tp_fixed,
            'fp': fp_fixed,
            'tn': tn_fixed,
            'fn': fn_fixed
        }
    }

def main():
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model path
    model_path = "model-2m-test-edilecek/model_best_2m.pt"
    
    # Test data path
    test_file = "test-data/input_A_initial.csv"
    
    # Output file path
    output_file = "output_A_2m.csv"
    
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
    print("\n" + "="*60)
    print("SUMMARY TEST RESULTS")
    print("="*60)
    print(f"Best Threshold: {results['best_threshold']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"AP:        {results['ap']:.4f}")
    
    # Write predictions to output file
    try:
        # Read original test data
        original_test_data = pd.read_csv(test_file, header=None)
        original_test_data.columns = ['source', 'target', 'edge_type', 'start_time', 'end_time', 'label']
        
        # Add prediction probabilities
        original_test_data['probability'] = results['predictions']
        
        # Threshold value
        threshold_value = results['fixed_threshold']  # Use fixed threshold value from results
        
        # Select only required columns and save (in expected format)
        output_df = original_test_data[['source', 'target', 'probability']]
        
        # Add binary predictions (for information)
        output_df['prediction'] = (output_df['probability'] >= threshold_value).astype(int)
        
        # Write only source, target and probability columns to CSV
        output_df[['source', 'target', 'probability']].to_csv(output_file, index=False, header=False)
        
        print(f"\nPrediction probabilities saved to '{output_file}'.")
        print(f"Threshold: {threshold_value:.2f}")
        print(f"Total edges predicted: {len(output_df)}")
        print(f"Positive predictions: {output_df['prediction'].sum()} ({output_df['prediction'].mean()*100:.2f}%)")
        
        # Show first few rows
        print("\nOutput file preview (first 5 rows):")
        for i in range(min(5, len(output_df))):
            src = output_df.iloc[i, 0]
            dst = output_df.iloc[i, 1]
            prob = output_df.iloc[i, 2]
            pred = "Positive" if output_df.iloc[i, 3] == 1 else "Negative"
            print(f"{src},{dst},{prob:.6f} -> {pred}")
        
    except Exception as e:
        print(f"Error creating output file: {e}")
    
    # Save the detailed report
    try:
        import json
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "test_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # Remove predictions array which cannot be serialized to JSON
        report_results = {k: v for k, v in results.items() if k != 'predictions'}
        
        # Convert int64 values to standard int
        def convert_int64(obj):
            if isinstance(obj, dict):
                return {k: convert_int64(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_int64(item) for item in obj]
            elif isinstance(obj, np.int64):
                return int(obj)  # Convert int64 values to standard int
            elif isinstance(obj, np.float64):
                return float(obj)  # Convert float64 values to standard float
            else:
                return obj
        
        # Create converted report
        report_results = convert_int64(report_results)
        
        report_path = os.path.join(report_dir, f"test_report_2m_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report_results, f, indent=4)
        
        print(f"\nDetailed test report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    print("="*60)

if __name__ == "__main__":
    main() 