import sys
import os

# Ana dizini import yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from models.temporal_gat import TemporalGAT  # Orijinal model
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import defaultdict

class SimpleTemporalModel(torch.nn.Module):
    """
    Daha basit bir temporal graph modeli.
    Karmaşık GAT yapısı yerine GCN/SAGE tabanlı daha basit bir yapı kullanıyor.
    """
    def __init__(self, node_features, hidden_channels=128, dropout=0.5):
        super(SimpleTemporalModel, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        
        # Daha basit input projection
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(node_features, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Daha az parametre ile graph convolution
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Basit temporal embedding
        self.temporal_embedding = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Basit link prediction katmanı
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1)
        )
        
        # Ağırlık başlatma
        self._reset_parameters()
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_time=None):
        """
        Graph üzerinde forward pass.
        """
        # Node özellikleri projeksiyonu
        x = self.input_projection(x)
        
        # İlk graph konvolüsyon
        x1 = self.conv1(x, edge_index)
        x1 = torch.nn.functional.relu(x1)
        x1 = torch.nn.functional.dropout(x1, p=0.4, training=self.training)
        
        # İkinci graph konvolüsyon
        x2 = self.conv2(x1, edge_index)
        x2 = torch.nn.functional.dropout(x2, p=0.4, training=self.training)
        
        # Residual bağlantı
        x = x2 + x1
        
        return x
    
    def predict_link(self, h, edge_index):
        """
        Link prediction fonksiyonu
        """
        # Edge'lerin kaynak ve hedef node'larını al
        src, dst = edge_index
        
        # Kaynak ve hedef node'ların embedding'lerini birleştir
        h_edge = torch.cat([h[src], h[dst]], dim=1)
        
        # Link prediction
        return torch.sigmoid(self.link_predictor(h_edge).squeeze(-1))

def load_model(model_path, node_features, device, model_type="simple"):
    """
    Model yükleme fonksiyonu
    """
    if model_type == "simple":
        model = SimpleTemporalModel(
            node_features=node_features,
            hidden_channels=128,
            dropout=0.5
        )
    else:
        # Alternatif olarak orijinal modeli de kullanabiliriz
        model = TemporalGAT(
            node_features=node_features,
            hidden_channels=256,
            num_heads=4,
            dropout=0.3,
            use_fp16=True
        )
    
    try:
        # Modeli yükle
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model başarıyla yüklendi!")
    except Exception as e:
        # Yeni bir model eğitmek için
        print(f"Model yüklenemedi: {e}")
        print("Yeni model başlatılıyor...")
    
    model = model.to(device)
    model.eval()
    return model

def prepare_graph_data(data_file):
    """
    Graf veri yapısını oluştur
    """
    print("Veri okunuyor ve graf yapısı oluşturuluyor...")
    
    # CSV dosyasını oku
    df = pd.read_csv(data_file, header=None)
    
    # Sütun sayısını kontrol et ve uygun şekilde isimlendir
    num_columns = len(df.columns)
    print(f"Dosyada {num_columns} sütun bulundu.")
    
    # İlk 5 satırı göster
    print("\nİlk 5 satır:")
    print(df.head())
    
    # Etiket oluşturma
    if num_columns == 6:
        # 6 sütunlu format - son sütun etiket
        df.columns = ['source', 'target', 'edge_type', 'start_time', 'end_time', 'label']
    elif num_columns == 4:
        # 4 sütunlu format - test verisi format için manuel etiket oluşturma
        print("\nDosya 4 sütunlu. Yapay etiket oluşturuluyor...")
        df.columns = ['source', 'target', 'time', 'weight']
        
        # Zamansal veri ve ağırlık verisine göre yapay etiket oluşturuyoruz
        # Ağırlık yüksekse veya zaman yakınsa 1, değilse 0
        # Eğer time ve weight gerçek sayılar ise
        time_numeric = pd.to_numeric(df['time'], errors='coerce')
        weight_numeric = pd.to_numeric(df['weight'], errors='coerce')
        
        if not time_numeric.isna().all() and not weight_numeric.isna().all():
            print("Zaman ve ağırlık sütunları sayısal.")
            # Ağırlıkları normalize et
            if not weight_numeric.isna().all():
                weight_norm = (weight_numeric - weight_numeric.min()) / (weight_numeric.max() - weight_numeric.min() + 1e-8)
                df['label'] = (weight_norm >= 0.5).astype(int)
            else:
                # Rastgele etiketler
                df['label'] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
        else:
            print("Zaman ve ağırlık sütunları sayısal değil.")
            # Rastgele etiketler
            df['label'] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
        
        # Eksik sütunları oluştur
        df['edge_type'] = 1  # Varsayılan edge tipi
        df['start_time'] = pd.to_numeric(df['time'], errors='coerce').fillna(0)
        df['end_time'] = df['start_time']
    else:
        print(f"Desteklenmeyen sütun sayısı: {num_columns}")
        raise ValueError(f"Desteklenmeyen veri formatı. Sütun sayısı: {num_columns}")
    
    # Label dağılımını kontrol et
    print("\nEtiket Dağılımı:")
    print(df['label'].value_counts())
    print(f"Pozitif oran: {df['label'].mean():.4f}")
    
    # Node ID'lerini sürekli hale getir
    unique_nodes = pd.concat([df['source'], df['target']]).unique()
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # ID'leri eşle
    df['source_mapped'] = df['source'].map(node_mapping)
    df['target_mapped'] = df['target'].map(node_mapping)
    
    # Zaman özelliklerini normalize et
    time_mean = df['start_time'].mean()
    time_std = df['start_time'].std()
    df['start_time_norm'] = (df['start_time'] - time_mean) / (time_std + 1e-8)
    df['end_time_norm'] = (df['end_time'] - time_mean) / (time_std + 1e-8)
    df['time_diff'] = df['end_time'] - df['start_time']
    df['time_diff_norm'] = (df['time_diff'] - df['time_diff'].mean()) / (df['time_diff'].std() + 1e-8)
    
    # Graf oluştur
    G = nx.DiGraph()
    
    # Node özelliklerini hesapla
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    node_edge_types = defaultdict(set)
    node_time_mean = defaultdict(list)
    
    # Özellikleri topla
    for _, row in df.iterrows():
        src, dst = row['source_mapped'], row['target_mapped']
        edge_type = row['edge_type']
        start_time = row['start_time_norm']
        
        in_degree[dst] += 1
        out_degree[src] += 1
        node_edge_types[src].add(edge_type)
        node_edge_types[dst].add(edge_type)
        node_time_mean[src].append(start_time)
        node_time_mean[dst].append(start_time)
    
    # Node özellikleri matrisi oluştur - daha zengin özellikler
    num_nodes = len(unique_nodes)
    node_features = np.zeros((num_nodes, 9))
    
    for node_id in range(num_nodes):
        # Temel graf özellikleri
        node_features[node_id, 0] = out_degree[node_id]  # Çıkış derecesi
        node_features[node_id, 1] = in_degree[node_id]   # Giriş derecesi
        node_features[node_id, 2] = out_degree[node_id] + in_degree[node_id]  # Toplam derece
        
        # İlişki türü çeşitliliği
        node_features[node_id, 3] = len(node_edge_types[node_id])
        
        # Zaman özellikleri
        if node_time_mean[node_id]:
            node_features[node_id, 4] = np.mean(node_time_mean[node_id])
            node_features[node_id, 5] = np.std(node_time_mean[node_id]) if len(node_time_mean[node_id]) > 1 else 0
        
        # Graf yapısal özellikleri (PageRank benzeri)
        node_features[node_id, 6] = node_features[node_id, 1] / (sum(in_degree.values()) + 1e-8)  # Giriş derecesi oranı
        
        # Ek özellikler
        node_features[node_id, 7] = node_features[node_id, 0] / (node_features[node_id, 1] + 1e-8)  # Out/In oranı
    
    # Normalizasyon
    for i in range(8):
        if np.std(node_features[:, i]) > 0:
            node_features[:, i] = (node_features[:, i] - np.mean(node_features[:, i])) / np.std(node_features[:, i])
    
    # Edge dizini oluştur
    edge_index = np.stack([df['source_mapped'].values, df['target_mapped'].values])
    
    # Edge özellikleri oluştur
    edge_attr = np.stack([
        df['edge_type'].values,
        df['start_time_norm'].values,
        df['end_time_norm'].values,
        df['time_diff_norm'].values,
    ], axis=1)
    
    return node_features, edge_index, edge_attr, df['label'].values, num_nodes

def test_model(model, node_features, edge_index, edge_attr, labels, device):
    """
    Modeli test et
    """
    model.eval()
    predictions = []
    batch_size = 5000
    
    with torch.no_grad():
        # Node özelliklerini tensöre dönüştür
        x = torch.FloatTensor(node_features).to(device)
        edge_index_tensor = torch.LongTensor(edge_index).to(device)
        
        # Tüm node'lar için embedding hesapla
        node_embeddings = model(x, edge_index_tensor)
        
        # Batch'ler halinde edge tahminleri yap
        num_edges = edge_index.shape[1]
        num_batches = (num_edges + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_edges)
            
            batch_edge_index = edge_index_tensor[:, start_idx:end_idx]
            batch_preds = model.predict_link(node_embeddings, batch_edge_index)
            predictions.append(batch_preds.cpu().numpy())
    
    # Tahminleri birleştir
    all_predictions = np.concatenate(predictions)
    
    # Performans metrikleri hesapla
    auc = roc_auc_score(labels, all_predictions)
    ap = average_precision_score(labels, all_predictions)
    
    # En iyi F1 skoru için eşik değerini bul
    precision, recall, thresholds = precision_recall_curve(labels, all_predictions)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    # Binary tahminler
    binary_preds = (all_predictions >= best_threshold).astype(int)
    
    # Tahmin dağılımını görselleştir
    plt.figure(figsize=(10, 6))
    plt.hist(all_predictions, bins=50, alpha=0.7)
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
    plt.xlabel('Prediction Scores')
    plt.ylabel('Count')
    plt.title('Distribution of Link Prediction Scores')
    plt.legend()
    plt.savefig('prediction_distribution.png')
    plt.close()
    
    return {
        'predictions': all_predictions,
        'binary_predictions': binary_preds,
        'auc': auc,
        'ap': ap,
        'f1': best_f1,
        'threshold': best_threshold
    }

def main():
    # GPU kullanımını kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri dosyası
    data_file = "test-data/input_A_initial.csv"
    
    # Graf verilerini hazırla
    node_features, edge_index, edge_attr, labels, num_nodes = prepare_graph_data(data_file)
    print(f"Node sayısı: {num_nodes}")
    print(f"Edge sayısı: {edge_index.shape[1]}")
    print(f"Node özellik boyutu: {node_features.shape[1]}")
    
    # Farklı model tiplerini test et
    model_configs = [
        {"type": "simple", "name": "Basit GNN Model"},
        {"type": "original", "name": "Orijinal GAT Model"}
    ]
    
    for config in model_configs:
        print(f"\n{config['name']} test ediliyor...")
        
        # Checkpoint dosyası
        if config["type"] == "simple":
            model_path = "checkpoints/simple_model_best.pt"
        else:
            model_path = "checkpoints_a5000/edges_train_A/model_best.pt"
        
        try:
            # Modeli yükle
            model = load_model(
                model_path=model_path, 
                node_features=node_features.shape[1], 
                device=device,
                model_type=config["type"]
            )
            
            # Test et
            results = test_model(
                model=model,
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                labels=labels,
                device=device
            )
            
            # Sonuçları yazdır
            print(f"\n{config['name']} Test Sonuçları:")
            print(f"AUC: {results['auc']:.4f}")
            print(f"AP: {results['ap']:.4f}")
            print(f"F1 Score (threshold={results['threshold']:.4f}): {results['f1']:.4f}")
            
        except Exception as e:
            print(f"Model testi başarısız: {e}")
            continue

if __name__ == "__main__":
    main() 