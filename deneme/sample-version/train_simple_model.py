import sys
import os

# Ana dizini import yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from tqdm import tqdm
import time
import pandas as pd

# Modeli doğrudan import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from improved_test_model import SimpleTemporalModel

def load_datasets():
    """
    Eğitim için gerekli verileri yükler.
    
    Returns:
        tuple: (node_features, edge_data, edge_type_features)
    """
    print("Veri dosyaları yükleniyor...")
    
    # Node özellikleri dosyası
    node_features_path = "data/node_features.csv"
    if os.path.exists(node_features_path):
        node_features_df = pd.read_csv(node_features_path, header=None)
        print(f"Node özellikler: {node_features_df.shape[0]} satır, {node_features_df.shape[1]} sütun")
    else:
        print(f"UYARI: {node_features_path} bulunamadı.")
        node_features_df = None
    
    # Edge tipi özellikleri
    edge_type_path = "data/edge_type_features.csv"
    if os.path.exists(edge_type_path):
        edge_type_df = pd.read_csv(edge_type_path, header=None)
        print(f"Edge tipi özellikleri: {edge_type_df.shape[0]} satır, {edge_type_df.shape[1]} sütun")
    else:
        print(f"UYARI: {edge_type_path} bulunamadı.")
        edge_type_df = None
    
    # Edges dosyası
    edges_path = "data/edges_train_A_prototype_sample.csv"
    if os.path.exists(edges_path):
        edges_df = pd.read_csv(edges_path, header=None)
        print(f"Edge verileri: {edges_df.shape[0]} satır, {edges_df.shape[1]} sütun")
    else:
        print(f"UYARI: {edges_path} bulunamadı.")
        edges_df = None
    
    return node_features_df, edges_df, edge_type_df

def prepare_graph_data(node_features_df, edges_df, edge_type_df):
    """
    Graf veri yapısını oluştur
    """
    print("Veri yapısı oluşturuluyor...")
    
    # Edge verilerini işle
    if edges_df is not None:
        num_columns = edges_df.shape[1]
        print(f"Edge dosyasında {num_columns} sütun bulundu.")
        
        # Sütun sayısına göre işlem yap
        if num_columns == 4:
            print("4 sütunlu edge formatı algılandı.")
            edges_df.columns = ['source', 'target', 'edge_type', 'time']
            
            # Eğitim için yapay etiketler oluştur
            edges_df['label'] = np.random.choice([0, 1], size=len(edges_df), p=[0.4, 0.6])
            
            # Zamansal bilgileri ekle
            edges_df['start_time'] = edges_df['time']
            edges_df['end_time'] = edges_df['time']
            
        elif num_columns == 6:
            print("6 sütunlu edge formatı algılandı.")
            edges_df.columns = ['source', 'target', 'edge_type', 'start_time', 'end_time', 'label']
            
        else:
            print(f"UYARI: Desteklenmeyen edge sütun sayısı: {num_columns}")
            return None, None, None, None, 0
        
        # Label dağılımını kontrol et
        print("\nEtiket Dağılımı:")
        print(edges_df['label'].value_counts())
        print(f"Pozitif oran: {edges_df['label'].mean():.4f}")
    else:
        print("Edge verisi olmadan devam edilemez.")
        return None, None, None, None, 0
    
    # Node özelliklerini kullan veya oluştur
    if node_features_df is not None:
        # Doğrudan node_features dosyasını kullan
        # İlk sütun node ID olabilir, kontrol et
        first_col = node_features_df.iloc[:, 0]
        if first_col.nunique() == len(first_col):
            print("Node features ilk sütunu ID olarak algılandı.")
            node_id_mapping = {id_val: idx for idx, id_val in enumerate(first_col)}
            node_features = node_features_df.iloc[:, 1:].values
            num_nodes = len(node_id_mapping)
            
            # Edge indekslerini eşle
            edges_df['source_mapped'] = edges_df['source'].map(node_id_mapping)
            edges_df['target_mapped'] = edges_df['target'].map(node_id_mapping)
            
            # Eksik eşleşmeleri kontrol et
            missing_source = edges_df['source_mapped'].isna().sum()
            missing_target = edges_df['target_mapped'].isna().sum()
            
            if missing_source > 0 or missing_target > 0:
                print(f"UYARI: {missing_source} kaynak ve {missing_target} hedef node eşleşmesi bulunamadı.")
                print("Eksik node'lar için yeni ID'ler oluşturuluyor...")
                
                # Eksik node'ları topla ve yeni ID'ler ata
                all_nodes = set(node_id_mapping.keys())
                edge_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
                missing_nodes = edge_nodes - all_nodes
                
                # Yeni node ID'leri oluştur
                next_id = num_nodes
                for node in missing_nodes:
                    node_id_mapping[node] = next_id
                    next_id += 1
                
                # Node özellik matrisini genişlet
                additional_features = np.zeros((len(missing_nodes), node_features.shape[1]))
                node_features = np.vstack([node_features, additional_features])
                num_nodes = next_id
                
                # Edge indekslerini tekrar eşle
                edges_df['source_mapped'] = edges_df['source'].map(node_id_mapping)
                edges_df['target_mapped'] = edges_df['target'].map(node_id_mapping)
        else:
            print("Node features doğrudan kullanılıyor (ID sütunu yok).")
            node_features = node_features_df.values
            num_nodes = len(node_features)
            
            # Node ID'lerini sürekli hale getir
            unique_nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()
            if len(unique_nodes) > num_nodes:
                print(f"UYARI: Edge'lerde {len(unique_nodes)} farklı node var ama sadece {num_nodes} node özelliği var.")
                print("Node özellik matrisi genişletiliyor...")
                
                # Matrisi genişlet
                additional_features = np.zeros((len(unique_nodes) - num_nodes, node_features.shape[1]))
                node_features = np.vstack([node_features, additional_features])
                num_nodes = len(unique_nodes)
            
            # ID'leri eşle
            node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
            edges_df['source_mapped'] = edges_df['source'].map(node_mapping)
            edges_df['target_mapped'] = edges_df['target'].map(node_mapping)
    else:
        # Node özelliklerini oluştur
        print("Node özellikleri dosyası bulunamadı, özellikler hesaplanıyor...")
        
        # Benzersiz node'ları bul
        unique_nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()
        node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
        num_nodes = len(unique_nodes)
        
        # ID'leri eşle
        edges_df['source_mapped'] = edges_df['source'].map(node_mapping)
        edges_df['target_mapped'] = edges_df['target'].map(node_mapping)
        
        # Node özellikleri matrisi oluştur - daha zengin özellikler
        node_features = np.zeros((num_nodes, 9))
        
        # Node özelliklerini hesapla
        from collections import defaultdict
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        node_edge_types = defaultdict(set)
        node_time_mean = defaultdict(list)
        
        # Özellikleri topla
        for _, row in edges_df.iterrows():
            src, dst = row['source_mapped'], row['target_mapped']
            edge_type = row['edge_type']
            start_time = row['start_time']
            
            in_degree[dst] += 1
            out_degree[src] += 1
            node_edge_types[src].add(edge_type)
            node_edge_types[dst].add(edge_type)
            node_time_mean[src].append(start_time)
            node_time_mean[dst].append(start_time)
        
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
            
            # Ek özellikler
            node_features[node_id, 6] = node_features[node_id, 1] / (sum(in_degree.values()) + 1e-8)  # Giriş derecesi oranı
            node_features[node_id, 7] = node_features[node_id, 0] / (node_features[node_id, 1] + 1e-8)  # Out/In oranı
        
        # Normalizasyon
        for i in range(8):
            if np.std(node_features[:, i]) > 0:
                node_features[:, i] = (node_features[:, i] - np.mean(node_features[:, i])) / np.std(node_features[:, i])
    
    # Edge indeksi oluştur
    edge_index = np.stack([edges_df['source_mapped'].values, edges_df['target_mapped'].values])
    
    # Edge tipi özelliklerini ekle
    if edge_type_df is not None:
        print("Edge tipi özellikleri kullanılıyor...")
        # Edge tipi feature vektörlerini oluştur
        # İlk sütun edge_type ID'si olabilir, kontrol et
        first_col = edge_type_df.iloc[:, 0]
        if first_col.nunique() == len(first_col):
            print("Edge type features ilk sütunu ID olarak algılandı.")
            edge_type_mapping = {int(id_val): idx for idx, id_val in enumerate(first_col)}
            edge_type_features = edge_type_df.iloc[:, 1:].values
            
            # Edge tiplerinin feature vektörlerini oluştur
            edge_features_list = []
            for et in edges_df['edge_type']:
                if et in edge_type_mapping:
                    # Edge tipi feature vektörü
                    et_idx = edge_type_mapping[et]
                    edge_features_list.append(edge_type_features[et_idx])
                else:
                    # Edge tipi bulunamadıysa sıfır vektörü
                    print(f"UYARI: Edge tipi {et} özelliklerde bulunamadı.")
                    edge_features_list.append(np.zeros(edge_type_features.shape[1]))
            
            edge_attr = np.array(edge_features_list)
        else:
            print("Edge type features doğrudan kullanılıyor (ID sütunu yok).")
            # Edge type özelliklerini doğrudan kullan
            edge_attr = np.zeros((len(edges_df), edge_type_df.shape[1]))
            for i, et in enumerate(edges_df['edge_type']):
                if 0 <= et < len(edge_type_df):
                    edge_attr[i] = edge_type_df.iloc[et].values
    else:
        print("Edge tipi özellikleri bulunamadı, basit özellikler kullanılıyor...")
        # Basit edge özellikleri oluştur
        # Zaman bilgisini normalize et
        time_mean = edges_df['start_time'].mean()
        time_std = edges_df['start_time'].std() + 1e-8
        time_norm = (edges_df['start_time'] - time_mean) / time_std
        
        # Edge tipi one-hot encoding
        edge_types = edges_df['edge_type'].unique()
        edge_type_one_hot = np.zeros((len(edges_df), len(edge_types)))
        for i, et in enumerate(edges_df['edge_type']):
            edge_type_idx = np.where(edge_types == et)[0][0]
            edge_type_one_hot[i, edge_type_idx] = 1
        
        # Edge özellikleri oluştur
        edge_attr = np.hstack([
            time_norm.values.reshape(-1, 1),
            edge_type_one_hot
        ])
    
    print(f"Veri hazırlama tamamlandı:")
    print(f"- Node sayısı: {num_nodes}")
    print(f"- Edge sayısı: {edge_index.shape[1]}")
    print(f"- Node özellik boyutu: {node_features.shape[1]}")
    print(f"- Edge özellik boyutu: {edge_attr.shape[1]}")
    
    return node_features, edge_index, edge_attr, edges_df['label'].values, num_nodes

def train_simple_model(node_features, edge_index, edge_attr, labels, num_nodes, device, val_ratio=0.2, test_ratio=0.1):
    """
    Basit bir temporal graph modelini eğitir.
    
    Args:
        node_features (np.ndarray): Node özellikleri [num_nodes, num_features]
        edge_index (np.ndarray): Edge indeksleri [2, num_edges]
        edge_attr (np.ndarray): Edge özellikleri [num_edges, num_edge_features]
        labels (np.ndarray): Edge etiketleri [num_edges]
        num_nodes (int): Graf içindeki toplam node sayısı
        device (torch.device): Model çalıştırılacak cihaz
        val_ratio (float): Validasyon seti oranı
        test_ratio (float): Test seti oranı
    
    Returns:
        dict: Eğitim geçmişi
        dict: Test metrikler
    """
    print("Basit model eğitimi başlıyor...")
    
    # Veriyi train, validation ve test olarak ayır
    num_edges = edge_index.shape[1]
    
    # Zaman bazlı bölünme için zamansal bilgiler kullanılabilir mi kontrol et
    try:
        # Edge attr'ın ilk sütunu genellikle zaman bilgisidir
        if edge_attr is not None and edge_attr.shape[1] > 0:
            print("Zamansal bölünme stratejisi kullanılıyor...")
            # Zamansal bilgilere göre sırala
            temporal_info = edge_attr[:, 0]  # İlk sütunu zaman bilgisi olarak kullan
            sorted_indices = np.argsort(temporal_info)
            
            # Zamansal sırayla verileri böl
            test_size = int(num_edges * test_ratio)
            val_size = int(num_edges * val_ratio)
            train_size = num_edges - val_size - test_size
            
            train_indices = sorted_indices[:train_size]
            val_indices = sorted_indices[train_size:train_size + val_size]
            test_indices = sorted_indices[train_size + val_size:]
            
            print("Zamansal bölünme başarılı!")
        else:
            raise ValueError("Edge özellikleri bulunamadı, rastgele bölünme kullanılacak.")
    except:
        print("Zamansal bölünme başarısız, rastgele bölünme kullanılıyor...")
        # Rastgele karıştır ve böl
        indices = np.random.permutation(num_edges)
        
        test_size = int(num_edges * test_ratio)
        val_size = int(num_edges * val_ratio)
        train_size = num_edges - val_size - test_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
    print(f"Eğitim: {train_size}, Validasyon: {val_size}, Test: {test_size} edge")
    
    # Veri setlerini oluştur
    train_edge_index = edge_index[:, train_indices]
    train_edge_attr = edge_attr[train_indices]
    train_labels = labels[train_indices]
    
    val_edge_index = edge_index[:, val_indices]
    val_edge_attr = edge_attr[val_indices]
    val_labels = labels[val_indices]
    
    test_edge_index = edge_index[:, test_indices]
    test_edge_attr = edge_attr[test_indices]
    test_labels = labels[test_indices]
    
    # Modeli oluştur
    model = SimpleTemporalModel(
        node_features=node_features.shape[1],
        hidden_channels=128,
        dropout=0.5  # Dropout oranını artır
    ).to(device)
    
    # Model parametrelerini göster
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model toplam parametre sayısı: {total_params:,}")
    
    # Optimizasyon parametreleri - weight decay artırıldı
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # 5e-4'ten 1e-3'e yükseltildi
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = torch.nn.BCELoss()
    
    # L1 Regularization ekle
    l1_lambda = 1e-5
    
    # Eğitim parametreleri
    epochs = 50
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Eğitim geçmişi
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_ap': [],
        'val_f1': []
    }
    
    # Tensörleri hazırla
    x = torch.FloatTensor(node_features).to(device)
    train_edge_index_tensor = torch.LongTensor(train_edge_index).to(device)
    val_edge_index_tensor = torch.LongTensor(val_edge_index).to(device)
    
    # Tahmini toplam süreyi hesapla
    epoch_times = []
    
    # Eğitim döngüsü
    epoch_pbar = tqdm(range(epochs), desc="Eğitim", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        model.train()
        
        # Forward pass
        node_embeddings = model(x, train_edge_index_tensor)
        
        # Pozitif edge'ler için tahmin
        pos_pred = model.predict_link(node_embeddings, train_edge_index_tensor)
        
        # Negatif edge'ler oluştur
        neg_edge_index = generate_negative_samples(train_edge_index, num_nodes, train_size)
        neg_edge_index_tensor = torch.LongTensor(neg_edge_index).to(device)
        
        # Negatif edge'ler için tahmin
        neg_pred = model.predict_link(node_embeddings, neg_edge_index_tensor)
        
        # Tüm tahminleri ve etiketleri birleştir
        all_pred = torch.cat([pos_pred, neg_pred])
        all_labels = torch.cat([
            torch.ones_like(pos_pred),
            torch.zeros_like(neg_pred)
        ])
        
        # Loss hesapla
        loss = criterion(all_pred, all_labels)
        
        # L1 regularization ekle
        l1_reg = 0
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        
        loss += l1_lambda * l1_reg
        
        # Geriye yayılım
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Eğitim kaybını kaydet
        train_loss = loss.item()
        history['train_loss'].append(train_loss)
        
        # Validasyon
        model.eval()
        with torch.no_grad():
            # Node embeddings hesapla
            node_embeddings = model(x, val_edge_index_tensor)
            
            # Pozitif edge'ler için tahmin
            val_pos_pred = model.predict_link(node_embeddings, val_edge_index_tensor)
            
            # Negatif edge'ler oluştur
            val_neg_edge_index = generate_negative_samples(val_edge_index, num_nodes, val_size)
            val_neg_edge_index_tensor = torch.LongTensor(val_neg_edge_index).to(device)
            
            # Negatif edge'ler için tahmin
            val_neg_pred = model.predict_link(node_embeddings, val_neg_edge_index_tensor)
            
            # Tüm tahminleri ve etiketleri birleştir
            val_all_pred = torch.cat([val_pos_pred, val_neg_pred]).cpu().numpy()
            val_all_labels = np.concatenate([
                np.ones_like(val_pos_pred.cpu().numpy()),
                np.zeros_like(val_neg_pred.cpu().numpy())
            ])
            
            # Loss ve metrikler
            val_loss = criterion(
                torch.tensor(val_all_pred, dtype=torch.float32),
                torch.tensor(val_all_labels, dtype=torch.float32)
            ).item()
            
            val_auc = roc_auc_score(val_all_labels, val_all_pred)
            val_ap = average_precision_score(val_all_labels, val_all_pred)
            val_f1 = f1_score(val_all_labels, (val_all_pred > 0.5).astype(int))
            
            # Eğitim geçmişini kaydet
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_ap'].append(val_ap)
            history['val_f1'].append(val_f1)
            
            # Öğrenme oranını ayarla
            scheduler.step(val_loss)
        
        # Epoch süresini ölç
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Kalan süreyi tahmin et
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs
        remaining_minutes = int(remaining_time // 60)
        remaining_seconds = int(remaining_time % 60)
        
        # İlerleme çubuğunu güncelle
        epoch_pbar.set_postfix({
            'Train Loss': f"{train_loss:.4f}",
            'Val Loss': f"{val_loss:.4f}",
            'Val AUC': f"{val_auc:.4f}",
            'Epoch Time': f"{epoch_duration:.2f}s",
            'Remaining': f"{remaining_minutes}m {remaining_seconds}s"
        })
            
        # Eğitim durumunu yazdır
        epoch_info = (f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Val F1: {val_f1:.4f} "
                f"[{epoch_duration:.2f}s, Kalan: {remaining_minutes}m {remaining_seconds}s]")
        
        # Epoch bilgilerini ayrıca yazdır
        if (epoch + 1) % 5 == 0:
            print(epoch_info)
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"\nYeni en iyi model kaydedildi! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nErken durdurma! {patience} epoch boyunca iyileşme olmadı.")
                break
    
    # En iyi modeli yükle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Modeli kaydet
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': best_val_loss,
        'val_auc': history['val_auc'][-1],
        'val_ap': history['val_ap'][-1],
        'val_f1': history['val_f1'][-1]
    }, "checkpoints/simple_model_best.pt")
    print("Model kaydedildi: checkpoints/simple_model_best.pt")
    
    # Test
    model.eval()
    with torch.no_grad():
        # Test edge'leri için embeddings
        node_embeddings = model(x, torch.LongTensor(edge_index).to(device))
        
        # Test edge'leri için tahmin
        test_edge_index_tensor = torch.LongTensor(test_edge_index).to(device)
        test_pred = model.predict_link(node_embeddings, test_edge_index_tensor).cpu().numpy()
        
        # Metrikler
        test_auc = roc_auc_score(test_labels, test_pred)
        test_ap = average_precision_score(test_labels, test_pred)
        
        # En iyi eşik değeri bul
        precision, recall, thresholds = precision_recall_curve(test_labels, test_pred)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        # Binary tahminler
        binary_preds = (test_pred >= best_threshold).astype(int)
        test_f1 = f1_score(test_labels, binary_preds)
        
        # Confusion matrix hesapla
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels, binary_preds)
        
        print(f"\nTest Sonuçları:")
        print(f"AUC: {test_auc:.4f}")
        print(f"AP: {test_ap:.4f}")
        print(f"F1 Score (threshold={best_threshold:.4f}): {test_f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        test_metrics = {
            'auc': test_auc,
            'ap': test_ap,
            'f1': test_f1,
            'threshold': best_threshold
        }
        
        # Test tahminlerini görselleştir
        plt.figure(figsize=(10, 6))
        plt.hist(test_pred[test_labels==1], bins=50, alpha=0.5, label='Positive Samples')
        plt.hist(test_pred[test_labels==0], bins=50, alpha=0.5, label='Negative Samples')
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
        plt.xlabel('Prediction Scores')
        plt.ylabel('Count')
        plt.title('Distribution of Link Prediction Scores on Test Set')
        plt.legend()
        plt.savefig('test_prediction_distribution.png')
        print("Test tahmin dağılımı 'test_prediction_distribution.png' olarak kaydedildi.")
        
    # Eğitim geçmişini görselleştir
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Val AUC')
    plt.plot(history['val_ap'], label='Val AP')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return history, test_metrics

def generate_negative_samples(edge_index, num_nodes, num_samples):
    """
    Negatif örnekler oluştur
    """
    # Pozitif edge'leri bir set olarak sakla
    existing_edges = set()
    for i in range(edge_index.shape[1]):
        existing_edges.add((edge_index[0, i], edge_index[1, i]))
    
    # Negatif edge'leri oluştur
    neg_edges = []
    while len(neg_edges) < num_samples:
        # Rastgele bir kaynak ve hedef node seç
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Kendine dönen kenarları ve zaten var olan kenarları atla
        if src != dst and (src, dst) not in existing_edges:
            neg_edges.append([src, dst])
            existing_edges.add((src, dst))
    
    return np.array(neg_edges).T  # [2, num_samples] formatında

def main():
    """
    Ana fonksiyon
    """
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri dosyalarını yükle
    node_features_df, edges_df, edge_type_df = load_datasets()
    
    # Graf verilerini hazırla
    node_features, edge_index, edge_attr, labels, num_nodes = prepare_graph_data(
        node_features_df, edges_df, edge_type_df
    )
    
    if node_features is None or edge_index is None:
        print("Veri hazırlama başarısız, program sonlandırılıyor.")
        return
    
    # Modeli eğit
    history, test_metrics = train_simple_model(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=labels,
        num_nodes=num_nodes,
        device=device
    )
    
    print("\nEğitim tamamlandı!")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test AP: {test_metrics['ap']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 