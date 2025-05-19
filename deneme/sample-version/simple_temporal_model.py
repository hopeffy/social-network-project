import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class SimpleTemporalModel(nn.Module):
    """
    Daha basit bir temporal graph modeli.
    Karmaşık GAT yapısı yerine GCN/SAGE tabanlı daha basit bir yapı kullanıyor.
    """
    def __init__(self, node_features, hidden_channels=128, dropout=0.3):
        super(SimpleTemporalModel, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        
        # Daha basit input projection
        self.input_projection = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Daha az parametre ile graph convolution
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Basit temporal embedding
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Basit link prediction katmanı
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
        # Ağırlık başlatma
        self._reset_parameters()
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_time=None):
        """
        Graph üzerinde forward pass.
        
        Args:
            x: Node özellikleri [num_nodes, node_features]
            edge_index: Edge indeksleri [2, num_edges]
            edge_time: Edge zaman bilgisi [num_edges] (opsiyonel)
            
        Returns:
            Node embeddings [num_nodes, hidden_channels]
        """
        # Node özellikleri projeksiyonu
        x = self.input_projection(x)
        
        # İlk graph konvolüsyon
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # İkinci graph konvolüsyon
        x2 = self.conv2(x1, edge_index)
        
        # Residual bağlantı
        x = x2 + x1
        
        return x
    
    def predict_link(self, h, edge_index):
        """
        Link prediction fonksiyonu
        
        Args:
            h: Node embeddings [num_nodes, hidden_channels]
            edge_index: Tahmin edilecek edge'ler [2, num_edges]
            
        Returns:
            Link olma olasılıkları [num_edges]
        """
        # Edge'lerin kaynak ve hedef node'larını al
        src, dst = edge_index
        
        # Kaynak ve hedef node'ların embedding'lerini birleştir
        h_edge = torch.cat([h[src], h[dst]], dim=1)
        
        # Link prediction
        return torch.sigmoid(self.link_predictor(h_edge).squeeze(-1)) 