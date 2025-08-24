import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalInformationFusionNormalized(nn.Module):
    """
    Módulo de "External Information" para dados já normalizados.
    
    Entradas (todas normalizadas):
    --------
    uid     : (B,)   torch.LongTensor  – id do usuário [0 … n_users-1]
    d_norm  : (B,)   torch.FloatTensor – dia normalizado [0,1]
    t_sin   : (B,)   torch.FloatTensor – timeslot sin [-1,1]
    t_cos   : (B,)   torch.FloatTensor – timeslot cos [-1,1]
    city    : (B,)   torch.LongTensor  – cidade codificada [0,1,2,3]
    poi_norm: (B,85) torch.FloatTensor – POIs normalizados [0,1]

    Saída:
    -----
    fused : (B, out_dim) torch.FloatTensor – concat de todos os embeddings + projeções
    """

    def __init__(
        self,
        n_users: int,          # número de usuários
        n_cities: int = 4,     # 4 cidades (A,B,C,D)
        user_emb_dim: int = 10,    # dim do embedding de usuário
        city_emb_dim: int = 4,     # dim do embedding de cidade
        temporal_dim: int = 8,     # dim das projeções temporais
        poi_out_dim: int = 10,     # dim da projeção POI
        disable_poi: bool = False  # para debug/ablation
    ):
        super().__init__()
        self.disable_poi = disable_poi
        
        # 🆕 Embeddings apenas para variáveis categóricas
        self.uid_emb = nn.Embedding(n_users, user_emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        
        # 🆕 Projeções para variáveis numéricas normalizadas
        self.day_proj = nn.Linear(1, temporal_dim)      # d_norm [0,1] → temporal_dim
        self.time_proj = nn.Linear(2, temporal_dim)     # [t_sin, t_cos] → temporal_dim
        self.poi_proj = nn.Linear(85, poi_out_dim)      # POI_norm[85] → poi_out_dim
        
        # Dimensão total de saída
        self.out_dim = user_emb_dim + city_emb_dim + temporal_dim * 2 + poi_out_dim
        
        # Inicialização mais estável
        self._init_weights()
        
    def _init_weights(self):
        """Inicialização Xavier para estabilidade"""
        nn.init.xavier_uniform_(self.day_proj.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.poi_proj.weight)
        nn.init.zeros_(self.day_proj.bias)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.zeros_(self.poi_proj.bias)
        
        # Embeddings com std menor
        nn.init.normal_(self.uid_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.city_emb.weight, mean=0.0, std=0.02)

    def forward(self, uid, d_norm, t_sin, t_cos, city, poi_norm):
        """
        Forward pass com dados normalizados
        
        Args:
            uid: (B,) LongTensor - IDs de usuário
            d_norm: (B,) FloatTensor - dias normalizados [0,1]
            t_sin: (B,) FloatTensor - seno do timeslot [-1,1]
            t_cos: (B,) FloatTensor - cosseno do timeslot [-1,1]
            city: (B,) LongTensor - cidades codificadas [0,1,2,3]
            poi_norm: (B,85) FloatTensor - POIs normalizados [0,1]
        """
        batch_size = uid.size(0)
        
        # Embeddings categóricos
        e_uid = self.uid_emb(uid)    # (B, user_emb_dim)
        e_city = self.city_emb(city) # (B, city_emb_dim)
        
        # Projeções temporais
        d_input = d_norm.unsqueeze(1)  # (B, 1)
        e_day = F.relu(self.day_proj(d_input))  # (B, temporal_dim)
        
        t_input = torch.stack([t_sin, t_cos], dim=1)  # (B, 2)
        e_time = F.relu(self.time_proj(t_input))      # (B, temporal_dim)
        
        # Projeção POI (com opção de desabilitar para debug)
        if self.disable_poi:
            e_poi = torch.zeros(batch_size, self.poi_proj.out_features,
                               device=poi_norm.device, dtype=poi_norm.dtype)
        else:
            e_poi = F.relu(self.poi_proj(poi_norm))   # (B, poi_out_dim)
        
        # Concatena tudo
        fused = torch.cat([e_uid, e_city, e_day, e_time, e_poi], dim=-1)
        return fused


class ExternalInformationDense(nn.Module):
    """
    Camada densa que reduz a dimensionalidade do vetor fundido.
    Agora com normalização e dropout para maior estabilidade.
    """
    def __init__(self, in_dim: int, out_dim: int = 20, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, out_dim)
        
        # Inicialização Xavier
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_dim) - saída do ExternalInformationFusion
        Returns:
            (batch_size, out_dim) - vetor denso final
        """
        x = self.layer_norm(x)
        x = self.dropout(x)
        return F.relu(self.fc(x))