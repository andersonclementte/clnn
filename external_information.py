import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalInformationFusionDTPC(nn.Module):
    """
    Módulo de “External Information” usando as colunas
    uid, d (dia), t (slot), city (letra) e poi (vetor de 85).
    
    Entradas
    --------
    uid   : (B,)   torch.LongTensor  – id do usuário [0 … n_users-1]
    d     : (B,)   torch.LongTensor  – dia, de 0 a n_days-1
    t     : (B,)   torch.LongTensor  – slot de tempo, de 0 a n_slots-1
    city  : (B,)   torch.LongTensor  – id da cidade, de 0 a n_cities-1
    poi   : (B,85) torch.FloatTensor – vetor de contagens de POIs

    Saída
    -----
    fused : (B, out_dim) torch.FloatTensor – concat de todos os embeddings + projeção POI
    """

    def __init__(
        self,
        n_users: int,      # número de usuários
        n_days: int,       # ex: 75
        n_slots: int,      # ex: 48
        n_cities: int,     # ex: 4
        emb_dim: int = 10,     # dimensão de cada embedding
        poi_in_dim: int = 85,  # tamanho do vetor POI
        poi_out_dim: int = 10  # dimensão de saída da projeção POI
    ):
        super().__init__()
        # Embeddings para cada variável categórica
        self.uid_emb   = nn.Embedding(n_users, emb_dim)
        self.day_emb   = nn.Embedding(n_days, emb_dim)
        self.slot_emb  = nn.Embedding(n_slots, emb_dim)
        self.city_emb  = nn.Embedding(n_cities, emb_dim)
        # Projeção do vetor de POIs
        self.poi_proj  = nn.Linear(poi_in_dim, poi_out_dim)
        # dimensão de saída: 4 embeddings + projeção POI
        self.out_dim   = emb_dim * 4 + poi_out_dim

    def forward(
        self,
        uid:  torch.LongTensor,
        d:    torch.LongTensor,
        t:    torch.LongTensor,
        city: torch.LongTensor,
        poi:  torch.FloatTensor
    ) -> torch.FloatTensor:
        # embeddings individuais → (B, emb_dim)
        e_uid  = self.uid_emb(uid)
        e_d    = self.day_emb(d)
        e_t    = self.slot_emb(t)
        e_city = self.city_emb(city)
        # projeção POI 85→poi_out_dim + ReLU
        e_poi  = F.relu(self.poi_proj(poi))
        # concatena tudo → (B, out_dim)
        fused = torch.cat([e_uid, e_d, e_t, e_city, e_poi], dim=-1)
        return fused

class ExternalInformationDense(nn.Module):
    """
    Camada densa que mapeia o vetor de 60 dimensões produzido pelo
    ExternalInformationFusion para um vetor de 20 dimensões.
    """
    def __init__(self, in_dim: int = 60, out_dim: int = 20):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor de shape (batch_size, in_dim)
        retorna: Tensor de shape (batch_size, out_dim)
        """
        return F.relu(self.fc(x))