import torch
import torch.nn as nn

class CoordLSTM(nn.Module):
    """
    LSTM que recebe sequências de coordenadas (x, y) e produz
    um vetor de saída de dimensão 20 para cada sequência.

    Args:
        input_size (int): número de features de entrada (2 para x e y).
        hidden_size (int): dimensão do espaço escondido e saída (20).
        num_layers (int): número de camadas LSTM empilhadas.
        bidirectional (bool): se True, usa LSTM bidirecional.
        dropout (float): dropout entre camadas (se num_layers > 1).
    """
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 20,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 🆕 ÚNICA ADIÇÃO: propriedade para compatibilidade com o pipeline
        self.output_dim = hidden_size * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor de shape (batch_size, seq_len, 2)
                              contendo as sequências de (x, y).

        Returns:
            torch.Tensor: tensor de shape (batch_size, hidden_size*num_directions)
                          correspondente ao último estado escondido.
        """
        # x -> (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)
        # h_n -> (num_layers * num_directions, batch, hidden_size)
        # Seleciona o último layer
        last_h = h_n.view(self.lstm.num_layers,
                          self.num_directions,
                          x.size(0),
                          self.hidden_size)[-1]
        # Se bidirecional, concatena ambos sentidos
        if self.num_directions == 2:
            # last_h shape: (2, batch, hidden_size)
            last_h = torch.cat([last_h[0], last_h[1]], dim=-1)  # (batch, hidden_size*2)
        else:
            last_h = last_h.squeeze(0)  # (batch, hidden_size)
        return last_h