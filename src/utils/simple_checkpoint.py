"""
Checkpoint simples - salva a cada época.
src/utils/simple_checkpoint.py
"""

import torch
import os
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, val_loss, save_dir="outputs/models/", 
                   filename=None, extra_data=None):
    """
    Salva checkpoint simples.
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador
        epoch: Época atual
        val_loss: Loss de validação
        save_dir: Diretório para salvar
        filename: Nome do arquivo (auto-gera se None)
        extra_data: Dados extras (ex: cluster_centers, config)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    filepath = os.path.join(save_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    
    # Adiciona dados extras se fornecidos
    if extra_data:
        checkpoint.update(extra_data)
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint salvo: {filepath}")
    
    return filepath


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Carrega checkpoint simples.
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador
        checkpoint_path: Caminho do checkpoint
        
    Returns:
        epoch: Época do checkpoint
    """
    # Compatibilidade PyTorch 2.6+
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except:
        import numpy as np
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray]):
            checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', 0)
    
    print(f"📂 Checkpoint carregado: época {epoch}, val_loss={val_loss:.6f}")
    
    return epoch


def get_latest_checkpoint(save_dir="outputs/models/"):
    """
    Encontra o checkpoint mais recente.
    
    Returns:
        Path do checkpoint ou None
    """
    checkpoints = list(Path(save_dir).glob("checkpoint_epoch_*.pt"))
    
    if not checkpoints:
        return None
    
    # Pega o mais recente
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)