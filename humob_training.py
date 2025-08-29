import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from humob_model import HuMobModel, discretize_coordinates
from humob_dataset import create_humob_loaders, create_test_loader


def compute_cluster_centers(
    parquet_path: str,
    cities: list[str] = ["A"],
    n_clusters: int = 1024,
    sample_size: int = 200_000,
    save_path: str = "cluster_centers.npy",
    chunk_size: int = 50_000,
    coord_columns: list = ['x_norm', 'y_norm']  # CORRIGIDO: usar coordenadas normalizadas
) -> torch.Tensor:
    """
    Calcula centros de cluster usando K-Means em coordenadas normalizadas.
    CORRIGIDO: agora trabalha com dados já normalizados [0,1].
    """
    
    # Se já existe, carrega
    if os.path.exists(save_path):
        print(f"📂 Carregando centros existentes: {save_path}")
        centers = np.load(save_path)
        return torch.from_numpy(centers.astype(np.float32))
    
    print(f"🔄 Calculando {n_clusters} centros para cidades {cities}...")
    
    # 1) Coleta coordenadas normalizadas
    pf = pq.ParquetFile(parquet_path)
    coords_list = []
    
    city_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    target_city_codes = [city_map[c] for c in cities if c in city_map]
    
    for batch in pf.iter_batches(batch_size=chunk_size):
        table = batch.to_pandas()
        
        # Filtra cidades
        if 'city_encoded' in table.columns:
            mask = table['city_encoded'].isin(target_city_codes)
        elif 'city' in table.columns:
            mask = table['city'].isin(cities)
        else:
            continue
            
        table = table[mask]
        if len(table) == 0:
            continue
        
        # Extrai coordenadas normalizadas
        if all(col in table.columns for col in coord_columns):
            coords = table[coord_columns].values
            # Remove pontos inválidos (ex: 999 normalizado)
            valid_mask = (coords >= 0) & (coords <= 1)
            valid_mask = valid_mask.all(axis=1)
            coords = coords[valid_mask]
            
            if len(coords) > 0:
                coords_list.append(coords)
    
    if not coords_list:
        print("❌ Nenhuma coordenada válida encontrada!")
        # Fallback: grid regular
        x = np.linspace(0, 1, int(np.sqrt(n_clusters)))
        y = np.linspace(0, 1, int(np.sqrt(n_clusters)))
        xx, yy = np.meshgrid(x, y)
        centers = np.stack([xx.flatten(), yy.flatten()], axis=1)[:n_clusters]
        centers = centers.astype(np.float32)
    else:
        coords = np.vstack(coords_list)
        print(f"📊 Coletadas {len(coords):,} coordenadas válidas")
        
        # 2) Amostra se muito grande
        if len(coords) > sample_size:
            idx = np.random.choice(len(coords), sample_size, replace=False)
            coords = coords[idx]
            print(f"🎲 Amostradas {sample_size:,} coordenadas")
        
        # 3) K-Means
        print("⚙️ Executando K-Means...")
        kmeans = KMeans(
            n_clusters=n_clusters, 
            n_init='auto', 
            random_state=42,
            max_iter=300
        ).fit(coords)
        
        centers = kmeans.cluster_centers_.astype(np.float32)
    
    # 4) Salva para reutilizar
    np.save(save_path, centers)
    print(f"💾 Centros salvos em: {save_path}")
    print(f"📏 Shape dos centros: {centers.shape}")
    print(f"📍 Range dos centros: x=[{centers[:,0].min():.3f}, {centers[:,0].max():.3f}], y=[{centers[:,1].min():.3f}, {centers[:,1].max():.3f}]")
    
    return torch.from_numpy(centers)


def train_humob_model(
    parquet_path: str,
    cluster_centers: torch.Tensor,
    device: torch.device,
    cities: list[str] = ["A"],
    n_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    sequence_length: int = 24,
    n_users: int = 100_000,
    save_path: str = "humob_model.pt"
):
    """Treina o modelo HuMob com dados normalizados."""
    print("🏋️ Iniciando treinamento do modelo HuMob...")
    
    # 1. Cria loaders
    train_loader, val_loader = create_humob_loaders(
        parquet_path=parquet_path,
        cities=cities,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    # 2. Instancia modelo CORRIGIDO
    model = HuMobModel(
        n_users=n_users,
        n_cities=4,  # A, B, C, D
        cluster_centers=cluster_centers,
        sequence_length=sequence_length,
        prediction_steps=1
    ).to(device)
    
    # 3. Setup de treino
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                 betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.MSELoss()

    def grad_norm(model):
        """Calcula norma total dos gradientes"""
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().float().norm(2).item() ** 2
        return (total ** 0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(n_epochs):
        # === TREINO ===
        model.train()
        train_loss_epoch = 0
        train_count = 0
        
        print(f"\n🔄 Época {epoch+1}/{n_epochs}")
        train_pbar = tqdm(train_loader, desc=f'Treino {cities}')
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [b.to(device) for b in batch]
                
                # Verificação de dados de entrada
                if not torch.isfinite(coords_seq).all() or not torch.isfinite(target_coords).all():
                    print("⚠️ Dados contêm NaN/Inf, pulando batch")
                    continue
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward CORRIGIDO: usar parâmetros normalizados
                pred = model.forward_single_step(uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq)
                target = target_coords.squeeze(1)
                
                # Verificação de predição
                if not torch.isfinite(pred).all():
                    print("⚠️ Predição contém NaN/Inf, pulando batch")
                    continue
                
                # Loss
                loss = criterion(pred, target)
                
                if not torch.isfinite(loss):
                    print(f"⚠️ Loss={loss.item()} não finito, pulando batch")
                    continue
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                pre_clip_norm = grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                post_clip_norm = grad_norm(model)
                
                if pre_clip_norm > 10:
                    print(f"⚠️ Gradiente alto: pre={pre_clip_norm:.1f} → post={post_clip_norm:.2f}")

                optimizer.step()
                
                train_loss_epoch += loss.item() * target.size(0)
                train_count += target.size(0)
                
                # Logs
                if batch_idx % 100 == 0:
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'GradNorm': f'{post_clip_norm:.2f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
                    
            except Exception as e:
                print(f"❌ Erro no batch {batch_idx}: {e}")
                continue
        
        # === VALIDAÇÃO ===
        model.eval()
        val_loss_epoch = 0
        val_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Val')
            for batch in val_pbar:
                try:
                    uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [b.to(device) for b in batch]
                    
                    pred = model.forward_single_step(uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq)
                    target = target_coords.squeeze(1)
                    
                    loss = criterion(pred, target)
                    
                    val_loss_epoch += loss.item() * target.size(0)
                    val_count += target.size(0)
                    
                    val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    continue
        
        # Médias
        avg_train_loss = train_loss_epoch / max(train_count, 1)
        avg_val_loss = val_loss_epoch / max(val_count, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Treino: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        print(f"Fusion weights: w_r={model.weighted_fusion.w_r.item():.3f}, w_e={model.weighted_fusion.w_e.item():.3f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Salva melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"💾 Novo melhor modelo! Loss: {best_val_loss:.4f}")
            
            torch.save({
                'state_dict': model.state_dict(),
                'centers': cluster_centers.cpu().numpy(),
                'config': {
                    'n_users': n_users,
                    'n_cities': 4,
                    'sequence_length': sequence_length,
                    'prediction_steps': 1,
                    'n_clusters': cluster_centers.shape[0]
                },
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
            }, save_path)
    
    print(f"\n✅ Treinamento concluído! Modelo salvo em: {save_path}")
    print(f"Melhor loss de validação: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses


def evaluate_model(
    parquet_path: str,
    checkpoint_path: str,
    device: torch.device,
    cities: list[str] = ["D"],
    n_samples: int = 5000,
    sequence_length: int = 24
):
    """Avalia o modelo em cidades específicas."""
    print(f"🎯 Avaliando modelo em cidades {cities}...")
    
    # 1. Carrega checkpoint
    import numpy as np, torch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
    #     ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    centers = torch.from_numpy(ckpt['centers']).to(device)
    config = ckpt['config']
    
    # 2. Instancia modelo
    model = HuMobModel(
        n_users=config['n_users'],
        n_cities=config['n_cities'],
        cluster_centers=centers,
        sequence_length=sequence_length,
        prediction_steps=1
    ).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # 3. Cria loader de teste
    test_loader = create_test_loader(
        parquet_path=parquet_path,
        cities=cities,
        batch_size=32,
        sequence_length=sequence_length
    )
    
    # 4. Avalia
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    coord_errors = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Eval')
        
        for batch in pbar:
            if total_samples >= n_samples:
                break
                
            try:
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [b.to(device) for b in batch]
                
                pred = model.forward_single_step(uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq)
                target = target_coords.squeeze(1)
                
                loss = criterion(pred, target)
                total_loss += loss.item() * target.size(0)
                total_samples += target.size(0)
                
                # Erro em células (após discretização)
                pred_discrete = discretize_coordinates(pred)
                target_discrete = discretize_coordinates(target)
                cell_error = torch.abs(pred_discrete - target_discrete).float().mean(dim=1)
                coord_errors.extend(cell_error.cpu().tolist())
                
                pbar.set_postfix({
                    'MSE': f'{loss.item():.4f}',
                    'Samples': total_samples
                })
                
            except Exception as e:
                continue
    
    avg_mse = total_loss / max(total_samples, 1)
    avg_cell_error = np.mean(coord_errors)
    
    print(f"\n📊 Resultados em {cities}:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  Erro médio em células: {avg_cell_error:.2f}")
    print(f"  Amostras avaliadas: {total_samples:,}")
    
    return avg_mse, avg_cell_error