import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

from humob_model import HuMobModel
from humob_dataset import create_humob_loaders


def load_checkpoint_safe(checkpoint_path: str, device: torch.device):
    """
    Carrega checkpoint com compatibilidade PyTorch 2.6+.
    CORRIGIDO: usa weights_only=False para numpy arrays.
    """
    try:
        # Primeiro tenta o método padrão
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return ckpt
    except Exception as e:
        if "weights_only" in str(e) or "multiarray" in str(e):
            print("⚠️  Usando safe_globals para compatibilidade PyTorch 2.6+")
            # Método alternativo com safe_globals
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
                return ckpt
        else:
            raise e


def finetune_model(
    parquet_path: str,
    pretrained_checkpoint: str,
    target_city: str,
    device: torch.device,
    n_epochs: int = 3,
    learning_rate: float = 1e-4,  # LR menor para fine-tuning
    batch_size: int = 32,
    sequence_length: int = 24,
    save_path: str = None,
    data_split: tuple = (0.0, 0.8)  # Dias 1-60 para fine-tuning (60/75 ≈ 0.8)
):
    """
    Fine-tuning do modelo pré-treinado em uma cidade específica.
    
    Args:
        parquet_path: Caminho para dados normalizados
        pretrained_checkpoint: Modelo pré-treinado (ex: humob_model_A.pt)
        target_city: Cidade para fine-tuning ("B", "C", ou "D")
        device: Device PyTorch
        n_epochs: Épocas de fine-tuning (menos que treinamento inicial)
        learning_rate: LR reduzido para fine-tuning
        batch_size: Tamanho do batch
        sequence_length: Comprimento da sequência temporal
        save_path: Onde salvar modelo fine-tuned (auto-gerado se None)
        data_split: Range de dias para fine-tuning [0,1] normalizado
    
    Returns:
        tuple: (modelo_fine_tuned, train_losses, val_losses)
    """
    print(f"🎯 FINE-TUNING NA CIDADE {target_city}")
    print("=" * 40)
    
    # 1. Carrega modelo pré-treinado com correção PyTorch 2.6+
    print("📂 Carregando modelo pré-treinado...")
    ckpt = load_checkpoint_safe(pretrained_checkpoint, device)
    
    centers = torch.from_numpy(ckpt['centers']).to(device)
    config = ckpt['config']
    
    # 2. Instancia modelo
    model = HuMobModel(
        n_users=config['n_users'],
        n_cities=config['n_cities'],
        cluster_centers=centers,
        sequence_length=sequence_length,
        prediction_steps=config.get('prediction_steps', 1)
    ).to(device)
    
    # Carrega pesos pré-treinados
    model.load_state_dict(ckpt['state_dict'])
    print(f"✅ Modelo pré-treinado carregado (loss: {ckpt.get('val_loss', 'N/A')})")
    
    # 3. Cria loaders para cidade alvo com dados de fine-tuning
    print(f"📊 Criando datasets para cidade {target_city}...")
    
    # CORRIGIDO: usa create_humob_loaders com parâmetros customizados
    from humob_dataset import HuMobNormalizedDataset
    from torch.utils.data import DataLoader
    
    # Dataset de treino (dias 0.0-0.8 da cidade alvo)
    train_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=[target_city],
        mode="train",
        sequence_length=sequence_length,
        train_days=data_split,  # Ex: (0.0, 0.8) para dias 1-60
        val_days=(0.8, 1.0),    # Não usado no treino
        max_sequences_per_user=30  # Menos sequências para fine-tuning
    )
    
    # Dataset de validação (últimos 20% dos dados disponíveis)
    val_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=[target_city],
        mode="val",
        sequence_length=sequence_length,
        train_days=data_split,
        val_days=(0.75, 0.8),  # Pequena fração para validação
        max_sequences_per_user=10
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    
    # 4. Setup de fine-tuning
    # LR menor e otimizador mais conservador
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),  # Menos agressivo
        weight_decay=1e-5,   # Regularização menor
        eps=1e-8
    )
    
    # Scheduler mais suave
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=learning_rate/10
    )
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"🔧 Setup fine-tuning:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {n_epochs}")
    print(f"   Data split: dias {data_split[0]:.1f}-{data_split[1]:.1f}")
    
    # 5. Loop de fine-tuning
    for epoch in range(n_epochs):
        # === TREINO ===
        model.train()
        train_loss_epoch = 0
        train_count = 0
        
        print(f"\n🔄 Fine-tune Época {epoch+1}/{n_epochs} - Cidade {target_city}")
        
        train_pbar = tqdm(train_loader, desc=f'Finetune {target_city}')
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [
                    b.to(device) for b in batch
                ]
                
                # Verificações de sanidade
                if not torch.isfinite(coords_seq).all() or not torch.isfinite(target_coords).all():
                    continue
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward
                pred = model.forward_single_step(
                    uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq
                )
                target = target_coords.squeeze(1)
                
                if not torch.isfinite(pred).all():
                    continue
                
                # Loss
                loss = criterion(pred, target)
                
                if not torch.isfinite(loss):
                    continue
                
                # Backward com gradient clipping mais suave
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # Clipping menor
                optimizer.step()
                
                train_loss_epoch += loss.item() * target.size(0)
                train_count += target.size(0)
                
                # Update progress
                if batch_idx % 50 == 0:
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.5f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
            except Exception as e:
                print(f"⚠️ Erro batch {batch_idx}: {e}")
                continue
        
        # === VALIDAÇÃO ===
        model.eval()
        val_loss_epoch = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val'):
                try:
                    uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [
                        b.to(device) for b in batch
                    ]
                    
                    pred = model.forward_single_step(
                        uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq
                    )
                    target = target_coords.squeeze(1)
                    
                    loss = criterion(pred, target)
                    val_loss_epoch += loss.item() * target.size(0)
                    val_count += target.size(0)
                    
                except Exception as e:
                    continue
        
        # Médias da época
        avg_train_loss = train_loss_epoch / max(train_count, 1)
        avg_val_loss = val_loss_epoch / max(val_count, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Cidade {target_city} - Treino: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
        
        # Update scheduler
        scheduler.step()
        
        # Salva se melhorou
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Nome automático do checkpoint
            if save_path is None:
                save_path = f"humob_model_finetuned_{target_city}.pt"
            
            print(f"💾 Novo melhor modelo para {target_city}! Loss: {best_val_loss:.5f}")
            
            torch.save({
                'state_dict': model.state_dict(),
                'centers': centers.cpu().numpy(),
                'config': config,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'city': target_city,
                'finetuned_from': pretrained_checkpoint,
                'timestamp': datetime.now().isoformat()
            }, save_path)
    
    print(f"\n✅ Fine-tuning cidade {target_city} concluído!")
    print(f"💾 Modelo salvo: {save_path}")
    print(f"📈 Melhor loss: {best_val_loss:.5f}")
    
    return model, train_losses, val_losses


def sequential_finetuning(
    parquet_path: str,
    base_checkpoint: str,
    cities: list[str] = ["B", "C", "D"],
    device: torch.device = None,
    n_epochs_per_city: int = 3,
    learning_rate: float = 1e-4,
    sequence_length: int = 24
):
    """
    Fine-tuning sequencial em múltiplas cidades.
    
    Processo: A (treinado) → fine-tune B → fine-tune C → fine-tune D
    Cada cidade usa o modelo da cidade anterior como ponto de partida.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔄 FINE-TUNING SEQUENCIAL")
    print("=" * 50)
    print(f"Base model: {base_checkpoint}")
    print(f"Cities: {' → '.join(cities)}")
    print(f"Epochs per city: {n_epochs_per_city}")
    
    results = {}
    current_checkpoint = base_checkpoint
    
    for i, city in enumerate(cities):
        print(f"\n🎯 Fine-tuning cidade {city} ({i+1}/{len(cities)})")
        
        # Nome do checkpoint desta cidade
        city_checkpoint = f"humob_model_finetuned_{city}.pt"
        
        try:
            # Fine-tuning
            model, train_losses, val_losses = finetune_model(
                parquet_path=parquet_path,
                pretrained_checkpoint=current_checkpoint,
                target_city=city,
                device=device,
                n_epochs=n_epochs_per_city,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
                save_path=city_checkpoint
            )
            
            results[city] = {
                'checkpoint': city_checkpoint,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                'status': 'success'
            }
            
            # Próxima cidade usa o modelo desta cidade
            current_checkpoint = city_checkpoint
            
        except Exception as e:
            print(f"❌ Erro no fine-tuning da cidade {city}: {e}")
            results[city] = {
                'checkpoint': None,
                'train_losses': [],
                'val_losses': [],
                'final_val_loss': float('inf'),
                'status': 'failed',
                'error': str(e)
            }
            # Continue com checkpoint atual em caso de erro
    
    # Relatório final
    print(f"\n📊 RELATÓRIO SEQUENCIAL")
    print("=" * 40)
    
    successful_cities = 0
    for city, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {city}: Loss final = {result['final_val_loss']:.5f}")
            successful_cities += 1
        else:
            print(f"❌ {city}: Falhou ({result.get('error', 'Unknown error')})")
    
    print(f"\n🎉 Conclusão: {successful_cities}/{len(cities)} cidades com sucesso")
    
    if successful_cities == len(cities):
        final_model = f"humob_model_finetuned_{cities[-1]}.pt"
        print(f"🏆 Modelo final disponível: {final_model}")
    
    return results


# Função de utilidade para comparar performance
def compare_models_performance(
    parquet_path: str,
    checkpoints: dict,  # {'model_name': 'path_to_checkpoint'}
    test_cities: list[str] = ["B", "C", "D"],
    device: torch.device = None,
    n_samples: int = 2000
):
    """
    Compara performance de múltiplos modelos nas cidades de teste.
    Útil para comparar zero-shot vs fine-tuned models.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📊 COMPARAÇÃO DE MODELOS")
    print("=" * 40)
    
    from humob_training import evaluate_model
    
    results = {}
    
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n🔍 Avaliando {model_name}...")
                
        results[model_name] = {}
        
        for city in test_cities:
            try:
                mse, cell_error = evaluate_model(
                    parquet_path=parquet_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    cities=[city],
                    n_samples=n_samples
                )
                results[model_name][city] = {'mse': mse, 'cell_error': cell_error}
                print(f"   {city}: MSE={mse:.4f}, Erro células={cell_error:.2f}")
                
            except Exception as e:
                print(f"   ❌ {city}: Erro - {e}")
                results[model_name][city] = {'mse': float('inf'), 'cell_error': float('inf')}
    
    # Relatório comparativo
    print(f"\n📋 COMPARAÇÃO DETALHADA")
    print("=" * 50)
    
    for city in test_cities:
        print(f"\n🏙️ Cidade {city}:")
        city_results = []
        for model_name in checkpoints.keys():
            if city in results[model_name]:
                mse = results[model_name][city]['mse']
                cell_err = results[model_name][city]['cell_error']
                city_results.append((model_name, mse, cell_err))
        
        # Ordena por MSE
        city_results.sort(key=lambda x: x[1])
        
        for i, (model_name, mse, cell_err) in enumerate(city_results):
            ranking = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}º"
            print(f"  {ranking} {model_name:20s}: MSE={mse:.4f}, Células={cell_err:.2f}")
    
    return results


# Função principal para execução completa
def main():
    """Execução do fine-tuning completo."""
    # Configurações - AJUSTE AQUI
    parquet_file = "humob_all_cities_v2_normalized.parquet"
    base_model = "humob_model_A.pt"  # Modelo treinado apenas em A
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 FINE-TUNING HUMOB CHALLENGE")
    print("=" * 50)
    
    # Verifica arquivos
    import os
    if not os.path.exists(parquet_file):
        print(f"❌ Dados não encontrados: {parquet_file}")
        return
    
    if not os.path.exists(base_model):
        print(f"❌ Modelo base não encontrado: {base_model}")
        print("Execute primeiro o treinamento com run_humob.py")
        return
    
    # Fine-tuning sequencial
    results = sequential_finetuning(
        parquet_path=parquet_file,
        base_checkpoint=base_model,
        cities=["B", "C", "D"],
        device=device,
        n_epochs_per_city=3,
        learning_rate=5e-5,  # LR bem baixo para fine-tuning
        sequence_length=24
    )
    
    # Comparação de performance se tudo deu certo
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    if successful > 0:
        print(f"\n🔍 COMPARANDO PERFORMANCE...")
        
        # Monta lista de checkpoints para comparar
        checkpoints = {'Zero-shot (A apenas)': base_model}
        
        for city, result in results.items():
            if result['status'] == 'success':
                checkpoints[f'Fine-tuned {city}'] = result['checkpoint']
        
        comparison = compare_models_performance(
            parquet_path=parquet_file,
            checkpoints=checkpoints,
            device=device,
            n_samples=3000
        )
        
        print("\n🎉 FINE-TUNING COMPLETO!")
        print("Agora você pode usar os modelos fine-tuned para submissão.")
    
    return results


if __name__ == "__main__":
    main()