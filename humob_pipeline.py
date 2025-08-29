import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from humob_model import HuMobModel, discretize_coordinates
from humob_dataset import HuMobNormalizedDataset
from humob_training import compute_cluster_centers, train_humob_model, evaluate_model


def run_full_pipeline(
    parquet_path: str,
    device: torch.device = None,
    n_clusters: int = 512,
    n_epochs: int = 5,
    sequence_length: int = 24,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_users_A: int = 100_000
):
    """
    Pipeline completo do HuMob Challenge:
    1. Calcula cluster centers
    2. Treina modelo na cidade A
    3. Avalia zero-shot em B, C, D
    4. Gera plots de resultado
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 PIPELINE COMPLETO HUMOB CHALLENGE")
    print(f"Device: {device}")
    print(f"Clusters: {n_clusters}")
    print(f"Epochs: {n_epochs}")
    print(f"Sequence length: {sequence_length}")
    print("=" * 60)
    
    # 1. CLUSTER CENTERS
    print("\n📍 ETAPA 1: Calculando cluster centers...")
    centers = compute_cluster_centers(
        parquet_path=parquet_path,
        cities=["A"],
        n_clusters=n_clusters,
        save_path=f"centers_A_{n_clusters}.npy"
    ).to(device)
    
    print(f"✅ Centers prontos: {centers.shape}")
    
    # 2. TREINAMENTO EM A
    print("\n🏋️ ETAPA 2: Treinando modelo na cidade A...")
    model, train_losses, val_losses = train_humob_model(
        parquet_path=parquet_path,
        cluster_centers=centers,
        device=device,
        cities=["A"],
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_users=n_users_A,
        save_path="humob_model_A.pt"
    )
    
    # 3. AVALIAÇÃO ZERO-SHOT EM B, C, D
    print("\n🎯 ETAPA 3: Avaliação zero-shot em B, C, D...")
    results = {}
    
    for city in ["B", "C", "D"]:
        print(f"\n--- Avaliando cidade {city} ---")
        try:
            mse, cell_error = evaluate_model(
                parquet_path=parquet_path,
                checkpoint_path="humob_model_A.pt",
                device=device,
                cities=[city],
                n_samples=3000,
                sequence_length=sequence_length
            )
            results[city] = {"mse": mse, "cell_error": cell_error}
        except Exception as e:
            print(f"❌ Erro avaliando cidade {city}: {e}")
            results[city] = {"mse": float('inf'), "cell_error": float('inf')}
    
    # 4. PLOTS DOS RESULTADOS
    print("\n📊 ETAPA 4: Gerando plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Curvas de treino
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='orange')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Curvas de Treinamento (Cidade A)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Pesos da fusão
    w_r = model.weighted_fusion.w_r.item()
    w_e = model.weighted_fusion.w_e.item()
    axes[0, 1].bar(['Static (w_r)', 'Dynamic (w_e)'], [w_r, w_e], 
                   color=['lightcoral', 'lightblue'])
    axes[0, 1].set_ylabel('Peso')
    axes[0, 1].set_title('Pesos da Fusão Ponderada')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MSE por cidade
    cities = list(results.keys())
    mse_values = [results[city]["mse"] for city in cities]
    axes[1, 0].bar(cities, mse_values, color=['red', 'green', 'purple'])
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Zero-shot por Cidade')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Erro em células por cidade
    cell_errors = [results[city]["cell_error"] for city in cities]
    axes[1, 1].bar(cities, cell_errors, color=['red', 'green', 'purple'])
    axes[1, 1].set_ylabel('Erro Médio (células)')
    axes[1, 1].set_title('Erro em Células por Cidade')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('humob_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. RELATÓRIO FINAL
    print("\n📋 RELATÓRIO FINAL")
    print("=" * 60)
    print(f"✅ Treinamento concluído em {n_epochs} épocas")
    print(f"📈 Loss final de treino: {train_losses[-1]:.4f}")
    print(f"📉 Loss final de validação: {val_losses[-1]:.4f}")
    print(f"⚖️ Pesos da fusão: w_r={w_r:.3f}, w_e={w_e:.3f}")
    print("\n🎯 Resultados Zero-shot:")
    for city in cities:
        print(f"  Cidade {city}: MSE={results[city]['mse']:.4f}, "
              f"Erro células={results[city]['cell_error']:.2f}")
    
    print(f"\n💾 Arquivos gerados:")
    print(f"   - centers_A_{n_clusters}.npy (cluster centers)")
    print(f"   - humob_model_A.pt (modelo treinado)")
    print(f"   - humob_results.png (gráficos)")
    
    # 6. VERIFICAÇÕES DE QUALIDADE
    print("\n🔍 VERIFICAÇÕES:")
    converged = val_losses[-1] < val_losses[0] * 0.8
    avg_mse = np.mean([results[city]["mse"] for city in cities if results[city]["mse"] != float('inf')])
    reasonable_mse = avg_mse < 0.1  # Para dados normalizados [0,1]
    
    print(f"  Convergiu? {'✅' if converged else '❌'} (loss diminuiu 20%+)")
    print(f"  MSE razoável? {'✅' if reasonable_mse else '❌'} (< 0.1 para [0,1])")
    
    if converged and reasonable_mse:
        print("\n🎉 PIPELINE EXECUTADO COM SUCESSO!")
        print("   Pronto para gerar submissões HuMob!")
    else:
        print("\n⚠️ ALGUNS PROBLEMAS DETECTADOS")
        print("   Considere ajustar hiperparâmetros ou verificar dados")
    
    return {
        'model': model,
        'centers': centers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'results_by_city': results,
        'checkpoint_path': 'humob_model_A.pt'
    }


def generate_humob_submission(
    parquet_path: str,
    checkpoint_path: str,
    device: torch.device,
    target_cities: list[str] = ["B", "C", "D"],
    submission_days: tuple = (61, 75),  # Dias 61-75 para HuMob
    sequence_length: int = 24,
    output_file: str = "humob_submission.csv",
    chunk_size: int = 1000
):
    """
    Gera arquivo de submissão para o HuMob Challenge.
    
    Formato esperado: uid, day, slot, x, y
    onde x,y são coordenadas discretas [0,199]
    """
    print(f"📄 Gerando submissão HuMob para cidades {target_cities}")
    print(f"Dias: {submission_days[0]}-{submission_days[1]} | Output: {output_file}")
    
    # 1. Carrega modelo
    import numpy as np, torch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    centers = torch.from_numpy(ckpt['centers']).to(device)
    config = ckpt['config']
    
    model = HuMobModel(
        n_users=config['n_users'],
        n_cities=config['n_cities'], 
        cluster_centers=centers,
        sequence_length=sequence_length,
        prediction_steps=1
    ).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # 2. Processa dados para submissão
    submission_rows = []
    
    # Cria dataset para dias observados (para construir histórico)
    observed_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=target_cities,
        mode="test",  # Usa todos os dados disponíveis
        sequence_length=sequence_length,
        max_sequences_per_user=float('inf')  # Sem limitação
    )
    
    # Agrupa por usuário para fazer rollout sequencial
    user_sequences = {}
    print("🔄 Coletando sequências por usuário...")
    
    for sample in tqdm(observed_ds):
        uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = sample
        uid_int = uid.item()
        
        if uid_int not in user_sequences:
            user_sequences[uid_int] = []
        
        user_sequences[uid_int].append({
            'd_norm': d_norm.item(),
            't_sin': t_sin.item(), 
            't_cos': t_cos.item(),
            'city': city.item(),
            'poi_norm': poi_norm.numpy(),
            'coords_seq': coords_seq.numpy(),
            'last_observed_day': d_norm.item() * 74,  # Desnormaliza
            'last_observed_slot': np.arctan2(t_sin.item(), t_cos.item()) / (2*np.pi) * 48
        })
    
    print(f"📊 Encontrados {len(user_sequences)} usuários únicos")
    
    # 3. Gera predições para cada usuário
    print("🎯 Gerando predições...")
    
    with torch.no_grad():
        for uid_int, sequences in tqdm(user_sequences.items()):
            if not sequences:
                continue
                
            # Pega a sequência mais recente como base
            latest_seq = max(sequences, key=lambda x: x['d_norm'])
            
            # Converte para tensors
            uid_tensor = torch.tensor([uid_int], dtype=torch.long, device=device)
            city_tensor = torch.tensor([latest_seq['city']], dtype=torch.long, device=device)
            poi_tensor = torch.tensor([latest_seq['poi_norm']], dtype=torch.float32, device=device)
            coords_seq_tensor = torch.tensor([latest_seq['coords_seq']], dtype=torch.float32, device=device)
            
            # Estado temporal inicial (último ponto observado)
            d_norm = torch.tensor([latest_seq['d_norm']], dtype=torch.float32, device=device)
            t_sin = torch.tensor([latest_seq['t_sin']], dtype=torch.float32, device=device)  
            t_cos = torch.tensor([latest_seq['t_cos']], dtype=torch.float32, device=device)
            
            # Calcula quantos passos até chegar no dia 61
            last_day = int(latest_seq['last_observed_day'])
            last_slot = int(latest_seq['last_observed_slot']) 
            
            # Rollout até o dia 75, slot 47
            current_day = max(last_day, submission_days[0])  # Começa no dia 61
            current_slot = 0
            
            # Total de predições necessárias
            total_slots = (submission_days[1] - submission_days[0] + 1) * 48
            
            for step in range(total_slots):
                # Atualiza estado temporal 
                d_norm_current = torch.tensor([current_day / 74.0], device=device)
                t_sin_current = torch.sin(torch.tensor([2 * np.pi * current_slot / 48], device=device))
                t_cos_current = torch.cos(torch.tensor([2 * np.pi * current_slot / 48], device=device))
                
                # Predição
                try:
                    pred_coords = model.forward_single_step(
                        uid_tensor, d_norm_current, t_sin_current, t_cos_current,
                        city_tensor, poi_tensor, coords_seq_tensor
                    )
                    
                    # Discretiza coordenadas
                    pred_discrete = discretize_coordinates(pred_coords, grid_size=200)
                    x, y = pred_discrete[0, 0].item(), pred_discrete[0, 1].item()
                    
                    # Adiciona à submissão
                    submission_rows.append({
                        'uid': uid_int,
                        'day': current_day,
                        'slot': current_slot,
                        'x': x,
                        'y': y
                    })
                    
                    # Atualiza sequência para próxima predição (teacher forcing)
                    new_point = pred_coords.unsqueeze(1)  # (1, 1, 2)
                    coords_seq_tensor = torch.cat([coords_seq_tensor[:, 1:, :], new_point], dim=1)
                    
                except Exception as e:
                    print(f"❌ Erro predizindo usuário {uid_int}, dia {current_day}, slot {current_slot}: {e}")
                    # Fallback: repete última posição conhecida
                    x, y = int(coords_seq_tensor[0, -1, 0] * 199), int(coords_seq_tensor[0, -1, 1] * 199)
                    submission_rows.append({
                        'uid': uid_int, 'day': current_day, 'slot': current_slot, 'x': x, 'y': y
                    })
                
                # Avança para próximo slot/dia
                current_slot += 1
                if current_slot >= 48:
                    current_slot = 0
                    current_day += 1
                    
                if current_day > submission_days[1]:
                    break
    
    # 4. Salva submissão
    df_submission = pd.DataFrame(submission_rows)
    df_submission = df_submission.sort_values(['uid', 'day', 'slot'])
    
    print(f"📊 Submissão gerada:")
    print(f"   Linhas: {len(df_submission):,}")
    print(f"   Usuários únicos: {df_submission['uid'].nunique():,}")
    print(f"   Dias: {df_submission['day'].min()}-{df_submission['day'].max()}")
    print(f"   Coordenadas x: [{df_submission['x'].min()}, {df_submission['x'].max()}]")
    print(f"   Coordenadas y: [{df_submission['y'].min()}, {df_submission['y'].max()}]")
    
    df_submission.to_csv(output_file, index=False)
    print(f"💾 Submissão salva em: {output_file}")
    
    return df_submission


# Função principal de execução
def main():
    """Execução principal do pipeline HuMob."""
    
    # Configurações
    parquet_file = "humob_all_cities_v2_normalized.parquet"  # Seu arquivo normalizado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 HUMOB CHALLENGE - PIPELINE PRINCIPAL")
    print(f"Arquivo: {parquet_file}")
    print(f"Device: {device}")
    
    # Executa pipeline completo
    results = run_full_pipeline(
        parquet_path=parquet_file,
        device=device,
        n_clusters=256,        # Reduzido para ser mais rápido
        n_epochs=3,           # Poucas épocas para teste
        sequence_length=12,   # Histórico de 12 slots (6 horas)
        batch_size=64,
        learning_rate=1e-3,
        n_users_A=100_000
    )
    
    # Se o treinamento foi bem-sucedido, gera submissão
    if results:
        print("\n📄 Gerando submissão HuMob...")
        submission_df = generate_humob_submission(
            parquet_path=parquet_file,
            checkpoint_path=results['checkpoint_path'],
            device=device,
            target_cities=["B", "C", "D"],
            submission_days=(61, 75),
            sequence_length=12,
            output_file=f"humob_submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        
        print(f"\n🎉 PIPELINE COMPLETO!")
        print(f"✅ Modelo treinado e salvo")
        print(f"✅ Submissão gerada: {len(submission_df):,} predições")
        print(f"✅ Pronto para envio ao HuMob Challenge!")
    
    return results


if __name__ == "__main__":
    results = main()