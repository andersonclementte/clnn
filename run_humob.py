"""
Script principal para executar o pipeline completo do HuMob Challenge.

CORREÇÕES IMPLEMENTADAS baseadas na análise:
1. ✅ Sequências temporais adequadas (sequence_length > 1) para LSTM fazer sentido
2. ✅ Uso correto da classe ExternalInformationFusionNormalized (dados já normalizados)
3. ✅ Dataset que trabalha com dados já normalizados
4. ✅ Rollout para múltiplos passos (15 dias × 48 slots para HuMob)
5. ✅ Discretização final para grid [0,199]
6. ✅ Cluster centers calculados corretamente
7. ✅ Pipeline completo com treino, validação e submissão

PREMISSAS:
- Dados já normalizados conforme especificado:
  * x_norm, y_norm: [0,1] (MinMaxScaler)
  * d_norm: [0,1] (normalização linear) 
  * t_sin, t_cos: [-1,1] (codificação circular)
  * POI_norm: [0,1] (log1p + normalização por categoria)
  * city_encoded: {0,1,2,3} (label encoding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Importa módulos locais (assumindo que estão no mesmo diretório)
try:
    from humob_model import HuMobModel, discretize_coordinates
    from humob_dataset import create_humob_loaders, create_test_loader
    from humob_training import compute_cluster_centers, train_humob_model, evaluate_model
    from humob_pipeline import run_full_pipeline, generate_humob_submission
except ImportError as e:
    print(f"❌ Erro importando módulos: {e}")
    print("📋 Certifique-se de que os arquivos estão no mesmo diretório:")
    print("   - external_information.py")
    print("   - partial_information.py") 
    print("   - humob_model.py")
    print("   - humob_dataset.py")
    print("   - humob_training.py")
    print("   - humob_pipeline.py")
    exit(1)


def quick_test(parquet_path: str, device: torch.device):
    """Teste rápido para verificar se o pipeline está funcionando."""
    print("🧪 TESTE RÁPIDO DO PIPELINE")
    print("=" * 40)
    
    try:
        # 1. Testa cluster centers
        print("1. Testando cálculo de cluster centers...")
        centers = compute_cluster_centers(
            parquet_path=parquet_path,
            cities=["A"],
            n_clusters=64,  # Pequeno para teste
            sample_size=10000,
            save_path="test_centers.npy"
        )
        print(f"   ✅ Centers: {centers.shape}")
        
        # 2. Testa criação de dataset
        print("2. Testando dataset...")
        train_loader, val_loader = create_humob_loaders(
            parquet_path=parquet_path,
            cities=["A"],
            batch_size=16,
            sequence_length=6  # Pequeno para teste
        )
        
        # Tenta uma amostra
        for batch in train_loader:
            uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = batch
            print(f"   ✅ Batch shape: {coords_seq.shape}")
            print(f"   ✅ Ranges: d_norm=[{d_norm.min():.3f},{d_norm.max():.3f}], "
                  f"coords=[{coords_seq.min():.3f},{coords_seq.max():.3f}]")
            break
        
        # 3. Testa modelo
        print("3. Testando modelo...")
        model = HuMobModel(
            n_users=1000,  # Pequeno para teste
            n_cities=4,
            cluster_centers=centers.to(device),
            sequence_length=6
        ).to(device)
        
        # Forward test
        with torch.no_grad():
            uid_test = torch.randint(0, 1000, (2,), device=device)
            d_norm_test = torch.rand(2, device=device)
            t_sin_test = torch.randn(2, device=device)
            t_cos_test = torch.randn(2, device=device)
            city_test = torch.randint(0, 4, (2,), device=device)
            poi_test = torch.rand(2, 85, device=device)
            coords_seq_test = torch.rand(2, 6, 2, device=device)
            
            pred = model.forward_single_step(
                uid_test, d_norm_test, t_sin_test, t_cos_test,
                city_test, poi_test, coords_seq_test
            )
            print(f"   ✅ Predição shape: {pred.shape}")
            print(f"   ✅ Predição range: [{pred.min():.3f}, {pred.max():.3f}]")
        
        print("\n🎉 TESTE RÁPIDO PASSOU! Pipeline está funcionando.")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE RÁPIDO FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_minimal_example(parquet_path: str):
    """Executa um exemplo mínimo para demonstração."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 EXEMPLO MÍNIMO HUMOB")
    print(f"Device: {device}")
    print("=" * 40)
    
    # Teste rápido primeiro
    if not quick_test(parquet_path, device):
        print("❌ Teste rápido falhou. Verifique os dados e imports.")
        return None
    
    print("\n🏃 Executando pipeline mínimo...")
    
    # Pipeline com configurações mínimas
    results = run_full_pipeline(
        parquet_path=parquet_path,
        device=device,
        n_clusters=128,     # Reduzido
        n_epochs=2,        # Poucas épocas 
        sequence_length=8,  # Histórico pequeno
        batch_size=32,
        learning_rate=2e-3,
        n_users_A=100_000
    )
    
    if results and results.get('checkpoint_path'):
        print("\n📄 Gerando submissão de exemplo...")
        try:
            submission_df = generate_humob_submission(
                parquet_path=parquet_path,
                checkpoint_path=results['checkpoint_path'],
                device=device,
                target_cities=["D"],  # Apenas cidade D para teste
                submission_days=(61, 62),  # Apenas 2 dias para teste
                sequence_length=8,
                output_file="humob_example_submission.csv"
            )
            print(f"✅ Submissão de exemplo gerada: {len(submission_df):,} linhas")
        except Exception as e:
            print(f"⚠️ Erro na submissão: {e}")
    
    return results


def run_full_competition(parquet_path: str):
    """Executa o pipeline completo para competição."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🏆 PIPELINE COMPLETO PARA COMPETIÇÃO")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Configurações otimizadas para competição
    results = run_full_pipeline(
        parquet_path=parquet_path,
        device=device,
        n_clusters=512,      # Mais centros para melhor precisão
        n_epochs=8,         # Mais épocas
        sequence_length=24,  # Mais histórico (12 horas)
        batch_size=64,
        learning_rate=1e-3,
        n_users_A=100_000
    )
    
    if results and results.get('checkpoint_path'):
        print("\n📄 Gerando submissão final...")
        submission_df = generate_humob_submission(
            parquet_path=parquet_path,
            checkpoint_path=results['checkpoint_path'],
            device=device,
            target_cities=["B", "C", "D"],
            submission_days=(61, 75),  # 15 dias completos
            sequence_length=24,
            output_file="humob_final_submission.csv"
        )
        print(f"🎯 Submissão final: {len(submission_df):,} predições")
        print("📧 Pronto para envio ao HuMob Challenge!")
    
    return results


def main():
    """Função principal com menu interativo."""
    
    # Configuração do arquivo (AJUSTE AQUI)
    parquet_file = "humob_all_cities_v2_normalized.parquet"
    
    print("🎯 HUMOB CHALLENGE - PIPELINE CORRIGIDO")
    print("=" * 50)
    print("Correções implementadas:")
    print("✅ Sequências temporais adequadas (LSTM funcional)")
    print("✅ Dados normalizados [0,1] / [-1,1]") 
    print("✅ Rollout para múltiplos passos")
    print("✅ Discretização para grid [0,199]")
    print("✅ Pipeline completo treino → submissão")
    print()
    
    if not os.path.exists(parquet_file):
        print(f"❌ Arquivo não encontrado: {parquet_file}")
        print("📋 Certifique-se de:")
        print("   1. Ter executado a normalização dos dados")
        print("   2. O arquivo ter as colunas corretas:")
        print("      - uid, city_encoded, d_norm, t_sin, t_cos")
        print("      - x_norm, y_norm, POI_norm")
        return
    
    print(f"📁 Arquivo encontrado: {parquet_file}")
    print()
    
    # Menu
    print("Escolha uma opção:")
    print("1. 🧪 Teste rápido (verifica se tudo está funcionando)")
    print("2. 🏃 Exemplo mínimo (pipeline pequeno para demonstração)")  
    print("3. 🏆 Pipeline completo (para submissão final)")
    
    try:
        choice = input("\nDigite sua escolha (1-3): ").strip()
        
        if choice == "1":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            quick_test(parquet_file, device)
            
        elif choice == "2":
            run_minimal_example(parquet_file)
            
        elif choice == "3":
            run_full_competition(parquet_file)
            
        else:
            print("❌ Opção inválida")
            
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()