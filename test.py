"""
Script para avaliar modelo já treinado sem treinar novamente.
Use depois que já executou o treinamento e tem o arquivo humob_model_A.pt.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Importa funções necessárias
from humob_training import evaluate_model
from humob_pipeline import generate_humob_submission


def evaluate_pretrained_model(
    parquet_path: str,
    checkpoint_path: str = "humob_model_A.pt",
    target_cities: list[str] = ["B", "C", "D"],
    device: torch.device = None
):
    """Avalia modelo já treinado nas cidades alvo."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 AVALIAÇÃO DE MODELO PRÉ-TREINADO")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Cidades alvo: {target_cities}")
    
    # Verifica se checkpoint existe
    import os
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint não encontrado: {checkpoint_path}")
        print("Execute o treinamento primeiro com run_humob.py")
        return None
    
    # Avalia em cada cidade
    results = {}
    
    for city in target_cities:
        print(f"\n--- Avaliando cidade {city} ---")
        try:
            mse, cell_error = evaluate_model(
                parquet_path=parquet_path,
                checkpoint_path=checkpoint_path,
                device=device,
                cities=[city],
                n_samples=5000,
                sequence_length=8  # Ajuste conforme usado no treino
            )
            results[city] = {"mse": mse, "cell_error": cell_error}
            print(f"✅ Cidade {city}: MSE={mse:.4f}, Erro células={cell_error:.2f}")
            
        except Exception as e:
            print(f"❌ Erro avaliando cidade {city}: {e}")
            results[city] = {"mse": float('inf'), "cell_error": float('inf')}
    
    # Relatório final
    print(f"\n📊 RELATÓRIO FINAL DE AVALIAÇÃO")
    print("=" * 40)
    
    valid_results = {k: v for k, v in results.items() 
                    if v["mse"] != float('inf')}
    
    if valid_results:
        avg_mse = np.mean([v["mse"] for v in valid_results.values()])
        avg_cell_error = np.mean([v["cell_error"] for v in valid_results.values()])
        
        print(f"Resultados por cidade:")
        for city, metrics in results.items():
            status = "✅" if metrics["mse"] != float('inf') else "❌"
            print(f"  {status} {city}: MSE={metrics['mse']:.4f}, "
                  f"Erro células={metrics['cell_error']:.2f}")
        
        print(f"\nMétricas médias:")
        print(f"  MSE médio: {avg_mse:.4f}")
        print(f"  Erro médio células: {avg_cell_error:.2f}")
        
        # Avaliação qualitativa
        if avg_mse < 0.01:
            print("🎉 Excelente performance!")
        elif avg_mse < 0.05:
            print("✅ Boa performance")
        else:
            print("⚠️ Performance pode melhorar")
            
    else:
        print("❌ Nenhuma avaliação bem-sucedida")
        return None
    
    return results


def generate_submission_only(
    parquet_path: str,
    checkpoint_path: str = "humob_model_A.pt", 
    target_cities: list[str] = ["B", "C", "D"],
    submission_days: tuple = (61, 75),
    device: torch.device = None
):
    """Gera apenas o arquivo de submissão sem treinar."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📄 GERAÇÃO DE SUBMISSÃO")
    print("=" * 30)
    
    output_file = f"humob_submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    try:
        submission_df = generate_humob_submission(
            parquet_path=parquet_path,
            checkpoint_path=checkpoint_path,
            device=device,
            target_cities=target_cities,
            submission_days=submission_days,
            sequence_length=8,  # Ajuste conforme treino
            output_file=output_file
        )
        
        print(f"✅ Submissão gerada: {output_file}")
        print(f"   Predições: {len(submission_df):,}")
        print(f"   Usuários: {submission_df['uid'].nunique():,}")
        
        return submission_df
        
    except Exception as e:
        print(f"❌ Erro gerando submissão: {e}")
        return None


def main():
    """Menu principal para avaliação."""
    
    # Configurações - AJUSTE AQUI
    parquet_file = "humob_all_cities_v2_normalized.parquet"
    checkpoint_file = "humob_model_A.pt"
    
    print("🔍 AVALIAÇÃO PÓS-TREINO - HUMOB")
    print("=" * 40)
    print(f"Dados: {parquet_file}")
    print(f"Modelo: {checkpoint_file}")
    
    # Verifica arquivos
    import os
    if not os.path.exists(parquet_file):
        print(f"❌ Arquivo de dados não encontrado: {parquet_file}")
        return
        
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint não encontrado: {checkpoint_file}")
        print("Execute o treinamento primeiro com run_humob.py")
        return
    
    print("\nOpções:")
    print("1. 📊 Avaliar modelo em B, C, D")
    print("2. 📄 Gerar submissão HuMob")
    print("3. 🎯 Fazer ambos (avaliação + submissão)")
    
    try:
        choice = input("\nEscolha (1-3): ").strip()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if choice == "1":
            evaluate_pretrained_model(
                parquet_path=parquet_file,
                checkpoint_path=checkpoint_file,
                device=device
            )
            
        elif choice == "2":
            generate_submission_only(
                parquet_path=parquet_file,
                checkpoint_path=checkpoint_file,
                device=device
            )
            
        elif choice == "3":
            # Faz avaliação
            results = evaluate_pretrained_model(
                parquet_path=parquet_file,
                checkpoint_path=checkpoint_file,
                device=device
            )
            
            # Se avaliação foi bem-sucedida, gera submissão
            if results and any(v["mse"] != float('inf') for v in results.values()):
                print("\n" + "="*50)
                generate_submission_only(
                    parquet_path=parquet_file,
                    checkpoint_path=checkpoint_file,
                    device=device
                )
            else:
                print("❌ Pulando submissão devido a erros na avaliação")
                
        else:
            print("Opção inválida")
            
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()