# HuMob Challenge 2024 - Human Mobility Prediction

## Visão Geral

Este projeto implementa uma solução para o **HuMob Challenge 2024**, uma competição internacional de predição de mobilidade humana urbana. O desafio consiste em prever trajetórias de movimento de usuários em 4 cidades metropolitanas japonesas (A, B, C, D) usando dados sintéticos mas realistas de 100k+ usuários ao longo de 75 dias.

### Tarefa Principal
- **Entrada**: Dados de mobilidade das cidades A (completa, dias 1-75) e B,C,D (parciais, dias 1-60)
- **Objetivo**: Prever movimento de usuários nas cidades B, C, D para os dias 61-75
- **Formato**: Grid discreto 200x200 (células de 500m x 500m), intervalos de 30 minutos

## Arquitetura do Modelo

### Abordagem Híbrida
O modelo combina duas fontes de informação:

1. **Informação Estática/Externa**: Contexto do usuário e ambiente
   - Embeddings de usuário, cidade, tempo
   - Informações de Points of Interest (POI) - 85 categorias
   - Projeções temporais (dia normalizado + codificação circular de horário)

2. **Informação Dinâmica**: Padrões de movimento
   - LSTM bidirecional processando sequências de coordenadas
   - Captura padrões temporais de mobilidade individual

3. **Fusão Inteligente**: 
   - Combinação ponderada aprendível entre contexto estático e dinâmico
   - Weights: w_r (estático) + w_e (dinâmico)

4. **Head de Destino**:
   - MLP que mapeia representação fusionada para probabilidades sobre cluster centers
   - Predição final via média ponderada de centros K-means

## Normalização dos Dados

Os dados passam por um processo de normalização rigoroso:

### Coordenadas Espaciais (x, y)
- **Método**: MinMaxScaler para range [0,1]
- **Transformação**: `x_norm = (x - x_min) / (x_max - x_min)`
- **Range original**: [0, 199] → **Range final**: [0.0, 1.0]

### Dimensão Temporal
- **Dias (d)**: Normalização linear `d_norm = d / 74` → [0.0, 1.0]
- **Horários (t)**: Codificação circular para capturar ciclicidade
  - `t_sin = sin(2π × t / 48)` → [-1.0, 1.0]
  - `t_cos = cos(2π × t / 48)` → [-1.0, 1.0]

### Points of Interest (POI)
- **Processo**: log1p + normalização por categoria
- **Transformação**: 
  1. `poi_log = log(1 + poi_count)` (comprime outliers)
  2. `poi_norm = poi_log / max_categoria` (equaliza importância)
- **Range original**: [0, 65535] → **Range final**: [0.0, 1.0]

### Cidades
- **Método**: Label encoding A→0, B→1, C→2, D→3

## Estrutura do Projeto

```
humob_project/
├── external_information.py    # Classes de fusão de informação externa
├── partial_information.py     # LSTM para sequências de coordenadas
├── humob_model.py            # Modelo principal híbrido
├── humob_dataset.py          # Dataset para dados normalizados
├── humob_training.py         # Funções de treino e cluster centers
├── humob_pipeline.py         # Pipeline completo
├── run_humob.py              # Script principal com menu
├── test.py                   # Avaliação de modelo pré-treinado
├── check_setup.py            # Verificação de setup
└── README.md                 # Este arquivo
```

## Como Usar

### Pré-requisitos
```bash
pip install torch numpy pandas pyarrow scikit-learn matplotlib tqdm
```

### Execução Rápida
1. **Verificar setup**:
   ```bash
   python check_setup.py
   ```

2. **Ajustar caminho dos dados** em `run_humob.py`:
   ```python
   parquet_file = "SEU_ARQUIVO_NORMALIZADO.parquet"
   ```

3. **Executar pipeline**:
   ```bash
   python run_humob.py
   ```

### Opções Disponíveis
- **Teste rápido**: Verifica se tudo está funcionando
- **Exemplo mínimo**: Pipeline pequeno (2 épocas, 128 clusters)
- **Pipeline completo**: Configuração competitiva (8 épocas, 512 clusters)
- **Avaliação apenas**: Testa modelo já treinado (opções 4-5 no menu)

### Avaliação Pós-Treino
```bash
python test.py  # Script dedicado para avaliação sem re-treino
```

## Status Atual

### Implementado
- ✅ Pré-treino na cidade A com dados normalizados
- ✅ Arquitetura híbrida (estático + dinâmico + fusão)
- ✅ Sequências temporais adequadas para LSTM
- ✅ Cluster centers via K-means
- ✅ Transfer learning zero-shot para B, C, D
- ✅ Geração de arquivo de submissão (formato HuMob)
- ✅ Discretização [0,1] → [0,199] para submissão

### Resultados Obtidos
- **Loss de treino**: ~0.0043 (MSE em dados normalizados)
- **Loss de validação**: ~0.0042 
- **Convergência**: Modelo converge bem em ~20 horas (2 épocas)
- **Fusão**: Modelo priorizou padrões dinâmicos (w_e=0.741 vs w_r=0.023)

## Próximos Passos

### 1. Fine-tuning Multi-Cidade (PRIORIDADE ALTA)
Atualmente o modelo faz apenas transfer learning zero-shot. Para melhorar performance:

```python
# Estratégia atual: A → (B,C,D) zero-shot
train_on_A() → evaluate_on_BCD()

# Estratégia melhorada: A → fine-tune → avaliar
train_on_A() → finetune_on_B() → finetune_on_C() → finetune_on_D()
```

#### Implementação Sugerida:
1. **Modificar `train_humob_model()`** para aceitar múltiplas cidades
2. **Adicionar função `finetune_model()`** que:
   - Carrega modelo pré-treinado em A
   - Fine-tune com learning rate reduzido em B, C, D (dias 1-60)
   - Usa menos épocas (1-2) para evitar overfitting
3. **Atualizar pipeline** para executar fine-tuning sequencial

### 2. Otimizações de Performance
- **Experimentar arquiteturas**: Attention mechanisms, Transformers
- **Hiperparâmetros**: Grid search em learning rate, dimensões de embedding
- **Regularização**: Dropout, weight decay, early stopping
- **Dados**: Augmentation temporal, ensemble methods

### 3. Análise e Debugging
- **Visualizações**: Plotar trajetórias preditas vs reais
- **Análise de erro**: Por usuário, horário, tipo de POI
- **Interpretabilidade**: Análise dos pesos de fusão por cidade

## Problemas Conhecidos

### 1. PyTorch 2.6+ Compatibility
**Erro**: `torch.load()` falha com numpy arrays nos checkpoints

**Solução Temporária**:
```python
# Em humob_training.py e humob_pipeline.py
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Ou use safe_globals:
with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
    ckpt = torch.load(checkpoint_path, map_location=device)
```

### 2. Limitações da Estratégia Atual
- **Zero-shot apenas**: Não usa dados disponíveis de B, C, D (dias 1-60)
- **Assume uniformidade**: Padrões de mobilidade idênticos entre cidades
- **Performance**: Pode ser inferior a métodos com fine-tuning

## Resultados da Competição

Para referência, métodos competitivos no HuMob 2024 utilizaram:
- **Modelos baseados em Transformer**: BERT espacial-temporal
- **Fine-tuning**: Pré-treino + fine-tuning por cidade
- **Ensemble**: Múltiplos modelos combinados
- **Features engenheiradas**: Padrões de recorrência, sazonalidade

## Contribuições

O código implementa correções importantes identificadas em revisão:
- Sequências temporais adequadas (sequence_length > 1) para LSTM
- Uso correto de dados já normalizados (sem re-normalização)
- Pipeline completo treino → avaliação → submissão
- Rollout autoregressivo para múltiplos passos
- Discretização correta para formato de submissão

## License

Este projeto foi desenvolvido para fins educacionais e de pesquisa no contexto do HuMob Challenge 2024.
