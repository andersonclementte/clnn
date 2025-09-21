#!/bin/bash
# Script com dados em pasta persistente

CIA_HOME="$HOME"
mkdir -p "$CIA_HOME"/{humob_outputs,humob_experiments}

echo "🚀 HuMob Challenge - RTX A4000 16GB"
echo "📊 PyTorch cu128 + CUDA 12.8"
echo "🏠 HOME: $CIA_HOME"

# ✅ Busca parquet na pasta persistente
PARQUET_FILE="$CIA_HOME/humob_data/processed/humob_all_cities_v2_normalized.parquet"
if [ ! -f "$PARQUET_FILE" ]; then
    echo "❌ Arquivo não encontrado: $PARQUET_FILE"
    echo "📋 Execute primeiro:"
    echo "   mkdir -p $CIA_HOME/humob_data/processed"
    echo "   cp $CIA_HOME/humob_project/data/processed/humob_all_cities_v2_normalized.parquet $CIA_HOME/humob_data/processed/"
    exit 1
fi

# Configurações flexíveis
CONTAINER_NAME="${CONTAINER_NAME:-humob}"
START_MLFLOW="${START_MLFLOW:-true}"
SHM_SIZE="${SHM_SIZE:-4g}"

echo "📊 Montando parquet: $(du -h "$PARQUET_FILE" | cut -f1)"

# Execução
CUDA_VISIBLE_DEVICES=0 docker run --rm -it \
  --gpus all \
  --name "$CONTAINER_NAME" \
  -p 5000:5000 \
  -e START_MLFLOW="$START_MLFLOW" \
  -v "$PARQUET_FILE:/workspace/data/processed/humob_all_cities_v2_normalized.parquet:ro" \
  -v "$CIA_HOME/humob_outputs:/workspace/outputs" \
  -v "$CIA_HOME/humob_experiments:/workspace/experiments" \
  --shm-size="$SHM_SIZE" \
  humob:cu128 "$@"