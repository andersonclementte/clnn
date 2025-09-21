#!/bin/bash
set -e
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

echo "🚀 HuMob Challenge - Container Iniciado"
echo "🐍 Python: $(python --version)"
echo "🔧 PyTorch: $(python -c 'import torch; print(torch.__version__)')"

python - <<'PY'
import torch
print("🔧 CUDA disponível:", "✅" if torch.cuda.is_available() else "❌")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "🔧 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
  echo "🔧 VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"
fi

# Aviso se o parquet não estiver montado
if [ ! -f "data/processed/humob_all_cities_v2_normalized.parquet" ]; then
  echo "⚠️ Dados não encontrados em data/processed/"
  echo "📋 Monte o volume com o parquet em /workspace/data/processed/"
fi

# MLflow UI opcional
if [ "${START_MLFLOW:-true}" = "true" ]; then
  echo "🔬 Iniciando MLflow UI (porta 5000)..."
  mlflow ui --host 0.0.0.0 --port 5000 &
fi

# Executa comando ou menu
if [ $# -gt 0 ]; then
  exec "$@"
else
  python run.py
fi
