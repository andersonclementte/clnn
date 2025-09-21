# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TZ=America/Sao_Paulo \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pacotes mínimos do SO
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git bash tini \
    build-essential pkg-config \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Instala dependências Python como root (evita problemas de permissão no /opt/conda)
COPY requirements.txt ./
RUN echo "🐍 Python version: $(python --version)" \
 && pip install --upgrade pip \
 && pip install mlflow pandas numpy scikit-learn matplotlib seaborn \
      pyarrow tqdm plotly \
 && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copia o projeto e instala em modo editable
COPY . .
RUN pip install -e .

# Configuração MLflow + pastas padrão
ENV MLFLOW_TRACKING_URI="file:/workspace/experiments/mlruns"
RUN mkdir -p /workspace/experiments/mlruns /workspace/outputs /workspace/data/processed

# Cria usuário não-root e entrega posse do workspace
RUN groupadd -g ${GID} humob && useradd -m -u ${UID} -g ${GID} humob \
 && chown -R humob:humob /workspace

# Copia entrypoint e dá permissão
COPY --chown=humob:humob entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

EXPOSE 5000

# Roda como usuário normal
USER humob

ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
