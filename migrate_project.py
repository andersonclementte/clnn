#!/usr/bin/env python3
"""
Script de migração automática para reorganizar o projeto HuMob Challenge.

Execução: python migrate_project.py
"""

import os
import shutil
import glob
from pathlib import Path


def create_directory_structure():
    """Cria a nova estrutura de diretórios."""
    directories = [
        'src',
        'src/models',
        'src/data', 
        'src/training',
        'src/evaluation',
        'src/utils',
        'scripts',
        'config',
        'data',
        'data/raw',
        'data/processed', 
        'outputs',
        'outputs/models',
        'outputs/submissions',
        'outputs/plots',
        'experiments',
        'experiments/logs',
        'experiments/mlruns',
        'docs',
        'tests'
    ]
    
    print("🏗️ Criando estrutura de diretórios...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return directories


def create_init_files():
    """Cria arquivos __init__.py necessários."""
    init_locations = [
        'src/__init__.py',
        'src/models/__init__.py', 
        'src/data/__init__.py',
        'src/training/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py'
    ]
    
    print("\n📄 Criando arquivos __init__.py...")
    for init_file in init_locations:
        with open(init_file, 'w') as f:
            f.write('"""HuMob Challenge - Human Mobility Prediction"""\n')
        print(f"   ✅ {init_file}")


def move_files():
    """Move arquivos para suas novas localizações."""
    
    # Mapeamento de arquivos (origem -> destino)
    file_mappings = {
        # Modelos
        'external_information.py': 'src/models/external_info.py',
        'partial_information.py': 'src/models/partial_info.py', 
        'humob_model.py': 'src/models/humob_model.py',
        
        # Data
        'humob_dataset.py': 'src/data/dataset.py',
        
        # Training
        'humob_training.py': 'src/training/train.py',
        'humob_finetuning.py': 'src/training/finetune.py',
        'humob_pipeline.py': 'src/training/pipeline.py',
        
        # Utils
        'mlflow_utils.py': 'src/utils/mlflow_tracker.py',
        'pytorch_compatibility.py': 'src/utils/pytorch_compat.py',
        
        # Scripts
        'run_humob.py': 'scripts/train.py',
        'test.py': 'scripts/evaluate.py',
        'check_setup.py': 'scripts/setup_check.py',
        
        # Docs
        'humob_architecture_diagram.html': 'docs/architecture.html',
    }
    
    print("\n🔄 Movendo arquivos...")
    for source, destination in file_mappings.items():
        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"   ✅ {source} → {destination}")
        else:
            print(f"   ⚠️ Não encontrado: {source}")
    
    # Move dados
    print("\n📊 Movendo arquivos de dados...")
    for parquet_file in glob.glob("*.parquet"):
        dest = f"data/processed/{parquet_file}"
        shutil.move(parquet_file, dest)
        print(f"   ✅ {parquet_file} → {dest}")
    
    # Move modelos
    print("\n🤖 Movendo modelos...")
    for pt_file in glob.glob("*.pt"):
        dest = f"outputs/models/{pt_file}"
        shutil.move(pt_file, dest) 
        print(f"   ✅ {pt_file} → {dest}")
        
    for npy_file in glob.glob("*.npy"):
        dest = f"outputs/models/{npy_file}"
        shutil.move(npy_file, dest)
        print(f"   ✅ {npy_file} → {dest}")
    
    # Move submissões
    print("\n📄 Movendo submissões...")
    for csv_file in glob.glob("*submission*.csv"):
        dest = f"outputs/submissions/{csv_file}"
        shutil.move(csv_file, dest)
        print(f"   ✅ {csv_file} → {dest}")
    
    # Move plots
    print("\n📈 Movendo gráficos...")
    for png_file in glob.glob("*.png"):
        dest = f"outputs/plots/{png_file}"
        shutil.move(png_file, dest)
        print(f"   ✅ {png_file} → {dest}")
    
    # Move logs
    print("\n📝 Movendo logs...")
    for log_file in glob.glob("*.txt"):
        if "log" in log_file.lower():
            dest = f"experiments/logs/{log_file}"
            shutil.move(log_file, dest)
            print(f"   ✅ {log_file} → {dest}")
    
    # Move MLflow
    if os.path.exists("mlruns"):
        if os.path.exists("experiments/mlruns"):
            shutil.rmtree("experiments/mlruns")
        shutil.move("mlruns", "experiments/mlruns")
        print("   ✅ mlruns/ → experiments/mlruns/")


def update_imports_in_file(filepath, import_mappings):
    """Atualiza imports em um arquivo específico."""
    if not os.path.exists(filepath):
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Substitui imports antigos por novos
        updated_content = content
        for old_import, new_import in import_mappings.items():
            updated_content = updated_content.replace(old_import, new_import)
        
        # Se houve mudanças, salva o arquivo
        if updated_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"   ✅ Imports atualizados: {filepath}")
                
    except Exception as e:
        print(f"   ⚠️ Erro atualizando {filepath}: {e}")


def update_all_imports():
    """Atualiza imports em todos os arquivos Python."""
    
    # Mapeamento de imports antigos para novos
    import_mappings = {
        'from src.models.external_info import': 'from src.models.external_info import',
        'from src.models.partial_info import': 'from src.models.partial_info import', 
        'from src.models.humob_model import': 'from src.models.humob_model import',
        'from src.data.dataset import': 'from src.data.dataset import',
        'from src.training.train import': 'from src.training.train import',
        'from src.training.finetune import': 'from src.training.finetune import',
        'from src.training.pipeline import': 'from src.training.pipeline import',
        'from src.utils.mlflow_tracker import': 'from src.utils.mlflow_tracker import',
        'from src.utils.pytorch_compat import': 'from src.utils.pytorch_compat import',
        
        'import src.models.external_info': 'import src.models.external_info',
        'import src.models.partial_info': 'import src.models.partial_info',
        'import src.models.humob_model': 'import src.models.humob_model',
        'import src.data.dataset': 'import src.data.dataset',
        'import src.training.train': 'import src.training.train',
        'import src.training.finetune': 'import src.training.finetune',
        'import src.training.pipeline': 'import src.training.pipeline',
        'import src.utils.mlflow_tracker': 'import src.utils.mlflow_tracker',
        'import src.utils.pytorch_compat': 'import src.utils.pytorch_compat',
    }
    
    print("\n🔄 Atualizando imports...")
    
    # Busca todos os arquivos Python na nova estrutura
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Pula diretórios que não queremos processar
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    for py_file in python_files:
        update_imports_in_file(py_file, import_mappings)


def create_main_runner():
    """Cria script principal atualizado na raiz."""
    runner_content = '''#!/usr/bin/env python3
"""
Script principal do HuMob Challenge - ponto de entrada único.
"""

import sys
from pathlib import Path

# Adiciona src ao Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importa e executa o script principal
if __name__ == "__main__":
    from scripts.train import main
    main()
'''
    
    with open('run.py', 'w') as f:
        f.write(runner_content)
    print("\n🚀 Criado script principal: run.py")


def create_config_files():
    """Cria arquivos de configuração."""
    
    # Setup.py para instalação
    setup_content = '''from setuptools import setup, find_packages

setup(
    name="humob-challenge",
    version="1.0.0", 
    description="Human Mobility Prediction - HuMob Challenge 2024",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.62.0",
        "pyarrow>=5.0.0",
        "mlflow>=2.0.0",
        "seaborn>=0.11.0"
    ],
    python_requires=">=3.8",
)'''
    
    with open('setup.py', 'w') as f:
        f.write(setup_content)
    print("\n📦 Criado setup.py")
    
    # Atualiza .gitignore
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data files
data/raw/*.parquet
data/processed/*.parquet

# Model outputs  
outputs/models/*.pt
outputs/models/*.npy

# Experiment data
experiments/mlruns/
experiments/logs/*.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("📝 Criado/atualizado .gitignore")


def main():
    """Executa migração completa."""
    print("🚀 MIGRAÇÃO AUTOMÁTICA DO PROJETO HUMOB")
    print("=" * 50)
    
    # Confirma antes de continuar
    response = input("Deseja continuar com a reorganização? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Migração cancelada")
        return
    
    try:
        create_directory_structure()
        create_init_files()
        move_files()
        update_all_imports()
        create_main_runner()
        create_config_files()
        
        print("\n🎉 MIGRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 50)
        print("✅ Estrutura de pastas criada")
        print("✅ Arquivos movidos para novas localizações") 
        print("✅ Imports atualizados")
        print("✅ Scripts principais criados")
        print("✅ Configurações atualizadas")
        
        print("\n📋 PRÓXIMOS PASSOS:")
        print("1. Execute: python run.py  # Testa se tudo funciona")
        print("2. Execute: pip install -e .  # Instala em modo desenvolvimento")
        print("3. Execute: python scripts/setup_check.py  # Verifica setup")
        
        print("\n🎯 NOVA ESTRUTURA PRONTA!")
        
    except Exception as e:
        print(f"\n❌ Erro durante migração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()