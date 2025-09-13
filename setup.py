from setuptools import setup, find_packages

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
)