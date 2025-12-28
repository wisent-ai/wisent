import os
from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read version from __init__.py
with open(os.path.join("wisent", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="wisent",
    version=version,
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",  # Replace with your email
    description="Monitor and influence AI Brains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wisent-ai/wisent",  # Replace with your GitHub repo
    packages=find_packages(exclude=["patches", "patches.*"]),  # Exclude patches directory
    include_package_data=True,
    package_data={
        "wisent": [
            "core/evaluators/benchmark_specific/coding/safe_docker/Dockerfile",
            "core/evaluators/benchmark_specific/coding/safe_docker/entrypoint.py",
            "parameters/lm_eval/*.json",
            "parameters/tasks/*.json",
            "examples/scripts/*.json",
            "scripts/*.sh",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "numpy>=1.21.0",
        "datasets>=2.0.0",
        "sentence-transformers>=2.0.0",
        "faiss-cpu>=1.7.0",
        "uncensorbench>=0.2.0",
        "pebble>=5.0.0",
        "latex2sympy2_extended>=1.0.0",
        "sae_lens>=0.1.0",
        "trl>=0.7.0",
    ],
    extras_require={
        "harness": [
            "lm-eval==0.4.8",
        ],
        "cuda": [
            "flash-attn>=2.5.0",
        ],
        "sparsify": [
            "sparsify>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wisent=wisent.core.main:main",
        ],
    },
    keywords="nlp, machine learning, language models, safety, guardrails, lm-evaluation-harness",
) 