# Semiparametric Treatment Interaction Model

[📄 中文文档](README_CN.md)

## Overview

This repository implements an innovative **semiparametric regression model** for analyzing **treatment-covariate interactions** in continuous treatment settings. It provides a comprehensive framework for estimating and evaluating **personalized treatment effects**, making it particularly useful in medical research, clinical trials, and precision medicine.

## Key Features

- 🔬 **Advanced Statistical Methods**: Implementation of **repeated Nadaraya-Watson regression estimators** for continuous treatment variables
- 📊 **Flexible Modeling Framework**: Support for various **kernel functions** and **bandwidth selection methods**
- 🛠️ **Optimization Suite**: Integration of multiple optimization methods (**Hyperopt, CMA-ES, Differential Evolution**)
- 🔍 **Cross-validation**: Complete framework for **model selection and hyperparameter tuning**
- 📈 **Visualization Tools**: Rich visualization capabilities including **heatmaps, ROC curves, and more**

## Project Structure

```
📦 semiparametric-treatment-interaction
 ┣ 📂 semiparametric_treatment_interaction/  # Main package directory
 ┃ ┣ 📜 model.py            # Core model training and prediction
 ┃ ┣ 📜 kernels.py          # Kernel function implementations
 ┃ ┣ 📜 objectives.py       # Objective functions for optimization
 ┃ ┣ 📜 optimizers.py       # Optimization algorithms
 ┃ ┣ 📜 utils.py            # Utility functions and data processing
 ┃ ┣ 📜 visualization.py    # Visualization tools
 ┃ ┗ 📜 __init__.py         # Package initialization
 ┣ 📂 examples/             # Example notebooks
 ┃ ┣ 📜 simulation.ipynb    # Simulation data analysis
 ┃ ┣ 📜 beta_xi_conf.ipynb  # Parameter confidence interval estimation
 ┃ ┗ 📜 diag_score_comparison.ipynb  # Diagnostic score comparison
 ┣ 📂 tests/                # Test suite
 ┣ 📂 docs/                 # Documentation
 ┣ 📂 figures/              # Figures produced
 ┣ 📜 setup.py             # Package installation configuration
 ┣ 📜 requirements.txt     # Project dependencies
 ┣ 📜 README.md            # English documentation
 ┣ 📜 README_CN.md         # Chinese documentation
 ┗ 📜 LICENSE             # MIT License
```

## Core Modules

### 1️⃣ Model Training (`model.py`)
- Cross-validation
- Model fitting
- Prediction functions

### 2️⃣ Kernel Functions (`kernels.py`)
- Nadaraya-Watson estimator
- High-dimensional kernel smoothing
- Numerically stable implementations

### 3️⃣ Optimization Methods (`optimizers.py`)
- Hyperopt optimization
- CMA-ES algorithm
- Differential Evolution
- Optuna framework

### 4️⃣ Objective Functions (`objectives.py`)
- Basic objective
- Lasso-regularized objective
- Specialized objectives for different optimizers

### 5️⃣ Utilities (`utils.py`)
- Data processing
- Evaluation metrics
- Helper functions

### 6️⃣ Visualization (`visualization.py`)
- Heatmap generation
- ROC curves
- Distribution plots
- 3D surface plots

## Applications

- 🏥 **Clinical Trial Analysis**
- 💊 **Personalized Medicine Research**
- 📊 **Biostatistics Research**
- 🔬 **Medical Research Data Analysis**
- 📈 **Continuous Treatment Effect Evaluation**

## Installation

### 1️⃣ Create and Activate a Virtual Environment (Recommended)

#### Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Install the Package in Development Mode
```bash
pip install -e .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper-reference,
  title={Learning Interactions Between Continuous Treatments and Covariates with a Semiparametric Model},
  author={***},
  journal={Conference on Health, Inference, and Learning (CHIL)},
  year={2025}
}
```

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contact

- **Project Maintainer**: **Muyan Jiang**
- 📧 Email: [muyan_jiang@berkeley.edu](mailto:muyan_jiang@berkeley.edu)
