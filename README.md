# Semiparametric Treatment Interaction Model

[中文文档](README_CN.md)

## Overview

This repository implements an innovative semiparametric regression model for analyzing treatment-covariate interactions in continuous treatment settings. The project provides a comprehensive framework for estimating and evaluating personalized treatment effects, particularly useful in medical research, clinical trials, and precision medicine.

## Key Features

- 🔬 **Advanced Statistical Methods**: Implementation of repeated Nadaraya-Watson regression estimators for continuous treatment variables
- 📊 **Flexible Modeling Framework**: Support for various kernel functions and bandwidth selection methods
- 🛠️ **Optimization Suite**: Integration of multiple optimization methods (Hyperopt, CMA-ES, Differential Evolution)
- 🔍 **Cross-validation**: Complete framework for model selection and hyperparameter tuning
- 📈 **Visualization Tools**: Rich visualization capabilities including heatmaps, ROC curves, and more

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
 ┣ 📜 setup.py             # Package installation configuration
 ┣ 📜 requirements.txt     # Project dependencies
 ┣ 📜 README.md            # English documentation
 ┣ 📜 README_CN.md         # Chinese documentation
 ┗ 📜 LICENSE             # MIT License
```
## Core Modules

1. **Model Training** (`model.py`)
   - Cross-validation
   - Model fitting
   - Prediction functions

2. **Kernel Functions** (`kernels.py`)
   - Nadaraya-Watson estimator
   - High-dimensional kernel smoothing
   - Numerically stable implementations

3. **Optimization Methods** (`optimizers.py`)
   - Hyperopt optimization
   - CMA-ES algorithm
   - Differential Evolution
   - Optuna framework

4. **Objective Functions** (`objectives.py`)
   - Basic objective
   - Lasso-regularized objective
   - Specialized objectives for different optimizers

5. **Utilities** (`utils.py`)
   - Data processing
   - Evaluation metrics
   - Helper functions

6. **Visualization** (`visualization.py`)
   - Heatmap generation
   - ROC curves
   - Distribution plots
   - 3D surface plots

## Applications

- 🏥 Clinical Trial Analysis
- 💊 Personalized Medicine Research
- 📊 Biostatistics Research
- 🔬 Medical Research Data Analysis
- 📈 Continuous Treatment Effect Evaluation

## Technical Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- See `requirements.txt` for full list

## Installation`

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper-reference,
  title={Learning Interactions Between Continuous Treatments and Covariates with a Semiparametric Model},
  author={Your Name},
  journal={Conference on Health, Inference, and Learning (CHIL)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Project Maintainer: [Muyan Jiang]
- Email: [muyan_jiang@berkeley.edu]
