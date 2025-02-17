"""
Semiparametric Treatment Interaction Model

A comprehensive framework for analyzing treatment-covariate interactions
in continuous treatment settings using semiparametric regression models.
"""

from .functions import (
    nw,
    high_dim_nw,
    stable_high_dim_nw,
    objective,
    objective_lasso,
    cross_validate,
    hyperopt_train,
    optuna_train,
    de_train,
    cma_train,
    y,
    predict
)

__version__ = '1.0.0'
__author__ = '[Your Name]'
__email__ = '[your.email@example.com]' 