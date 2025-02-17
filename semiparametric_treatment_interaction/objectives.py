"""
Objective functions module.
Contains objective functions for optimization, including basic and Lasso-regularized objectives,
as well as specialized objectives for different optimization methods (Hyperopt, Optuna, etc.).
"""

# Standard libraries
import time
import warnings
# Data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotly.graph_objects as go
# Optimization modules
from scipy.stats import norm, binom
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import optuna
import cma
# Machine learning and evaluation
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (mean_squared_error, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score)
# Additional tools
import numdifftools as nd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
from hyperopt import fmin as hyperopt_fmin
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from .kernels import high_dim_nw

def objective(XI, X, Y, T, h, K=norm.pdf):
    """
    Compute the kernel-smoothed objective value for parameter estimation.

    Args:
        XI (np.ndarray): Parameter vector for transforming features.
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Outcome vector.
        T (np.ndarray): Treatment vector.
        h (float): Bandwidth parameter for kernel smoothing.
        K (callable, optional): Kernel function (default: norm.pdf).

    Returns:
        float: Average squared error from the kernel-smoothed residuals.
    """
    n = X.shape[0]
    exi = X - high_dim_nw(np.dot(X, XI) - T, X, h, K)
    eyi = Y - high_dim_nw(np.dot(X, XI) - T, Y, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    ObjVal = np.sum((eyi - np.dot(exi, beta)) ** 2)
    return 1/n * ObjVal

def objective_lasso(XI, X, Y, T, h, lam, K=norm.pdf, weights = None):
    """
    Compute the Lasso-regularized kernel-smoothed objective value.

    Args:
        XI (np.ndarray): Parameter vector for feature transformation.
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Outcome vector.
        T (np.ndarray): Treatment vector.
        h (float): Bandwidth parameter for kernel smoothing.
        lam (float): Regularization parameter for Lasso.
        K (callable, optional): Kernel function (default: norm.pdf).
        weights (np.ndarray, optional): Weights for each observation; defaults to ones if not provided.

    Returns:
        float: Regularized objective value combining the weighted squared error and the L1 penalty.
    """
    n = X.shape[0]
    
    # Set default weights to ones if not provided
    if weights is None:
        weights = np.ones_like(Y)
    exi = X - high_dim_nw(np.dot(X, XI) - T, X, h, K)
    eyi = Y - high_dim_nw(np.dot(X, XI) - T, Y, h, K)
    beta = (exi.T @ eyi) / np.sum(exi ** 2, axis=0)
    
    # Incorporate weights into the objective value calculation
    weighted_errors = weights * (eyi - np.dot(exi, beta))
    ObjVal = np.sum(weighted_errors ** 2)
    
    # L1 penalty term for Lasso
    lasso_penalty = lam * np.sum(np.abs(XI))
    
    return 1/n * (ObjVal) + lasso_penalty

def hyperopt_objective(args, X, Y, T, h, objective_func, lam, K, weights = None):
    """
    Compute the hyperopt objective by converting spherical coordinates to the XI parameter
    and evaluating the given objective function.

    Returns:
        dict: Dictionary with keys 'loss' (the computed loss), 'status' (STATUS_OK flag),
            'eval_time' (timestamp), 'XI' (transformed parameters as a list), and 'ObjVal' (loss value).
    """
    XI = np.array(spherical_to_cartesian(*args))
    loss = objective_func(XI, X, Y, T, h, lam, K, weights)
    return {'loss': loss, 'status': STATUS_OK, 'eval_time': time.time(), 'XI': XI.tolist(), 'ObjVal': loss}

def optuna_objective(trial, X, Y, T, h, objective_func, lam, K, weights=None):
    """
    Compute the Optuna objective for hyperparameter tuning using spherical coordinates.
    """
    p = X.shape[1]
    num_angles = p - 1  # number of angles needed
    # For the first angle, restrict to [0, pi/2] to enforce a positive first coordinate.
    angle0 = trial.suggest_uniform("angle_0", 0, np.pi/2)
    angles = [angle0]
    # The remaining angles in [0, pi]
    for i in range(1, num_angles):
        angles.append(trial.suggest_uniform(f"angle_{i}", 0, np.pi))
    XI = np.array(spherical_to_cartesian(*angles))
    loss = objective_func(XI, X, Y, T, h, lam, K, weights)
    return loss

def de_objective(angles, X, Y, T, h, objective_func, lam, K, weights=None):
    """		
    Compute the Differential Evolution objective using spherical coordinates.
    """
    XI = np.array(spherical_to_cartesian(*angles))
    return objective_func(XI, X, Y, T, h, lam, K, weights)

def cma_objective(angles, X, Y, T, h, objective_func, lam, K, weights=None):
    """
    Compute the CMA-ES objective using spherical coordinates.
    """
    XI = np.array(spherical_to_cartesian(*angles))
    return objective_func(XI, X, Y, T, h, lam, K, weights)

def spherical_to_cartesian(*angles):
    """
    Convert angles in n-sphere to Cartesian coordinates.
    E.g. for 3D: r, theta, phi -> x, y, z
    """
    dim = len(angles) + 1
    cart_coords = [np.sin(angles[0])]
    
    for i in range(1, dim - 1):
        product = np.sin(angles[i])
        for j in range(i):
            product *= np.cos(angles[j])
        cart_coords.append(product)
    
    last_coord = 1
    for angle in angles:
        last_coord *= np.cos(angle)
    cart_coords.append(last_coord)
    
    return cart_coords 