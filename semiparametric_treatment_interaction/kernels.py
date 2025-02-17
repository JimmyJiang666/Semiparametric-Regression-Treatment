"""
Kernel functions module.
Contains kernel functions used for smoothing and estimation, including Gaussian kernel,
Nadaraya-Watson estimator, and high-dimensional kernel smoothing implementations.
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

### Model Training Functions ###

def nw(x, X, Y, h, K=norm.pdf):
    """
    Kernel-weighted smoothing for a given target x.

    Args:
        x (np.ndarray): 1D array representing the target point.
        X (np.ndarray): 2D array of sample points.
        Y (np.ndarray): 1D or 2D array of responses corresponding to X.
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function to compute weights (default: norm.pdf).

    Returns:
        np.ndarray: The weighted sum of Y computed via kernel smoothing.
    """
    x = x.reshape(-1, 1) # Ensure x has two dimensions
    X_diff = (X - x.T) / h
    Kx = K(X_diff) / h
    W = Kx / Kx.sum(axis=1, keepdims=True)
    return np.dot(W, Y)

def high_dim_nw(ZZ, Y, h, K=norm.pdf):
    """
    High-dimensional kernel smoothing using broadcasting.

    Args:
        ZZ (np.ndarray): 1D array of values to compute pairwise differences.
        Y (np.ndarray): 1D array of response values corresponding to ZZ.
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function to compute weights (default: norm.pdf).

    Returns:
        np.ndarray: The kernel-smoothed response computed as the weighted sum of Y.
    """
    # Use broadcasting instead of np.tile for pairwise differences
    diff = (ZZ[:, None] - ZZ[None, :]) / h
    M = K(diff) / h
    np.fill_diagonal(M, 0)
    normalized_M = M / M.sum(axis=1, keepdims=True)
    return normalized_M @ Y

def stable_high_dim_nw(ZZ, Y, h, K=norm.pdf, eps=1e-10):
    """
    Compute normalized kernel weights for input data using the log-sum-exp trick for numerical stability.

    Args:
        ZZ (np.ndarray): 1D array of shape (n,) for computing pairwise differences.
        Y (np.ndarray): 1D array of responses of shape (n,).
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function, defaults to the Gaussian pdf.
        eps (float, optional): Small constant for numerical stability, defaults to 1e-10.

    Returns:
        np.ndarray: Weighted sum computed as the dot product of normalized weights and Y.
    """
    # Compute the pairwise differences in a vectorized way.
    diff = (ZZ[:, None] - ZZ[None, :]) / h
    
    # Compute log kernel values: note that for the Gaussian,
    # log(K(u)) = -0.5 * u**2 - 0.5*log(2*pi)
    log_K = -0.5 * diff**2 - 0.5 * np.log(2*np.pi)
    # Adjust for the scaling by h (since K(diff)/h is used)
    log_K = log_K - np.log(h)
    
    # Set the diagonal to -inf so it does not contribute to the sum.
    np.fill_diagonal(log_K, -np.inf)
    
    # Compute the log-sum-exp for each row.
    max_log_K = np.max(log_K, axis=1, keepdims=True)
    # Use max subtraction to stabilize the sum.
    sum_exp = np.sum(np.exp(log_K - max_log_K), axis=1, keepdims=True)
    log_denom = max_log_K + np.log(sum_exp + eps)  # add eps inside log for extra stability
    
    # Now compute the normalized weights
    log_weights = log_K - log_denom
    weights = np.exp(log_weights)
    
    return weights @ Y 