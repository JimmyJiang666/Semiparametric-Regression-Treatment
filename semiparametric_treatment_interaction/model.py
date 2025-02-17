"""
Model module.
Contains core model training and prediction functions, including cross-validation,
model fitting, and prediction methods.
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

from .kernels import nw, high_dim_nw
from .objectives import objective, objective_lasso
from .optimizers import hyperopt_train

def cross_validate(X_train, T_train, Y_train, h_values, n_cv=5, K=norm.pdf):
    """
    Perform cross-validation over candidate bandwidth values and return the one that minimizes the mean squared error.

    Args:
        X_train (np.ndarray): Training feature matrix.
        T_train (np.ndarray): Training treatment vector.
        Y_train (np.ndarray): Training outcome vector.
        h_values (iterable): List or array of candidate bandwidth values.
        n_cv (int, optional): Number of cross-validation folds (default: 5).
        K (callable, optional): Kernel function to compute weights (default: norm.pdf).

    Returns:
        float: The optimal bandwidth (h) that minimizes the average mean squared error.
    """
    # Initialize the KFold cross-validator
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    h_mses = {}
    # Loop over each h value
    for h in h_values:
        mses = []
        print(f"Looking at h={h} now...\n")
        # Loop over each fold
        for train_index, val_index in kf.split(X_train):
            # Split the data into training and validation for the current fold
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            T_train_fold, T_val_fold = T_train[train_index], T_train[val_index]
            Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]
            
            # Train the model on the current training set
            _, XI_opt_fold, BETA_opt_fold = train(X_train_fold, Y_train_fold, T_train_fold, h, K)
            
            # Define y function for the current training set and trained parameters
            def y_fold(x, t):
                x_array = np.array(x).reshape(1, -1)  
                nw_part = nw(np.dot(x_array, XI_opt_fold) - t, np.dot(X_train_fold, XI_opt_fold) - T_train_fold, Y_train_fold - np.dot(BETA_opt_fold, X_train_fold.T), h)
                return np.dot(BETA_opt_fold, x_array.T) + nw_part

            # Make predictions on the validation set
            Y_pred_fold = np.array([y_fold(x, t) for x, t in zip(X_val_fold, T_val_fold)])
            
            # Create a mask of non-NaN values in Y_pred_fold
            non_nan_mask = ~np.isnan(Y_pred_fold)
            Y_val_fold = Y_val_fold.reshape(-1)
            Y_pred_fold = Y_pred_fold.reshape(-1)
            non_nan_mask = non_nan_mask.reshape(-1)
            # Filter both Y_val_fold and Y_pred_fold using the non-NaN mask
            Y_val_fold_filtered = Y_val_fold[non_nan_mask]
            Y_pred_fold_filtered = Y_pred_fold[non_nan_mask]
            # Calculate the mean squared error using the filtered values            
            mse = mean_squared_error(Y_val_fold_filtered, Y_pred_fold_filtered)

            mses.append(mse)
        
        # Store the average MSE for this h value
        h_mses[h] = np.mean(mses)
    
    # Find the h with the smallest average MSE
    best_h = min(h_mses, key=h_mses.get)
    
    return best_h

def train(X_train, Y_train, T_train, h, K=norm.pdf):
    """
    Train the semiparametric model using the specified kernel function and bandwidth.

    Args:
        X_train (np.ndarray): Training feature matrix.
        Y_train (np.ndarray): Training outcome vector.
        T_train (np.ndarray): Training treatment vector.
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function (default: norm.pdf).

    Returns:
        tuple: (best_loss, XI_opt, BETA_opt) containing the best loss value and optimal parameters.
    """
    # Train using hyperopt with default objective function
    XI_opt, BETA_opt, trials, best_loss = hyperopt_train(X_train, Y_train, T_train, h, objective_func=objective, lam=0.0, K=K)
    return best_loss, XI_opt, BETA_opt

def y(x, t, X_train, T_train, Y_train, XI_opt, BETA_opt, best_h):
    """
    Compute a single prediction via kernel smoothing using optimal parameters.

    Args:
        x (np.ndarray): Feature vector.
        t (float): Treatment value for the observation.
        X_train (np.ndarray): Training feature matrix.
        T_train (np.ndarray): Training treatment vector.
        Y_train (np.ndarray): Training outcome vector.
        XI_opt (np.ndarray): Optimized XI parameter vector.
        BETA_opt (np.ndarray): Optimized beta parameter vector.
        best_h (float): Bandwidth parameter.

    Returns:
        float: Predicted outcome.
    """
    x_array = np.array(x).reshape(1, -1)  # Ensuring x has shape (1, p), where p is the number of features
    nw_part = nw(np.dot(x_array, XI_opt) - t, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)
    return np.dot(x_array, BETA_opt) + nw_part

def predict(XX, TT, X_train, T_train, Y_train, XI_opt, BETA_opt, best_h):
    """
    Predict outcomes for multiple observations using kernel smoothing.

    Args:
        XX (np.ndarray): Matrix of feature vectors for prediction.
        TT (np.ndarray): Vector of treatment values corresponding to XX.
        X_train (np.ndarray): Training feature matrix.
        T_train (np.ndarray): Training treatment vector.
        Y_train (np.ndarray): Training outcome vector.
        XI_opt (np.ndarray): Optimized XI parameter vector.
        BETA_opt (np.ndarray): Optimized beta parameter vector.
        best_h (float): Bandwidth parameter.

    Returns:
        np.ndarray: Array of predicted outcomes.
    """
    return np.array([y(x, t, X_train, T_train, Y_train, XI_opt, BETA_opt, best_h) for x, t in zip(XX, TT)]) 