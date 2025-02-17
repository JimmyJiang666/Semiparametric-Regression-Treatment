"""
Optimization methods module.
Contains various optimization algorithms for model training, including Hyperopt,
Optuna, Differential Evolution, and CMA-ES implementations.
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
from .objectives import *
from .kernels import *


def hyperopt_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf, max_evals = 100, weights = None):
    """
    Perform hyperparameter optimization using hyperopt with spherical coordinates to determine
    the optimal XI and corresponding beta parameters.

    Args:
        X_train (np.ndarray): Training feature matrix.
        Y_train (np.ndarray): Training outcome vector.
        T_train (np.ndarray): Training treatment vector.
        h (float): Bandwidth parameter.
        objective_func (callable): Objective function to optimize.
        lam (float): Regularization parameter for Lasso.
        K (callable, optional): Kernel function (default: norm.pdf).
        max_evals (int, optional): Maximum number of evaluations for hyperopt (default: 100).
        weights (np.ndarray, optional): Observation weights; defaults to ones if None.

    Returns:
        tuple: A tuple (XI_best, beta, trials, best_loss) where:
            XI_best (np.ndarray): Optimized XI parameter vector.
            beta (np.ndarray): Corresponding beta parameter vector.
            trials (hyperopt.Trials): Hyperopt trials object containing optimization details.
            best_loss (float): Best loss value achieved.
    """

    # Define the search space for hyperopt using spherical coordinates
    dim = X_train.shape[1]
    # Restricting the range of the first angle to ensure positive first coordinate in XI
    angles_space = [hp.uniform(f'angle_0', 0, np.pi/2)]
    angles_space += [hp.uniform(f'angle_{i}', 0, np.pi) for i in range(1, dim-2)]
    angles_space += [hp.uniform('angle_last', 0, 2*np.pi)]
    # Create a Trials object
    trials = Trials()
    # spark_trials = SparkTrials(parallelism=4)
    # Use hyperopt to find the best XI values using the angles_space
    best = hyperopt_fmin(
            lambda args: hyperopt_objective(args, X_train, Y_train, T_train, h, objective_func, lam, K, weights),
            space=angles_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials)
    # Convert the result dictionary to a numpy array
    angles_best = [best[f'angle_{i}'] for i in range(dim-2)] + [best['angle_last']]
    XI_best = spherical_to_cartesian(*angles_best)
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)

    return np.array(XI_best), np.array(beta), trials, trials.best_trial['result']['loss']

def calculate_heatmaps_hyperopt(X, T, Y, beta, xi, n, p, best_h, f0, max_evals = 200):
    """
    Fit model parameters via hyperopt and generate corresponding ground truth and estimand heatmaps.
    """
    # Split data into training and test sets
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, Z_test = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p + 1]

    # Fit model and find optimal XI and BETA
    XI_opt, BETA_opt, _, _ = hyperopt_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=0.0, max_evals = max_evals)

    # Define the ranges for the heatmap
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    # Function using the model
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)

    # Generate heatmap data
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)

    return XI_opt, BETA_opt, Z_f0, Z_f

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

def optuna_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf, n_trials=100, weights=None):
    """
	Perform hyperparameter optimization using Optuna and spherical coordinates.
	"""

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, Y_train, T_train, h, objective_func, lam, K, weights),
                   n_trials=n_trials)
    
    best_trial = study.best_trial
    p = X_train.shape[1]
    num_angles = p - 1
    angles_best = [best_trial.params[f"angle_{i}"] for i in range(num_angles)]
    XI_best = spherical_to_cartesian(*angles_best)
    
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    
    return np.array(XI_best), np.array(beta), study, study.best_value

def calculate_heatmaps_optuna(X, T, Y, beta, xi, n, p, best_h, f0, n_trials=200):
    """
	Split data, optimize model parameters with Optuna, and compute corresponding heatmap grids.
    """
    # Split data into training and test sets.
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, _ = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p + 1]

    # Fit model using optuna_train.
    XI_opt, BETA_opt, study, best_value = optuna_train(X_train, Y_train, T_train, best_h,
                                                       objective_func=objective_lasso, lam=0.0, 
                                                       n_trials=n_trials)
    # Define the ranges for the heatmap.
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    
    # Define the function using the model.
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)

    # Generate heatmap data.
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)

    return XI_opt, BETA_opt, Z_f0, Z_f

def de_objective(angles, X, Y, T, h, objective_func, lam, K, weights=None):
    """		
    Compute the Differential Evolution objective using spherical coordinates.
    """
    XI = np.array(spherical_to_cartesian(*angles))
    return objective_func(XI, X, Y, T, h, lam, K, weights)

def de_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf):
    """
    Perform model fitting using Differential Evolution and return optimized parameters.
    """
    p = X_train.shape[1]
    num_angles = p - 1  # For a p-dimensional unit vector
    # Set bounds: first angle in [0, pi/2] to enforce XI[0] >= 0; others in [0, pi]
    bounds = [(0, np.pi/2)] + [(0, np.pi)] * (num_angles - 1)
    
    result = differential_evolution(
        de_objective,
        bounds,
        args=(X_train, Y_train, T_train, h, objective_func, lam, K),
        maxiter=100,
        polish=True
    )
    best_angles = result.x
    XI_best = np.array(spherical_to_cartesian(*best_angles))
    
    # Compute residuals and obtain beta via least-squares as before.
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    return XI_best, beta, result.fun

def calculate_heatmaps_de(X, T, Y, beta, xi, n, p, best_h, f0, maxiter=100):
    """
    Uses Differential Evolution to fit the model and computes heatmap grids.
    """
    # Split data into training and (unused) test parts.
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, _ = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p+1]
    
    # Fit the model using DE.
    XI_opt, BETA_opt, best_loss = de_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=0.0, K=norm.pdf)
    
    # Define grid ranges for the heatmap.
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    
    # Define the model function f using the fitted parameters.
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)
    
    # Evaluate f0 (the ground truth function) on the grid.
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    # Evaluate the model-based function f on the grid.
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    
    return XI_opt, BETA_opt, Z_f0, Z_f

def cma_objective(angles, X, Y, T, h, objective_func, lam, K, weights=None):
    """
    Compute the CMA-ES objective using spherical coordinates.
    """
    XI = np.array(spherical_to_cartesian(*angles))
    return objective_func(XI, X, Y, T, h, lam, K, weights)

def cma_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf, sigma=0.1, maxiter=100):
    """
    Perform model fitting using CMA-ES and return optimized parameters.
    """
    p = X_train.shape[1]
    num_angles = p - 1
    # Starting guess: use pi/4 for all angles.
    x0 = np.full(num_angles, np.pi/4)
    # Set bounds: first angle [0, pi/2] and others [0, pi].
    lower_bounds = [0] + [0]*(num_angles - 1)
    upper_bounds = [np.pi/2] + [np.pi]*(num_angles - 1)
    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': maxiter,
        'verb_disp': 1,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    
    def wrapped_objective(angles):
        return cma_objective(angles, X_train, Y_train, T_train, h, objective_func, lam, K)
    
    es.optimize(wrapped_objective)
    best_angles = es.result.xbest
    XI_best = np.array(spherical_to_cartesian(*best_angles))
    
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    return XI_best, beta, es.result.fbest

def calculate_heatmaps_cma(X, T, Y, beta, xi, n, p, best_h, f0, sigma=0.1, maxiter=100):
    """
    Uses CMA-ES to fit the model and computes heatmap grids.
    """
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, _ = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p+1]
    
    XI_opt, BETA_opt, best_obj = cma_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=0.0, K=norm.pdf, sigma=sigma, maxiter=maxiter)
    
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)
    
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    
    return XI_opt, BETA_opt, Z_f0, Z_f

