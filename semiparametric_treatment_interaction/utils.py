"""
Utility functions module.
Contains helper functions for data processing, evaluation metrics, and statistical utilities.
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

### Data Extraction Functions ###
def extract_RotGBSG_df(df):
    """
    Extract RotGBSG data with logit-transformed survival probability.
    """
    X = np.array(df[["0", "1", "2", "6", "4", "5"]])
    T = np.array(df["3"]).reshape(-1, 1)
    Y_prob = np.array(df["survival_prob"]).reshape(-1, 1)
    Y = np.log((Y_prob + .000005) / (1.00005 - Y_prob))
    event_time = np.array(df["time"]).reshape(-1, 1)
    delta_reverse = 1 - np.array(df["delta"])
    return X, T, Y.reshape(-1,), Y_prob.reshape(-1,), delta_reverse.reshape(-1,), event_time

def extract_RotGBSG_df_wo_prob(df):
    """
    Extract RotGBSG data from a DataFrame without survival probability transformation.
    """
    X = np.array(df[["0", "1", "2", "6", "4", "5"]])
    T = np.array(df["3"]).reshape(-1, 1)
    event_time = np.array(df["time"]).reshape(-1, 1)
    delta_reverse = 1 - np.array(df["delta"])
    return X, T, delta_reverse.reshape(-1,), event_time

def extract_mimic(df, t_name):
    """
    Extract MIMIC survival data from a DataFrame.
    """
    T = np.array(df[t_name]).reshape(-1, 1)
    Y_prob = np.array(df["survival_prob"]).reshape(-1, 1)
    delta = np.array(df["day_28_flag"]).reshape(-1, 1)
    Y = np.log(Y_prob / (1.0 - Y_prob))
    # Drop the specified columns and then extract the rest
    X = df.drop(columns=[t_name, "survival_prob", "day_28_flag", "mort_day"]).values
    return X, T, Y, Y_prob, delta

### Evaluation Metrics Functions ###
def roc_result(Y_test_class, Y_pred_scores):
    # Calculate false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(Y_test_class, Y_pred_scores)
    
    # Compute the AUC (Area Under the Curve) score
    roc_auc = roc_auc_score(Y_test_class, Y_pred_scores)
    print(f"AUC for our classifier is: {roc_auc}")
    
    # Plotting the ROC
    plt.figure(figsize=(10, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    
    # Annotating some thresholds
    for i, thresh in enumerate(thresholds):
        if i % 50 == 0:  # plot every 50th threshold; this reduces clutter
            plt.annotate(f"{thresh:.2f}", (fpr[i], tpr[i]), fontsize=9, ha="right")
    
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def reg_result(A,B):
    plt.figure(figsize=(10, 6))
    plt.plot(A, label="Actual values")
    plt.plot(B, label="Predicted values")
    # plt.plot(delta_test, label="delta values")
    # plt.plot(Y_test_class, label="softlabel class values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of Actual and Predicted Values")
    plt.legend()
    plt.show()

def compute_c_index(y_true_class, y_pred_prob):
    n = len(y_true_class)
    concordant = 0
    permissible = 0
    tied = 0

    for i in range(n):
        for j in range(i+1, n):
            if y_true_class[i] != y_true_class[j]:
                permissible += 1
                if y_pred_prob[i] == y_pred_prob[j]:
                    tied += 1
                elif y_true_class[i] == 1 and y_pred_prob[i] > y_pred_prob[j]:
                    concordant += 1
                elif y_true_class[j] == 1 and y_pred_prob[j] > y_pred_prob[i]:
                    concordant += 1

    c_index = (concordant + 0.5 * tied) / permissible
    print("# permissible: ", permissible)
    return c_index

def cm_result(Y_test_class,Y_pred_class):
    print(confusion_matrix(Y_test_class, Y_pred_class))
    print(f"Precision Score: {precision_score(Y_test_class, Y_pred_class)}")
    print(f"Recall Score: {recall_score(Y_test_class, Y_pred_class)}")
    print(f"F1 Score: {f1_score(Y_test_class, Y_pred_class)}")
    # print(f"ROCAUC Score: {roc_auc_score(Y_test_class, Y_pred_scores)}")

def get_cindex(Y_test_class,Y_pred):
    Y_pred_prob = log_odds_to_prob(Y_pred)
    c_index_value = compute_c_index(Y_test_class, Y_pred_prob)
    return c_index_value

def cindex_result(Y_test_class,Y_pred):
    Y_pred_prob = log_odds_to_prob(Y_pred)
    c_index_value = compute_c_index(Y_test_class, Y_pred_prob)
    print("c_index_value: ", c_index_value)

### Helper Functions ###
def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = np.std(data)
    ci = std_err * 1.96 / np.sqrt(n)  # 1.96 corresponds to 95% CI
    return m, m - ci, m + ci

def log_odds_to_prob(log_odds):
    odds = np.exp(log_odds)
    prob = odds / (1 + odds)
    return prob

def epanechnikov_kernel(u):
    abs_u = np.abs(u)
    mask = abs_u <= 1
    return 0.75 * (1 - u**2) * mask

def even_resample(X, T, Y, Y_prob, delta, num_bins=10, final_sample_size=None):
    df_resample = pd.DataFrame({
        'index': range(X.shape[0])  # Include an index
    })

    for i in range(X.shape[1]):
        df_resample[f'X_col_{i}'] = X[:, i]

    df_resample['T'] = T
    df_resample['Y'] = Y
    df_resample['Y_prob'] = Y_prob
    df_resample['delta'] = delta
    
    labels = range(num_bins)
    df_resample['T_bins'] = pd.cut(df_resample['T'], bins=num_bins, labels=labels)
    
    if final_sample_size is None:
        samples_per_bin = len(df_resample) // num_bins
    else:
        samples_per_bin = final_sample_size // num_bins
    
    subsamples = []
    for _, group in df_resample.groupby('T_bins'):
        if len(group) >= samples_per_bin:
            subsamples.append(group.sample(samples_per_bin, replace=False))
        else:
            subsamples.append(group.sample(samples_per_bin, replace=True))

    subsample_df = pd.concat(subsamples).sample(frac=1).reset_index(drop=True)
    subsample_df = subsample_df.drop(columns=['T_bins'])
    
    X_resampled = subsample_df.filter(like='X_col').values
    T_resampled = subsample_df['T'].values
    Y_resampled = subsample_df['Y'].values
    Y_prob_resampled = subsample_df['Y_prob'].values
    delta_resampled = subsample_df['delta'].values
    sampled_indices = subsample_df['index'].values  # Extract the indices of the sampled rows
    
    return X_resampled, T_resampled, Y_resampled, Y_prob_resampled, delta_resampled, sampled_indices

def smote_resample(X, T, Y, Y_prob, delta, num_bins=10):
    df_resample = pd.DataFrame({
        'index': range(X.shape[0])
    })

    for i in range(X.shape[1]):
        df_resample[f'X_col_{i}'] = X[:, i]

    df_resample['T'] = T
    df_resample['Y'] = Y
    df_resample['Y_prob'] = Y_prob
    df_resample['delta'] = delta
    
    labels = range(num_bins)
    df_resample['T_bins'] = pd.cut(df_resample['T'], bins=num_bins, labels=labels)
    df_resample['T_bins'] = df_resample['T_bins'].astype(int)  # Convert bins to integer type for SMOTE

    min_samples = df_resample['T_bins'].value_counts().min()
    k_neighbors = max(1, min_samples - 1)  # Ensure k is at least 1 and less than min_samples
    # sm = SMOTE(random_state=0, k_neighbors=k_neighbors)
    sm = SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors), random_state=0)

    columns = [col for col in df_resample.columns if col != 'T_bins']
    
    X_resampled_full, _ = sm.fit_resample(df_resample[columns], df_resample['T_bins'])
    subsample_df = pd.DataFrame(X_resampled_full, columns=columns)

    X_resampled = subsample_df.filter(like='X_col').values
    T_resampled = subsample_df['T'].values
    Y_resampled = subsample_df['Y'].values
    Y_prob_resampled = subsample_df['Y_prob'].values
    delta_resampled = subsample_df['delta'].values
    sampled_indices = subsample_df['index'].values

    return X_resampled, T_resampled, Y_resampled, Y_prob_resampled, delta_resampled, sampled_indices

def psd_xi(xi, X, Y, T,best_h,XI_opt):
    Hessian_matrix = nd.Hessian(lambda xi: objective(xi, X, Y, T, best_h))(XI_opt)
    eigenvalues = np.linalg.eigvals(Hessian_matrix)
    is_positive_semi_definite = np.all(eigenvalues >= 0)
    return is_positive_semi_definite