"""
Visualization module.
Contains functions for generating various plots and visualizations, including heatmaps,
ROC curves, distribution plots, and 3D surface plots.
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

def heatmap_with_dots(X_test, XI_opt, T_test, BETA_opt, Y_test_class, X_train, T_train, Y_train, h, f):
    """Plot a heatmap with overlaid data points for classification."""

    coord1 = np.dot(X_test, BETA_opt)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    Z_flat_f = np.array([f(arg1, arg2,X_train, XI_opt, T_train, Y_train, BETA_opt, h) for arg1, arg2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    
    # Plotting the heatmap
    c2 = axs.imshow(Z_f, extent=[coord1.min(), coord1.max(), coord2.min(), coord2.max()], origin='lower', aspect='auto', cmap='inferno')
    fig.colorbar(c2, ax=axs)
    
    # Adding the contour where function value is 0
    cs = axs.contour(ARG1, ARG2, Z_f, levels=[0], colors='white', linestyles='dashed')
    axs.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    
    # Compute the coordinates
    coord1 = np.dot(X_test, BETA_opt)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    
    # Filter and plot dots based on Y_test_class
    mask_class0 = Y_test_class == 0
    mask_class1 = Y_test_class == 1
    
    # Plot dots for class 0
    axs.scatter(coord1[mask_class0], coord2[mask_class0], c='red', label='Class 0', s=50, edgecolors='black')
    
    # Plot dots for class 1
    axs.scatter(coord1[mask_class1], coord2[mask_class1], c='blue', label='Class 1', s=50, edgecolors='black')
    
    # axs.set_title('Heatmap for f with Dots')
    axs.set_xlabel(r'$X^T \beta$',fontsize = 20)
    axs.set_ylabel(r'$X^T \xi - \tau$', fontsize = 20)
    
    # Show legend
    axs.legend(loc='upper right', fontsize = 12)
    
    plt.show()

def slider_plot(X_test_selected_final, XI_opt, T_test, BETA_opt, Y_test_class, X_train_selected_final, T_train, Y_train, best_h,f):
    """
    Display an interactive slider heatmap plot for visualizing model predictions.
    """
    score_1_test = np.dot(X_test_selected_final,BETA_opt)
    score_2_test = np.dot(X_test_selected_final, XI_opt) 
    Y_score_test = np.squeeze([f(a1,a2, X_train_selected_final, XI_opt, T_train, Y_train, BETA_opt, best_h) for a1,a2 in zip(score_1_test,(score_2_test - T_test))]) # this is equiv to Y_pred
    # Assuming you have your data loaded
    coord1 = score_2_test
    coord2 = T_test
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    def heatmap_data(score_1_fixed_value):
        Z = np.array([f(score_1_fixed_value, arg1 - arg2,X_train_selected_final, XI_opt, T_train, Y_train, BETA_opt, best_h) for arg1, arg2 in zip(ARG1.ravel(), ARG2.ravel())])
        return Z.reshape(ARG1.shape)

    all_heatmaps = [heatmap_data(score_1_value) for score_1_value in np.arange(score_1_test.min(), score_1_test.max(), 1.0)]
    global_min = np.min(all_heatmaps)
    global_max = np.max(all_heatmaps)
    heatmap = heatmap_data(np.mean(score_1_test))
    fig = go.Figure(data=go.Heatmap(z=heatmap, x=arg1_range, y=arg2_range, colorscale="Inferno", zmin=global_min, zmax=global_max))
    # Add slider (remains unchanged from previous code)
    steps = []
    for score_1_value in np.arange(score_1_test.min(), score_1_test.max(), 1.0):
        step = dict(
            args=[{"z": [heatmap_data(score_1_value)], "colorscale": "Inferno", "zmin": global_min, "zmax": global_max}],
            label=str(round(score_1_value, 2)),
            method="restyle"
        )
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Score 1 Value:"},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(sliders=sliders)
    fig.show()

def plot_dist_comparison(A, B, col_names):
    """
    Compare distributions between two datasets with overlaid markers.
    """
    # Convert to pandas DataFrame if they are numpy arrays
    if isinstance(A, np.ndarray):
        A = pd.DataFrame(A, columns=col_names)
    if isinstance(B, np.ndarray):
        B = pd.DataFrame(B, columns=col_names)
    
    # Check if the input dataframes have the specified columns
    for col in col_names:
        if col not in A.columns or col not in B.columns:
            raise ValueError(f"Column {col} not found in the input dataframes.")
    
    # Determine the number of rows based on the length of col_names
    num_rows = int(np.ceil(len(col_names) / 4.0))

    # Plotting
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(24, num_rows * 4))
    
    # Make sure axes is always a 2D array
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    
    for i, column in enumerate(col_names):
        row_idx, col_idx = divmod(i, 4)
        ax = axes[row_idx, col_idx]
        
        A[column].hist(ax=ax, bins=30, alpha=0.75, label='A', color='blue')  # Increased bins to 30
        
        for val in B[column]:
            ax.axvline(x=val, color='red', alpha=0.5)
        
        ax.set_title(f'Distribution of {column} with overlay from B')
        ax.legend()

    # Hide any unused subplot axes
    for j in range(i+1, num_rows*4):
        row_idx, col_idx = divmod(j, 4)
        axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.show()

def plot_scores_dist(score_1,score_2,T,labels):
    """
    Plot scatter distributions of scores and treatments, color-coded by class labels.
    """
    mask_0 = labels == 0
    mask_1 = labels == 1
    fig, ax = plt.subplots()
    plt.scatter(score_2[mask_0], T[mask_0], c=score_1[mask_0], marker='x', cmap='viridis', label='death')
    plt.scatter(score_2[mask_1], T[mask_1], c=score_1[mask_1], marker='o', cmap='viridis', label='survive')
    ax.set_xlabel("score_2")
    ax.set_ylabel("initiation")
    cbar = plt.colorbar()
    cbar.set_label('score_1')
    plt.legend()
    plt.show()

def surface_3d_plot(ARG1, ARG2, Z_f0, Z_f):
    """
    Render a 3D surface plot comparing ground truth and estimand surfaces.
    """
    # Plot both surfaces
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', frame_on=False)

    # Adjusting the gridlines, panes, and background
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.xaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1.0))
    ax.yaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1.0))
    ax.zaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1.0))
    
    ax.grid(True, color='k', linestyle=':', linewidth=0.5)

    # Plotting the surfaces with new colormaps
    f0_surface = ax.plot_surface(ARG1, ARG2, Z_f0, cmap='Blues', edgecolor='none', alpha=0.7)
    f_surface = ax.plot_surface(ARG1, ARG2, Z_f, cmap='Reds', edgecolor='none', alpha=0.6)

    # Creating proxies for the legend
    f0_proxy = mlines.Line2D([], [], color='dodgerblue', linewidth=2, label='Ground Truth')
    f_proxy = mlines.Line2D([], [], color='salmon', linewidth=2, label='Estimand')

    # Add the legend to the plot
    ax.legend(handles=[f0_proxy, f_proxy], loc='upper right')

    # Labeling axes
    ax.set_xlabel(r'$X^T \beta$', labelpad=10)
    ax.set_ylabel(r'$X^T \xi - \tau$', labelpad=10)
    ax.set_zlabel(r'$f(X^T \beta, X^T \xi - \tau)$', labelpad=10)
    # ax.set_title("Fancy 3D Surface Plot", pad=20)

    plt.show()

def surface_heatmap_plot(ARG1, ARG2, Z_f0, Z_f):
    """
    Display side-by-side heatmaps for ground truth and estimand surfaces.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate global min and max
    vmin_val = min(Z_f0.min(), Z_f.min())
    vmax_val = max(Z_f0.max(), Z_f.max())

    # Heatmap for Z_f0
    c1 = axs[0].imshow(Z_f0, extent=[ARG1.min(), ARG1.max(), ARG2.min(), ARG2.max()], origin='lower', aspect='auto', cmap='viridis', interpolation='bicubic', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(c1, ax=axs[0])
    axs[0].set_title('Heatmap for Ground Truth')
    axs[0].set_xlabel(r'$X^T \beta$')
    axs[0].set_ylabel(r'$X^T \xi - \tau$')

    # Heatmap for Z_f
    c2 = axs[1].imshow(Z_f, extent=[ARG1.min(), ARG1.max(), ARG2.min(), ARG2.max()], origin='lower', aspect='auto', cmap='viridis', interpolation='bicubic', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(c2, ax=axs[1])
    axs[1].set_title(r'Heatmap for Estimand')
    axs[1].set_xlabel(r'$X^T \beta$')
    axs[1].set_ylabel(r'$X^T \xi - \tau$')
    
    plt.tight_layout()
    plt.show()

def surface_3d_plot_single(f, X_test, XI_opt, T_test, BETA_opt):
    """
    Render a 3D surface plot for a single function.
    """
    coord1 = np.dot(BETA_opt, X_test.T)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    # Evaluate f on the grid
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(ARG1, ARG2, Z_f, cmap='viridis', edgecolor='none')
    fig.colorbar(surf)
    
    ax.set_title('3D Surface Plot for f')
    ax.set_xlabel(r'$X^T \beta$')
    ax.set_ylabel(r'$X^T \xi - \tau$')
    ax.set_zlabel('f Value')
    
    plt.show()

def surface_heatmap_plot_single(f, X_test, XI_opt, T_test, BETA_opt):
    """
    Display a heatmap for a single function based on kernel smoothing.
    """
    coord1 = np.dot(BETA_opt, X_test.T)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    
    c2 = axs.imshow(Z_f, extent=[coord1.min(), coord1.max(), coord2.min(), coord2.max()], origin='lower', aspect='auto', cmap='viridis')
    fig.colorbar(c2, ax=axs)
    
    axs.set_title('Heatmap for f')
    axs.set_xlabel(r'$X^T \beta$')
    axs.set_ylabel(r'$X^T \xi - \tau$')
    
    plt.show()

def plot_roc(Y_test_class, Y_pred_class):
    """
    Plot the ROC curve with AUC annotation.
    """
    fpr, tpr, threshold = roc_curve(Y_test_class, Y_pred_class)
    roc_auc = auc(fpr, tpr)
    print(f"AUC for our classifier is: {roc_auc}")
    # Plotting the ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def show_dist(data):
    """
    Display a histogram of the data distribution.
    """
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Distribution of the numpy array')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def show_dist_matrix(X, column_names=None):
    """
    Plot histograms for each column of a matrix with optional titles.
    """
    num_cols = X.shape[1]
    
    # Define the number of rows and columns for your subplots
    nrows = int(np.ceil(np.sqrt(num_cols)))
    ncols = int(np.ceil(num_cols / nrows))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()  # Flatten axes for easier indexing
    
    for i in range(num_cols):
        axes[i].hist(X[:, i], bins=30, density=True, alpha=0.6, color='g')
        
        # If column names are provided, use them as titles for the subplots
        if column_names is not None:
            axes[i].set_title(column_names[i])
        else:
            axes[i].set_title(f"Column {i+1}")
        
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    # Remove any remaining empty subplots
    for i in range(num_cols, nrows*ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def show_dist_comparison(X_original, X_imputed, column_names=None):
    """
    Compare distributions of original and imputed data using overlaid histograms.
    """
    num_cols = X_original.shape[1]
    
    # Define the number of rows and columns for your subplots
    nrows = int(np.ceil(np.sqrt(num_cols)))
    ncols = int(np.ceil(num_cols / nrows))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()  # Flatten axes for easier indexing
    
    for i in range(num_cols):
        # Plot the original data in green
        axes[i].hist(X_original[:, i], bins=30, density=True, alpha=0.5, color='g', label='Original')
        # Overlay the imputed data in red
        axes[i].hist(X_imputed[:, i], bins=30, density=True, alpha=0.5, color='r', label='Imputed')
        
        # If column names are provided, use them as titles for the subplots
        if column_names is not None:
            axes[i].set_title(column_names[i])
        else:
            axes[i].set_title(f"Column {i+1}")
        
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    # Remove any remaining empty subplots
    for i in range(num_cols, nrows*ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()