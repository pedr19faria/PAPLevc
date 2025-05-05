#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod 
        
    
from pyqubo import Array

import pprint

import neal

import itertools

from joblib import Parallel, delayed


# ### Precision Vectors
# 
# * Binary
# * Linearly Spaced
# * Constant
# * Uniform
# * Normal

# In[2]:


def create_uni_P(K, dm1):
    p = np.random.uniform(low = 0, high = 0.5, size = (1,K))
    I = np.eye(dm1)
    P = np.kron(I, p)
    return P
def create_normal_P(K, dm1):
    p = np.random.normal(loc = 0, scale = 0.5, size = (1,K))
    p2 = np.abs(p) 
    I = np.eye(dm1)
    P = np.kron(I, p2)
    return P
def create_linspace_P(K, dm1):
    p = np.linspace(0.1, 0.75 , K)
    I = np.eye(dm1)
    P = np.kron(I, p)
    return P
def create_binary_P(K, dm1):
    p = np.array([2**(-(i+1)) for i in range(K)])
    I = np.eye(dm1)
    P = np.kron(I,p)
    return P
def create_constant_P(K, dm1):
    c = 1/K
    p = np.array([c for i in range(K)])
    I = np.eye(dm1)
    P = np.kron(I,p)
    return P    
    


# ### Data generation


def generate_linear_regression_data(num_samples, num_features, w_range = (0, 1),x_range = (0,10), noise_std = 0.):
    """
    Generates random data for a linear regression problem.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of features (excluding bias term).
        w_range (tuple): Range for randomly generating the true coefficients w.
        noise_std (float): Standard deviation of Gaussian noise added to Y.
    
    Returns:
        X (numpy.ndarray): Feature matrix of shape (num_samples, num_features).
        Y (numpy.ndarray): Target values of shape (num_samples, 1).
        w_true (numpy.ndarray): True regression coefficients of shape (num_features + 1, 1).
    """
    
    X = np.random.uniform(x_range[0], x_range[1], size=(num_samples, num_features -1))
    
    # Add bias column (intercept)
    X_bias = np.hstack((np.ones((num_samples, 1)), X))  # Shape (num_samples, num_features + 1)

    # Generate true weights including bias term
    w_true = np.random.uniform(w_range[0], w_range[1], size=(num_features, 1))
    ### tirei o arrendodamento, para que em geral haja um erro no coeficiente

    # Generate noise
    eps_noise = np.random.normal(0, noise_std, size=(num_samples, 1))

    # Compute target values Y
    Y = np.dot(X_bias, w_true) + eps_noise

    return X, Y, w_true


# ### Qubo Exact Solver

# In[4]:


def qubo_binary_linear_regression(X, Y, P): 
    """
    """
    (N, d) = X.shape
    X = np.hstack((np.ones((N, 1)), X))
    (N, dm1) = X.shape
    
    S = X @ P  # S -> dim(N, K(d+1))
    (N, Kd) = S.shape

    A = S.T @ S  # A -> dim(K(d+1), K(d+1))
    b = -2 * (S.T @ Y).ravel()  # b -> dim(K(d+1),1) flattened
    Q = A + np.diag(b)  # QUBO matrix

    # **1. Construct BQM directly (without symbolic compilation)**
    bqm = dimod.BinaryQuadraticModel('BINARY')
    for i in range(Kd):
        bqm.add_linear(i, Q[i, i])  # Add linear terms
        for j in range(i + 1, Kd):
            bqm.add_quadratic(i, j, 2*Q[i, j])  # Add quadratic terms
    
    # **3. Exact Solver**
    sampler_exact = dimod.ExactSolver()
    sampleset_exact = sampler_exact.sample(bqm)
    best_sample = sampleset_exact.first.sample  # Extract best binary solution
    what = np.array([best_sample[i] for i in range(Kd)]).reshape(Kd, 1)
    whatc = P @ what  # Regression coefficients

    return whatc,X,Y


# ### Linear Regression Solver

# In[5]:


def lin_reg_sol(data_X, data_Y):
    (N, d) = data_X.shape
    ##np.concatenate
    X = np.hstack((np.ones((N, 1)), data_X))
    Y = data_Y
    XTX = np.dot(X.T,X)
    inv = np.linalg.inv(XTX)
    invXt = np.dot(inv, X.T)
    w = np.dot(invXt,Y)
    return w, X, Y


# ### Error Metrics

# In[6]:


def compute_r2(X, y_true, w):
    """
    Compute the R-squared (coefficient of determination).
    
    Parameters:
        y_true (numpy array): True target values.
        y_pred (numpy array): Predicted target values.
    
    Returns:
        float: R-squared value.
    """
    y_pred = np.dot(X,w)
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0  # Avoid division by zero

def MSE(X, Y, w):### 1/N((Xw - Y)^2)

    yhat = np.dot(X,w)
    res = yhat - Y
    mse_e = np.power(res,2).mean()

    return mse_e

#def error(w,w_true): ### ||Xw - Y||/N(^(1/2))
    


# ### Generating the combinations

# In[7]:


def comb_with_data(data_gen_lr, combinations, w_range = (0,1), x_range = (0,10)):
    for i in range(len(combinations)):
        (feature, sample, K,noise, trial) = combinations[i]
        X, Y,w = data_gen_lr(sample, feature, w_range = (w_range[0], w_range[1]), x_range = (x_range[0], x_range[1]), noise_std = noise)
        
        combinations[i] = combinations[i] + (X, Y,w,)
    return combinations


# ### Running the Experiment

# In[8]:


def run_experiment_uniform(dm1, N, K, noise,trial,X,Y,w_true):
    P = create_uni_P(K, dm1)
    w_found, X_aug, Y = qubo_binary_linear_regression(X, Y, P)
    r2 = compute_r2(X_aug,Y,w_found)
    mse = MSE(X_aug,Y, w_found)
    w_er = np.dot(X_aug,w_found) - np.dot(X_aug,w_true)
    
        
    vary = np.var(Y)
    varw = np.var(w_er)
    snr = 10*np.log10(vary/varw)
    return {
        "n_samples": N,
        "n_features": dm1,
        "K": K,
        "noise": noise,
        "trial": trial,
        "mse": mse,
        "r2": r2,
        "snr": snr
    }
def run_experiment_normal(dm1, N, K, noise,trial, X, Y, w_true):
    P = create_normal_P(K, dm1)
    
    w_found, X_aug, Y = qubo_binary_linear_regression(X, Y, P)
    r2 = compute_r2(X_aug,Y,w_found)
    mse = MSE(X_aug,Y, w_found)
    w_er = np.dot(X_aug,w_found) - np.dot(X_aug,w_true)
    
    vary = np.var(Y)
    varw = np.var(w_er)
    snr = 10*np.log10(vary/varw)
    return {
        "n_samples": N,
        "n_features": dm1,
        "K": K,
        "noise": noise,
        "trial": trial,
        "mse": mse,
        "r2": r2,
        "snr": snr
    }
def run_experiment_linspace(dm1, N, K, noise,trial, X, Y, w_true):
    P = create_linspace_P(K, dm1)
    
    w_found, X_aug, Y = qubo_binary_linear_regression(X, Y, P)
    r2 = compute_r2(X_aug,Y,w_found)
    mse = MSE(X_aug,Y, w_found)
    w_er = np.dot(X_aug,w_found) - np.dot(X_aug,w_true)
        
    vary = np.var(Y)
    varw = np.var(w_er)
    snr = 10*np.log10(vary/varw)
    return {
        "n_samples": N,
        "n_features": dm1,
        "K": K,
        "noise": noise,
        "trial": trial,
        "mse": mse,
        "r2": r2,
        "snr": snr
    }
def run_experiment_binary(dm1, N, K, noise,trial, X, Y, w_true):
    P = create_binary_P(K, dm1)
    w_found, X_aug, Y = qubo_binary_linear_regression(X, Y, P)
    r2 = compute_r2(X_aug,Y,w_found)
    mse = MSE(X_aug,Y, w_found)
    w_er = np.dot(X_aug,w_found) - np.dot(X_aug,w_true)
        
    vary = np.var(Y)
    varw = np.var(w_er)
    snr = 10*np.log10(vary/varw)
    return {
        "n_samples": N,
        "n_features": dm1,
        "K": K,
        "noise": noise,
        "trial": trial,
        "mse": mse,
        "r2": r2,
        "snr": snr
    }
def run_experiment_constant(dm1, N, K, noise,trial, X, Y, w_true):
    P = create_constant_P(K, dm1)
    w_found, X_aug, Y = qubo_binary_linear_regression(X, Y, P)
    r2 = compute_r2(X_aug,Y,w_found)
    mse = MSE(X_aug,Y, w_found)
    w_er = np.dot(X_aug,w_found) - np.dot(X_aug,w_true)
    vary = np.var(Y)
    varw = np.var(w_er)
    snr = 10*np.log10(vary/varw)
    return {
        "n_samples": N,
        "n_features": dm1,
        "K": K,
        "noise": noise,
        "trial": trial,
        "mse": mse,
        "r2": r2,
        "snr": snr
    }
def linear_regression(dm1, N, K, noise, trial, X, Y, w_true):
    w_found, X_aug, Y = lin_reg_sol(X, Y)
    r2 = compute_r2(X_aug,Y,w_found)
    mse = MSE(X_aug,Y, w_found)
    w_er = np.dot(X_aug,w_found) - np.dot(X_aug,w_true)
        
    vary = np.var(Y)
    varw = np.var(w_er)
    snr = 10*np.log10(vary/varw)
    return {
        "n_samples": N,
        "n_features": dm1,
        "K": K,
        "noise": noise,
        "trial": trial,
        "mse": mse,
        "r2": r2,
        "snr":snr
    }


# ### Getting Averages


import numpy as np
import pandas as pd

def get_av(dataset, error1='r2', error2='mse', error3='snr'):
    df = pd.DataFrame(dataset)
    param_cols = ['K', 'n_samples', 'noise', 'n_features']
    grouped = df.groupby(param_cols)

    result_list = []

    for params, group in grouped:
        err1_mean = group[error1].mean()
        err2_mean = group[error2].mean()
        err3_mean= group[error3].mean()
        

        result_dict = dict(zip(param_cols, params))
        result_dict.update({
            error1: err1_mean, #r2
            error2: err2_mean, #,se
            error3: err3_mean, #snr
        })

        result_list.append(result_dict)

    return result_list


def return_mean(result, number_of_trials = 50):
    df = pd.DataFrame(result)
    param_cols = ['K', 'n_samples', 'noise', 'n_features']
    grouped = df.groupby(param_cols, as_index = False).mean()
    df_mean = pd.DataFrame(grouped).drop('trial', axis = 1)
    
    return df_mean


def get_filtered_data(filters,  # e.g., {"K": 2, "noise": 0}
                      df_bin=None, df_const=None, df_linspace=None,
                      df_uniform=None, df_normal=None, df_linreg=None):
    """Filters multiple DataFrames based on a dictionary of column filters."""
    
    def apply_filters(df):
        if df is None:
            return None
        cond = pd.Series([True] * len(df))
        for col, val in filters.items():
            cond &= (df[col] == val)
        return df[cond]

    filtered_data = {
        "binary": apply_filters(df_bin),
        "constant": apply_filters(df_const),
        "linspace": apply_filters(df_linspace),
        "uniform": apply_filters(df_uniform),
        "normal": apply_filters(df_normal),
        "linreg": apply_filters(df_linreg),
    }
    
    return filtered_data



