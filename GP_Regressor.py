# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:39:19 2024

@author: gh00616

"""

from pyomo.environ import exp, sqrt
import numpy as np
from numpy.linalg import inv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct

# def GPR(x, y, x_init, kernel_type):
#     if kernel_type == 'RBF':
#         kernel = ConstantKernel() * (RBF(1.0, (1e-4, 1e4)) + WhiteKernel(noise_level=1e-5))
#     elif kernel_type == 'matern12':
#         kernel = ConstantKernel() * (Matern(length_scale=1.0, nu=1.5) +  WhiteKernel(noise_level=1e-5))
#     elif kernel_type == 'matern52':
#         kernel = ConstantKernel() * (Matern(length_scale=1.0, nu=2.5) +  WhiteKernel(noise_level=1e-5))
#     elif kernel_type == 'matern72':
#         kernel = ConstantKernel() * (Matern(length_scale=1.0, nu=3.5) +  WhiteKernel(noise_level=1e-5))
#     else:
#         raise('Assigned kernel function is not supported!')
        
def GPR(x, y, x_init, kernel_type):
    if kernel_type == 'RBF':
        kernel = ConstantKernel() * (RBF(1.0, (1e-4, 1e2)) + WhiteKernel(noise_level=1e-15, noise_level_bounds=(1e-15, 1e-1)))
    elif kernel_type == 'matern12':
        kernel = ConstantKernel() * (Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=1e-15, noise_level_bounds=(1e-15, 1e-1)))
    elif kernel_type == 'matern52':
        kernel = ConstantKernel() * (Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-15, noise_level_bounds=(1e-15, 1e-1)))
    elif kernel_type == 'matern72':
        kernel = ConstantKernel() * (Matern(length_scale=1.0, nu=3.5) + WhiteKernel(noise_level=1e-15, noise_level_bounds=(1e-15, 1e-1)))
    else:
        raise ValueError("Assigned kernel function is not supported!")
    
    gp = GaussianProcessRegressor(
        alpha=1e-10,
        copy_X_train=True,
        kernel=kernel,
        n_restarts_optimizer=20, # More restarts for better hyperparameter tuning
        normalize_y=True, # Helps numerical stability
        optimizer='fmin_l_bfgs_b',
        random_state=None
    )

    gp.fit(x, y)

    sigma = max(sqrt(gp.kernel_.get_params()['k1'].constant_value), 1e-5)  # Avoid sqrt(0)
    length_scale = gp.kernel_.get_params().get('k2__k1__length_scale', 1.0)

    m, std = gp.predict(x_init, return_std=True)
    m, covM = gp.predict(x_init, return_cov=True)
    
    # Add jitter to covariance matrix for numerical stability
    covM += np.eye(len(covM)) * 1e-6
    
    M3 = [y[j] - m[j] for j in range(len(y))]

    return sigma, length_scale, m, covM, M3

def generate_pyomo_gp_expression(x_train, y_train, x_unknown, kernel_type):
    
    # Train the GP model and get hyperparameters and covariance
    sigma, length_scale, m, covM, M = GPR(x_train, y_train, x_train, kernel_type)
    
    def squared_exponential_kernel(x_test, x_train_point, l, sigma):
        """Squared-Exponential (RBF) Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(0, len(x_test))]
        squared_dist = sum(d**2 for d in diff)  # Pyomo-compatible squared distance
        return sigma**2 * exp(-squared_dist / (2 * l**2))
    
    def matern12_kernel(x_test, x_train_point, l, sigma):
        """Matérn-1/2 Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(len(x_test))]
        safe_distance = sqrt(sum(d**2 for d in diff) + 1e-10)  # Prevents sqrt(0)
        return sigma**2 * exp(-safe_distance / l)
    
    def matern52_kernel(x_test, x_train_point, l, sigma):
        """Matérn-5/2 Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(len(x_test))]
        safe_distance = sqrt(sum(d**2 for d in diff) + 1e-10)
        term1 = 1 + sqrt(5) * safe_distance / l + (5 / 3) * (safe_distance**2 / l**2)
        term2 = exp(-sqrt(5) * safe_distance / l)
        return sigma**2 * term1 * term2


    def matern72_kernel(x_test, x_train_point, l, sigma):
        """Matérn-7/2 Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(len(x_test))]
        euclidean_dist = sqrt(sum(d**2 for d in diff))
        term1 = 1 + sqrt(7) * euclidean_dist / l + (14 / 5) * (euclidean_dist**2 / l**2) + (7 / 15) * (euclidean_dist**3 / l**3)
        term2 = exp(-sqrt(7) * euclidean_dist / l)
        return sigma**2 * term1 * term2

    # Create the kernel matrix for the unknown input
    kernel_mat = []
    
    if kernel_type == 'RBF':
        for k in range(0, len(x_train)):
            kernel = squared_exponential_kernel(x_unknown, x_train[k], length_scale, sigma)
            kernel_mat.append(kernel)
    elif kernel_type == 'matern12':
        for k in range(0, len(x_train)):
            kernel = matern12_kernel(x_unknown, x_train[k], length_scale, sigma)
            kernel_mat.append(kernel)
    elif kernel_type == 'matern52':
        for k in range(0, len(x_train)):
            kernel = matern52_kernel(x_unknown, x_train[k], length_scale, sigma)
            kernel_mat.append(kernel)
    elif kernel_type == 'matern72':
        for k in range(0, len(x_train)):
            kernel = matern72_kernel(x_unknown, x_train[k], length_scale, sigma)
            kernel_mat.append(kernel)

    # Pyomo-compatible matrices
    M1 = kernel_mat  # Pyomo symbolic expressions
    M2 = covM  # Numerical matrix from the GP
    M3 = M  # Numerical residuals

    # Construct the final Pyomo-compatible expression
    y1 = sum(
        M1[i] * sum(M2[i, j] * M3[j] for j in range(len(M3))) for i in range(len(M1))
    )
    known_matrix = M2@M3
    return y1, kernel_matrix, known_matrix, sigma, length_scale

def kernel_matrix(x_train, y_train, x_unknown, kernel_type, sigma, l):
    
    def squared_exponential_kernel(x_test, x_train_point, l, sigma):
        """Squared-Exponential (RBF) Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(0, len(x_test))]
        squared_dist = sum(d**2 for d in diff)  # Pyomo-compatible squared distance
        return sigma**2 * exp(-squared_dist / (2 * l**2))
    
    def matern12_kernel(x_test, x_train_point, l, sigma):
        """Matérn-1/2 Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(len(x_test))]
        safe_distance = sqrt(sum(d**2 for d in diff) + 1e-10)  # Prevents sqrt(0)
        return sigma**2 * exp(-safe_distance / l)

    
    def matern52_kernel(x_test, x_train_point, l, sigma):
        """Matérn-5/2 Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(len(x_test))]
        safe_distance = sqrt(sum(d**2 for d in diff) + 1e-10)
        term1 = 1 + sqrt(5) * safe_distance / l + (5 / 3) * (safe_distance**2 / l**2)
        term2 = exp(-sqrt(5) * safe_distance / l)
        return sigma**2 * term1 * term2

    def matern72_kernel(x_test, x_train_point, l, sigma):
        """Matérn-7/2 Kernel with Pyomo compatibility."""
        diff = [x_test[j] - x_train_point[j] for j in range(len(x_test))]
        safe_distance = sqrt(sum(d**2 for d in diff) + 1e-10)
        term1 = 1 + sqrt(7) * safe_distance / l + (14 / 5) * (safe_distance**2 / l**2) + (7 / 15) * (safe_distance**3 / l**3)
        term2 = exp(-sqrt(7) * safe_distance / l)
        return sigma**2 * term1 * term2


    # Create the kernel matrix for the unknown input
    UKM = []
    
    if kernel_type == 'RBF':
        for k in range(0, len(x_train)):
            kernel = squared_exponential_kernel(x_unknown, x_train[k], l, sigma)
            UKM.append(kernel)
    elif kernel_type == 'matern12':
        for k in range(0, len(x_train)):
            kernel = matern12_kernel(x_unknown, x_train[k], l, sigma)
            UKM.append(kernel)
    elif kernel_type == 'matern52':
        for k in range(0, len(x_train)):
            kernel = matern52_kernel(x_unknown, x_train[k], l, sigma)
            UKM.append(kernel)
    elif kernel_type == 'matern72':
        for k in range(0, len(x_train)):
            kernel = matern72_kernel(x_unknown, x_train[k], l, sigma)
            UKM.append(kernel)
    
    return UKM

