# MIT License
#
# Copyright (c) 2025 Nicolaj Bösel-Schmid
# Contact: nicolaj.schmid@epfl.ch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def matrix_inv(
    matrix:np.ndarray,
):
    """
    Calculate the inverse of a flattened matrix.
    Args:
        matrix (np.ndarray): Flattened matrix (N*N,).
    Returns:
        np.ndarray: Flattened inverted matrix (N*N,).
    """
    N = int(np.sqrt(matrix.size))
    return np.linalg.inv(matrix.reshape(N, N)).flatten()

def matrix_det(
    matrix:np.ndarray,
):
    """
    Calculate the determinant of a flattened matrix.
    Args:
        matrix (np.ndarray): Flattened matrix (N*N,).
    Returns:
        np.ndarray: Flattened inverted matrix (N*N,).
    """
    N = int(np.sqrt(matrix.size))
    return np.linalg.det(matrix.reshape(N, N)).flatten()

def gaussian_1D(
    x:np.ndarray,
    mu:float,
    sigma:float,
    normalize:bool=False,
):
    """
    Calculate the Gaussian function.
    Args:
        x (np.ndarray): Input values.
        mu (float): Mean.
        sigma (float): Standard deviation.
        normalize (bool): If false, return the normal distribution. If true, normalize the
                            output to [0,1], i.e. do not scale Gaussian by the std.
    Returns:
        np.ndarray: Gaussian function.
    """
    exp = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    if normalize:
        return exp
    return exp / (sigma * np.sqrt(2 * np.pi))

def gaussian(
    X:np.ndarray,
    mu:np.ndarray,
    sigma:np.ndarray,
    normalize:bool=False,
):
    """
    Calculate the Gaussian function.
    Args:
        x (np.ndarray): Input values (n_samples, n_dims).
        mu (np.ndarray): Mean (n_dims,) or (n_samples, n_dims).
        sigma (np.ndarray): Covariance matrix (n_dims,n_dims) or (n_samples,n_dims,n_dims).
        normalize (bool): If false, return the normal distribution. If true, normalize the 
                            output to [0,1], i.e. do not scale Gaussian by the determinant.
    Returns:
        np.ndarray: Gaussian function (n_samples,).
    """
    if X.shape[0] == 0:
        return np.array([])
    
    X = X - mu

    if sigma.ndim == 2:
        exp = np.exp(-0.5 * np.sum(X * (np.linalg.inv(sigma) @ X.T).T, axis=1)) 

        if normalize:
            return exp
        return exp / np.sqrt(np.linalg.det(sigma) * (2*np.pi)**X.shape[1])
    
    if sigma.ndim == 3:
        sigma_inv = np.apply_along_axis(
            func1d=matrix_inv, 
            axis=1, 
            arr=sigma.reshape(X.shape[0], X.shape[1]**2),
        ).reshape(X.shape[0], X.shape[1], X.shape[1]) # (n_samples, n_dims, n_dims)
        exp = np.exp(-0.5 * np.einsum('ni,nij,nj->n', X, sigma_inv, X)) 

        if normalize:
            return exp

        sigma_det = np.apply_along_axis(
            func1d=matrix_det, 
            axis=1, 
            arr=sigma.reshape(X.shape[0], X.shape[1]**2),
        ).reshape(X.shape[0]) # (n_samples,)
        return exp / np.sqrt(sigma_det * (2*np.pi)**X.shape[1])
    
    raise ValueError("Invalid sigma shape.")