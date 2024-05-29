# pcisa/pcisa.py

import os

import numpy as np
import anndata
import pandas as pd

import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="pcisa",
        description='Principal Component Analysis in Python'
    )
    parser.add_argument('--data', type=str, required=True, help='Path to input data file, expects matrix-like', metavar="data")
    parser.add_argument('--n_pcs', type=int, required=True, help='Number of principal components calculated', metavar="n_pcs")

    parser.add_argument('--standardize', action='store_true', help='Standardize the input data before running PCA')
    parser.add_argument('--output', type=str, help='Path to output file, .csv format default')
    parser.add_argument('--plot', action='store_true', help='Plot the PCA results')

    args = parser.parse_args()

    print(f'Running PCA on {args.data} with {args.n_pcs} principal components.')

    run_pca(args.data, args.n_pcs)

def run_pca(data: str, n_pcs: int, output: str = None):
    """
    Preprocess input and run PCA function
    
    Parameters
    ----------
    data : str
        Path to input data file, expects matrix-like
        
    n_pcs : int
        Number of principal components to calculate
        
    output : str
        Path to output file, .csv format default
        
    Returns
    -------
    None
    """
    # Load data and perform preprocessing
    # TODO: add try and catch for filetype not readable by anndata (and output error message)
    adata = anndata.read_h5ad(data)
    df = pd.DataFrame(data=adata.X)
    # CHECKPOINT TEST 1:
    print('data loaded successfully')
    
    ## TODO: add preprocessing based on user input

    # Run PCA calculation
    pca = pca_calculation(df, n_pcs)

    # Save results to .csv
    if output is None:
        base = os.path.splitext(data)
        output = f'{base}_pca.csv'
    try:
        pca.to_csv(output)
    except AttributeError:
        pca = pd.DataFrame(data=pca)
        pca.to_csv(output)
    except: 
        print('Error saving PCA results to .csv')

    print(f'PCA results saved to {output}')
    return

def pca_calculation(data: pd.DataFrame, n_pcs: int):
    """
    Perform PCA on input data and return the principal components
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data for PCA, expects matrix-like
        
    n_pcs : int
        Number of principal components to calculate
        
    Returns
    -------
    pca : np.array
        Principal components of the input data
    """

    # center datapoints / normalize data:
    for col in data.columns:
        avg = data[col].fillna(0).mean()
        std = data[col].fillna(0).std()
        for i in range(len(data[col])):
            if pd.isna(data[col][i]):
                data[col][i] = 0
            else:
                data[col][i] = (data[col][i] - avg) / std
    # CHECKPOINT TEST 2:
    print('data normalized successfully')

    # TODO: check mean_subtracted implementation (justify):
    # for col in data.columns:
    #     data[col] = data[col] - data.mean(axis=0)

    # calculate covariance matrix:
    cov_matrix = data.cov() # TODO: check implementation versus using np.dot and vectorizing (speed and memory considerations)
    # CHECKPOINT TEST 3:
    print('covariance matrix calculated successfully')

    # calculate eigenvectors and eigenvalues:
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # CHECKPOINT TEST 4:
    print('eigenvectors and eigenvalues calculated successfully')

    # sort eigenvectors by eigenvalues:
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # CHECKPOINT TEST 5:
    print('eigenvectors sorted successfully')

    # select top n_pcs eigenvectors, using user input:
    pcs = eigenvectors[:,: n_pcs]
    # CHECKPOINT TEST 6:
    print('principal components selected successfully')

    # project data onto eigenvectors and give as output:
    pca = np.dot(data.values, pcs)
    # CHECKPOINT TEST 7:
    print('PCA calculated successfully')

    # the commented code below is a placeholder
    # pca = None
    # principalComponents = pca.fit_transform(data)
    # pcsDF = pd.DataFrame(data = principalComponents)
    print(type(pca))
    print(pca)
    return pca