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

    print(f'Running PCA on {args.input} and saving results to {args.output}')

    run_pca(args.data, args.n_pcs)

def run_pca(data: str, n_pcs: int, output: str = None):
    ## TODO: add proper documentation
    # Load data and perform preprocessing
    # TODO: add try and catch for filetype not readable by anndata (and output error message)
    adata = anndata.read(data)
    df = pd.DataFrame(adata.X)
    
    ## TODO: add preprocessing based on user input

    # Run PCA calculation
    pca = pca_calculation(df, n_pcs)

    # Save results to .csv
    if output is None:
        base = os.path.splitext(data)
        output = f'{base}_pca.csv'
    pca.to_csv(output)
    print(f'PCA results saved to {output}')

def pca_calculation(data: pd.DataFrame, n_pcs: int):
    ## TODO: add proper documentation

    # center datapoints / normalize data:
    for col in data.columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    # TODO: check mean_subtracted implementation (justify):
    for col in data.columns:
        data[col] = data[col] - data[col].mean()

    # calculate covariance matrix:
    cov_matrix = data.cov() # TODO: check implementation versus using np.dot and vectorizing (speed and memory considerations)

    # calculate eigenvectors and eigenvalues:
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # sort eigenvectors by eigenvalues:
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # select top n_pcs eigenvectors, using user input:
    pcs = eigenvectors[:,: n_pcs]

    # project data onto eigenvectors and give as output:
    pca = np.dot(data.values, pcs)

    # the commented code below is a placeholder
    # pca = None
    # principalComponents = pca.fit_transform(data)
    # pcsDF = pd.DataFrame(data = principalComponents)
    return pca