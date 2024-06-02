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
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to input data file, expects matrix-like', metavar="data")
    parser.add_argument('-n', '--n_pcs', type=int, required=True, help='Number of principal components calculated', metavar="n_pcs")

    parser.add_argument('-s', '--standardize', action='store_true', help='Standardize the input data before running PCA')
    parser.add_argument('-o', '--output', type=str, help='Output file name', metavar="output")
    parser.add_argument('-d', '--output-dir', type=str, help='Path to output file, .csv format default', metavar="outdir")
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the PCA results')

    args = parser.parse_args()

    print(f'Running PCA on {args.data} with {args.n_pcs} principal components.')
    # print running options if they are set
    for arg in vars(args):
        if getattr(args, arg) is not None:
            print(f'Option {arg}: {getattr(args, arg)}')

    print('Running PCA! Please wait.')

    run_pca(args.data, args.n_pcs)

def run_pca(data: str, n_pcs: int, output: str = "pca_results.csv", outdir: str = None):
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
    
    ## TODO: add preprocessing based on user input

    # Run PCA calculation
    pca = pca_calculation(df, n_pcs)

    # Save results to .csv in main directory or user-specified location
    if outdir is None:
        outdir = os.path.join(os.getcwd(), output)
    try:
        pca.to_csv(outdir)
    except AttributeError:
        pca = pd.DataFrame(data=pca)
        pca.to_csv(outdir)
    except: 
        print('Error saving PCA results to .csv')

    print(f'PCA results saved to {outdir}')
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
    
    # TODO: setting values through chained assignment warning !
    # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy !
    # behavior will change in pandas 3.0 !
    for col in data.columns:
        avg = data[col].fillna(0).mean()
        std = data[col].fillna(0).std()
        # fix chained assignment by using .loc accessor (check)
        for i in range(len(data[col])):
            if pd.isna(data[col][i]):
                data.loc[i, col] = 0
            else:
                data.loc[i, col] = (data[col][i] - avg) / std

    # TODO: check mean_subtracted implementation (justify):
    # for col in data.columns:
    #     data[col] = data[col] - data.mean(axis=0)

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