# pcisa/pcisa.py

import os

import numpy
import anndata
import pandas

def run_pca(data: str, n_pcs: int, output: str = None):
    ## TODO: add proper documentation
    # Load data and perform preprocessing
    adata = anndata.read(data)
    df = pandas.DataFrame(adata.X)
    
    ## TODO: add preprocessing based on user input

    # Run PCA calculation
    pca = pca_calculation(df, n_pcs)

    # Save results to .csv
    if output is None:
        base = os.path.splitext(data)
        output = f'{base}_pca.csv'
    pca.to_csv(output)
    print(f'PCA results saved to {output}')

def pca_calculation(data: pandas.DataFrame, n_pcs: int):
    ## TODO: add proper documentation
    ## TODO: implement PCA calculation, output "pca" variable in pandas DataFrame format

    # center datapoints:


    # calculate covariance matrix:


    # calculate eigenvectors and eigenvalues:


    # sort eigenvectors by eigenvalues:


    # select top n_pcs eigenvectors:


    # project data onto eigenvectors:


    # return projected data:

    pca = None # placeholder

    principalComponents = pca.fit_transform(data)
    pcsDF = pandas.DataFrame(data = principalComponents)
    return pcsDF