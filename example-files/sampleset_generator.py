import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import random

# generating a poisson distribution anndata object (mimicks sparse count information)
def generate_sample(size: tuple, seed: int = 0):
    """
    Generate a random sparse count matrix
    
    Parameters
    ----------
    size : tuple
        Tuple of (n_cells, n_genes) for the size of the count matrix

    seed : int
        Seed for random number generation
        
    Returns
    -------
    anndata.AnnData
        Anndata object with sparse count matrix
    """
    counts = csr_matrix(np.random.poisson(1, size=size), dtype=np.float32)
    adata = ad.AnnData(counts)
    return adata

# generate sample dataset with set seed = 0
sample = generate_sample((100, 2000), 0)
sample.write('example-files/poisson.h5ad', compression='gzip')