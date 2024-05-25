import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

# generating a poisson distribution anndata object (mimicks sparse count information)
counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
adata = ad.AnnData(counts)

# save to h5ad file
adata.write('example-files/poisson.h5ad', compression='gzip')