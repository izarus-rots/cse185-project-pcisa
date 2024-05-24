import numpy as np
import pandas as pd

# use a poisson distribution to generate a high-dimensional matrix
def generate_poisson_matrix(dimensions, lambd):
    shape = tuple(dimensions)
    matrix = np.random.poisson(lambd, shape)
    return matrix

# Example usage
dimensions = (10, 10, 10)  # dimensions
lambd = 5  # lambda parameter
matrix = generate_poisson_matrix(dimensions, lambd)
# print(matrix)

# output to csv using pandas
np.save('example-files/poisson_matrix.npy', matrix)