import numpy as np
import pandas as pd

# use a poisson distribution to generate a high-dimensional matrix
def generate_poisson_matrix(dimensions, lambd):
    shape = tuple(dimensions)
    matrix = np.random.poisson(lambd, shape)
    return matrix

# # Example usage
# dimensions = (10, 10, 10)  # dimensions
# lambd = 5  # lambda parameter
# matrix = generate_poisson_matrix(dimensions, lambd)

# two dimensional example! OK to reduce using principal components..
dimensions = (1000, 1000)
lambd = 5
matrix = generate_poisson_matrix(dimensions, lambd)

# output to csv using pandas
# np.save('example-files/poisson_matrix.npy', matrix)
df = pd.DataFrame(matrix)
df.to_csv('example-files/poisson_matrix.csv', index=False)