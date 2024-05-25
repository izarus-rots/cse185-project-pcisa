# pcisa/__main__.py

# import argparse
# from .pcisa import run_pca

# def main():
#     parser = argparse.ArgumentParser(description='Principal Component Analysis in Python')
#     parser.add_argument('--data', type=str, required=True, help='Path to input data file, expects matrix-like')
#     parser.add_argument('--n_pcs', type=int, required=True, help='Number of principal components calculated')

#     parser.add_argument('--standardize', action='store_true', help='Standardize the input data before running PCA')
#     parser.add_argument('--output', type=str, help='Path to output file, .csv format default')
#     parser.add_argument('--plot', action='store_true', help='Plot the PCA results')
#     args = parser.parse_args()

#     print(f'Running PCA on {args.input} and saving results to {args.output}')

#     run_pca(args.input, args.output)