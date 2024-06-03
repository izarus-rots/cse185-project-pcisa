# pcisa/__main__.py

import argparse
import os
from .pcisa import run_pca
from .quickplot import quickplot

def main():
    parser = argparse.ArgumentParser(
        prog="pcisa",
        description='Principal Component Analysis in Python'
    )
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to input data file, expects matrix-like', metavar="file")
    parser.add_argument('-n', '--n_pcs', type=int, required=True, help='Number of principal components calculated', metavar="n_pcs")

    parser.add_argument('-s', '--standardize', action='store_true', help='Standardize the input data before running PCA')
    parser.add_argument('-o', '--output', type=str, help='Output file name', metavar="output")
    parser.add_argument('-d', '--outputdir', type=str, help='Path to output file, .csv format default', metavar="outputdir")
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the PCA results')

    args = parser.parse_args()

    print(f'Running PCA on {args.file} with {args.n_pcs} principal components.')
    # print running options if they are set
    for arg in vars(args):
        if getattr(args, arg) is not None:
            print(f'Option {arg}: {getattr(args, arg)}')

    print('Running PCA! Please wait.')

    run_pca(args.file, args.n_pcs, args.output, args.outputdir)

    if args.plot is not False:
        print('You set the argument "plot" to True. Plotting now: ')
        # check that file exists:
        if args.outputdir is not None:
            csv_path = os.path.join(args.outputdir, args.output)
            if os.path.exists(csv_path):
                ## TODO:    edit plot argument so that user can specify which PCs to plot;
                ##          also optionally add a way to change the name of the output
                quickplot(csv_path, [1, 2], 'quickPCA.png')
            else:
                print(f"Output directory with output filename '{csv_path}' does not exist.")
        else:
            csv_path = os.path.join(os.getcwd(), args.output) if args.output is not None else os.path.join(os.getcwd(), "pca_results.csv")
            quickplot(csv_path, [1, 2], 'quickPCA.png')