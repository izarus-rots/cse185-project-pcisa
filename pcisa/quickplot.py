import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def quickplot(csvpath: str, pcs : list, name : str = None):
    # ASSUMES pcs IS GIVEN AS AN INTEGER LIST! TODO: ADD CHECK!
    df = pd.read_csv(csvpath)

    plt.figure(figsize=(10,10))
    sns.scatterplot(x=df.columns[pcs[0]], y=df.columns[pcs[1]], data=df)
    plt.title('PCA Plot:' + str(pcs[0]) + ' vs. ' + str(pcs[1]))
    plt.savefig(name)

    print(f'PCA plot saved to {name} in {os.getcwd()}')