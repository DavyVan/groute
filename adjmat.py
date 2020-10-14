import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
import seaborn as sns
import gc

def main():
    # args
    parser = argparse.ArgumentParser(description="Adjencency Matrix Visualization.")
    parser.add_argument('-i', type=str, required=True, dest='inputfile', help='Input file path')
    parsed_arg = parser.parse_args()
    print(parsed_arg)

    # read from file
    # read the first line: <# TBs> <# conns>
    fd = open(parsed_arg.inputfile, 'r')
    header = np.fromfile(fd, dtype=int, count=2, sep=' ')
    TB_num = header[0]
    conn_num = header[1]
    # read indptr
    indptr = np.fromfile(fd, dtype=int, count=TB_num+1, sep=' ')
    # read indices
    indices = np.fromfile(fd, dtype=int, count=conn_num, sep=' ')
    # read data
    data = np.fromfile(fd, dtype=int, count=conn_num, sep=' ')

    print(len(indices), len(data))

    csr = csr_matrix((data, indices, indptr), shape=(TB_num, TB_num))
    del indptr
    del indices
    del data
    gc.collect()
    coo = csr.tocoo(copy=False)

    ax = sns.heatmap(coo.toarray(), vmax=100.0, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_xlim(xmax=TB_num)
    ax.set_ylim(ymax=TB_num)
    ax.set_xticks(np.arange(0, TB_num, step=2000))
    ax.set_yticks(np.arange(0, TB_num, step=2000))
    ax.set_xticklabels(np.arange(0, TB_num, step=2000).astype(str))
    ax.set_yticklabels(np.arange(0, TB_num, step=2000).astype(str))
    plt.savefig("/data/qfan005/adj.parmat.png", format='png', dpi=200.0)

if __name__ == "__main__":
    main()