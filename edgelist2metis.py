import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import sys

'''
The format of METIS:
nv ne fmt ncon(nwgt)
n lines: vsize w1 ... wncon v1 e1 ... vk ek
(index of v start from 1 not 0)
'''
def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("Usage:")
        print("edgelist2csr.py <input edgelist file> <output csr file>")
        return
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    # read file
    print("Reading file...")
    edgelist = pd.read_csv(inputfile, delim_whitespace=True, header=None)

    # to coo, then csr
    print("Construct COO and convert to CSR...")
    edgelist_np = edgelist.values
    coo = coo_matrix((edgelist_np[:,0], (edgelist_np[:,0], edgelist_np[:,1])))
    csr = coo.tocsr()

    # resize to square
    print("Resizing...")
    maxDim = max(csr.get_shape())
    csr.resize((maxDim, maxDim))
    
    # transform it to undirected graph
    print("Transforming to undirected...")
    csrT = csr.transpose()
    csr = csr + csrT

    # write to file
    print("Writing to file...")
    fd = open(outputfile, "w")

    # header line with 2 parameters
    fd.write(str(len(csr.indptr)) + " " + str(len(csr.indices)//2))     # METIS doesn't count a bidirection edge twice
    fd.write("\n")

    # vertices
    for i in range(len(csr.indptr)-1):
        # no vsize
        # no vwgt
        startIdx = csr.indptr[i]
        endIdx = csr.indptr[i+1] # exclusive
        for j in range(startIdx, endIdx):
            fd.write(str(csr.indices[j]) + " ")
        fd.write("\n")
    fd.write("\n")      # An extra \n is necessary to end file properly

    fd.close()


if __name__ == "__main__":
    main()