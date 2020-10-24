# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:12:09 2020
Q1 - PageRank
@author: Adi Pall
"""
import numpy as np
from numpy import random
from scipy import sparse

def power_method2(pT, alpha, steps):
    """ simple implementation of the power method, using a fixed
        number of steps. 
        
        Args:
            pT - the transposed probability matrix
            alpha - the teleportation prob
            steps - the number of iterations to take
        Returns:
            x, r - the eigenvector/value pair such that a*x = r*x
    """
    n = pT.shape[0]
    E = np.ones((n,n))
    pt_mod = alpha*pT # first half of M
    pt_mod2 = ((1-alpha)/n)*E # second half of M
    x = random.rand(n)
    x = x.transpose()
    for it in range(steps):
        x = pt_mod.dot(x)  # dot them separately
        x2 = pt_mod2.dot(x)
        x = x+x2 # recombine
        x /= sum(x) # normalize 
    return x

def read_graph_data(fname):
    'load in graph data, and return dictionary of names and adjacency list'
    fcur = open(fname,'r')
    adj = dict()
    names = dict()
    line = fcur.readline()
    while line:
        parts = line.split(' ')
        if parts[0] == 'n': # determine if reading edge or vertex
            idx = int(parts[1])
            names[idx] = parts[2].strip() # remove what it considers to be space at end
        if parts[0] == 'e':
            idx_i = int(parts[1])
            adj.setdefault(idx_i,[]) # sets the type for this given key to be a list
            adj[idx_i].append(int(parts[2]))
        line = fcur.readline() # read next line
    fcur.close()
    
    return names, adj

def turn_sparse(fname):
    """turn the loaded dictionary data into sparse matrix, and keep names
       (kinda wraps read_graph_data(fname))
    """
    names, adj = read_graph_data(fname)
    m = len(names) # size for sparse matrix (nodes x nodes)
    row = []
    col = []
    data = []
    for key in adj:
        n = len(adj[key])
        p = 1/n
        for i in range(n):
            row.append(key)
            col.append(adj[key][i])
            data.append(p)
    mat = sparse.coo_matrix((data, (row, col)), shape=(m, m))
    # should likely use csr or csc here for better code efficiency, but need to move to
    # other assignments
    mat_trans = mat.transpose()
    return mat_trans, names
    
if __name__ == "__main__":   # example from lecture (2x2 matrix)

    mat_trans , names = turn_sparse('california.txt')
    # teleportation alpha = 0.9 (just grabbed this from slides)
    x = power_method2(mat_trans,0.9,100) # steps change result by a bit, enter 1000
    # for a little better stationary result, just takes long
    # could likely make code more efficient somewhere
    x_sort = np.flip(np.sort(x), axis = 0)
    x_idx_sort = np.flip(np.argsort(x), axis = 0)
    print("top ten:")
    print("----------")
    for i in x_idx_sort[0:10]:
        print(f"{names[i]} at p = {x[i]}")
# output       
# top ten:
# ----------
# http://search.ucdavis.edu/ at p = 0.1378932742949024
# http://www.gene.com/ae/bioforum/ at p = 0.11048181692052911
# http://vision.berkeley.edu/VSP/index.shtml at p = 0.08578305080058735
# http://www.lib.uci.edu/ at p = 0.07585548111235535
# http://www.ucdavis.edu/ at p = 0.07378068892956909
# http://www.uci.edu/ at p = 0.06943617659679627
# http://www.students.ucr.edu/ at p = 0.04820550427998829
# http://www.scag.org at p = 0.038118352037494814
# http://spectacle.berkeley.edu/ at p = 0.03247489044324612
# http://rsv.ricoh.com/ at p = 0.030464669618797638
        
