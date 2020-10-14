# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:29:03 2020
Homework 7 - Q2
@author: Adi Pall
"""

import numpy as np
from numpy.random import rand   # for a random number

def norm(x):
    "get vector norm -> sqrt(sum of squares)"
    return np.sqrt(sum((v**2 for v in x)))

def stationary(pt, steps = 100, tol = 1e-5):
    """Power method to find stationary distribution.
       Given the largest eigenvalue is 1, finds the eigenvector.
       Default steps = 100
       Default tolerance = 1e-5
    """
    x = rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    count_it = steps # if it finishes the for loop without breaking, steps = iter
    for it in range(steps):
        temp = x
        x = np.dot(pt, x)
        # to get overall vector error, need sqrt(sum of squares)
        err = abs(norm(x)-norm(temp))
        if (err <= tol):
            count_it = it
            break
    return x, count_it+1 # return dist and steps taken (iter+1 since iter starts @ 0)

def pagerank_Q2(alpha):
    """ page rank example (five nodes).
        takes alpha as an input, where 1-alpha is the chance of teleporting """
    n = 5
    p_matrix = np.array([[0, 1/3, 1/3, 1/3, 0],
                         [1/2, 0, 0, 0, 1/2],
                         [1/2, 1/2, 0, 0, 0],
                         [1/2, 1/2, 0, 0, 0],
                         [0, 1, 0, 0, 0]])
    # note: there are no dead ends in this markov chain
    pt_mod = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            pt_mod[j, k] = alpha*p_matrix[k, j] + (1-alpha)*(1/n)

    dist, steps = stationary(pt_mod)
    print("distribution:", dist)
    print("steps taken:", steps)
    
if __name__ == '__main__':
    # Q2b
    pagerank_Q2(0.95)
    # need about 20 steps to get to the point when taking another step changes 
    # distribution by less than 1e-5, which is how I am defining "stationary"
    
    # additionally, the distribution makes sense, given that most (four) of the
    # arrows are pointing to node 1, and then second most (three) are pointing
    # to node 0
    # resulting dist: [0.271 0.358 0.096 0.096 0.18 ]
    # then, ordering: 1,0,4,2-3 (tie)
    print('---------------------------------')
    # Q2c
    pagerank_Q2(0.01)
    # node 1 will always be the highest ranked, but the trend is:
    # as alpha decreases, the distributions get closer and closer together
    # lim of alpha -> 0 translates to lim stationary dist -> 1/5
