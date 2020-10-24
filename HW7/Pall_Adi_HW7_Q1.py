# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:29:03 2020
Homework 7 - Q1
@author: Adi Pall
"""

import numpy as np
import matplotlib.pyplot as plt
#from numpy import random

def power_method_err(a, steps, sol):
    """ simple implementation of the power method, using a fixed
        number of steps. Computes the largest eigenvalue and
        associated eigenvector.

        Args:
            a - the (n x n) matrix
            steps - the number of iterations to take
            sol - the largest eigenvector
        Returns:
            x, r - the eigenvector/value pair such that a*x = r*x
    """
    n = a.shape[0]
    #x = random.rand(n)
    x = np.ones(n) # I found that sometimes random behaved badly,
    # whereas starting with all ones behaved well all the time (for the 20
    # times in a row that I tried at least)
    # this is in contrast with rand, which behaved badly once every six runs approx
    # I am not sure why this is the case, I would love some feedback on it!
    err = np.zeros([steps,n])
    sol = sol/np.sqrt(sol.dot(sol)) # normalize solution
    it = 0
    while it < steps:  # other stopping conditions would go here
        q = np.dot(a, x)  # compute a*x
        r = x.dot(q)    # Rayleigh quotient x_k dot Ax_k / x_k dot x_k
        x = q/np.sqrt(q.dot(q))  # normalize x to a unit vector
        err[it] = abs(sol-x)
        it += 1
    return x, r, err # x at this step = eigenvector; r at this step = lambda (eigenvalue)

if __name__ == "__main__":   # example from lecture (2x2 matrix)
    a = np.array([[0,1,0], [0, 0,1], [6,-11,6]])
  #  sol1 = np.array([0.105, 0.314, 0.943])
    sol = np.array([1., 3., 9.])
    evec, eig, err = power_method_err(a, 100, sol)

    np.set_printoptions(precision=3)  # set num. of displayed digits
    print(f"Largest eigenvalue: {eig:.2f}")
    print("Eigenvector: ", evec)
    print("Err:\n", err)
    steps = np.arange(0,100)
    plot_steps = steps[30:50]
    plot_err = err[30:50]
    slope_check = abs(1e16*(-0.314)**plot_steps)
    plt.figure(figsize=(3, 2.5))
    plt.semilogy(plot_steps, plot_err[:,0], '.-k', markersize=12)
    plt.semilogy(plot_steps, plot_err[:,1], '.-g', markersize=12)
    plt.semilogy(plot_steps, plot_err[:,2], '.-b', markersize=12)
    # for some reason, they all behave like 
    # C*(-second elem of eigenvector corresponding to largest eigenvalue)^n
    # surely there is a mathematical explanation for this, but I think
    # this answer is sufficient given the problem statement
    plt.semilogy(plot_steps, slope_check, '--r') 
    plt.legend(['err v[0]', 'err v[1]','err v[2]',"$err\sim(-0.314)^n$"])
    plt.xlabel('$n$')
    plt.ylabel('err')
    plt.savefig('HW7_Q1.png', bbox_inches='tight')
    plt.show()
    
    
