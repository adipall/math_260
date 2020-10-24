# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:12:09 2020
Q2 - RK4
@author: Adi Pall (but mostly Prof. Wong)
"""

import numpy as np
import matplotlib.pyplot as plt

def ode_func(t, y):
    """ ode function for lecture example (y' = 2ty) """
    return 2*t*y

def sol_true(t, y0):
    """ true solution for the example, given y(0) = y0 """
    return y0*np.exp(t**2)

def rk4(func, a, b, y0, h):
    """ 4th order RK method with fixed step size input, using while loop"""
    y = y0
    yvals = [y]
    tvals = [a]
    t = a # t0 = t = a
    while t < b - 1e-12:  # (avoids rounding error where t misses b slightly)
        k1 = h*func(t,y)
        k2 = h*func(t + 0.5*h, y + 0.5*k1)
        k3 = h*func(t + 0.5*h, y + 0.5*k2)
        k4 = h*func(t + h, y + k3)
        
        y = y + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
        yvals.append(y) # update yvals tracking
        
        t = t + h 
        tvals.append(t)
        
    return tvals, yvals

def example_plot():
    """ Solve y' = 2ty, y(0) = 1 using RK4 method.
        (Example from lecture with a plot)
    """
    b = 1
    h = 0.01
    y0 = 1
    t, y_approx = rk4(ode_func, 0, b, y0, h)
    t_true = np.linspace(0, b, 200)  
    y_true = sol_true(t_true, y0)  

    print("{:.2f} \t {:.2f}".format(t[-1], y_approx[-1]))

    plt.figure()
    plt.plot(t, y_approx, '.--r')
    plt.plot(t_true, y_true, '-k')
    plt.legend(['approx', 'actual sol'])
    plt.ylabel('y')
    plt.xlabel('t')
    
def convergence_ex():
    """ Solve y' = 2ty, y(0) = 1 using RK4 method.
        Use the true solution to compute the max error,
        and show that it is O(h^4)
    """
    hvals = [(0.1)*2**(-k) for k in range(8)]
    y0 = 1
    b = 1

    err = [0]*len(hvals)
    for k in range(len(hvals)):  # err[k] is the max error given spacing h[k]
        t, u = rk4(ode_func, 0, b, y0, hvals[k])

        # compute errors at each t (point_errs), then max. error
        # (use zip/list comprehension trick to iterate over t and u)
        point_errs = [abs(sol_true(t1, y0) - u1) for t1, u1 in zip(t, u)]
        err[k] = max(point_errs)  # max error

    plot_hvals = [(1e-4)*2**(-4*k) for k in range(8)]
    plt.figure()
    plt.loglog(hvals, plot_hvals, '--r')
    plt.loglog(hvals, err, '.--k')
    plt.legend(['slope 4', 'max. err.'])
    plt.xlabel('$h$')
    plt.ylabel('$err$')
    plt.savefig('convergence_HW8.png')
    
if __name__ == "__main__":
    example_plot()
    convergence_ex()
    

