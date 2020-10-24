# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:12:09 2020
Q3 - Explicit trapz vs forward euler for oscillating system
@author: Adi Pall
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def ode_func(t, v):
    '2nd order ODE turned into first order sys'
    return np.array([v[1], -v[0]])

def fwd_euler_sys(f, a, b, y0, h):
    """ Forward euler, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            t - a list of the times for the computed solution
            y - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)  # copy!
    t = a
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]
    while t < b - 1e-12:
        y += h*f(t, y)
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)

    return tvals, yvals

def trapz_solver(f, t, b, y0, h):
    """ Explicit trapz, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            t - a list of the times for the computed solution
            y - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)
    tvals = [t]
    yvals = [[v] for v in y]
    while t < b - 1e-12:
        # part a
        # I hope that I correctly understood the instructions.
        # I interpreted this as a question designed to highlight the situation
        # where explicit trapz could behave better than euler
        y_apprx = y + h*f(t, y) # compute approx of next step directly from val at tn
        # uses euler method for that approximation
        # part b
        y += (h/2)*(f(t,y)+f(t+h,y_apprx)) # trapezoidal method to get next y
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)

    return tvals, yvals

def circle_ex_trapz(h, periods=6):
    """ circular motion example, solving
        x' = y,  y' = -x
        with solutions x(t) = A*cos t, y(t) = A*sin(t)
        (note: in code, v is the vector (x,y))
    """
    v_init = [1.0, 0]
    t, v = trapz_solver(ode_func, 0, periods*2*pi, v_init, h)
    x = v[0]
    y = v[1]

    # plot vs. t
    plt.figure()
    plt.plot(t, x, '-k', t, y, '-r')
    plt.legend(["$\\theta(t)$", "$\\theta'(t)$"])
    plt.xlabel("$t$")
    plt.show()

    # phase plane plot
    plt.figure()
    t = np.linspace(0, 2*pi, 100)
    plt.plot(x, y, '-k', np.cos(t), np.sin(t), '--b')
    plt.legend(["approx.", "exact"])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    
def circle_ex_euler(h, periods=6):
    """ circular motion example, solving
        x' = y,  y' = -x
        with solutions x(t) = A*cos t, y(t) = A*sin(t)
        (note: in code, v is the vector (x,y))
    """
    v_init = [1.0, 0]
    t, v = fwd_euler_sys(ode_func, 0, periods*2*pi, v_init, h)
    x = v[0]
    y = v[1]

    # plot vs. t
    plt.figure()
    plt.plot(t, x, '-k', t, y, '-r')
    plt.legend(["$\\theta(t)$", "$\\theta'(t)$"])
    plt.xlabel("$t$")
    plt.show()

    # phase plane plot
    plt.figure()
    t = np.linspace(0, 2*pi, 100)
    plt.plot(x, y, '-k', np.cos(t), np.sin(t), '--b')
    plt.legend(["approx.", "exact"])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    
if __name__ == "__main__":
    # comparing using same step size of course
    # part c
    circle_ex_euler(0.01) # euler starts to incur serious error for 6 periods
    circle_ex_trapz(0.01) # trapz stays right on the exact solution circle for 6 periods
