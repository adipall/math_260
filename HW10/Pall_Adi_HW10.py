""" Adi Pall - HW10
    11/6/2020
    
    Template code by Prof. Wong
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def rk4(func, interval, y0, h):
    """ RK4 method, to solve the system y' = f(t,y) of m equations
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
    t = interval[0]
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]
    while t < interval[1] - 1e-12:
        f0 = func(t, y)
        f1 = func(t + 0.5*h, y + 0.5*h*f0)
        f2 = func(t + 0.5*h, y + 0.5*h*f1)
        f3 = func(t + h, y + h*f2)
        y += (1.0/6)*h*(f0 + 2*f1 + 2*f2 + f3)
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)

    return tvals, yvals

def minimize(f, r0, tol=1e-6, d = 0.01, steps=100):
    """
    Inputs:
            f - the function f(x)
            r0 - initial params (2d)
            tolerance - default = 1e-6
            d - default = 0.01, step size for moving between parameter values
            steps - default = 100, max iterations

    Returns:
            r - the minimized parameters
            it - the # iterations
            pts - the collection of past parameter values            
    """
    r = np.array(r0)
    v = np.zeros(2)
    it = 0
    err_s = 100
    pts = [np.array(r)]
    while err_s > tol and it < steps:
        fx = f(r)
        v[0] = (f((r[0]+d,r[1]))-f((r[0]-d,r[1])))/(2*d) # dE/d(alpha)
        v[1] = (f((r[0],r[1]+d))-f((r[0],r[1]-d)))/(2*d) # dE/d(beta)
        v /= max(np.abs(v))  # normalize v to unit vector
        alpha = 1  # pick an alpha [could be different scheme for doing so!]
        while (r[0] - alpha*v[0])<0 or (r[1] - alpha*v[1])<0:
            alpha /=2
        # the problem could be that we need to add checking for positive alpha and beta
        while alpha > 1e-10 and f(r - alpha*v) >= fx:
            alpha /= 2
        print(f"it={it}, r[0]={r[0]:.8f}, f={fx:.8f}")
        r -= alpha*v
        print(r)
        pts.append(np.array(r))
        err_s = max(np.abs(alpha*v))
        it += 1

    return r, it, pts

def read_sir_data(fname):
    """ Reads the SIR data for the HW problem.
        Inputs:
            fname - the filename (should be "sir_data.txt")
        Returns:
            t, x - the data (t[k], I[k]), where t=times, I= # infected
            pop - the initial susceptible population (S(0))
    """
    with open(fname, 'r') as fp:
        parts = fp.readline().split()
        pop = float(parts[0])
        npts = int(float(parts[1]))
        t = np.zeros(npts)
        x = np.zeros(npts)

        for k in range(npts):
            parts = fp.readline().split()
            t[k] = float(parts[0])
            x[k] = float(parts[1])
            
    return t, x, pop

def SIR(x, params):
    " SIR Differential Equation model "
    P = x[0]+x[1]+x[2]
    dS = -1*params[0]*x[0]*x[1]/P
    dI = params[0]*x[0]*x[1]/P - params[1]*x[1]
    dR = params[1]*x[1]
    return np.array((dS,dI,dR))

def err(r):
    """ Least-squares error E(r) for SIR """
    err = 0
    x0 = np.array((100.0, 5.0, 0.0)) # Initial Values: Pop(0) = 100, I(0) = 5
    h = .1
    t, x = rk4(lambda t, x: SIR(x, r),
               (0, 140), x0, h)
    for k in range(len(data)):
        err += (x[1][::67][k] - data[k])**2 # grab only the points where data exists
    return err

tvals, data, pop = read_sir_data('sir_data.txt')

if __name__ == '__main__':
    
    # trying out the rk4 method using guessed alpha, betas to find a close guess
    # for the minimize method
    alpha = 0.1
    beta = 0.03 # this gets the I(t) curve closest to the model data
    x0 = np.array((100.0, 5.0, 0.0))
    # I(0) taken from data vals (k=0)
    # h = (tvals[1]-tvals[0])/(2**len(tvals))
    h = (tvals[1]-tvals[0])/(2**3) # minimum step size, use .1 to be safe
    h = 0.1
    t, x = rk4(lambda t, x: SIR(x, (alpha,beta)),
               (0, 140), x0, h)

    plt.figure(1)
    plt.plot(t, x[0], '-k', t, x[1], '-r',t, x[2], '-g', tvals, data, '.k', markersize=12)
    plt.xlabel('t')
    plt.ylabel('People')
    plt.legend(['S(t)', 'I(t)', 'R(t)','Data'])
    plt.show()

    # using minimize and least squares error now
    r0 = (0.15, 0.035)
    r_model, it, pts = minimize(lambda r:  err(r),
                           r0, tol=1e-8)

    t, x = rk4(lambda t, x: SIR(x, (r_model)),
               (0, 140), x0, h)

    fig2 = plt.figure(2)
    plt.plot(t, x[0], '-k', t, x[1], '-r',t, x[2], '-g',tvals, data, '.k', markersize=12)
    plt.xlabel('t')
    plt.title('Using gradient descent & line search')
    plt.ylabel('People')
    plt.legend(['S(t)', 'I(t)', 'R(t)','Data'])
    plt.show()
    fig2.savefig('gradient_descent_SIR.png')
    
    print("-----------------")
    print("Using scipy:")
    
    sol = opt.minimize(lambda r:  err(r),(0.1,0.03), method = 'Nelder-Mead')
    print(sol.x)
    
    t, x = rk4(lambda t, x: SIR(x, (sol.x)),
               (0, 140), x0, h)
    
    fig3 = plt.figure(3)
    plt.plot(t, x[0], '-k', t, x[1], '-r',t, x[2], '-g',tvals, data, '.k', markersize=12)
    plt.xlabel('t')
    plt.ylabel('People')
    plt.title('Using scipy')
    plt.legend(['S(t)', 'I(t)', 'R(t)','Data'])
    plt.show()
    fig3.savefig('scipy_SIR.png')
