import numpy as np
import decimal
import matplotlib.pyplot as plt

def trapezoid(f, a, b, pieces, graph=False):
    """Find the integral of the function f between a and b using pieces trapezoids
    Args:
        f: function to integrate
        a: lower bound of integral
        b: upper bound of integral
        pieces: number of pieces to chop [a,b] into
        
    Returns:
        estimate of integral
    """
    integral = 0
    h = b - a
    if (graph):
        x = np.linspace(a,b,100)
        plt.plot(x,f(x),label="f(x)")
        ax = plt.subplot(111)
    #initialize the left function evaluation
    fa = f(a)
    for i in range(pieces):
        #evaluate the function at the left end of the piece
        fb = f(a+(i+1)*h/pieces)
        integral += 0.5*h/pieces*(fa + fb)
        if (graph):
            verts = [(a+i*h/pieces,0),(a+i*h/pieces,fa), (a+(i+1)*h/pieces,fb),(a+(i+1)*h/pieces,0)]
            poly = Polygon(verts, facecolor='0.8', edgecolor='k')
            ax.add_patch(poly)
        #now make the left function evaluation the right for the next step
        fa = fb
        
    if (graph):
        ax.set_xticks((a,b))
        ax.set_xticklabels(('a','b'))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        if (pieces > 1):
            plt.title("Trapezoid Rule with " + str(pieces) + " pieces")
        else:
            plt.title("Trapezoid Rule with " + str(pieces) + " piece")
        plt.show()
    return integral

def trapezoid_rec(f, a, b, epsilon = 1.0e-6, old = -1.0e16, depth=0, graph=False):
    """Find the integral of the function f between a and b using pieces trapezoids
    Args:
        f: function to integrate
        a: lower bound of integral
        b: upper bound of integral
        old: keeps track of how much the estimate changes from recursion to level
        depth: how many levels of recursion do we have, python only allows so many
        
    Returns:
        estimate of integral
    """
    h = b - a
    #break interval into two pieces and do trapezoid on each
    if (graph) and (depth == 0):
        x = np.linspace(a,b,100)
        plt.plot(x,f(x),label="f(x)")
        
    new_estimate_left = 0.25*h*(f(a) + f(a+0.5*h))
    new_estimate_right = 0.25*h*(f(a+0.5*h) + f(b))

    
    
    #check to see if the sum of the two pieces is close to the old guess
    if (np.fabs(new_estimate_left + new_estimate_right - old) > epsilon) and (depth < 100):
        #if not, then call trapezoid_rec again on each half-interval
        integral = (trapezoid_rec(f,a,a+0.5*h,epsilon=epsilon, old=new_estimate_left,depth=depth+1,graph=graph) + 
                    trapezoid_rec(f,a+0.5*h,b,epsilon=epsilon,old=new_estimate_right,depth=depth+1,graph=graph))

        return integral
    else:
        #halving the interval didn't change much so we can stop
            
        if (graph):
            ax = plt.subplot(111)
            verts = [(a,0),(a,f(a)), (a+0.5*h,f(a+0.5*h)),(a+0.5*h,0)]
            poly = Polygon(verts, facecolor='0.8', edgecolor='k')
            ax.add_patch(poly)
            verts = [(a+0.5*h,0),(a+0.5*h,f(a+0.5*h)), (b,f(b)),(b,0)]
            poly = Polygon(verts, facecolor='0.8', edgecolor='k')
            ax.add_patch(poly)
            
        if (graph) and (depth == 0):
            ax.set_xticks((a,b))
            ax.set_xticklabels(('a','b'))
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Trapezoid Rule with recursion")
            plt.show()
        return new_estimate_left + new_estimate_right

def quadratic_interp(a,f,x):
    """Compute at quadratic interpolant
    Args:
        a: array of the 3 points
        f: array of the value of f(a) at the 3 points
    Returns:
        The value of the linear interpolant at x
    """
    answer = (x-a[1])*(x-a[2])/(a[0]-a[1])/(a[0]-a[2])*f[0] 
    answer += (x-a[0])*(x-a[2])/(a[1]-a[0])/(a[1]-a[2])*f[1] 
    answer += (x-a[0])*(x-a[1])/(a[2]-a[0])/(a[2]-a[1])*f[2] 
    return answer

def simpsons(f, a, b, pieces, graph=False):
    """Find the integral of the function f between a and b using Simpson's rule
    Args:
        f: function to integrate
        a: lower bound of integral
        b: upper bound of integral
        pieces: number of pieces to chop [a,b] into
        
    Returns:
        estimate of integral
    """
    integral = 0
    h = b - a
    one_sixth = 1.0/6.0
    if (graph):
        x = np.linspace(a,b,100)
        plt.plot(x,f(x),label="f(x)")
        ax = plt.subplot(111)
    
    #initialize the left function evaluation
    fa = f(a)
    for i in range(pieces):
        #evaluate the function at the left end of the piece
        fb = f(a+(i+1)*h/pieces)
        fmid = f(0.5*(a+(i+1)*h/pieces+ a+i*h/pieces))
        integral += one_sixth*h/pieces*(fa + 4*fmid + fb)
        if (graph):
            ix = np.arange(a+i*h/pieces, a+(i+1)*h/pieces, 0.001)
            iy = quadratic_interp(np.array([a+i*h/pieces,0.5*(a+(i+1)*h/pieces+ a+i*h/pieces),a+(i+1)*h/pieces]),
                                  np.array([fa,fmid,fb]),ix)
            verts = [(a+i*h/pieces,0)] + list(zip(ix,iy)) + [(a+(i+1)*h/pieces,0)]
            poly = plt.Polygon(verts, facecolor='0.8', edgecolor='k')
            ax.add_patch(poly)
        #now make the left function evaluation the right for the next step
        fa = fb
        
    if (graph):
        ax.set_xticks((a,b))
        ax.set_xticklabels(('a','b'))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Simpsons Rule with " + str(pieces) + " pieces")
        plt.show()
    return integral

def RichardsonExtrapolation(fh, fhn, n, k):
    """Compute the Richardson extrapolation based on two approximations of order k
    where the finite difference parameter h is used in fh and h/n in fhn.
    Inputs:
    fh:  Approximation using h
    fhn: Approximation using h/n
    n:   divisor of h
    k:   original order of approximation
    
    Returns:
    Richardson estimate of order k+1"""
    n = decimal.Decimal(n)
    k = decimal.Decimal(k)
    numerator = decimal.Decimal(n**k * decimal.Decimal(fhn) - decimal.Decimal(fh))
    denominator = decimal.Decimal(n**k - decimal.Decimal(1.0))
    return float(numerator/denominator)

def Romberg(f, a, b, MaxLevels = 10, epsilon = 1.0e-6, PrintMatrix = False):
    """Compute the Romberg integral of f from a to b
    Inputs:
    f:  integrand function
    a: left edge of integral
    b: right edge of integral
    MaxLevels: Number of levels to take the integration to
    
    Returns:
    Romberg integral estimate"""
    
    estimate = np.zeros((MaxLevels,MaxLevels))
    
    estimate[0,0] = trapezoid(f,a,b,pieces=1)
    count = 1
    converged = 0
    while not(converged):
        estimate[count,0] = trapezoid(f,a,b,pieces=2**count)
        for extrap in range(count):
            estimate[count,1+extrap] = RichardsonExtrapolation(estimate[count-1,extrap],
                                                               estimate[count,extrap],2,2**(extrap+1))
        
        converged = np.fabs(estimate[count,count] - estimate[count-1,count-1]) < epsilon
        if (count == MaxLevels-1): converged = 1
        count += 1
    if (PrintMatrix):
        print(estimate[0:count,0:count])
    return estimate[count-1, count-1]

def RombergSimpson(f, a, b, MaxLevels = 10, epsilon = 1.0e-6, PrintMatrix = False):
    """Compute the Romberg integral of f from a to b
    Inputs:
    f:  integrand function
    a: left edge of integral
    b: right edge of integral
    MaxLevels: Number of levels to take the integration to
    
    Returns:
    Romberg integral estimate"""
    
    estimate = np.zeros((MaxLevels,MaxLevels))
    
    estimate[0,0] = simpsons(f,a,b,pieces=1)
    count = 1
    converged = 0
    while not(converged):
        estimate[count,0] = simpsons(f,a,b,pieces=2**count)
        for extrap in range(count):
            estimate[count,1+extrap] = RichardsonExtrapolation(estimate[count-1,extrap],
                                                               estimate[count,extrap],n=2,k=2+2.0**(extrap+1))
        
        converged = np.fabs(estimate[count,count] - estimate[count-1,count-1]) < epsilon
        if (count == MaxLevels-1): converged = 1
        count += 1
    if (PrintMatrix):
        print(estimate[0:count,0:count])
    return estimate[count-1, count-1]