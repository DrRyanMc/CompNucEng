import numpy as np

def newton(f,fprime,x0,epsilon=1.0e-6, LOUD=False):
    """Find the root of the function f via Newton-Raphson method
    Args:
        f: function to find root of
        fprime: derivative of f
        x0: initial guess
        epsilon: tolerance
        
    Returns:
        estimate of root
    """
    x = x0
    if (LOUD):
        print("x0 =",x0)
    iterations = 0
    while (np.fabs(f(x)) > epsilon):
        
        if (LOUD):
            print("x_",iterations+1,"=",x,"-",f(x),"/",fprime(x),"=",x - f(x)/fprime(x))
        x = x - f(x)/fprime(x)
        iterations += 1
    print("It took",iterations,"iterations")
    return x #return estimate of root

def inexact_newton(f,x0,delta = 1.0e-7, epsilon=1.0e-6, LOUD=False):
    """Find the root of the function f via Newton-Raphson method
    Args:
        f: function to find root of
        x0: initial guess
        delta: finite difference parameter
        epsilon: tolerance
        
    Returns:
        estimate of root
    """
    x = x0
    if (LOUD):
        print("x0 =",x0)
    iterations = 0
    while (np.fabs(f(x)) > epsilon):
        fx = f(x)
        fxdelta = f(x+delta)
        slope = (fxdelta - fx)/delta
        if (LOUD):
            print("x_",iterations+1,"=",x,"-",fx,"/",slope,"=",x - fx/slope)
        x = x - fx/slope
        iterations += 1
    print("It took",iterations,"iterations")
    return x #return estimate of root

def secant(f,x0,delta = 1.0e-7, epsilon=1.0e-6, LOUD=False):
    """Find the root of the function f via Newton-Raphson method
    Args:
        f: function to find root of
        x0: initial guess
        delta: finite difference parameter
        epsilon: tolerance
        
    Returns:
        estimate of root
    """
    x = x0
    if (LOUD):
        print("x0 =",x0)
    #first time use inexact Newton
    x_old = x
    fold = f(x_old)
    fx = fold
    slope = (f(x_old+delta) - fold)/delta
    x = x - fold/slope
    if (LOUD):
        print("Inexact Newton\nx_",1,"=",x,"-",fx,"/",slope,"=",x - fx/slope,"\nStarting Secant")
    fx = f(x)
    iterations = 1 
    while (np.fabs(fx) > epsilon):
        slope = (fx - fold)/(x - x_old)
        fold = fx
        x_old = x
        if (LOUD):
            print("x_",iterations+1,"=",x,"-",fx,"/",slope,"=",x - fx/slope)
        x = x - fx/slope
        fx = f(x)
        iterations += 1
    print("It took",iterations,"iterations")
    return x #return estimate of root

def newton_system(f,x0,delta = 1.0e-7, epsilon=1.0e-6, LOUD=False):
    """Find the root of the function f via inexact Newton-Raphson method
    Args:
        f: function to find root of
        x0: initial guess
        delta: finite difference parameter
        epsilon: tolerance
        
    Returns:
        estimate of root
    """
    def Jacobian(f,x,delta = 1.0e-7):
        N = x0.size
        J = np.zeros((N,N))
        idelta = 1.0/delta #division is slower than multiplication
        x_perturbed = x.copy() #copy x to add delta
        fx = f(x) #only need to evaluate this once
        for i in range(N):
            x_perturbed[i] += delta
            col = (f(x_perturbed) - fx) * idelta
            x_perturbed[i] = x[i]
            J[:,i] = col
        return J
        
    x = x0
    if (LOUD):
        print("x0 =",x0)
    iterations = 0
    fx = f(x)
    while (np.linalg.norm(fx) > epsilon):
        J = Jacobian(f,x,delta)
        
        RHS = -fx;
        delta_x = GaussElimPivotSolve(J,RHS)
        x = x + delta_x
        fx = f(x)
        if (LOUD):
            print("Iteration",iterations+1,": x =",x," norm(f(x)) =",np.linalg.norm(fx))
        iterations += 1
    print("It took",iterations,"iterations")
    return x #return estimate of root