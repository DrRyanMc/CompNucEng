import numpy as np

def JacobiSolve(A,b,tol=1.0e-6,max_iterations=100,LOUD=False):
    """Solve a linear system by Jacobi iteration.
    Note: system must be diagonally dominant
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    converged = False
    iteration = 1
    x = np.random.rand(N) #random initial guess 
    x_new = np.zeros(N)
    while not(converged):
        x = x_new.copy() #replace old value
        x_new *= 0 #reset x_new
        for row in range(N):
            x_new[row] = b[row]
            for column in range(N):
                if column != row:
                    x_new[row] -= A[row,column]*x[column]
            x_new[row] /= A[row,row]
        relative_change = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x_new

def JacobiSolve_Short(A,b,tol=1.0e-6,max_iterations=100,LOUD=False):
    """Solve a linear system by Jacobi iteration.
    This implementation removes the for loops to make it faster
    Note: system must be diagonally dominant
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    converged = False
    iteration = 1
    x = np.random.rand(N) #random initial guess 
    x_new = np.zeros(N)
    while not(converged):
        x = x_new.copy() #replace old value
        x_new *= 0 #reset x_new
        #update is (b - whole row * x + diagonal part * x)/diagonal
        x_new = (b - np.dot(A,x)+ A.diagonal()*x)/A.diagonal()
        relative_change = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x_new

def Gauss_Seidel_Solve(A,b,tol=1.0e-6,max_iterations=100,LOUD=False):
    """Solve a linear system by Gauss-Seidel iteration.
    Note: system must be diagonally dominant
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    converged = False
    iteration = 1
    x = np.random.rand(N) #random initial guess 
    x_new = np.zeros(N)
    while not(converged):
        x = x_new.copy() #replace old value
        for row in range(N):
            x_new[row] = b[row]
            for column in range(N):
                if column != row:
                    #only change from before is that I use x_new in the update
                    x_new[row] -= A[row,column]*x_new[column]
            x_new[row] /= A[row,row]
        relative_change = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x_new

def SOR_Solve(A,b,tol=1.0e-6,omega=1,max_iterations=100,LOUD=False):
    """Solve a linear system by Gauss-Seidel iteration with SOR.
    Note: system must be diagonally dominant
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        omega: the over-relaxation parameter
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    converged = False
    iteration = 1
    x = np.random.rand(N) #random initial guess 
    x_new = np.zeros(N)
    while not(converged):
        x = x_new.copy() #replace old value
        for row in range(N):
            x_new[row] = b[row]
            for column in range(N):
                if column != row:
                    x_new[row] -= A[row,column]*x_new[column]
            x_new[row] /= A[row,row]
            x_new[row] = (1.0-omega) * x[row] + omega*x_new[row]
        relative_change = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x_new

def SOR_Solve_Opt(A,b,tol=1.0e-6,max_iterations=100,LOUD=False):
    """Solve a linear system by Gauss-Seidel iteration with SOR and automatic \omega
    Note: system must be diagonally dominant
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    converged = False
    iteration = 1
    omega = 1
    l = 5
    p = 2
    x = np.random.rand(N) #random initial guess 
    x_new = np.zeros(N)
    while not(converged):
        x = x_new.copy() #replace old value
        for row in range(N):
            x_new[row] = b[row]
            for column in range(N):
                if column != row:
                    x_new[row] -= A[row,column]*x_new[column]
            x_new[row] /= A[row,row]
            x_new[row] = (1.0-omega) * x[row] + omega*x_new[row]
        relative_change = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        #record change after iteration k
        if (l==iteration):
            dxl = np.linalg.norm(x_new-x)
        if (l + p == iteration):
            dxlp = np.linalg.norm(x_new-x)
            omega = 2.0/(1.0+np.sqrt(1-(dxlp/dxl)**(1.0/p)))
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x_new

def CG(A,b,tol=1.0e-6,max_iterations=100,LOUD=False):
    """Solve a linear system by Conjugate Gradient
    Note: system must be SPD
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    converged = False
    iteration = 1
    x = np.random.rand(N) #random initial guess 
    r = b - np.dot(A,x)
    s = r.copy()
    while not(converged):
        denom = np.dot(s, np.dot(A,s))
        alpha = np.dot(s,r)/denom
        x = x + alpha*s            
        r = b - np.dot(A,x)
        beta = - np.dot(r,np.dot(A,s))/denom
        s = r + beta * s
        relative_change = np.linalg.norm(r)
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x

def JacobiTri(A,b,tol=1.0e-6,max_iterations=100,LOUD=False):
    """Solve a linear system by Jacobi iteration.
    Note: system must be diagonally dominant
    Args:
        A: N by N array
        b: array of length N
        tol: Relative L2 norm tolerance for convergence
        max_iterations: maximum number of iterations
    Returns:
        The approximate solution to the linear system
    """
    [Nrow, Ncol] = A.shape
    assert 3 == Ncol
    N = Nrow
    converged = False
    iteration = 1
    x = np.random.rand(N) #random initial guess 
    x_new = np.zeros(N)
    while not(converged):
        x = x_new.copy() #replace old value
        for i in range(1,N-1):
            x_new[i] = (b[i] - A[i,0]*x[i-1] - A[i,2]*x[i+1])/A[i,1]
        i = 0
        x_new[0] = (b[i]  - A[i,2]*x[i+1])/A[i,1]
        i = N-1
        x_new[i] = (b[i] - A[i,0]*x[i-1])/A[i,1]
        relative_change = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        if (LOUD):
            print("Iteration",iteration,": Relative Change =",relative_change)
        if (relative_change < tol) or (iteration >= max_iterations):
            converged = True
        iteration += 1
    return x_new