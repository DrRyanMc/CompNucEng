import numpy as np

def LU_factor(A,LOUD=True):
    """Factor in place A in L*U=A. The lower triangular parts of A
    are the L matrix.  The L has implied ones on the diagonal.

    Args:
        A: N by N array
    Returns:
        a vector holding the order of the rows, relative to the original order
    Side Effects:
        A is factored in place.
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    #create scale factors 
    s = np.zeros(N)
    count = 0
    row_order = np.arange(N)
    for row in A:
        s[count] = np.max(np.fabs(row))
        count += 1
    if LOUD:
        print("s =",s)
    if LOUD:
        print("Original Matrix is\n",A)
    for column in range(0,N):
        #swap rows if needed
        largest_pos = np.argmax(np.fabs(A[column:N,column]/s[column])) + column
        if (largest_pos != column):
            if (LOUD):
                print("Swapping row",column,"with row",largest_pos)
                print("Pre swap\n",A)
            swap_rows(A,column,largest_pos)
            #keep track of changes to RHS
            tmp = row_order[column]
            row_order[column] = row_order[largest_pos]
            row_order[largest_pos] = tmp
            #re-order s
            tmp = s[column]
            s[column] = s[largest_pos]
            s[largest_pos] = tmp
            if (LOUD):
                print("A =\n",A)
        for row in range(column+1,N):
            mod_row = A[row]
            factor = mod_row[column]/A[column,column]
            mod_row = mod_row - factor*A[column,:]
            #put the factor in the correct place in the modified row
            mod_row[column] = factor
            #only take the part of the modified row we need
            mod_row = mod_row[column:N]
            A[row,column:N] = mod_row
    return row_order
    
def LU_solve(A,b,row_order):
    """Take a LU factorized matrix and solve it for RHS b

    Args:
        A: N by N array that has been LU factored with
        assumed 1's on the diagonal of the L matrix
        b: N by 1 array of righthand side
        row_order:  list giving the re-ordered equations
        from the the LU factorization with pivoting
    Returns:
        x: N by 1 array of solutions
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    assert b.size == Ncol
    assert row_order.max() == Ncol-1
    N = Nrow
    
    #reorder the equations
    tmp = b.copy()
    for row in range(N):
        b[row_order[row]] = tmp[row]
        
    x = np.zeros(N)
    #temporary vector for L^-1 b
    y = np.zeros(N)
    #forward solve
    for row in range(N):
        RHS = b[row]
        for column in range(0,row):
            RHS -= y[column]*A[row,column]
        y[row] = RHS
    #back solve
    for row in range(N-1,-1,-1):
        RHS = y[row]
        for column in range(row+1,N):
            RHS -= x[column]*A[row,column]
        x[row] = RHS/A[row,row]
    return x
    
def inversePower(A,B,epsilon=1.0e-6,LOUD=False):
    """Solve the generalized eigenvalue problem
    Ax = l B x using inverse power iteration
    Inputs
    A: The LHS matrix (must be invertible)
    B: The RHS matrix
    epsilon: tolerance on eigenvalue
    Outputs:
    l: the smallest eigenvalue of the problem
    x: the associated eigenvector
    """
    N,M = A.shape
    assert(N==M)
    #generate initial guess
    x = np.random.random((N))
    x = x / np.linalg.norm(x) #make norm(x)==1
    l_old = 0
    converged = 0
    #compute LU factorization of A
    row_order = LU_factor(A,LOUD=False)
    iteration = 1;
    while not(converged):
        b = LU_solve(A,np.dot(B,x),row_order)
        l = np.linalg.norm(b)
        x = b/l
        converged = (np.fabs(l-l_old) < epsilon)
        l_old = l
        if (LOUD):
            print("Iteration:",iteration,"\tMagnitude of l =",1.0/l)
        iteration += 1
    #need to check sign of l
    return 1.0/l, x

def create_grid(R,I):
    """Create the cell edges and centers for a 
    domain of size R and I cells
    Args:
        R: size of domain
        I: number of cells
        
    Returns:
        Delta_r: the width of each cell
        edges: the cell edges of the grid
        centers: the cell centers of the grid
    """
    Delta_r = float(R)/I
    centers = np.arange(I)*Delta_r + 0.5*Delta_r
    edges = np.arange(I+1)*Delta_r
    return Delta_r, centers, edges

def DiffusionEigenvalue(R,I,D,Sig_a,nuSig_f, geometry,epsilon = 1.0e-8):
    """Solve a neutron diffusion eigenvalue problem in a 1-D geometry
    using cell-averaged unknowns
    Args:
        R: size of domain
        I: number of cells
        D: name of function that returns diffusion coefficient for a given r
        Sig_a: name of function that returns Sigma_a for a given r
        nuSig_f: name of function that returns nu Sigma_f for a given r
        geometry: shape of problem 
                0 for slab
                1 for cylindrical
                2 for spherical
        
    Returns:
        k: the multiplication factor of the system
        phi:  the fundamental mode with norm 1
        centers: position at cell centers
        
    """
    #create the grid
    Delta_r, centers, edges = create_grid(R,I)
    A = np.zeros((I+1,I+1))
    B = np.zeros((I+1,I+1))
    #define surface areas and volumes
    assert( (geometry==0) or (geometry == 1) or (geometry == 2))
    if (geometry == 0):
        #in slab it's 1 everywhere except at the left edge
        S = 0.0*edges+1
        S[0] = 0.0 #to enforce Refl BC
        #in slab its dr
        V = 0.0*centers + Delta_r
    elif (geometry == 1):
        #in cylinder it is 2 pi r
        S = 2.0*np.pi*edges
        #in cylinder its pi (r^2 - r^2)
        V = np.pi*( edges[1:(I+1)]**2 
                   - edges[0:I]**2 )
    elif (geometry == 2):
        #in sphere it is 4 pi r^2
        S = 4.0*np.pi*edges**2
        #in sphere its 4/3 pi (r^3 - r^3)
        V = 4.0/3.0*np.pi*( edges[1:(I+1)]**3
                   - edges[0:I]**3 )
    
    #Set up BC at R
    A[I,I] = 1.0
    A[I,I-1] = 1.0
    
    #fill in rest of matrix
    for i in range(I):
        r = centers[i]
        A[i,i] = (0.5/(Delta_r * V[i])*((D(r)+D(r+Delta_r))*S[i+1]) +
                  Sig_a(r))
        B[i,i] = nuSig_f(r)
        if (i>0):
            A[i,i-1] = -0.5*(D(r)+D(r-Delta_r))/(Delta_r * V[i])*S[i] 
            A[i,i] += 0.5/(Delta_r * V[i])*((D(r)+D(r-Delta_r))*S[i])
        A[i,i+1] = -0.5*(D(r)+D(r+Delta_r))/(Delta_r * V[i])*S[i+1]
    
    #find eigenvalue
    l,phi = inversePower(A,B,epsilon)
    k = 1.0/l
    #remove last element of phi because it is outside the domain
    phi = phi[0:I]
    return k, phi, centers

def swap_rows(A, a, b):
    """Rows two rows in a matrix, switch row a with row b
    
    args: 
    A: matrix to perform row swaps on
    a: row index of matrix
    b: row index of matrix
    
    returns: nothing
    
    side effects:
    changes A to rows a and b swapped
    """
    assert (a>=0) and (b>=0)
    N = A.shape[0] #number of rows
    assert (a<N) and (b<N) #less than because 0-based indexing
    temp = A[a,:].copy()
    A[a,:] = A[b,:].copy()
    A[b,:] = temp.copy()
