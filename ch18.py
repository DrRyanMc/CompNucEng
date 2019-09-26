import numpy as np

def BackSub(aug_matrix,x):
    """back substitute a N by N system after Gauss elimination

    Args:
        aug_matrix: augmented matrix with zeros below the diagonal
        x: length N vector to hold solution
    Returns:
        nothing
    Side Effect:
        x now contains solution
    """
    N = x.size
    for row in range(N-1,-1,-1):
        RHS = aug_matrix[row,N]
        for column in range(row+1,N):
            RHS -= x[column]*aug_matrix[row,column]
        x[row] = RHS/aug_matrix[row,row]
    return
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
def GaussElimPivotSolve(A,b,LOUD=0):
    """create a Gaussian elimination with pivoting matrix for a system

    Args:
        A: N by N array
        b: array of length N
    Returns:
        solution vector in the original order
    """
    [Nrow, Ncol] = A.shape
    assert Nrow == Ncol
    N = Nrow
    #create augmented matrix
    aug_matrix = np.zeros((N,N+1))
    aug_matrix[0:N,0:N] = A
    aug_matrix[:,N] = b
    #augmented matrix is created
    
    #create scale factors 
    s = np.zeros(N)
    count = 0
    for row in aug_matrix[:,0:N]: #don't include b
        s[count] = np.max(np.fabs(row))
        count += 1
    if LOUD:
        print("s =",s)
    if LOUD:
        print("Original Augmented Matrix is\n",aug_matrix)
    #perform elimination
    for column in range(0,N):
        
        #swap rows if needed
        largest_pos = np.argmax(np.fabs(aug_matrix[column:N,column]/s[column])) + column
        if (largest_pos != column):
            if (LOUD):
                print("Swapping row",column,"with row",largest_pos)
                print("Pre swap\n",aug_matrix)
            swap_rows(aug_matrix,column,largest_pos)
            #re-order s
            tmp = s[column]
            s[column] = s[largest_pos]
            s[largest_pos] = tmp
            if (LOUD):
                print("A =\n",aug_matrix)
        #finish off the row
        for row in range(column+1,N):
            mod_row = aug_matrix[row,:]
            mod_row = mod_row - mod_row[column]/aug_matrix[column,column]*aug_matrix[column,:]
            aug_matrix[row] = mod_row
    #now back solve
    x = b.copy()
    if LOUD:
        print("Final aug_matrix is\n",aug_matrix)
    BackSub(aug_matrix,x)
    return x


def create_grid(R,I):
    """Create the cell edges and centers for a 
    domain of size R and I cells
    Args:
        R: size of domain
        I: number of cells
        
    Returns:
        Delta_r: the width of each cell
        centers: the cell centers of the grid
        edges: the cell edges of the grid
    """
    Delta_r = float(R)/I
    centers = np.arange(I)*Delta_r + 0.5*Delta_r
    edges = np.arange(I+1)*Delta_r
    return Delta_r, centers, edges

def DiffusionSolver(R,I,D,Sig_a,nuSig_f, Q,BC, geometry):
    """Solve the neutron diffusion equation in a 1-D geometry
    using cell-averaged unknowns
    Args:
        R: size of domain
        I: number of cells
        D: name of function that returns diffusion coefficient for a given r
        Sig_a: name of function that returns Sigma_a for a given r
        nuSig_f: name of function that returns nu Sigma_f for a given r
        Q: name of function that returns Q for a given r
        BC: Boundary Value of phi at r=R
        geometry: shape of problem 0 for slab
                1 for cylindrical
                2 for spherical
        
    Returns:
        centers: the cell centers of the grid
        phi:  cell-average value of the scalar flux
        
    """
    #create the grid
    Delta_r, centers, edges = create_grid(R,I)
    A = np.zeros((I+1,I+1))
    b = np.zeros(I+1)
    #define surface areas and volumes
    assert( (geometry==0) or (geometry == 1) or (geometry == 2))
    if (geometry == 0):
        #in slab it's 1 everywhere except at the left edge
        S = 0.0*edges+1
        S[0] = 0.0 #this will enforce reflecting BC
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
    b[I] = 2.0*BC
    #fill in rest of matrix
    for i in range(I):
        r = centers[i]
        A[i,i] = (0.5/(Delta_r * V[i])*((D(r)+D(r+Delta_r))*S[i+1]) +
                  Sig_a(r) - nuSig_f(r))
        if (i>0):
            A[i,i-1] = -0.5*(D(r)+D(r-Delta_r))/(Delta_r * V[i])*S[i] 
            A[i,i] += 0.5/(Delta_r * V[i])*((D(r)+D(r-Delta_r))*S[i])
        A[i,i+1] = -0.5*(D(r)+D(r+Delta_r))/(Delta_r * V[i])*S[i+1]
        b[i] = Q(r)
    
    #solve system
    phi = GaussElimPivotSolve(A,b)
    #remove last element of phi because it is outside the domain
    phi = phi[0:I]
    return centers, phi

