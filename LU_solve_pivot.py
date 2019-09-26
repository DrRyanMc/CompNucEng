import numpy as np

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
    
# LU_factor function from chapter 8
def LU_factor(A,LOUD=True):
    """Factor in place A in L*U=A. The lower triangular parts of A
    are the L matrix. The L has implied ones on the diagonal.
    
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
        row_order: list giving the re-ordered equations
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