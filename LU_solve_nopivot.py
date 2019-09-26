import numpy as np

def LU_factor(A):
	"""Factor in place A in L*U=A. The lower triangular parts of A
	are the L matrix.  The L has implied ones on the diagonal.
	Args:
		A: N by N array
	Side Effects:
		A is factored in place.
	"""
	[Nrow, Ncol] = A.shape
	assert Nrow == Ncol
	N = Nrow
	for column in range(0,N):
		for row in range(column+1,N):
			mod_row = A[row]
			factor = mod_row[column]/A[column,column]
			mod_row = mod_row - factor*A[column,:]
			#put the factor in the correct place in the modified row
			mod_row[column] = factor
			#only take the part of the modified row we need
			mod_row = mod_row[column:N]
			A[row,column:N] = mod_row
	return
	
def LU_solve(A,b):
	"""Take a LU factorized matrix and solve it for RHS b Args:
	A: N by N array that has been LU factored with
	assumed 1's on the diagonal of the L matrix
	b: N by 1 array of righthand side
	Returns:
	x: N by 1 array of solutions
	"""
	[Nrow, Ncol] = A.shape
	assert Nrow == Ncol
	N = Nrow

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