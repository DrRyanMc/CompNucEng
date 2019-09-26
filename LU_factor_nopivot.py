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