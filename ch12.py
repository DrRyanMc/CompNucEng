import numpy as np

def bisection(f,a,b,epsilon=1.0e-6):
	"""Find the root of the function f via bisection where the root lies within [a,b]
	Args:
		f: function to find root of
		a: left-side of interval
		b: right-side of interval
		epsilon: tolerance
	Returns:
		estimate of root
	"""

	assert (b>a)
	assert (f(a)*f(b) < 0)
	delta = b - a
	print("We expect",int(np.ceil(np.log(delta/epsilon)/np.log(2))),"iterations")
	iterations = 0
	while (delta > epsilon):
		c = (a+b)*0.5
		if (f(a)*f(c) < 0):
			b = c
		elif (f(b)*f(c) < 0):
			a=c
		else:
			return c
		delta = b-a
		iterations += 1
	print("It took",iterations,"iterations")
	return c #return midpoint of interval

def false_position(f,a,b,epsilon=1.0e-6):
	"""Find the root of the function f via false position where the root lies within [a,b]
	Args:
		f: function to find root of
		a: left-side of interval
		b: right-side of interval
		epsilon: tolerance
	Returns:
		estimate of root
	"""
	assert (b>a)
	assert (f(a)*f(b) < 0)
	delta = b - a
	iterations = 0
	residual = 1.0
	while (np.fabs(residual) > epsilon):
		m = (f(b)-f(a))/(b-a)
		c = a - f(a)/m
		if (f(a)*f(c) < 0):
			b=c
		elif (f(b)*f(c) < 0):
			a=c
		else:
			print("It took",iterations,"iterations")
			return c
		residual = f(c)
		iterations += 1
	print("It took",iterations,"iterations")
	return c #return c

def ridder(f,a,b,epsilon=1.0e-6):
	"""Find the root of the function f via Ridder's Method where the root lies within [a,b]
	Args:
		f: function to find root of
		a: left-side of interval
		b: right-side of interval
		epsilon: tolerance
	Returns:
		estimate of root
	"""
	assert (b>a)
	assert (f(a)*f(b) < 0)
	delta = b - a
	iterations = 0
	residual = 1.0
	while (np.fabs(residual) > epsilon):
		c = 0.5*(b+a)
		d = 0.0
		if (f(a) - f(b) > 0):
			d = c + (c-a)*f(c)/np.sqrt(f(c)**2-f(a)*f(b))
		else:
			d = c - (c-a)*f(c)/np.sqrt(f(c)**2-f(a)*f(b))
		#now see which part of interval root is in
		if (f(a)*f(d) < 0):
			b=d
		elif (f(b)*f(d) < 0):
			a=d
		residual = f(d)
		iterations += 1
	print("It took",iterations,"iterations")
	return d #return c