import numpy as np
import sys
file = sys.argv[1]
first = True
f = open(file, 'r')
i = 0
for l in f:
	l = l.split()
	if first:
		m = int(l[0])
		n = int(l[1])
		a = np.zeros((m,n))
		first = False
	else:
		for j in range(n):
			a[i][j] = float(l[j])
		i += 1

f.close()
print (m,n)
print a.T
print a.T.shape
u, s, vh = np.linalg.svd(a.T, full_matrices=False)
print u
print s