import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)

n = input('Please type a N : ')

#Creating a rank 1 Array
a=int(n)

M=np.random.random((a,a))

#Print M
print(M)

MI=np.linalg.inv(M)
#Print M inverse
print(MI)

#Their product
P=np.matmul(M,MI)
print(P)
