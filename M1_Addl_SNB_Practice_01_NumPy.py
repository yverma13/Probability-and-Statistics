import numpy as np

A = np.arange(10)
a = [1,2,3]
np.shape(a)
A.shape

np.array(([1,2],[3,4]))
np.arange(24).reshape(4,6)

np.ones(10)*5
np.full(10,5)

np.zeros(30).shape

arr = np.arange(20,41)
arr[(arr % 2) !=0]

np.identity(5)
np.eye(5,)

np.random.standard_normal(20)

l = [1,2,3]
np.array(l)
np.array(list(range(20)))

np.random.random((4,4,4))
np.random.rand(4,4,4)

b = np.random.rand(10,10)
b.max(), np.max(b), b.min()

np.tile([[1,0],[0,1]],(8,8))

A = np.arange(1,6)
B = np.arange(6,15,2)
print(A,B)

A+B
A-B
A**3

X = np.array([1,45,60,90])
print(np.sin(X[0]))
print(np.cos(X))
print(np.tan(X))

X[0]

X = np.array([4.1, 2.5, 44.5, 25.9, -1.1, -9.5, -6.9])
np.round(X), np.ceil(X), np.floor(X)

X = np.array([1, 2, 3])
Y = np.array([4, 5, 6])
np.divide(X, Y)

np.true_divide(X,Y)
np.floor_divide(X,Y)

X ** Y
np.power(X , Y)

X = np.array([8, 4, 6, 3, 66, 12, 1, 5])
np.clip(X,4,8)

len(X), X.ndim, X.shape

X = np.arange(-1,14,0.25)
X1 = X.reshape(3,20)
X1

print("sum is",np.sum(X1))
print("\nsum of all elements row wise\n",np.sum(X1, axis=0))
print("\nsum of all elements col wise\n",np.sum(X1, axis=1))
print("\nmax and min",X1.max(), X1.min())
print("\n min row wise",np.min(X1, axis=0))
print("\n mean row wise",np.mean(X1, axis=0))
print("\n standard deviation column wise",np.std(X1, axis=1))