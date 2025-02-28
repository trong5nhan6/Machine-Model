import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def cost(x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

def grad(x):
	m = A.shape[0]
	return 1/m * A.dot(A.dot(x) - b)

def gradient_descent(x_init, learning_rate, iteration):
	x_list = [x_init]

	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1])
		x_list.append(n_new)
	return x_list

A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T

fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(-1,30), ylim=(-10,60))
plt.plot(A,b, 'ro')

ones = np.ones((A.shape[0],1), dtype=np.int8)
x_quare = np.array([A[:,0]**2]).T
A = np.concatenate((x_quare,A,ones), axis=1)
print(A)

x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
x0_gd = np.linspace(1,25,10000)
y0_gd = x[0][0]*x0_gd*x0_gd+ x[1][0]*x0_gd + x[2][0]
plt.plot(x0_gd, y0_gd, color='green')

C = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).T
d = np.array([[2,7,9,23,22,29,29,35,40,6,39,42,15,10,6]]).T

ones1 = np.ones((C.shape[0],1), dtype=np.int8)
x1_quare = np.array([C[:,0]**2]).T
C = np.concatenate((x1_quare,C,ones1), axis=1)
z = np.linalg.inv(C.transpose().dot(C)).dot(C.transpose()).dot(d)

# x0_init = np.array([[1.],[5.],[10.],[15.],[20.]])
x0_init = np.linspace(1,15,10000)
y0_init = z[0][0]*x0_init*x0_init+ z[1][0]*x0_init + z[2][0]
plt.plot(x0_init,y0_init, color="black")

iteration = 90
learning_rate = 0.0001

# x_list = gradient_descent(x0_init, learning_rate, iteration)

# #draw x_list
# for i in range(len(x_list)):
# 	y0_x_list = x_list[i][0] + x_list[i][1]*x0_gd
# 	plt.plot(x0_gd,y0_x_list, color='black', alpha = 0.3)

# print(len(x_list))

plt.show()