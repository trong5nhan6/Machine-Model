import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model

A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

lr = linear_model.LinearRegression()
lr.fit(A,b)
plt.plot(A,b, 'ro')

x0 = np.array([1,46]).T
y0 = x0*lr.coef_[0][0] + lr.intercept_
print(y0)
print(lr.coef_)
plt.plot(x0, y0)
plt.show()


