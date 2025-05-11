import numpy as np
import numpy.linalg as npl
import math
from matplotlib import pyplot as plt

x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75])
y = np.array([-6.1, -5.1, -3.9, -2.6, -0.9, 0.8])

#считаем нужные суммы
sum_x = sum(i for i in x)
sum_x_sq = sum(i**2 for i in x)
sum_xy = sum(i * j for i, j in zip(x, y))
sum_y = sum(i for i in y)

sum_lnx = sum(math.log(i) for i in x)
sum_lnx_sq = sum(math.log(i)**2 for i in x)
sum_lnx_y = sum(math.log(i) * j for i, j in zip(x, y))

sum_x3 = sum(i**3 for i in x)
sum_x4 = sum(i**4 for i in x)
sum_x2_y = sum(i**2 * j for i, j in zip(x, y))

#аппроксимации тремя способами: линейным, логарифмическим и параболическим
def count_linear():
  a1 = np.array([sum_x_sq, sum_x])
  a2 = np.array([sum_x, len(x)])
  A = np.array([a1, a2])
  b = np.array([sum_xy, sum_y])
  return npl.solve(A, b)

def count_ln():
  a1 = np.array([sum_lnx_sq, sum_lnx])
  a2 = np.array([sum_lnx, len(x)])
  A = np.array([a1, a2])
  b = np.array([sum_lnx_y, sum_y])
  return npl.solve(A, b)

def count_parabola():
  a1 = np.array([sum_x4, sum_x3, sum_x_sq])
  a2 = np.array([sum_x3, sum_x_sq, sum_x])
  a3 = np.array([sum_x_sq, sum_x, len(x)])
  A = np.array([a1, a2, a3])
  b = np.array([sum_x2_y, sum_xy, sum_y])
  return npl.solve(A, b)


lin = count_linear()
logarithm = count_ln()
par = count_parabola()

#вычисление погрешностей
lin_pogr = sum((j - (lin[0]*i + lin[1]))**2 for i,j in zip(x, y))
log_pogr = sum((j - (logarithm[0]*math.log(i) + logarithm[1]))**2 for i,j in zip(x, y))
par_pogr = sum((j - (par[0]*i**2 + par[1]*i + par[2]))**2 for i,j in zip(x, y))

print('Погрешности: линейной аппроксимации -- ', lin_pogr, '  Логарифмической аппроксимации-- ', log_pogr, ' Квадратичной аппроксимации -- ', par_pogr)

X = lin[0]*x + lin[1]
Y = logarithm[0]*np.log(x) + logarithm[1]
Z = par[0]*x**2 + par[1]*x + par[2]

#псотроение графика
plt.scatter(x,y)
plt.plot(x, X, color='r')
plt.plot(x, Y, color='g')
plt.plot(x, Z, color='y')

plt.legend(["исходные точки", "y=ax+b", "y=a*ln(x)+b", "y=ax^2+bx+c"])
plt.show()
