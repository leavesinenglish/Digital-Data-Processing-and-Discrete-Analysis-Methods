import numpy as np
import scipy.integrate as integrate
import print_solve as solve

y_exact = lambda x: 1
a = 0
b = 1 + 0.001
Lambda = -1
K = lambda x, s: (x ** 2) * s + (x ** 3) * (s ** 2) / 2 + (x ** 4)(s ** 3) / 6
f = lambda x: np.exp(x) - x
h = 0.05
x = np.arange(a, b, h)
x = x.reshape(len(x), 1)
n = len(x)
alpha = lambda x: [x ** 2, (x ** 3), (x ** 4)]
beta = lambda s: [s, (s ** 2) / 2, (s ** 3) / 6]
y = []
for i in range(n):
    y.append([])
    y[i].append(y_exact(x[i]))
y = np.array(y).reshape((n, 1))


def b_fun(t, m, f):
    return beta(t)[m] * f(t)


def Aij_fun(t, m, k):
    return beta(t)[m] * alpha(t)[k]


def degenerate_kernel_method(f, t, Lambda):
    m = len(alpha(0))
    M = np.zeros((m, m))
    r = np.zeros((m, 1))
    for i in range(m):
        r[i] = integrate.quad(b_fun, a, b, args=(i, f))[0]
        for j in range(m):
            M[i][j] = -Lambda * integrate.quad(Aij_fun, a, b, args=(i, j))[0]
        M[i][i] += 1
    c = np.linalg.solve(M, r)
    return Lambda * (c[0] * alpha(t)[0] + c[1] * alpha(t)[1] + c[2] * alpha(t)[2]) + f(t)


def print_solve_2():
    y_approx = degenerate_kernel_method(f, x, Lambda)
    solve.print_solve(x, y, y_approx, "Метод вырожденных ядер")
