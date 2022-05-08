import numpy as np
import print_solve as solve

y_exact = lambda x: x
a = 0
b = 1 + 0.001
L = 1 / 2
K = lambda x, s: x * s
f = lambda x: 5 * x / 6
h = 1 / 10
lam = 1 / 2
x = np.arange(a, b, h)
x = x.reshape(len(x), 1)
n = len(x)

# exact solution
y = []
for i in range(n):
    y.append([])
    y[i].append(y_exact(x[i]))
y = np.array(y).reshape(n, 1)


def quadrature_method(K, f, a, b, h, lam):
    x = np.arange(a, b, h)
    x = x.reshape(len(x), 1)
    n = len(x)
    wt = 1 / 2
    wj = 1
    A = np.zeros((n, n))
    for i in range(n):
        A[i][0] = -lam * h * wt * K(x[i], x[0])
        for j in range(1, n - 1):
            A[i][j] = -lam * h * wj * K(x[i], x[j])
        A[i][n - 1] = -lam * h * wt * K(x[i], x[n - 1])
        A[i][i] += 1
    B = np.zeros((n, 1))
    for i in range(n):
        B[i][0] = f(x[i])
    return np.linalg.solve(A, B)


def print_solve_1():
    y_approx = quadrature_method(K, f, a, b, h, lam)
    solve.print_solve(x, y, y_approx, "Метод квадратур")
