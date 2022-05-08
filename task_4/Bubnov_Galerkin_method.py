import numpy as np
import scipy.integrate as integrate
import print_solve as solve

a = -1
b = 1 + 0.001
h = 0.05
lam = 1
x = np.arange(a, b, h)
f = lambda x: 1
f1 = lambda x: x
f2 = lambda x: x * x
p1 = lambda x: 1
p2 = lambda x: x
y_exact = lambda x: 1 + 6 * x * x
K = lambda x, s: x * x + x * s
y = y_exact(x)


def Bubnov_Galerkin_method(K, f1, f2, a, b):
    Aij = []
    Bi = []
    Aij.append([])
    Aij[0].append(integrate.quad(lambda t: f1(t) * f1(t), a, b)[0] - lam *
                  integrate.dblquad(lambda t, s: (f1(t) * K(t, s) * f1(s)), a, b, lambda t: a, lambda t: b)[0])
    Aij[0].append(integrate.quad(lambda t: f1(t) * f2(t), a, b)[0] - lam *
                  integrate.dblquad(lambda t, s: (f1(t) * K(t, s) * f2(s)), a, b, lambda t: a, lambda t: b)[0])
    Aij.append([])
    Aij[1].append(integrate.quad(lambda t: f2(t) * f1(t), a, b)[0] - lam *
                  integrate.dblquad(lambda t, s: (f2(t) * K(t, s) * f1(s)), a, b, lambda t: a, lambda t: b)[0])
    Aij[1].append(integrate.quad(lambda t: f2(t) * f2(t), a, b)[0] - lam *
                  integrate.dblquad(lambda t, s: (f2(t) * K(t, s) * f2(s)), a, b, lambda t: a, lambda t: b)[0])

    Bi.append(lam * integrate.dblquad(lambda t, s: (f1(t) * K(t, s) * f(s)), a, b, lambda t: a, lambda t: b)[0])
    Bi.append(lam * integrate.dblquad(lambda t, s: (f2(t) * K(t, s) * f(s)), a, b, lambda t: a, lambda t: b)[0])
    ci = np.linalg.solve(np.asarray(Aij), np.asarray(Bi))
    return 1 + ci[0] * f1(x) + ci[1] * f2(x)


def print_solve_3_2():
    y_approx = Bubnov_Galerkin_method(K, f1, f2, a, b)
    solve.print_solve(x, y, y_approx, "Метод Бубнова-Галеркина")
