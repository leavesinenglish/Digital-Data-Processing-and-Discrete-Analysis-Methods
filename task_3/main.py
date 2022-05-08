import numpy as np
import matplotlib.pyplot as mpl


class Heat_equation_solver:
    def __init__(self, l, h, t, kappa, f, f1, f2, dt=0.0):
        self.kappa = kappa
        self.h = h
        self.n = int(l / h)
        if dt == 0:
            self.dt = self.h ** 2 / (2 * self.kappa)
        else:
            self.dt = dt
        self.m = int(t / self.dt)
        self.solution = np.zeros((self.m, self.n))
        self.f = f
        self.f1 = f1
        self.f2 = f2

    def explicit_scheme(self):
        for i in range(self.n):
            self.solution[0][i] = self.f(i * self.h)
        for t in range(self.m - 1):
            for x in range(1, self.n - 1):
                self.solution[t + 1][x] = self.solution[t][x] + self.kappa * self.dt / (self.h ** 2) * (
                        self.solution[t][x + 1] - 2 * self.solution[t][x] + self.solution[t][x - 1])
                self.solution[t + 1][0] = self.f1(self.dt * (t + 1))
                self.solution[t + 1][self.n - 1] = self.f2(self.dt * (t + 1))
        return self.solution

    def implicit_scheme(self):
        self.n -= 1
        self.m -= 1
        for i in range(self.n + 1):
            self.solution[0][i] = self.f(i * self.h)
        alpha = np.zeros(self.n)
        beta = np.zeros(self.n)
        for t in range(self.m):
            alpha[0] = 0
            beta[0] = self.f1(self.dt * (t + 1))
            for x in range(1, self.n):
                a = -self.kappa * self.dt / (self.h ** 2)
                b = 1 + 2 * self.kappa * self.dt / (self.h ** 2)
                c = a
                alpha[x] = -a / (b + c * alpha[x - 1])
                beta[x] = (self.solution[t][x] - c * beta[x - 1]) / (b + c * alpha[x - 1])
            self.solution[t + 1][self.n] = self.f2(self.dt * (t + 1))
            for x in range(self.n, 0, -1):
                self.solution[t + 1][x - 1] = alpha[x - 1] * self.solution[t + 1][x] + beta[x - 1]
        return self.solution


def test_explicit():
    l = 10
    h = 0.1
    kappa = 3
    t = 5
    f = lambda x: 10
    f1 = lambda x: 10
    f2 = lambda x: 10
    solution = Heat_equation_solver(l, h, t, kappa, f, f1, f2).explicit_scheme()
    mesh = mpl.pcolormesh(solution)
    mpl.title("Зависимость температуры от координаты и времени, явная схема")
    mpl.xlabel("х")
    mpl.ylabel("t")
    mpl.colorbar(mesh)
    mpl.show()


def test_implicit():
    l = 10
    h = 0.1
    kappa = 3
    dt = 0.00125 * 4 / 3
    t = 5
    f = lambda x: 10
    f1 = lambda x: 10
    f2 = lambda x: 10
    solution = Heat_equation_solver(l, h, t, kappa, f, f1, f2, dt=dt).implicit_scheme()
    mesh = mpl.pcolormesh(solution)
    mpl.title("Зависимость температуры от координаты и времени неявная схема")
    mpl.xlabel("х")
    mpl.ylabel("t")
    mpl.colorbar(mesh)
    mpl.show()


test_explicit()
test_implicit()
