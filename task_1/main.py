import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

T = 2 * np.pi
w= 2 * np.pi / T


def a_n(n, f):
    return integrate.quad(lambda x: 2 * f(x) * np.cos(n * x * w) / T, 0, T)[0]


def b_n(n, f):
    return integrate.quad(lambda x: 2 * f(x) * np.sin(n * x * w) / T, 0, T)[0]


def coefs(n, f, coef_function):
    coef = np.zeros(n)
    for i in range(n):
        coef[i] = coef_function(i, f)
    return coef


def function_values(f, n, begin, end):
    func = np.zeros(n)
    x = begin
    for i in range(n):
        if x <= end:
            x += 2 * np.pi / n
            func[i] = f(x)
    return func


def fourier_series(t, a_nth, b_nth):
    fourier = np.zeros(len(t))
    for i in range(len(t)):
        fourier[i] = a_nth[0] / 2
        for j in range(1, len(a_nth)):
            fourier[i] += a_nth[j] * np.cos(j * w * t[i]) + b_nth[j] * np.sin(j * w * t[i])
    return fourier


def func(t):
    return 1 if t < T / 2 else 0 #+ (np.random.rand())/2


def main():
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(13)

    plt.subplot(1, 3, 1)
    a = coefs(10, func, a_n)
    b = coefs(10, func, b_n)
    plt.xlabel('n')
    plt.ylabel('a(n), b(n)')
    plt.plot(a, ".", label='a(n)')
    plt.plot(b, ".", label='b(n)')
    plt.legend()

    plt.subplot(1, 3, 2)
    meandr = function_values(func, 100, 0, 2 * np.pi)
    fourier = fourier_series(np.arange(0, 2 * np.pi, 2 * np.pi / 100), a, b)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.plot(meandr, label='function')
    plt.plot(fourier, '.', label='fourier')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.xlabel('t')
    plt.ylabel('delta(t)')
    plt.plot(np.abs(meandr - fourier), label='delta')
    plt.legend()

    plt.show()
    return 0


main()
