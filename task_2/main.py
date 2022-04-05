import numpy as np
import matplotlib.pyplot as mpl

precision = 0.1
a = -10
b = 10
interval = np.arange(a, b, precision)
N = 2
wc = 3
w_0 = 1
w_1 = 20
board_1 = 3*w_0
board_2 = 6*w_0

def butterworth_function(w):
    f = 1
    for i in range(N//2):
        f *= ((w / wc) ** 2 - 2 * 1j * np.sin(np.pi * (2 * i + 1) / (2 * N)) * (w / wc) - 1) * (-1)
    H = 1 / f
    if N%2 != 0:
        H = H/(1j*w - 1)
    return H

def butterworth_function_high(w):
    f = 1
    for i in range(N//2):
        f *= ((1 / w) ** 2 - 2 * 1j * np.sin(np.pi * (2 * i + 1) / (2 * N)) * (1 / w) - 1) * (-1)
    H = 1 / f
    if N%2 != 0:
        H = H/(1j*w/wc - 1)
    return H

def stripe_function(w):
    f = 1
    for i in range(N // 2):
        f *= (((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w)) ** 2 - 2 * 1j * np.sin(np.pi * (2 * i + 1) / (2 * N)) * ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w))) * (-1)
    H = 1 / f
    if N % 2 != 0:
        H = H / (1j * ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w)) - 1)
    return H

def reject_function(w):
    f = 1
    for i in range(N // 2):
        f *= ((1 / ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w))) ** 2 - 2 * 1j * np.sin(np.pi * (2 * i + 1) / (2 * N)) * (1 / ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w))) - 1) * (-1)
    H = 1 / f
    if N % 2 != 0:
        H = H / (1j * (1 / ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w))) - 1)
    return H

def H_butterworth():
    return np.vectorize(butterworth_function)(interval)

def H_butterworth_high():
    return np.vectorize(butterworth_function_high)(interval)

def H_stripe():
    return np.vectorize(stripe_function)(interval)

def H_reject():
    return np.vectorize(reject_function)(interval)

def H_butterworth_2():
    return np.vectorize(lambda w: 1 / ((w/wc) ** 2 + 1.41421 * w/wc + 1))(interval)

def H_butterworth_2_high():
    return np.vectorize(lambda w: 1 / ((wc/w) ** 2 + 1.41421 * wc/w + 1))(interval)

def H_stripe_2():
    return np.vectorize(lambda w: 1 / (((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w)) ** 2 + 1.41421 * ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w)) + 1))(interval)

def H_reject_2():
    return np.vectorize(lambda w: 1 / ((1 / ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w))) ** 2 + 1.41421 * (1 / ((w ** 2 + board_1*board_2) / ((board_1 - board_2) * w))) + 1))(interval)

def func_to_arr(func):
    return np.vectorize(func)(interval)

def fft_butterworth_filter(signal, filter_func):
    a, b = np.split(np.fft.fft(signal), 2)
    f = np.concatenate([b, a])
    f = f * filter_func()
    a, b = np.split(f, 2)
    return np.concatenate([b, a])

def filter(signal, filter_func):
    return np.fft.ifft(fft_butterworth_filter(signal, filter_func))


def filtrate_func(f, h_filter, h_filter_2):
    filtered_1 = filter(func_to_arr(f), h_filter)
    filtered_2 = filter(func_to_arr(f), h_filter_2)

    fig = mpl.figure()
    fig.set_figheight(10)
    fig.set_figwidth(8)

    mpl.subplot(5, 1, 1)
    mpl.plot(np.vectorize(lambda w: 1 / (1 + (w / wc) ** (2 * N)))(interval))
    mpl.title("filter")

    mpl.subplot(5, 1, 2)
    mpl.plot(func_to_arr(f))
    mpl.title("Signal")

    mpl.subplot(5, 1, 3)
    mpl.plot(np.abs(np.fft.fft(func_to_arr(f))))
    mpl.title("Spectrum")

    mpl.subplot(5, 1, 4)
    mpl.plot(np.abs(fft_butterworth_filter(func_to_arr(f), h_filter)))
    # mpl.plot(np.abs(fft_butterworth_filter(func_to_arr(f), h_filter_2)))
    mpl.title("Spectrum after filtration")

    mpl.subplot(5, 1, 5)
    mpl.plot(filtered_1)
    # mpl.plot(filtered_2)
    mpl.title("Signal after filtration")

    mpl.subplots_adjust(hspace=1.0)
    mpl.show()


def harmonic(a, b):
    return lambda x: a * np.cos(x * w_0) + b * np.cos(x * w_1)


#filtrate_func(harmonic(1,1), H_butterworth, H_butterworth_2)
#filtrate_func(harmonic(1,1), H_butterworth_high, H_butterworth_2_high)
#filtrate_func(harmonic(1,1), H_stripe, H_stripe_2)
filtrate_func(harmonic(1,1), H_reject, H_reject_2)