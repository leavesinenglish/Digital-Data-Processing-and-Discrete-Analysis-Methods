import matplotlib.pyplot as plt


def print_solve(x, y, y_approx, name):
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(13)
    plt.title(name)
    plt.plot(x, y, '-g', linewidth=2, label='y_exact')
    plt.plot(x, y_approx, 'or', label='y_approx')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend('l', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 1), loc='best')
    plt.ylim([0, max(y) + 0.1])
    plt.rcParams["figure.dpi"] = 300
    plt.show()
