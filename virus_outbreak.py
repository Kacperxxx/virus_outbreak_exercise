import numpy as np
import matplotlib.pyplot as plt

S = 1000
I = 1
R = 0
beta = 0.5
gamma = 0.1
T = 200
dt = 1
N = S + I + R


def f_S(S, I):
    return -(beta * I * S) / N


def f_R(I):
    return gamma * I


def f_I(S, I):
    return (beta * I * S) / N - gamma * I


def ode_FE(f_S, f_R, f_I, dt, T):
    steps = int(round(float(T) / dt))
    x_S = np.zeros(steps + 1)
    x_R = np.zeros(steps + 1)
    x_I = np.zeros(steps + 1)
    t = np.linspace(0, steps * dt, len(x_S))
    x_S[0] = S
    x_R[0] = R
    x_I[0] = I
    for n in range(steps):
        x_S[n + 1] = x_S[n] + dt * f_S(x_S[n], x_I[n])
        x_R[n + 1] = x_R[n] + dt * f_R(x_I[n])
        x_I[n + 1] = x_I[n] + dt * f_I(x_S[n], x_I[n])
    return x_S, x_R, x_I, t


def main():
    s, r, i, t = ode_FE(f_S, f_R, f_I, dt, T)

    plt.plot(t, s, label='s')
    plt.plot(t, r, label='r')
    plt.plot(t, i, label='i')

    plt.grid(color='k', linewidth=0.2, linestyle='--')
    plt.legend(loc='center right')

    plt.show()


if __name__ == '__main__':
    main()

# gamma is recovery rate taking into account the passage of time of the disease
# beta is infectivity rate