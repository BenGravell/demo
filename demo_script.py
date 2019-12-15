import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def cost_value(x, a=1, b=100):
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2


def cost_gradient(x, a=1, b=100):
    return -2*np.array([(a-x[0]) + 4*b*x[0]*(x[1]-x[0]**2), b*(x[1]-x[0]**2)])


def gradient_step(cost_gradient, step_size, x):
    return x - step_size * cost_gradient(x)


def minimize(cost_gradient, x0, step_size=0.02, num_iters=100):
    n = np.array(x0).size
    x_history = np.full([num_iters, n], np.nan)
    t_history = np.arange(num_iters)
    x_history[0] = x0
    gradient_step_partial = partial(gradient_step, cost_gradient, step_size)
    for i in range(num_iters-1):
        x_history[i+1] = gradient_step_partial(x_history[i])
    return t_history, x_history


if __name__ == "__main__":
    x0 = np.array([-2, 2])
    t_history, x_history = minimize(cost_gradient, x0)
    c_history = cost_value(x_history)
    x_cost_sample = np.linspace(-4.5, 4.5, 1000)
    c_cost_sample = cost_value(x_cost_sample)
    plt.plot(x_cost_sample, c_cost_sample)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=t_history.min(), vmax=t_history.max())
    plt.scatter(x_history, c_history, s=10, c=cmap(norm(t_history)))
    plt.colorbar()
    plt.show()
