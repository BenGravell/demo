import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def cost_value_parametric(a, b, x):
    x1, x2 = x[0], x[1]
    return (a-x1)**2 + b*(x2-x1**2)**2


def cost_gradient_parametric(a, b, x):
    x1, x2 = x[0], x[1]
    return 2*np.array([-(a-x1)-2*b*x1*(x2-x1**2),
                       b*(x2-x1**2)])


def gradient_step_parametric(cost_gradient, step_size, x):
    return x - step_size*cost_gradient(x)


def gradient_descent(cost_gradient, step_size, num_iters, x0):
    n = x0.size
    x_history = np.full([num_iters, n], np.nan)
    x_history[0] = x0
    gradient_step = partial(gradient_step_parametric, cost_gradient, step_size)
    for i in range(num_iters-1):
        x_history[i+1] = gradient_step(x_history[i])
    return x_history


if __name__ == "__main__":
    # Problem definition
    a = 1
    b = 100
    cost_value = partial(cost_value_parametric, a, b)
    cost_gradient = partial(cost_gradient_parametric, a, b)
    x0 = np.array([2, -2])
    step_size = 0.0001
    num_iters = 100000

    # Cost value
    x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    c_grid = cost_value(np.stack([x1_grid, x2_grid]))
    plt.contourf(x1_grid, x2_grid, c_grid**0.25, levels=200)

    # Gradient descent
    x_history = gradient_descent(cost_gradient, step_size, num_iters, x0)
    c_history = cost_value(x_history)
    plt.plot(x_history[:,0], x_history[:,1], color='w', linewidth=4)
    bbox_props = dict(boxstyle='round', facecolor='w', edgecolor='k')
    plt.annotate('start', x0+0.2, bbox=bbox_props)
    plt.annotate('end', x_history[-1]+0.2, bbox=bbox_props)

    # Plot options
    plt.colorbar()
    plt.axis('square')
    plt.show()
