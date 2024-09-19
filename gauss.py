import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.widgets

def gaussian(sigma, x, x0):
    dx = x - x0
    return np.exp(-dx.T @ np.linalg.inv(sigma) @ dx)

def phi0(x):
    return x[1] - 0.5


sigma0 = np.array([[0.01, 0], [0, 2e-3]])
sigma1 = sigma0

x0 = np.array([0.4, 0.5])
x1 = np.array([0.6, 0.5])
x2 = np.array([0.8, 0.5])
x3 = np.array([0.2, 0.5])
def phi_G(x, q):
    return phi0(x) + q * (gaussian(sigma0, x, x0) - gaussian(sigma1, x, x1) + gaussian(sigma0, x, x2) - gaussian(sigma1, x, x3))

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

xs, ys = np.meshgrid(x, y)

def phi_q(q):
    z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            z[i, j] = phi_G(np.array([xs[i, j], ys[i, j]]), q)


    return z

fig, ax = plt.subplots()
slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="lightgray")  # Slider position
slider = matplotlib.widgets.Slider(slider_ax, 'q', -1.0, 1.0, valinit=0.0)

plt.subplot()
z0 = phi_q(0.0)
plt.contourf(x, y, z0, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')

def update(val):
    z = phi_q(val)
    plt.subplot(1, 1, 1)
    plt.contourf(x, y, z, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')

slider.on_changed(update)
plt.show()

