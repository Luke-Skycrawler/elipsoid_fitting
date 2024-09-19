import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.widgets
import warp as wp
res = 100
sigma0 = np.array([[5e-3, 0], [0, 1e-3]])
sigma1 = sigma0
x0 = np.array([0.4, 0.5])
x1 = np.array([0.6, 0.5])
x2 = np.array([0.8, 0.5])
x3 = np.array([0.2, 0.5])


@wp.func
def gaussian_wp(sigma: wp.mat22, x: wp.vec2, x0: wp.vec2) -> float:
    dx = x - x0
    return wp.exp(- wp.dot(dx, wp.inverse(sigma) @ dx))

@wp.func 
def phi0_wp(x: wp.vec2) -> float:
    return x[1] - 0.5


@wp.func 
def phi_G_wp(x: wp.vec2, q: float, sigma: wp.array(dtype = wp.mat22), xbar: wp.array(dtype = wp.vec2)) -> float:    
    phi = float(phi0_wp(x))
    for i in range(sigma.shape[0]):
        sgn = wp.select(i %  2== 0, 1.0, -1.0)
        phi += q * sgn * gaussian_wp(sigma[i], x, xbar[i])
    return phi




def gaussian(sigma, x, x0):
    dx = x - x0
    return np.exp(-dx.T @ np.linalg.inv(sigma) @ dx)

def phi0(x):
    return x[1] - 0.5

def phi_G(x, q):
    return phi0(x) + q * (gaussian(sigma0, x, x0) - gaussian(sigma1, x, x1) + gaussian(sigma0, x, x2) - gaussian(sigma1, x, x3))
    

x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)

xs, ys = np.meshgrid(x, y)

class Ellipsoid2D: 
    def __init__(self, m = 4, q = None):
        self.G = wp.zeros((m), dtype = wp.mat22, requires_grad= True)
        self.x = wp.zeros((m), dtype = wp.vec2, requires_grad = True)
        self.z = wp.zeros((res, res), dtype = float)
        wp.launch(init, (m), inputs = [self.G, self.x])

    def phi_q(self, q):
        wp.launch(phi_q_, (res, res), inputs = [self.G, self.x, q], outputs = [self.z])
        return self.z.numpy()

@wp.kernel
def phi_q_(G: wp.array(dtype = wp.mat22), x: wp.array(dtype = wp.vec2), q: float, z: wp.array2d(dtype = float)):
    i, j = wp.tid()
    xi = wp.vec2(float(i) / float(res), float(j) / float(res))
    z[i, j] = phi_G_wp(xi, q, G, x)

@wp.kernel
def init(G: wp.array(dtype = wp.mat22), x: wp.array(dtype =wp.vec2)):
    i = wp.tid()
    G[i] = wp.diag(wp.vec2(5e-3, 1e-3))
    x[i] = wp.vec2(0.2 * float(i + 1), 0.5)

def phi_q(q):
    z = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            z[i, j] = phi_G(np.array([xs[i, j], ys[i, j]]), q)


    return z

def express_ability_test():
    fig, ax = plt.subplots()
    slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="lightgray")  # Slider position
    slider = matplotlib.widgets.Slider(slider_ax, 'q', -1.0, 1.0, valinit=0.0)

    plt.subplot()
    z0 = phi_q(0.0)
    plt.contourf(x, y, z0, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')
    e2d = Ellipsoid2D()
    
    def update(val):
        # z = phi_q(val)
        z = e2d.phi_q(val)
        plt.subplot(1, 1, 1)
        plt.contourf(x, y, z, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    express_ability_test()
