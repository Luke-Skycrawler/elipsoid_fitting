import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.widgets
import warp as wp
from adam import Adam
res = 100
alpha = 1e-2
inverse = False
epochs = 3000
sigma0 = np.array([[5e-3, 0], [0, 1e-3]])
sigma1 = sigma0
bump = 2e-2
x0 = np.array([0.4, 0.5])
x1 = np.array([0.6, 0.5])
x2 = np.array([0.8, 0.5])
x3 = np.array([0.2, 0.5])
m = 16

@wp.func
def gaussian_wp(sigma: wp.mat22, x: wp.vec2, x0: wp.vec2) -> float:
    dx = x - x0
    sig = wp.select(inverse, sigma, wp.inverse(sigma))
    return wp.exp(- wp.dot(dx, sig @ dx))
    # return wp.exp(- wp.dot(dx, sigma @ dx))

@wp.func 
def phi0_wp(x: wp.vec2) -> float:
    return x[1] - 0.5


@wp.func 
def phi_G_wp(x: wp.vec2, q: float, sigma: wp.array(dtype = wp.mat22), xbar: wp.array(dtype = wp.vec2), k: wp.array(dtype = float)) -> float:    
    phi = float(phi0_wp(x))
    for i in range(sigma.shape[0]):
        sgn = wp.select(i %  2== 0, 1.0, -1.0)
        phi += q * sgn * gaussian_wp(sigma[i], x, xbar[i]) * k[i]
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
    def __init__(self, m = m, q = None, alpha = alpha):
        self.G = wp.zeros((m), dtype = wp.mat22, requires_grad= True)
        self.x = wp.zeros((m), dtype = wp.vec2, requires_grad = True)
        self.k = wp.zeros((m), dtype = float, requires_grad= True)
        self.z = wp.zeros((res, res), dtype = float)
        self.loss = wp.zeros((1), float, requires_grad = True)
        self.m = m
        self.q = q
        wp.launch(init, (m), inputs = [self.G, self.x])
        self.alpha = alpha
        self.optimizer = Adam([self.k, self.G, self.x], lr = alpha)

    def phi_q(self, q):
        wp.launch(phi_q_, (res, res), inputs = [self.G, self.x, q, self.z, self.k])
        return self.z.numpy().T

    def step(self, epoch, img):
        tape = wp.Tape()
        self.loss.zero_()
        tape.zero()
        with tape:
            self.forward()
        tape.backward(loss = self.loss)
        self.update()
        print(f"epoch {epoch}, loss = {self.loss.numpy()}")
        # print(f"grad x = {self.x.grad.numpy()}")

        # z = self.phi_q(1.0)
        # plt.subplot(1, 1, 1)
        # plt.contourf(x, y, z, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')
        # plt.plot(x, self.q.numpy(), 'b')


        return img
        
    def forward(self):
        wp.launch(loss, (res), inputs = [self.G, self.x, self.q, self.k, self.loss])

    def update(self):
       wp.launch(gradient_descend, (self.m), inputs = [self.G, self.x, self.k, self.G.grad, self.x.grad, self.k.grad, self.alpha]) 

    def optimize(self, epoch, img):
        

        tape = wp.Tape()
        self.loss.zero_()
        tape.zero()
        with tape:
            self.forward()
        tape.backward(loss = self.loss)

        self.optimizer.step([self.k.grad, self.G.grad, self.x.grad])
        tape.zero()
        print(f"epoch {epoch}, loss = {self.loss.numpy()}")
        return img

@wp.kernel
def gradient_descend(G: wp.array(dtype = wp.mat22), x: wp.array(dtype = wp.vec2), k: wp.array(dtype = float), G_grad: wp.array(dtype = wp.mat22), x_grad: wp.array(dtype = wp.vec2), k_grad: wp.array(dtype = float), alpha: float):
    i = wp.tid()
    G[i] -= alpha * 0.5 * (G_grad[i] + wp.transpose(G_grad[i]))
    x[i] -= alpha * x_grad[i]
    k[i] -= alpha * k_grad[i]

@wp.kernel
def loss(G: wp.array(dtype = wp.mat22), x: wp.array(dtype =wp.vec2), q_target: wp.array(dtype = float), k: wp.array(dtype = float), l: wp.array(dtype = float)):
    i = wp.tid()
    xi = wp.vec2(float(i) / float(res), q_target[i])
    phii = phi_G_wp(xi, 1.0, G, x, k)
    wp.atomic_add(l, 0, phii ** 2.0)

@wp.kernel
def phi_q_(G: wp.array(dtype = wp.mat22), x: wp.array(dtype = wp.vec2), q: float, z: wp.array2d(dtype = float), k: wp.array(dtype = float)):
    i, j = wp.tid()
    xi = wp.vec2(float(i) / float(res), float(j) / float(res))
    z[i, j] = phi_G_wp(xi, q, G, x, k)

@wp.kernel
def init(G: wp.array(dtype = wp.mat22), x: wp.array(dtype =wp.vec2)):
    i = wp.tid()
    G[i] = wp.select(not inverse, wp.diag(wp.vec2(5e-3, 1e-3)), wp.diag(wp.vec2(200.0, 1000.0)))
    x[i] = wp.vec2((float(i) + 0.5) / float(m), 0.5)

def phi_q(q):
    z = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            z[i, j] = phi_G(np.array([xs[i, j], ys[i, j]]), q)


    return z

def express_ability_test():
    fig, ax = plt.subplots()
    button = matplotlib.widgets.Button(plt.axes([0.1, 0.05, 0.1, 0.075]), 'Optimize')

    plt.subplot()
    z0 = phi_q(0.0)
    plt.contourf(x, y, z0, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')
    siny = np.sin(m * np.pi * y) * bump + 0.5
    e2d = Ellipsoid2D(q = wp.from_numpy(siny, dtype = float, shape = (res)))

    def update(val):
        # z = phi_q(val)
        z = e2d.phi_q(val)
        plt.subplot(1, 1, 1)
        plt.contourf(x, y, z, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')
        plt.plot(x, siny, 'b')

    # epoch = 0
    # def optimize(event):
    #     nonlocal epoch
    #     e2d.step(epoch, 0)
    #     epoch += 1
    #     z = e2d.phi_q(1.0)
    #     plt.subplot(1, 1, 1)
    #     plt.contourf(x, y, z, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')
    #     plt.plot(x, siny, 'b')

    # button.on_clicked(optimize)

    for i in range(epochs):
        # e2d.step(i, 0)
        e2d.optimize(i, 0)
        if i % (epochs // 5) == 0:
            z = e2d.phi_q(1.0)
            plt.subplot(1, 1, 1)
            plt.contourf(x, y, z, levels = np.arange(-0.5, 0.5, 0.1), cmap = 'coolwarm')
            plt.plot(x, siny, 'b')
            plt.show()

    slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="lightgray")  # Slider position
    slider = matplotlib.widgets.Slider(slider_ax, 'q', -1.0, 1.0, valinit=0.0)
    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    express_ability_test()
