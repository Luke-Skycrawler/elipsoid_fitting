import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import warp as wp
res = 10
alpha = 1e-6
epochs = 1000
n, m = 3, 5

sigma0 = np.array([[5e-3, 0], [0, 1e-3]])
sigma1 = sigma0
x0 = np.array([0.4, 0.5])
x1 = np.array([0.6, 0.5])
x2 = np.array([0.8, 0.5])
x3 = np.array([0.2, 0.5])


@wp.func
def gaussian_wp(sigma: wp.mat33, x: wp.vec3, x0: wp.vec3) -> float:
    dx = x - x0
    return wp.exp(- wp.dot(dx, wp.inverse(sigma) @ dx))




@wp.func 
def phi0_wp(x: wp.vec3) -> float:
    return x[2]


@wp.func 
def phi_G_wp(x: wp.vec3, q: float, sigma: wp.array(dtype = wp.mat33), xbar: wp.array(dtype = wp.vec3), k: wp.array(dtype = float)) -> float:    
    phi = float(phi0_wp(x))
    for i in range(sigma.shape[0]):
        # sgn = wp.select(i %  2== 0, 1.0, -1.0)
        phi += q * gaussian_wp(sigma[i], x, xbar[i]) * k[i]
    return phi




    

class Ellipsoid3D: 
    def __init__(self, ng = m * n, q = None, alpha = alpha):
        self.G = wp.zeros((ng), dtype = wp.mat33, requires_grad= True)
        self.x = wp.zeros((ng), dtype = wp.vec3, requires_grad = True)
        self.k = wp.zeros((ng), dtype = float, requires_grad= True)
        # self.k = wp.ones((ng), dtype = float, requires_grad= True)
        self.z = wp.zeros((res, res), dtype = float)
        self.loss = wp.zeros((1), float, requires_grad = True)
        self.m = ng
        self.q = q
        wp.launch(init, (ng), inputs = [self.G, self.x])
        self.alpha = alpha

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
        if epoch % 50 == 0:
            self.phi_q(1.0)
            img.set_array(self.z.numpy().T)
        # return self.z.numpy().T
        return [img]
        
    def forward(self):
        wp.launch(loss, (res), inputs = [self.G, self.x, self.q, self.k, self.loss])

    def update(self):
       wp.launch(gradient_descend, (self.m), inputs = [self.G, self.x, self.k, self.G.grad, self.x.grad, self.k.grad, self.alpha]) 

@wp.kernel
def gradient_descend(G: wp.array(dtype = wp.mat33), x: wp.array(dtype = wp.vec3), k: wp.array(dtype = float), G_grad: wp.array(dtype = wp.mat33), x_grad: wp.array(dtype = wp.vec3), k_grad: wp.array(dtype = float), alpha: float):
    i = wp.tid()
    G[i] -= alpha * 0.5 * (G_grad[i] + wp.transpose(G_grad[i]))
    x[i] -= alpha * x_grad[i]
    k[i] -= alpha * k_grad[i]

@wp.kernel
def loss(G: wp.array(dtype = wp.mat33), x: wp.array(dtype =wp.vec3), q_target: wp.array2d(dtype = float), k: wp.array(dtype = float), l: wp.array(dtype = float)):
    i, j = wp.tid()
    xi = wp.vec3(float(i) / float(res), float(j) / float(res), q_target[i, j])
    phii = phi_G_wp(xi, 1.0, G, x, k)
    wp.atomic_add(l, 0, phii ** 2.0)

@wp.kernel
def phi_q_(G: wp.array(dtype = wp.mat33), x: wp.array(dtype = wp.vec3), q: float, z: wp.array2d(dtype = float), k: wp.array(dtype = float)):
    i, j = wp.tid()
    xi = wp.vec3(float(i) / float(res), float(j) / float(res), 0.0)
    z[i, j] = phi_G_wp(xi, q, G, x, k)

@wp.kernel
def init(G: wp.array(dtype = wp.mat33), x: wp.array(dtype =wp.vec3)):
    i = wp.tid()
    ii = i % m
    jj = i // m

    G[i] = wp.diag(wp.vec3(3e-3, 5e-3, 1e-3))
    x[i] = wp.vec3((float(ii) + 0.5) / float(m), (float(jj) + 0.5) / float(n), 0.0)
    
    

def build_grid():
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    xs = np.outer(np.ones_like(x), x)
    ys = np.outer(y, np.ones_like(y))
        
    plt.subplot()
    deform = lambda x, y: np.sin(m * np.pi * x) * np.sin(n * np.pi * y) * 0.1
    grid_deform = np.vectorize(deform)
    sinxsiny = grid_deform(xs, ys)
    return sinxsiny
    
def fitting_test():
    fig, ax = plt.subplots()

    sinmxsinny = build_grid()
    e2d = Ellipsoid3D(q = wp.from_numpy(sinmxsinny, dtype = float, shape = (res)))

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
        e2d.step(i, 0)
        if i % 50 == 0:
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
    # express_ability_test()
    sinxsiny = build_grid()
    print(sinxsiny)

    np.set_printoptions(precision = 4, linewidth = 2000, suppress = True)
    elipsoids = Ellipsoid3D(q = wp.from_numpy(sinxsiny, dtype = float, shape = (res, res)))

    fig = plt.figure()
    pnp = sinxsiny
    img = plt.imshow(pnp, origin="lower", animated=True, interpolation="antialiased", cmap = 'gray')
    # plt.show()
    
    # z = elipsoids.phi_q(1.0)
    # img.set_array(z)
    # print(z)
    # print(elipsoids.x.numpy())
    # print(elipsoids.G.numpy())
    # plt.show()
    forward = lambda i: elipsoids.step(i, img)
    seq = anim.FuncAnimation(fig, forward, frames = epochs, blit = True, repeat = False)
    plt.show(block = True)
    plt.close()
    # img = plt.imshow(sinxsiny, origin="lower", animated=True, interpolation="antialiased", cmap = 'gray')
    # plt.show()
    # for i in range(epochs):
    #     pnp = elipsoids.step(i, img)
    #     if i % 50 == 0:
    #         img.set_array(pnp)
    #         plt.show()

