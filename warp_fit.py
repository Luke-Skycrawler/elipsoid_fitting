import warp as wp
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt
nk = 5
N = 32
dx = 1.0 / N
max_epoch = 500
alpha = 2e-5
m = (nk - 0) ** 2
zscale = 0.05
class Membrane:
    def __init__(self, N = N):
        self.N = N
        self.pixels = wp.zeros((self.N, self.N))
        

    def run(self):
        wp.launch(init, (N, N), inputs = [self.pixels])


@wp.kernel
def init(pixels: wp.array2d(dtype = float)):
    i, j = wp.tid()

    r = zscale * wp.cos(float(nk) * wp.pi * float(i) / float(N)) * wp.cos(float(nk) * wp.pi * float(j) / float(N))

    pixels[i, j] = r


class ElipsoidFitter:   
    def __init__(self, q, m = m):
        self.m = m
        self.G = wp.zeros((m), dtype = wp.mat33, requires_grad= True)
        self.x = wp.zeros((m), dtype = wp.vec3, requires_grad = True)
        self.loss = wp.zeros((1), requires_grad= True)
        self.canvas = wp.zeros((N, N), dtype = float)

        self.q = q
        wp.launch(init_Gx, (m), inputs = [self.G, self.x])

    def fit(self, q):

        print("init")
        print(f"x = {self.x.numpy()[0]}")
        print(f"G = {self.G.numpy()[0]}")
        for epoch in range(max_epoch):
            self.step()

        print("fit done")
        print(f"x = {self.x.numpy()[12]}")
        print(f"G = {self.G.numpy()[12]}")

    def step(self, epoch, img):
        tape = wp.Tape()

        self.loss.zero_()
        tape.zero()
        with tape:
            self.forward(self.q)

        if epoch % 10 == 0:
            print(f"epoch {epoch} loss: {self.loss.numpy()}")
            print(f"x = {self.x.numpy()[12]}")
            print(f"G = {self.G.numpy()[12]}")

        tape.backward(loss = self.loss)
        self.update()
        self.draw(img)

        return [img]
        


    def draw(self, img):
        canvas = self.canvas
        wp.launch(draw, (N, N), inputs = [elipsoids.G, elipsoids.x, canvas])
        img.set_array(-canvas.numpy() / zscale)


    def update(self):
        wp.launch(gradient_descend, (m), inputs = [self.G, self.x, self.G.grad, self.x.grad, alpha])


    def forward(self, q):
        wp.launch(loss, (N, N), inputs = [self.G, self.x, q, self.loss])

@wp.kernel
def gradient_descend(G: wp.array(dtype = wp.mat33), x: wp.array(dtype = wp.vec3), G_grad: wp.array(dtype = wp.mat33), x_grad: wp.array(dtype = wp.vec3), alpha: float):
    i = wp.tid()
    G[i] -= alpha * 0.5 * (G_grad[i] + wp.transpose(G_grad[i]))
    x[i] -= alpha * x_grad[i]

@wp.kernel
def init_Gx(G: wp.array(dtype = wp.mat33), x: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    ii = i // nk
    jj = i % nk
    fnk = float(nk) * 2.0
    G[i] = wp.diag(wp.vec3(fnk, fnk, 1.0 / zscale))
    x[i] = wp.vec3(float(ii) / float(nk), float(jj) / float(nk), 0.0)


@wp.func 
def phi_j(G: wp.array(dtype = wp.mat33), x_bar: wp.array(dtype = wp.vec3), x: wp.vec3, j: int, q: float):
    Gx = G[j] @ (x - x_bar[j])

    return q * wp.min(0.0, (wp.length(Gx) - 1.0))

@wp.func
def phi_at_q(xi: wp.vec3, qi: float, q: float, G: wp.array(dtype = wp.mat33), x_bar: wp.array(dtype = wp.vec3)) -> float:
    phi = qi
    Qi = xi + wp.vec3(0.0, 0.0, qi)
    for j in range(m):
        phi += phi_j(G, x_bar, Qi, j, q)
    return phi

@wp.kernel
def loss(G: wp.array(dtype = wp.mat33), x: wp.array(dtype = wp.vec3), q: wp.array2d(dtype = float), l: wp.array(dtype = float)):
    i, j = wp.tid()
    xi = wp.vec3(float(i) * dx, float(j) * dx, 0.0)
    phii = q[i, j] + phi_at_q(xi, q[i, j], 1.0, G, x)
    phii_ = -q[i, j] - phi_at_q(xi, -q[i, j], -1.0, G, x)

    wp.atomic_add(l, 0, phii ** 2.0 + phii_ ** 2.0)
    
@wp.kernel
def draw(G: wp.array(dtype = wp.mat33), x: wp.array(dtype = wp.vec3), canvas: wp.array2d(dtype = float)):

    i, j = wp.tid()
    xi = wp.vec3(float(i) * dx, float(j) * dx, 0.0)
    phii = phi_at_q(xi, 0.0, 1.0, G, x)
    canvas[i, j] = phii
        
if __name__ == "__main__":
    np.set_printoptions(precision = 4, linewidth = 2000, suppress = True)
    membrane = Membrane()
    membrane.run()
    elipsoids = ElipsoidFitter(membrane.pixels)

    fig = plt.figure()
    
    # elipsoids.fit(membrane.pixels)


    pnp = membrane.pixels.numpy()
    img = plt.imshow(pnp, origin="lower", animated=True, interpolation="antialiased", cmap = 'gray')
    forward = lambda i: elipsoids.step(i, img)
    seq = anim.FuncAnimation(fig, forward, frames = max_epoch, blit = True, repeat = False)
    plt.show(block = True)
    plt.close()

