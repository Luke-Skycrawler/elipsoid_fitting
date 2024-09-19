import taichi as ti 
import math


nk = 5
N = 32
dx = 1.0 / N
max_epoch = 100

@ti.data_oriented
class Membrane:
    def __init__(self, N = N):
        self.N = N
        self.pixels = ti.field(dtype = float, shape = (self.N, self.N))
        
    @ti.kernel
    def init(self):
        for I in ti.grouped(ti.ndrange(self.N, self.N)):
            r = 1.0
            for k in ti.static(range(2)):
                r *= ti.cos(nk * math.pi * I[k] / self.N)

            self.pixels[I] = r

    def run(self):
        self.init()
    
@ti.data_oriented
class ElipsoidFitter:
    def __init__(self, m = nk * nk * 2):
        self.m = m
        self.G = ti.Matrix.field(3, 3, dtype = float, shape = m, needs_grad= True)
        self.x = ti.Vector.field(3, dtype = float, shape = m, needs_grad = True)

        # self.nabla_G = ti.Matrix.field(3, 3, dtype = float, shape = m, needs_grad = True)
        # self.nabla_x = ti.Vector.field(3, dtype = float, shape = m, needs_grad = True)


        self.alpha = 1e-3

        self.loss = ti.field(dtype = float, shape = (), needs_grad= True)
        self.phi = ti.field(dtype = float, shape = (N, N), needs_grad = True)
        self.init()
        print("init done")

    
    @ti.kernel
    def reduce(self, q: ti.template()):
        for I in ti.grouped(q):
            for i in range(self.m):
                self.loss[None] += self.x[i].norm_sqr()

    @ti.func
    def phi_j(self, x, j, q):
        # bulging displacment
        # G = self.G[j]
        # xj = self.x[j]

        # return ti.min(0.0, q * (ti.math.length(self.G[j] @ (x - self.x[j])) - 1.0))
        Gx = self.G[j] @ (x - self.x[j])
        return ti.min(0.0, q * (Gx.norm() - 1.0))
        # return ti.min(0.0, q * (self.x[j].norm_sqr() - 1.0))
        
    @ti.func
    def _phi(self, xi, qi, q): 
        phi = qi

        Qi = xi + ti.Vector.unit(3, 2, float) * qi
        for j in range(self.m):
            # rij = Qi - self.x[j]
            phi += self.phi_j(Qi, j, q)

        return phi
        
    @ti.kernel
    def phi_sqr(self, q: ti.template()):
        # for I in ti.grouped(q):
        for I in ti.grouped(ti.ndrange(N, N)):
            xi = ti.Vector([I[0], I[1], 0], int) * dx
            # self.loss[None] += (self._phi(xi, q[I], 1.0)) ** 2
            self.phi[I] = q[I]

            for j in range(self.m):
                # Qi = xi + ti.Vector.unit(3, 2, float) * q[I]
                # Gx = self.G[j] @ (Qi - self.x[j])
                # Gx = self.x[j]
                # self.phi[I] += ti.min(0.0, 1.0 * (self.G[j] @ (ti.Vector([I[0], I[1], 0], int) * dx + ti.Vector.unit(3, 2, float) * q[I] - self.x[j])).norm() - 1.0)
                self.phi[I] += self.x[j].norm_sqr()
                # self.phi[I] += 1.0
                # phi += ti.min(0.0, 1.0 * (Gx.norm_sqr() - 1.0))

            # self.loss[None] += self.phi[I] ** 2

            # self.loss[None] += self.foo()
            # self.loss[None] += self.x[0].norm_sqr()
            # self.loss[None] += self.x[0].norm_sqr()
            mat = ti.Matrix.diag(3, 2.0)
            self.loss[None] += (mat @ self.x[0]).norm_sqr() * self.phi[I]


    @ti.func
    def foo(self):
        l = 0.0
        for i in range(self.m):
            l += self.x[i].norm_sqr()
        return l
        


    @ti.kernel
    def init(self):
        for i in self.x:
            ii = i // nk
            jj = i % nk
            self.x[i] = ti.Vector([ii * dx, jj * dx, 0.0])
            self.G[i] = ti.Matrix.identity(float, 3)

            
    # @ti.kernel
    # def compute_grad(self, xi: ti.template(), qi: ti.template()):
    #     for i in self.x:
    #         self.nabla_G[i] = ti.Matrix.zero(float, 3, 3)
    #         self.nabla_x[i] = ti.Vector.zero(float, 3)
        
    #     for j in self.x:
    #         for i in 



    @ti.kernel    
    def update(self):
        for i in self.x:
            # self.G[i] -= self.alpha * self.nabla_G[i]
            # self.x[i] -= self.alpha * self.nabla_x[i]
            self.G[i] -= self.alpha * self.G.grad[i]
            self.x[i] -= self.alpha * self.x.grad[i]

    def fit(self, q):
        print("fitting")
        for epoch in range(max_epoch):
            print(f"epoch #{epoch}")
            self.loss.fill(0.0)
            self.loss.grad.fill(0.0)
            self.G.grad.fill(0.0)
            self.x.grad.fill(0.0)
            with ti.ad.Tape(loss = self.loss):
                self.phi_sqr(q)
                # self.reduce(q)

            print(f"grad got")
            print(f"epoch {epoch} loss = {self.loss[None]}")
            self.update()

if __name__ == '__main__':
    ti.init(arch = ti.cpu)
    membrane = Membrane()
    membrane.run()
    # ti.tools.imshow(membrane.pixels)
    elipsoids = ElipsoidFitter()
    elipsoids.fit(membrane.pixels)
