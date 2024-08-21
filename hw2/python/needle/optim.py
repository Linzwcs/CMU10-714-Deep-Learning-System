"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params: list[ndl.Tensor]):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = param.grad.data
            W = param.data
            grad_with_wd = grad + self.weight_decay * W

            u_0 = self.u.get(param, 0)
            u_1 = self.momentum * u_0 + (1 - self.momentum) * grad_with_wd
            self.u[param] = u_1.data

            W -= self.lr * u_1
            param.data = W
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = param.grad.data
            W = param.data
            grad_with_wd = grad + self.weight_decay * W

            u_0 = self.m.get(param, 0)
            v_0 = self.v.get(param, 0)

            u_1 = self.beta1 * u_0 + (1 - self.beta1) * grad_with_wd
            v_1 = self.beta2 * v_0 + (1 - self.beta2) * grad_with_wd**2
            # assert type(u_1) == ndl.Tensor
            str(u_1), str(v_1)  # there are some bug in unit_test?

            self.m[param] = u_1
            self.v[param] = v_1

            u_1 /= 1 - self.beta1**self.t
            v_1 /= 1 - self.beta2**self.t

            W -= self.lr * (u_1 / (v_1**0.5 + self.eps))
            param.data = W
        ### END YOUR SOLUTION
