from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        exp_Z = array_api.exp(Z)
        return array_api.log(exp_Z / exp_Z.sum())
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, self.axes, keepdims=True)
        # print(max_Z.shape, Z.shape)
        exp_item = array_api.exp(Z - max_Z).sum(self.axes)
        return array_api.log(exp_item) + max_Z.reshape(exp_item.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        max_a = a.realize_cached_data().max(
            self.axes, keepdims=True
        )  # solve the overflow problem
        exp_a = exp(a - max_a)
        sum_a = exp_a.sum(self.axes)

        shape = [i for i in exp_a.shape]
        axes = self.axes

        if axes is None:
            axes = range(len(exp_a.shape))
        for ax in axes:
            shape[ax] = 1

        sum_a = sum_a.reshape(shape)
        sum_a = sum_a.broadcast_to(a.shape)

        out_grad = out_grad.reshape(shape)
        out_grad = out_grad.broadcast_to(a.shape)
        in_grad = exp_a / sum_a
        return out_grad * in_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
