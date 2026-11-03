"""
Tiny scalar automatic-differentiation engine (Lab 5, tasks 1-3).

Every arithmetic operation on `Value` objects creates a new node and records
its parents plus a closure (`_backward`) that knows the local derivative rule.
Calling `.backward()` on the output topologically sorts the graph and runs the
local rules in reverse order, accumulating gradients with `+=`.
"""
import math


class Value:
    """A node in the computation graph holding a scalar.

    Attributes:
        data      -- the scalar value computed in the forward pass
        grad      -- d(output)/d(this node), filled in by backward()
        _prev     -- parent nodes (the inputs of the op that produced this node)
        _op       -- name of the op that produced this node (for debugging/printing)
        _backward -- local derivative rule: pushes self.grad into the parents' grads
    """

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None  # leaves have no local rule
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self._op or 'leaf'})"

    # ------------------------------------------------------------------ core ops
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only constant int/float exponents"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # d relu(x)/dx = 1 if x > 0 else 0 (0 at the kink, same as PyTorch)
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------- extra ops needed for task 5 (BCE)
    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        # numerically stable forward
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------- composite ops built from the core
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ------------------------------------------------------------------ backward
    def topo_order(self):
        """Iterative DFS post-order: every node appears after all of its parents."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        return order

    def backward(self):
        """Reverse-mode autodiff from this (scalar output) node."""
        order = self.topo_order()   # 1. topological sort
        self.grad = 1.0             # 2. seed d(out)/d(out) = 1
        for node in reversed(order):  # 3. visit nodes in reverse topological order
            node._backward()          # 4. local rules use += -> gradients accumulate


def vsum(values):
    """Sum a list of Values into one graph node chain (like tensor.sum())."""
    total = Value(0.0)
    for v in values:
        total = total + v
    return total


def zero_grad(params):
    for p in params:
        p.grad = 0.0
