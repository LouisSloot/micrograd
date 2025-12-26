import math
import numpy as np

class Value:
    
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data
        self.grad = 0 # stores dLoss/dself
        self._backward = lambda: None # node-specific function that determines how to propogate gradient from parent to children
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __radd__(self, other): # called when given other + self
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other): # called when given other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**(-1)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (other * -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data**(other-1))) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad 
        out._backward = _backward
            
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad # out.data == math.exp(x)
        out._backward = _backward

        return out

    def backward(self):
        # calculate d_self/d_ancestors for any node self
        topo_graph = []
        visited = set()

        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child)
                topo_graph.append(v)
        topo_sort(self)

        self.grad = 1.0
        for node in reversed(topo_graph):
            node._backward()