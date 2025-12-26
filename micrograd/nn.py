import random
from micrograd.engine import Value

class Module():

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, n_in):
        # initialize (n_in) weights and bias randomly
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # (w dot x) + b computes neuron activation on input vector x
        activation = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, n_in, n_out):
        # create FC layer of (n_out) Neuron objects
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
         

class MLP(Module):
    
    def __init__(self, n_in, n_outs):
        # create MLP with (n_in) inputs, (len(n_outs)-1) hidden layers, and 1 output layer, each with size according to n_outs[i]
        size = [n_in] + n_outs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outs))]

    def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]