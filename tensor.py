class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = float(data)
        self.requires_grad = requires_grad
        self.grad = 0.0

        # graph
        self._prev = []
        self._backward = lambda: None

    # --------------------
    # ADDITION
    # --------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._prev = [self, other]
        out._backward = _backward
        return out

    # --------------------
    # MULTIPLICATION
    # --------------------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._prev = [self, other]
        out._backward = _backward
        return out

    # --------------------
    # NEGATION (THIS FIXES -y)
    # --------------------
    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._prev = [self]
        out._backward = _backward
        return out

    # --------------------
    # BACKWARD PASS
    # --------------------
    def backward(self):
        self.grad = 1.0

        visited = set()
        order = []

        def topo(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for p in node._prev:
                topo(p)
            order.append(node)

        topo(self)

        for node in reversed(order):
            node._backward()

    # --------------------
    # DEBUG PRINT
    # --------------------
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
