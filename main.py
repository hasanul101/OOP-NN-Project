from core.tensor import Tensor

# dataset: y = 2x + 1
xs = [1.0, 2.0, 3.0, 4.0]
ys = [3.0, 5.0, 7.0, 9.0]

# parameters (learnable)
w = Tensor(0.0, True)
b = Tensor(0.0, True)

lr = 0.01

for epoch in range(50):

    total_loss = 0

    for x_val, y_val in zip(xs, ys):

        x = Tensor(x_val)
        y = Tensor(y_val)

        # forward pass
        pred = w * x + b
        loss = (pred + (-y)) * (pred + (-y))

        # backward pass
        loss.backward()

        total_loss += loss.data

        # update parameters
        w.data -= lr * w.grad
        b.data -= lr * b.grad

        # reset gradients
        w.grad = 0
        b.grad = 0

    print(f"epoch {epoch}, loss {total_loss}")
