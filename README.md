# OOP-NN-Project

This project implements a basic Tensor class with autograd functionality.

### Current features:
- Scalar tensor operations
- Gradient tracking
- Backward propagation
- Linear function demonstration

### Next steps:
- Extend to vector operations
- Build simple neural network components

Evaluation:
The model was tested on a simple linear dataset (y = 2x + 1).
The loss decreases over epochs, and parameters converge to expected values.

- Final result:
w ≈ 2.0
b ≈ 1.0
Loss decreases over epochs → model converges

The design is centered around a Tensor class that encapsulates both data and gradient information. This allows implementing automatic differentiation through backpropagation. Operations such as addition and multiplication construct a computational graph implicitly.
