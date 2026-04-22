# OOP-NN-Project

## Description
This project implements a simplified Tensor engine with automatic differentiation using object-oriented programming principles, inspired by deep learning frameworks like PyTorch.

The core component is a custom `Tensor` class that supports forward operations and backward propagation through a dynamically constructed computation graph.

---

## Features
- Scalar tensor operations (addition, multiplication, negation)
- Automatic gradient tracking (autograd)
- Backward propagation using chain rule
- Manual gradient descent optimization
- Linear regression training example
- ReLU activation function (extension)
- Training loss visualization

---

## Class Diagram
![Class Diagram](class_diagram.png)

---

## Training Loss Curve
![Loss Curve](loss_curve.png)

---

## Evaluation

The model was trained on a simple linear dataset:

y = 2x + 1

Results:
- The loss decreases consistently over epochs
- The model converges to expected parameters:

w ≈ 2.0  
b ≈ 1.0  

This demonstrates correct gradient computation and parameter optimization.

---

## Design Overview

The design is centered around a `Tensor` class that encapsulates:
- data (value)
- gradient
- computation graph connections

Each operation constructs part of a computation graph, and gradients are computed using backpropagation via the `backward()` method.

---

## ReLU Extension

A ReLU (Rectified Linear Unit) activation function was implemented as part of the Tensor class to demonstrate support for non-linear operations.

ReLU was tested separately and not integrated into the training pipeline, as the primary objective was to model a linear regression task.

---

## Key Concepts Demonstrated

- Automatic differentiation
- Computational graph construction
- Chain rule for gradient computation
- Gradient descent optimization
