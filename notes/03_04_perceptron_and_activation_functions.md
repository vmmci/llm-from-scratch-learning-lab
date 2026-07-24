# 03-04 - Perceptron and Activation Functions

## Goal

Understand the basic neuron computation and how activation functions change its behavior.

This unit combines:

- 03 - Perceptron
- 04 - Activation Functions

## Part 03 - Perceptron

A perceptron is the simplest computational unit in a neural network.

It takes an input, multiplies it by a weight, adds a bias, and produces an output.

```text
f(x) = wx + b

 Where:

x = input
w = weight
b = bias
Why the Perceptron Matters

The perceptron shows how a model can store simple behavior using learnable values.

The weight controls how strongly the input affects the output.

The bias shifts the output up or down.

Perceptron Example
weight = 1
input = 2
bias = 3

f(x) = 1 * 2 + 3 = 5
What I Implemented
A simple wx + b function.
A perceptron class.
Calling the perceptron with multiple inputs.
Visualizing the output as a linear function.
Part 04 - Activation Functions

The perceptron output is linear.

Activation functions add non-linearity after the linear computation.

output = activation(wx + b)

Without activation functions, neural networks would struggle to model non-linear patterns.

Why Activation Functions Matter

Activation functions help neural networks model:

image edges
XOR-like logic
sentiment intensity
conditional behavior
feature composition
ReLU
ReLU(x) = max(0, x)

ReLU keeps positive values and turns negative values into zero.

Example:

ReLU(5) = 5
ReLU(-3) = 0
ReLU²
ReLU²(x) = max(0, x)^2

ReLU² keeps only positive values, then squares them.

Example:

ReLU²(2) = 4
ReLU²(-2) = 0
Other Activation Functions Covered
Leaky ReLU
GeLU
Tanh
Sigmoid
Swish
SwiGLU
Combined ReLU
Important Observation

Different activation functions behave similarly around some positive values, but they diverge strongly around negative values and near zero.

This region is important because it decides whether a signal is blocked, weakened, preserved, or amplified.

My Understanding

The perceptron creates a linear score.

The activation function decides how that score should pass forward.

Together, they form the basic neuron behavior used in larger neural networks.

Reference

Based on Justin Angel's "Building LLMs to Develop Intuition" workshop.
