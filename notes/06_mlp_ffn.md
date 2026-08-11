# 06 — MLP / Feed-Forward Networks

## Overview

This unit extends the idea of a single perceptron into a multi-layer neural
network.

The main progression is:

```text
Single-input perceptron
        ↓
Multi-input perceptron
        ↓
Multiple neurons
        ↓
Hidden layers
        ↓
Multi-Layer Perceptron (MLP)
        ↓
Feed-Forward Network (FFN)
        ↓
FFN blocks used inside larger neural architectures
```

The unit also connects neural networks to linear algebra.

Instead of thinking about every neuron as an isolated calculation, entire
layers can be represented using vectors and matrices.

---

# From a Single-Input Perceptron to Multiple Inputs

In the earlier perceptron unit, a simple neuron was represented as:

$$
y = wx + b
$$

where:

- $x$ is the input
- $w$ is the weight
- $b$ is the bias

With multiple inputs, each input has its own weight.

The neuron computes a weighted sum:

$$
f(x)
=
\sum_{i=1}^{n} w_i x_i + b
$$

For example:

$$
f(x)
=
w_1x_1
+
w_2x_2
+
w_3x_3
+
b
$$

The important idea is that the neuron first combines information from all of
its inputs.

Conceptually:

```text
x1 ──× w1 ──┐
             │
x2 ──× w2 ──┼── sum ── + bias ── output
             │
x3 ──× w3 ──┘
```

Each input contributes to the final result according to its weight.

---

## Vector Form

The same computation can be written more compactly using vectors.

If:

$$
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

and:

$$
\mathbf{w}
=
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
$$

then the neuron can be written as:

$$
y
=
\mathbf{w}^{T}\mathbf{x}
+
b
$$

This is the first major connection between neural networks and linear algebra.

A neuron is not only a diagram of connected circles.

Its core computation is a dot product followed by a bias and, usually, an
activation function.

---

# From One Neuron to a Layer

A neural-network layer contains multiple neurons.

Each neuron receives the outputs of the previous layer and applies its own
weights.

Instead of computing one weighted sum, a layer computes many weighted sums at
once.

For one neuron:

$$
y
=
\mathbf{w}^{T}\mathbf{x}
+
b
$$

For an entire layer:

$$
\mathbf{y}
=
W\mathbf{x}
+
\mathbf{b}
$$

where:

- $\mathbf{x}$ is the input vector
- $W$ is the weight matrix
- $\mathbf{b}$ is the bias vector
- $\mathbf{y}$ is the output vector

This matrix form is much more useful when implementing neural networks.

---

# Multi-Layer Perceptron

A **Multi-Layer Perceptron (MLP)** connects several layers together.

A simple MLP can contain:

```text
Input layer
     ↓
Hidden layer
     ↓
Output layer
```

The output of one layer becomes the input of the next layer.

The naming convention reflects this flow:

- the first layer receives the **input**
- intermediate layers are **hidden layers**
- the final layer produces the **output**

For example:

```text
2 inputs
   ↓
2 hidden neurons
   ↓
1 output
```

can be written as:

```text
2 → 2 → 1
```

---

## Fully Connected Layers

When every neuron in one layer is connected to every neuron in the next layer,
the layer is called **fully connected**.

Conceptually:

```text
Input layer          Hidden layer        Output

    x1 ─────────────── h1 ──────────────── y
      ╲              ╱
       ╲            ╱
        ╲          ╱
    x2 ────────── h2
```

In PyTorch, a fully connected linear transformation is commonly represented
using:

```python
nn.Linear(input_features, output_features)
```

For example:

```python
nn.Linear(2, 2)
```

maps two input features to two output features.

---

# Activation Functions Inside an MLP

If multiple linear layers are stacked without nonlinear activation functions,
the entire network still behaves like a linear transformation.

Activation functions introduce nonlinear behavior between layers.

A common structure is:

$$
\mathbf{h}
=
\sigma(W_1\mathbf{x}+\mathbf{b}_1)
$$

followed by:

$$
\mathbf{y}
=
W_2\mathbf{h}
+
\mathbf{b}_2
$$

where $\sigma$ is an activation function.

In the coding examples, ReLU is used:

```python
x = self.hidden(x)
x = self.relu(x)
x = self.output(x)
```

The data therefore flows through:

```text
Linear transformation
        ↓
ReLU
        ↓
Linear transformation
```

This combines ideas from the earlier perceptron and activation-function units.

---

# Experiment 1 — 2 → 2 → 1 MLP

The first coded MLP contains:

```text
2 inputs
   ↓
2 hidden units
   ↓
ReLU
   ↓
1 output
```

The PyTorch structure is:

```python
self.hidden = nn.Linear(2, 2)
self.relu = nn.ReLU()
self.output = nn.Linear(2, 1)
```

The forward pass is:

```python
def forward(self, x):
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return x
```

This illustrates an important principle:

> A neural network is a sequence of transformations applied to a tensor.

Each layer transforms the representation produced by the previous layer.

---

# Logical Functions with MLPs

The unit also uses small MLPs to implement logical functions.

These examples make it easier to see how weights, biases, hidden units, and
activation functions interact.

The tested logical functions were:

- XOR
- AND
- OR

---

## XOR

The XOR truth table is:

| A | B | XOR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The implementation uses a hidden layer and ReLU before producing the final
output.

Structure:

```text
2 inputs
   ↓
2 hidden neurons
   ↓
ReLU
   ↓
1 output
```

The example demonstrates how multiple neurons and a nonlinear hidden layer can
combine simple intermediate computations into a more complex input-output
mapping.

---

## AND

The AND truth table is:

| A | B | AND |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

The coded model again uses:

```text
2 → 2 → 1
```

with explicitly chosen weights and biases.

The exercise shows that network behavior is determined by the values of its
parameters, not only by the number of layers.

---

## OR

The OR truth table is:

| A | B | OR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

The architecture remains similar, while changing the weights and biases changes
the function represented by the network.

This gives a useful mental model:

```text
Architecture
     +
Weights
     +
Biases
     +
Activation functions
     ↓
Network behavior
```

---

# Linear Algebra Behind Neural Networks

MLPs are naturally expressed using linear algebra.

The unit reviewed:

- one-dimensional tensors
- two-dimensional tensors
- matrix multiplication
- transpose

These are not separate mathematical topics from neural networks.

They are the language used to implement neural-network layers efficiently.

---

## 1D Tensor

Example:

```python
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
```

This can be interpreted as a vector.

Conceptually:

$$
\begin{bmatrix}
1 & 2 & 3 & 4 & 5
\end{bmatrix}
$$

---

## 2D Tensor

Example:

```python
tensor_2d = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Its shape is:

```text
2 × 3
```

and it can be interpreted as the matrix:

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

---

## Transpose

Transposing a matrix swaps its axes.

If:

$$
A
=
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

then:

$$
A^T
=
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
$$

In PyTorch:

```python
transposed_tensor = tensor_2d.T
```

The shape changes from:

```text
2 × 3
```

to:

```text
3 × 2
```

Transpose is important because matrix dimensions must align correctly for
matrix multiplication.

---

# Matrix Multiplication

The code also demonstrates matrix multiplication using:

```python
matrix_a @ matrix_b
```

For matrices:

$$
A_{m\times n}
$$

and:

$$
B_{n\times p}
$$

the multiplication:

$$
AB
$$

produces:

$$
C_{m\times p}
$$

The inner dimensions must match:

```text
(m × n) @ (n × p)
        ↑   ↑
        must match
```

For example:

```text
(2 × 3) @ (3 × 2)
```

produces:

```text
(2 × 2)
```

This dimension rule becomes extremely important when building neural-network
layers.

---

# Neural Layers as Matrix Multiplication

A fully connected layer is fundamentally a matrix operation.

Conceptually:

$$
Y
=
XW
+
b
$$

or, depending on the convention used:

$$
Y
=
WX
+
b
$$

The exact orientation depends on how vectors and batches are represented, but
the core idea remains the same:

> A layer transforms an input representation through learned matrix weights.

This explains why matrix multiplication is one of the dominant computations in
modern neural networks.

---

# Growing and Shrinking Representations

A linear layer can change the dimensionality of a representation.

For example:

```text
768 dimensions
      ↓
3072 dimensions
```

increases the feature dimension.

The reverse transformation:

```text
3072 dimensions
       ↓
768 dimensions
```

reduces it again.

In the unit, these operations are described as **upsampling** and
**downsampling** within the FFN.

Conceptually:

```text
Input representation
        ↓
Expand
        ↓
Larger hidden representation
        ↓
Nonlinear transformation
        ↓
Project back
        ↓
Output representation
```

This grow-and-shrink structure is a central idea in the FFN implementation used
in the exercise.

---

# ReLU² Feed-Forward Network

The main FFN exercise uses:

```text
Input dimension: 768
Upsample ratio:   4×
Hidden dimension: 3072
Output dimension: 768
```

because:

$$
768 \times 4 = 3072
$$

The architecture is therefore:

```text
768
 ↓
Linear
 ↓
3072
 ↓
ReLU
 ↓
Square
 ↓
Linear
 ↓
768
```

or more compactly:

```text
768 → 3072 → ReLU² → 768
```

---

## Mathematical Form

The first linear transformation expands the representation:

$$
\mathbf{h}
=
W_{\text{up}}\mathbf{x}
$$

The nonlinear activation is:

$$
\mathbf{a}
=
\mathrm{ReLU}(\mathbf{h})^2
$$

The second linear transformation projects the representation back:

$$
\mathbf{y}
=
W_{\text{down}}\mathbf{a}
$$

Combined:

$$
\boxed{
\mathbf{y}
=
W_{\text{down}}
\left[
\mathrm{ReLU}
\left(
W_{\text{up}}\mathbf{x}
\right)
\right]^2
}
$$

---

## PyTorch Implementation

The implemented structure is:

```python
self.upsample_weights = nn.Linear(
    input_dimension,
    input_dimension * upsample_ratio,
    bias=False
)

self.downsample_weights = nn.Linear(
    input_dimension * upsample_ratio,
    input_dimension,
    bias=False
)
```

The forward pass is:

```python
x = self.upsample_weights(x)

x = F.relu(x)
x = x.square()

x = self.downsample_weights(x)
```

The important shape flow is:

```text
[768]
   ↓
[3072]
   ↓
ReLU²
   ↓
[768]
```

---

# Why Expand Then Contract?

The FFN does not simply preserve the original representation throughout the
computation.

It first moves the input into a larger intermediate feature space.

Conceptually:

```text
Original representation
          ↓
     expand features
          ↓
nonlinear transformation
          ↓
   project back down
```

The larger hidden dimension gives the network a larger intermediate
representation in which transformations can be performed before returning to
the model dimension.

The activation between the two linear projections is essential because it
introduces nonlinear behavior.

---

# Workshop FFN Design Decisions

The design used in this unit applies the following choices:

```text
Input dimension:   768
Hidden dimension:  3072
Expansion ratio:   4×
Activation:        ReLU²
Bias:              disabled in the FFN exercise
```

The resulting structure is:

```text
Input
 768
  │
  ▼
Linear projection
  │
  ▼
3072
  │
  ▼
ReLU²
  │
  ▼
Linear projection
  │
  ▼
 768
Output
```

This architecture turns the earlier perceptron and activation-function concepts
into a reusable neural-network component.

---

# A Deeper MLP — 5 → 8 → 16 → 1

The final MLP exercise introduces more than one hidden layer.

Architecture:

```text
Input
  5
  ↓
Hidden Layer 1
  8
  ↓
ReLU
  ↓
Hidden Layer 2
 16
  ↓
ReLU
  ↓
Output
  1
```

The PyTorch structure is:

```python
self.layer1 = nn.Linear(5, 8)
self.relu1 = nn.ReLU()

self.layer2 = nn.Linear(8, 16)
self.relu2 = nn.ReLU()

self.output_layer = nn.Linear(16, 1)
```

This demonstrates that an MLP is not restricted to a single hidden layer.

The output of each layer becomes the representation processed by the next
layer.

---

## Shape Flow

For one input vector:

```text
[5]
 ↓
[8]
 ↓
[16]
 ↓
[1]
```

The shape changes because each `nn.Linear` layer defines a new output feature
dimension.

This provides a practical way to read neural-network architectures:

```text
5 → 8 → 16 → 1
```

is not just a list of numbers.

It describes the dimensions through which the representation flows.

---

## Parameter Count

For a linear layer with bias:

$$
\text{parameters}
=
(\text{input features}
\times
\text{output features})
+
\text{output features}
$$

For the `5 → 8 → 16 → 1` network:

First layer:

$$
5\times8+8=48
$$

Second layer:

$$
8\times16+16=144
$$

Output layer:

$$
16\times1+1=17
$$

Total:

$$
48+144+17=209
$$

So this small MLP contains:

$$
\boxed{209\text{ trainable parameters}}
$$

This illustrates how parameter count grows as layers become wider or deeper.

---

# Connecting the Previous Units

The progression across the previous units can now be summarized as:

```text
Unit 03
Perceptron
y = wx + b
        ↓
Unit 04
Activation functions
σ(wx + b)
        ↓
Unit 05
How these computations execute efficiently on hardware
        ↓
Unit 06
Many neurons and transformations organized into an MLP / FFN
```

The ideas are cumulative.

The FFN is constructed from concepts already studied:

- weighted sums
- biases
- activation functions
- tensor operations
- matrix multiplication
- efficient hardware execution

---

# Connection to Large Language Models

The FFN studied here is not an isolated neural-network concept.

Feed-forward blocks are important components of larger language-model
architectures.

A simplified representation is:

```text
Model representation
       ↓
Linear expansion
       ↓
Nonlinear activation
       ↓
Linear projection
       ↓
Updated representation
```

The operation is applied to high-dimensional representations rather than to
simple scalar inputs.

This creates a useful progression:

```text
Perceptron
   ↓
Neuron layer
   ↓
MLP
   ↓
FFN block
   ↓
Large neural architecture
```

Later transformer units will place FFN-style transformations alongside other
components such as attention, normalization, and residual connections.

---

# Key Takeaways

1. A multi-input perceptron computes a weighted sum of its inputs plus a bias.

2. The vector form of a neuron is a dot product.

3. An entire neural-network layer can be represented using matrix
   multiplication.

4. A Multi-Layer Perceptron connects multiple layers sequentially.

5. The outputs of one layer become the inputs to the next layer.

6. Fully connected layers connect each unit in one layer to the units in the
   next layer.

7. Nonlinear activation functions allow stacked layers to represent more than
   a single linear transformation.

8. PyTorch's `nn.Linear` represents the core linear transformation used in an
   MLP.

9. Tensor shapes provide a concise way to understand the structure of a neural
   network.

10. Matrix multiplication and transpose are fundamental operations for
    implementing neural-network layers.

11. FFNs can expand a representation into a larger hidden dimension and then
    project it back to the original dimension.

12. The FFN exercise used a `4×` expansion:

    ```text
    768 → 3072 → 768
    ```

13. ReLU² applies ReLU and then squares the resulting positive activations.

14. The FFN exercise used linear projections without bias.

15. Logical-function exercises demonstrate how different parameters can make
    the same basic architecture represent different mappings.

16. Deeper MLPs are constructed by repeatedly composing linear transformations
    and nonlinear activations.

17. MLPs and FFNs connect the basic neuron concepts from earlier units to the
    larger architectures used later in the LLM roadmap.

---

# Final Mental Model

A useful way to think about this unit is:

```text
A perceptron computes one weighted sum.

A layer computes many weighted sums.

A matrix represents all of those weights together.

An activation makes the transformation nonlinear.

An MLP composes several of these transformations.

An FFN can expand, transform, and project a representation.

Large neural networks repeatedly use these building blocks.
```

Or mathematically:

$$
\boxed{
\text{MLP}
=
\text{Linear}
\rightarrow
\text{Nonlinearity}
\rightarrow
\text{Linear}
\rightarrow
\cdots
}
$$

The important shift in this unit is from thinking about individual neurons to
thinking about **transformations of entire representations using matrices**.
