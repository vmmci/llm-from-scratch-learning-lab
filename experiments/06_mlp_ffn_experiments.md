# Experiment 06 — MLP / Feed-Forward Network Experiments

## Objective

Explore how simple perceptrons scale into multi-layer perceptrons and
feed-forward networks.

The experiments focus on:

- logical mappings with small MLPs
- information flow across hidden layers
- tensor shape transformations
- parameter counting
- representation expansion and contraction
- ReLU² feed-forward networks

---

## Experiment 1 — `2 → 2 → 1` MLP

The first model uses:

```text
2 input features
      ↓
2 hidden units
      ↓
ReLU
      ↓
1 output
```

Architecture:

```python
nn.Linear(2, 2)
nn.ReLU()
nn.Linear(2, 1)
```

The weights and biases were manually assigned rather than trained.

For the input:

```text
[1.0, 0.5]
```

the expected output was approximately:

```text
1.69
```

### Observation

The network is a sequence of transformations:

```text
input
  ↓
weighted sums
  ↓
nonlinearity
  ↓
weighted sum
  ↓
output
```

This extends the single-perceptron idea from earlier units into a layered
network.

---

## Experiment 2 — XOR

The XOR mapping is:

| A | B | XOR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A `2 → 2 → 1` MLP with ReLU was configured manually to reproduce this mapping.

### Observation

The hidden layer allows intermediate features to be created before the final
output is calculated.

The network therefore performs more than a single weighted sum.

Conceptually:

```text
Input
  ↓
Hidden representation
  ↓
Nonlinear transformation
  ↓
Output
```

---

## Experiment 3 — AND and OR

The same general MLP architecture was configured with different weights and
biases to represent AND and OR.

### AND

| A | B | AND |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

### OR

| A | B | OR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

### Observation

The architecture alone does not determine the function.

The behavior depends on:

```text
Architecture
     +
Weights
     +
Biases
     +
Activation functions
     ↓
Represented function
```

The same layer structure can therefore express different mappings when its
parameters change.

---

## Experiment 4 — ReLU² Feed-Forward Network

The main FFN experiment used:

```text
Input dimension:  768
Expansion ratio:  4×
Hidden dimension: 3072
Output dimension: 768
Activation:       ReLU²
Bias:             False
```

Architecture:

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

Compactly:

```text
768 → 3072 → ReLU² → 768
```

Implementation:

```python
self.upsample_weights = nn.Linear(
    768,
    3072,
    bias=False
)

self.downsample_weights = nn.Linear(
    3072,
    768,
    bias=False
)
```

The activation is:

```python
x = F.relu(x)
x = x.square()
```

---

## Shape Flow

For a single input representation:

```text
Input
[768]
   ↓
Upsample
[3072]
   ↓
ReLU²
[3072]
   ↓
Downsample
[768]
```

The network therefore preserves the external representation size while using a
larger intermediate feature space internally.

### Observation

The FFN can be understood as:

```text
Expand
  ↓
Transform nonlinearly
  ↓
Project back
```

This is more useful than thinking of the component only as a collection of
individual neurons.

---

## FFN Parameter Count

Because both FFN linear layers use `bias=False`:

### Upsampling matrix

```text
768 × 3072
```

Parameters:

\[
768 \times 3072
=
2,359,296
\]

### Downsampling matrix

```text
3072 × 768
```

Parameters:

\[
3072 \times 768
=
2,359,296
\]

### Total

\[
2,359,296
+
2,359,296
=
4,718,592
\]

Therefore the ReLU² FFN contains:

\[
\boxed{4,718,592\text{ trainable parameters}}
\]

This parameter count comes entirely from the two linear projection matrices.

---

## Experiment 5 — Deeper MLP

The deeper model used:

```text
5 → 8 → 16 → 1
```

with ReLU between the hidden layers.

Architecture:

```text
Input: 5
   ↓
Linear
   ↓
Hidden: 8
   ↓
ReLU
   ↓
Linear
   ↓
Hidden: 16
   ↓
ReLU
   ↓
Linear
   ↓
Output: 1
```

---

## Parameter Count

For a linear layer with bias:

\[
\text{parameters}
=
(\text{input features}\times\text{output features})
+
\text{output features}
\]

### Layer 1

```text
5 → 8
```

\[
5\times8+8=48
\]

### Layer 2

```text
8 → 16
```

\[
8\times16+16=144
\]

### Output layer

```text
16 → 1
```

\[
16\times1+1=17
\]

### Total

\[
48+144+17=209
\]

The model therefore contains:

\[
\boxed{209\text{ trainable parameters}}
\]

---

## Experiment 6 — Linear Algebra

The final part of the notebook reviewed the tensor operations underlying neural
network layers.

### Transpose

A tensor with shape:

```text
2 × 3
```

becomes:

```text
3 × 2
```

after transposition.

### Matrix multiplication

The experiment used:

```python
matrix_a @ matrix_b
```

with:

```text
A: 2 × 3
B: 3 × 2
```

producing:

```text
C: 2 × 2
```

The inner dimensions must match:

```text
(2 × 3) @ (3 × 2)
        ↑   ↑
        matching dimensions
```

### Observation

This is directly related to `nn.Linear`.

Neural-network layers are naturally implemented through vector and matrix
operations rather than by calculating each neuron separately.

---

## Main Findings

The experiments produced several useful observations:

1. A perceptron generalizes naturally from one input to many inputs.

2. Multiple neurons can be grouped into layers and represented using matrices.

3. Nonlinear activation functions allow stacked linear layers to represent more
   complex mappings.

4. XOR, AND, and OR illustrate how parameters determine network behavior.

5. Tensor shapes provide a useful way to reason about architecture.

6. The `5 → 8 → 16 → 1` model contains 209 trainable parameters.

7. The ReLU² FFN expands the representation by `4×`:

   ```text
   768 → 3072 → 768
   ```

8. The ReLU² FFN contains 4,718,592 trainable parameters when both linear
   projections use no bias.

9. Matrix multiplication is the core linear-algebra operation behind fully
   connected layers.

10. A feed-forward network is better understood as a transformation of an
    entire representation rather than as isolated neuron calculations.

---

## Connection to the LLM Roadmap

This experiment connects the earlier units:

```text
Perceptron
    ↓
Activation functions
    ↓
GPU execution
    ↓
MLP / FFN
```

to the larger model architecture that will be constructed later.

The FFN pattern:

```text
Linear expansion
      ↓
Nonlinear activation
      ↓
Linear projection
```

becomes a reusable component inside larger neural-network systems.

---

## Conclusion

The main conceptual shift in this unit is from reasoning about individual
neurons to reasoning about **matrix-based transformations of representations**.

A perceptron performs one weighted sum.

A layer performs many weighted sums simultaneously.

An MLP composes several layers.

An FFN expands, transforms, and projects an entire representation.

These ideas provide the foundation for understanding the larger blocks that
will appear later in the LLM architecture.
