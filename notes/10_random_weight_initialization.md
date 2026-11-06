# 10 — Random Weight Initialization

## Overview

Neural networks need initial parameter values before training can begin.

Those values are usually random, but **how they are randomized matters**. In a deep network, activations and gradients repeatedly pass through matrix multiplications and nonlinear activation functions. If the scale of the initial weights is poorly chosen, values can grow too large or shrink toward zero as they move through the network.

This produces two classic failure modes:

- **exploding activations / gradients**
- **vanishing activations / gradients**

The workshop connects this directly to practical training cost:

```text
Bad initialization
      ↓
unstable or very slow optimization
      ↓
more training iterations / failed runs
      ↓
more GPU time
      ↓
more compute cost
```

The goal of weight initialization is therefore not simply to "pick random numbers." It is to choose a distribution and scale that help signals and gradients remain numerically useful across layers.

---

# Why Random Initialization Is Needed

If every neuron in a layer started with identical parameters, neurons would initially behave identically and receive the same gradient updates.

Random initialization breaks this symmetry.

Conceptually:

```text
same architecture
+
different initial weights
↓
different neuron responses
↓
different gradients
↓
different learned features
```

But unrestricted randomness creates another problem: repeated multiplication can amplify or suppress values.

For a deep network:

```text
input
  ↓
W₁x
  ↓
activation
  ↓
W₂h₁
  ↓
activation
  ↓
W₃h₂
  ↓
...
```

the scale introduced by each weight matrix compounds across layers.

---

# Exploding and Vanishing Values

Suppose values are repeatedly multiplied through many layers.

If the effective scale is consistently larger than one, magnitudes can grow rapidly:

```text
1.5
→ 2.25
→ 3.375
→ 5.06
→ ...
```

This is the intuition behind **exploding** activations or gradients.

If the effective scale is consistently smaller than one:

```text
0.5
→ 0.25
→ 0.125
→ 0.0625
→ ...
```

the values shrink toward zero.

This is the intuition behind **vanishing** activations or gradients.

During backpropagation, gradients are also propagated through repeated multiplications. If they become extremely small, early layers receive almost no useful learning signal. If they become extremely large, optimization can become unstable.

---

# Numerical Precision Matters

Computers represent neural-network values using finite-precision floating-point numbers.

Extremely small values can eventually become numerically insignificant, while extremely large values can overflow or destabilize computation.

So initialization influences both:

- mathematical signal propagation
- numerical behavior on real hardware

This is one reason initialization quality can affect training time and GPU cost.

---

# Random Distributions

The workshop compares two common ways of generating random values.

## Uniform Distribution

A uniform distribution gives every value in a specified interval equal probability.

Example:

```text
Uniform(-1, 1)
```

Conceptually, the probability density is flat across the range.

Uniform randomness alone does **not** guarantee that the scale is suitable for a deep network. The interval must still be chosen with the network architecture in mind.

---

## Normal Distribution

A normal distribution follows a bell-shaped curve centered around a mean.

It is characterized mainly by:

- mean `μ`
- standard deviation `σ`

For a standard normal distribution:

```text
mean = 0
std  = 1
```

Approximately:

- 68% of values fall within ±1 standard deviation
- 95% fall within ±2 standard deviations
- 99.7% fall within ±3 standard deviations

Initialization methods such as He/Kaiming and Xavier/Glorot choose the standard deviation based on layer dimensions rather than using an arbitrary fixed scale.

---

# Activation Functions Change the Initialization Problem

Initialization cannot be considered independently of the activation function.

The workshop emphasizes two broad cases:

```text
ReLU-like activations
        ↓
He / Kaiming initialization

Tanh / sigmoid-style activations
        ↓
Xavier / Glorot initialization
```

The reason is that different activation functions transform the distribution of activations differently.

---

# ReLU and Zero Activations

ReLU is:

[
operatorname{ReLU}(x)=max(0,x)
]

Negative pre-activations become zero.

With a roughly zero-centered input distribution, a substantial fraction of values can therefore be clipped to zero.

This changes the variance of the signal that survives the activation.

A good initialization for ReLU should compensate for this effect so that activation variance does not collapse as depth increases.

---

# He / Kaiming Initialization

He/Kaiming initialization is designed for ReLU-like activations.

For the normal version shown in the workshop:

[
oxed{
sigma = sqrt{rac{2}{	ext{fan}_{in}}}
}
]

where:

[
	ext{fan}_{in}=	ext{number of inputs to the layer}
]

In PyTorch:

```python
nn.init.kaiming_normal_(
    layer.weight,
    nonlinearity="relu"
)
```

The intuition is:

```text
more inputs to a neuron
      ↓
smaller individual weight scale
      ↓
better control of total activation variance
```

This helps compensate for the variance reduction introduced by ReLU.

---

# Xavier / Glorot Initialization

Xavier/Glorot initialization balances the number of inputs and outputs of a layer.

For the normal version shown in the workshop:

[
oxed{
sigma =
sqrt{
rac{2}
{	ext{fan}_{in}+	ext{fan}_{out}}
}
}
]

where:

- `fan_in` = number of layer inputs
- `fan_out` = number of layer outputs

In PyTorch:

```python
nn.init.xavier_normal_(layer.weight)
```

The workshop associates Xavier initialization primarily with tanh/sigmoid-style networks.

The goal is to keep signal variance reasonably balanced between the forward and backward passes.

---

# Fan-In and Fan-Out

For a linear layer:

[
y=xW^T
]

with a weight matrix:

[
Winmathbb{R}^{d_{out}	imes d_{in}}
]

we have:

```text
fan_in  = d_in
fan_out = d_out
```

For example:

```text
Linear(512 → 256)

fan_in  = 512
fan_out = 256
```

Therefore:

He standard deviation:

[
sqrt{rac{2}{512}}
]

Xavier standard deviation:

[
sqrt{rac{2}{512+256}}
]

The initialization scale automatically changes with layer width.

---

# One-Hot Encoding Demonstration

The workshop uses a synthetic arithmetic task to demonstrate initialization.

The model receives one-hot encoded values for:

- `A`
- `B`
- `C`

and predicts:

[
(A	imes B)mod(C+1)
]

The saved notebook implementation uses:

```text
A: 100 positions
B: 100 positions
C:  19 positions
-----------------
   219 input dimensions
```

with 19 output classes.

> The workshop slide/initial display cell shows a 220-input / 20-output variant, while the actual recorded training runs instantiate the network with 219 inputs and 19 outputs. The experiment notes use the configuration that actually generated the recorded results.

---

# PyTorch Initialization API

The workshop maps the concepts to `torch.nn.init`.

Examples:

```python
nn.init.uniform_(weight, -1, 1)

nn.init.kaiming_normal_(
    weight,
    nonlinearity="relu"
)

nn.init.xavier_normal_(weight)
```

Biases are reset to zero in the experiment:

```python
nn.init.zeros_(bias)
```

This makes the initialization strategy explicit and reproducible.

---

# Workshop Experiment: Initialization + Activation

The notebook compares four combinations:

1. Uniform[-1,1] + ReLU
2. He/Kaiming + ReLU
3. Xavier + Tanh
4. Xavier + ReLU

The recorded run produced:

| Initialization | Activation | Recorded result |
|---|---|---:|
| He/Kaiming | ReLU | 100% test accuracy at epoch **25** |
| Xavier | ReLU | 100% test accuracy at epoch **49** |
| Uniform[-1,1] | ReLU | 100% test accuracy at epoch **85** |
| Xavier | Tanh | 98.3% test accuracy after 100 epochs; did not reach 100% |

The ordering supports the workshop's main intuition:

[
oxed{	ext{He + ReLU was the fastest combination in this run}}
]

However, the exact recorded run is more nuanced than some of the slide labels.

The slide describes Uniform + ReLU as not converging, but this saved run eventually reached 100% at epoch 85.

The slide describes Xavier + Tanh as converging slowly, while the saved run reached 98.3% by epoch 100 but did not satisfy the notebook's strict 100% convergence criterion.

This is a useful experimental lesson:

> Initialization changes optimization behavior and convergence speed, but a qualitative rule is not a guarantee about the exact epoch or outcome of one stochastic training run.

---

# Why This Matters for LLMs

Large language models contain many stacked transformations.

Even though modern architectures include mechanisms such as normalization, residual connections, carefully selected activations, and architecture-specific initialization schemes, the same underlying issue remains:

```text
many layers
+
many matrix multiplications
+
backpropagation through depth
↓
signal scale must remain controlled
```

Bad scaling can waste large amounts of compute.

For small experiments, that may mean minutes.

For large training runs:

```text
GPU time → infrastructure cost → money
```

Initialization is therefore both a mathematical design choice and a systems/efficiency choice.

---

# Connection to Earlier Units

## Unit 04 — Activation Functions

This unit extends the activation-function discussion.

Previously:

```text
weighted sum → activation
```

Now:

```text
initial weight distribution
        +
activation function
        ↓
signal propagation across many layers
```

---

## Unit 06 — MLP / FFN

An MLP repeatedly applies matrix multiplications and nonlinearities.

Initialization determines the scale of those signals before learning has even begun.

---

## Unit 08 — Backpropagation and Optimization

Backpropagation relies on gradients surviving through many operations.

Initialization affects whether those gradients remain useful, vanish, or explode.

---

## Unit 05 — GPU Performance

A poor initialization can make a run converge much more slowly or fail.

That connects numerical choices directly to hardware utilization:

```text
initialization
→ convergence behavior
→ number of training steps
→ GPU time
→ cost
```

---

# Practical Rule of Thumb

For the workshop examples:

```text
ReLU / ReLU-like activation
        ↓
He / Kaiming

Tanh / sigmoid-style activation
        ↓
Xavier / Glorot
```

The deeper lesson is broader:

> Initialization should be chosen together with the architecture and activation function, with the goal of preserving useful signal and gradient scales through the network.

---

# Key Takeaways

1. Neural-network weights need symmetry-breaking random initialization.
2. Arbitrary random scales can cause exploding or vanishing values.
3. Uniform and normal distributions describe how values are sampled, but architecture-aware scaling is what makes initialization useful.
4. ReLU changes activation statistics by clipping negative values to zero.
5. He/Kaiming initialization is designed to work well with ReLU-like activations.
6. Xavier/Glorot uses both fan-in and fan-out to balance variance.
7. In the recorded workshop experiment, He + ReLU converged fastest.
8. Initialization can directly affect training time, GPU utilization, and compute cost.
9. Experimental results should be recorded separately from simplified qualitative claims.
10. Weight initialization is a bridge between statistics, neural-network architecture, backpropagation, and systems efficiency.
