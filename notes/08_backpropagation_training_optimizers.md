# 08 — Backpropagation, Training Loops, and Optimizers

## Overview

The previous units established the main pieces required for a neural network:

```text
Inputs
  ↓
Linear transformations
  ↓
Activation functions
  ↓
MLP / FFN
  ↓
Prediction
  ↓
Loss
```

A model can now produce a prediction and measure how wrong that prediction is.

The missing question is:

> How should the model change its parameters so that the loss becomes smaller?

This unit introduces the mechanisms that make learning possible:

- derivatives
- gradients
- the chain rule
- backpropagation
- PyTorch autograd
- optimizers
- learning rate
- batch size
- train/validation splits
- the training loop

The complete learning process becomes:

```text
Forward pass
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Optimizer
     ↓
Updated parameters
     ↓
Repeat
```

---

# From Loss to Learning

A loss function tells us how poor a model prediction is.

Suppose a model has parameters:

\[
\theta
\]

and produces a loss:

\[
L(\theta)
\]

The goal of training is to find parameter values that reduce:

\[
L
\]

The loss therefore defines the objective.

However, knowing the loss alone is not enough.

We also need to know:

> If a parameter changes slightly, how will the loss change?

This is where derivatives and gradients become important.

---

# Derivatives

A derivative measures the slope of a function.

For:

\[
y=f(x)
\]

the derivative is:

\[
\frac{dy}{dx}
\]

It describes how much the output changes when the input changes slightly.

Conceptually:

```text
Positive derivative
→ increasing x increases y

Negative derivative
→ increasing x decreases y

Derivative near zero
→ locally flat region
```

When the function is the loss:

\[
L(\theta)
\]

the derivative tells us how the loss responds to a parameter.

---

# Gradient Descent Intuition

Imagine the loss function as a landscape.

```text
High loss
   \
    \
     ● current parameters
      \
       \
        ● lower loss
         \
          ● minimum
```

The gradient points toward increasing loss.

To reduce loss, we move in the opposite direction.

The basic update rule is:

\[
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta L
\]

where:

- \(\theta_t\) = current parameters
- \(\nabla_\theta L\) = gradient of loss
- \(\eta\) = learning rate

Conceptually:

```text
Current parameter
       ↓
Calculate gradient
       ↓
Move opposite gradient
       ↓
New parameter
       ↓
Hopefully lower loss
```

---

# Stochastic Gradient Descent

Instead of calculating the gradient over the entire dataset before every
update, **Stochastic Gradient Descent (SGD)** estimates the gradient using
individual examples or small batches.

The important idea is:

> Parameters should not be changed randomly.

They should be changed according to how each parameter affects the loss.

The gradient provides that information.

---

# The Chain Rule

Neural networks contain sequences of transformations.

For example:

\[
x
\rightarrow
h
\rightarrow
y
\rightarrow
L
\]

Suppose:

\[
h=f(x)
\]

and:

\[
L=g(h)
\]

To determine how \(x\) affects the loss, we use the chain rule:

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial h}
\frac{\partial h}{\partial x}
\]

The derivative is propagated through the chain of computations.

---

# Chain Rule in a Neural Network

Consider a simple neuron:

\[
z=wx+b
\]

followed by an activation:

\[
a=\sigma(z)
\]

and loss:

\[
L=L(a)
\]

To calculate how the weight affects the final loss:

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
\]

Each term answers a smaller question:

```text
How does loss change with activation?
                ×
How does activation change with pre-activation?
                ×
How does pre-activation change with weight?
```

This is the core mathematical idea behind backpropagation.

---

# Backpropagation

Backpropagation applies the chain rule through a neural network from the output
back toward the input layers.

The forward pass computes:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
  ↓
Loss
```

Backpropagation travels in the opposite conceptual direction:

```text
Loss
  ↓
Output gradients
  ↓
Layer 2 gradients
  ↓
Layer 1 gradients
```

The goal is to calculate:

\[
\frac{\partial L}{\partial \theta}
\]

for every trainable parameter.

---

# Forward Pass vs Backward Pass

The two phases have different jobs.

## Forward pass

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
```

The forward pass answers:

> What does the model currently predict?

## Backward pass

```text
Loss
  ↓
Gradient calculation
  ↓
Every trainable parameter
```

The backward pass answers:

> Which parameters contributed to the error, and in what direction should they
> change?

---

# PyTorch Autograd

Manually calculating the chain rule for a large neural network would be
impractical.

PyTorch provides **automatic differentiation** through its autograd system.

During the forward computation, PyTorch records operations involving tensors
that require gradients.

Conceptually:

```text
Tensor
  ↓
Operation
  ↓
Tensor
  ↓
Operation
  ↓
Loss
```

These relationships form a **computational graph**.

Then:

```python
loss.backward()
```

uses that graph to compute gradients automatically.

---

# Computational Graph

Suppose the calculation is:

\[
x
\rightarrow
wx+b
\rightarrow
ReLU
\rightarrow
output
\rightarrow
loss
\]

PyTorch tracks the operations required to connect the final loss back to the
parameters.

When `.backward()` is called, gradients are propagated through this graph.

The result is stored in:

```python
parameter.grad
```

for trainable parameters.

---

# Autograd Has a Memory Cost

Automatic differentiation makes backpropagation much easier to implement.

However, the framework needs information from the forward pass to compute
gradients later.

Therefore training often requires significantly more memory than a simple
forward-only inference pass.

Conceptually:

```text
Inference
Forward computation
        ↓
Output
```

versus:

```text
Training
Forward computation
        ↓
Store information for backward pass
        ↓
Loss
        ↓
Backward computation
        ↓
Gradients
```

This is one reason training large language models requires substantial memory.

---

# Updating Parameters

Calculating gradients does not automatically change the weights.

The optimizer performs the update.

The basic sequence is:

```python
optimizer.zero_grad()

output = model(x)

loss = criterion(output, y)

loss.backward()

optimizer.step()
```

Each line has a specific role.

---

## `optimizer.zero_grad()`

PyTorch gradients accumulate by default.

Before computing the gradients for the next batch:

```python
optimizer.zero_grad()
```

clears the previous gradients.

---

## Forward Pass

```python
output = model(x)
```

runs the current model.

---

## Loss Calculation

```python
loss = criterion(output, y)
```

compares the prediction to the target.

---

## Backpropagation

```python
loss.backward()
```

calculates the gradients.

---

## Optimizer Step

```python
optimizer.step()
```

uses those gradients to update the parameters.

The complete training step is therefore:

```text
Clear gradients
      ↓
Forward pass
      ↓
Calculate loss
      ↓
Backward pass
      ↓
Update parameters
```

---

# Optimizers

Gradient descent provides the basic update principle.

Modern optimizers modify or extend that idea to improve training behavior.

The workshop presents the following progression.

---

## SGD

SGD performs updates using the gradient and a learning rate.

Conceptually:

\[
\theta
\leftarrow
\theta-\eta g
\]

where:

\[
g=\nabla_\theta L
\]

SGD is simple, but training may be slow or sensitive to the learning rate.

---

## Momentum

Momentum incorporates information from previous updates.

Instead of responding only to the current gradient, the optimizer develops a
direction based on recent movement.

This can help training continue more efficiently when gradients repeatedly
point in similar directions.

Conceptually:

```text
Current gradient
      +
Previous movement
      ↓
Updated direction
```

---

## AdaGrad

AdaGrad adjusts learning behavior separately for different parameters using
accumulated gradient information.

The important shift is from:

```text
One learning behavior for every parameter
```

toward:

```text
Parameter-specific adaptation
```

---

## RMSProp

RMSProp focuses more strongly on recent gradient statistics rather than
allowing very old accumulated gradients to dominate indefinitely.

This makes the adaptive scaling more responsive to recent training behavior.

---

## Adam

Adam combines ideas related to:

- momentum
- adaptive per-parameter scaling

This makes it a widely used optimizer for neural-network training.

---

## AdamW

The workshop uses **AdamW** as the main optimizer.

AdamW is an Adam-style optimizer with improved handling of weight decay /
regularization behavior.

In PyTorch:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)
```

---

## Muon

The slides also introduce **Muon** as a more recent optimizer direction.

The key motivation presented is that neural-network parameter updates can
become concentrated in a limited number of dominant directions.

Muon aims to produce better-distributed update behavior in matrix-valued
parameters.

The purpose of introducing it here is not to replace AdamW in the workshop,
but to show that optimizer research continues to evolve.

---

# Learning Rate

The learning rate determines the size of the parameter update.

The generic update is:

\[
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta L
\]

where:

\[
\eta
\]

is the learning rate.

---

## Learning Rate Too Small

If the learning rate is too small:

```text
Current point
   ↓
tiny step
   ↓
tiny step
   ↓
tiny step
```

training may make progress extremely slowly.

This is **undershooting**.

---

## Learning Rate Too Large

If the learning rate is too large, updates may repeatedly jump across the
minimum:

```text
        minimum
      ↙       ↘
   update     update
      ↘       ↙
```

This can cause oscillation or instability.

This is **overshooting**.

---

# Learning-Rate Sweep

The notebook tests multiple learning rates by training a fresh model for each
value.

The tested values span:

```text
1e-5
5e-5
1e-4
5e-4
1e-3
3e-3
1e-2
3e-2
1e-1
3e-1
```

The experiment records how many epochs are required to reach perfect
validation accuracy, with a maximum number of epochs.

The purpose is not simply to choose the numerically largest learning rate.

The goal is to find a value that provides:

```text
fast convergence
        +
stable training
```

Very small learning rates can fail to make meaningful progress within the
training budget.

---

# Variable Learning Rates

The best learning rate does not necessarily remain constant throughout
training.

Modern training often uses a **learning-rate schedule**.

A common pattern is:

```text
Warmup
   ↓
Stable / peak learning rate
   ↓
Decay
```

---

## Warmup

During warmup, the learning rate begins small and increases gradually.

Conceptually:

```text
LR
│       ______
│      /
│     /
│    /
│___/
└──────────── steps
```

This can reduce instability at the beginning of training.

---

## Decay

Later in training, the learning rate can decrease.

The model then makes progressively smaller adjustments as it approaches a
useful solution.

One common schedule is **cosine decay**.

Conceptually:

```text
large updates early
        ↓
smaller updates later
```

---

# GPT-2-Style Learning-Rate Decision

The workshop's GPT-2-style design slide uses a schedule built around:

```text
Warmup
    ↓
Stable learning rate
    ↓
Cosine decay
```

The slide gives an example with approximately:

```text
stable LR:       6e-4
warmup:          2000 steps
cosine decay to: 6e-5
```

The exact values are architecture- and training-dependent.

The important idea is that learning rate itself becomes a scheduled
hyperparameter rather than a permanently fixed constant.

---

# Batch Size

A batch contains multiple training examples used to calculate one update.

For a dataset of \(N\) examples:

```text
Batch size = B
```

means the optimizer sees approximately:

\[
\frac{N}{B}
\]

batches per epoch.

---

# Small Batch

A smaller batch produces more parameter updates per epoch.

Its gradient estimate is generally noisier.

Conceptually:

```text
Small batch
    ↓
more updates
    ↓
noisier gradient estimate
```

---

# Large Batch

A larger batch averages information over more examples.

Conceptually:

```text
Large batch
    ↓
smoother gradient estimate
    ↓
fewer updates per epoch
```

Neither extreme is universally optimal.

Batch size is a training hyperparameter.

---

# Batch Size During Training vs Inference

The meaning of batch size depends on the phase.

## Training

Batch size determines how many examples contribute to one gradient update.

## Inference

Batch size determines how many examples can be processed concurrently.

Therefore batch-size decisions depend on:

- memory
- hardware utilization
- gradient behavior
- throughput
- latency requirements

---

# Batch-Size Sweep

The notebook tests:

```text
4
8
16
32
64
128
256
512
1024
2000
```

while holding the learning rate fixed.

The experiment measures:

```text
Batch size
     ↓
Epochs required to converge
```

This provides empirical evidence that batch size can strongly affect training
speed.

---

# Train / Validation Split

A model should not be evaluated only on the same samples used to update its
parameters.

The notebook splits the dataset into:

```text
70% training
30% validation
```

The training set is used for gradient updates.

The validation set is used to measure performance on held-out samples.

Conceptually:

```text
Dataset
  ├── Training data
  │      ↓
  │   Backpropagation
  │   Weight updates
  │
  └── Validation data
         ↓
      Evaluation only
```

---

# Training Mode vs Evaluation Mode

PyTorch models explicitly switch between modes.

During training:

```python
model.train()
```

During evaluation:

```python
model.eval()
```

Evaluation also uses:

```python
with torch.no_grad():
```

because gradients are not needed.

This avoids unnecessary autograd computation.

---

# Unit Dataset — `(A + B) % 5`

The practical training experiment constructs all ordered pairs:

\[
A,B\in\{0,\ldots,99\}
\]

and predicts:

\[
(A+B)\bmod5
\]

There are:

\[
100\times100=10,000
\]

examples.

The five output classes are perfectly balanced:

```text
class 0 → 2000 examples
class 1 → 2000 examples
class 2 → 2000 examples
class 3 → 2000 examples
class 4 → 2000 examples
```

This is useful because the model must learn a structured modular relationship.

---

# Input Representation

The model does not receive \(A\) and \(B\) directly as two scalar values.

Instead, each pair is represented using a 200-dimensional one-hot vector.

```text
Dimensions   0–99  → A
Dimensions 100–199 → B
```

For example:

```text
A = 7
B = 23
```

activates:

```text
x[7]       = 1
x[100+23]  = 1
```

and every other input dimension is zero.

---

# Mod5 MLP

The default network is:

```text
200
 ↓
Linear
 ↓
32
 ↓
ReLU
 ↓
16
 ↓
ReLU
 ↓
5 logits
```

or:

```text
200 → 32 → 16 → 5
```

The output contains one logit for each modulo-5 class.

The default architecture contains:

\[
7045
\]

trainable parameters.

---

# Cross-Entropy Objective

Because `(A+B) % 5` is a five-class classification problem, the training loop
uses:

```python
nn.CrossEntropyLoss()
```

The model produces five logits:

```text
class 0
class 1
class 2
class 3
class 4
```

Cross entropy compares these logits with the correct modulo class.

---

# The Training Loop

A general neural-network training loop follows the same pattern.

```text
Setup
 ├── model
 ├── dataset
 ├── optimizer
 └── loss function

Repeat for each epoch
 └── Repeat for each batch
       ├── zero gradients
       ├── forward pass
       ├── compute loss
       ├── backpropagation
       └── optimizer step
```

In code:

```python
for epoch in range(epochs):

    model.train()

    for x, y in train_loader:

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()
```

This loop is one of the most reusable patterns in deep learning.

---

# Baseline Training Result

Using the default architecture:

```text
200 → 32 → 16 → 5
```

with:

```text
Optimizer:  AdamW
Loss:       CrossEntropyLoss
LR:         1e-3
Batch size: 32
```

the recorded experiment progressed from approximately random five-class
performance toward perfect validation accuracy.

The model reached:

```text
100% validation accuracy
```

at approximately:

```text
epoch 6
```

This shows the complete training process working end to end.

---

# Model Capacity

The unit also varies hidden-layer dimensions.

Examples include:

```text
4 → 2
8 → 4
16 → 8
32 → 16
64 → 32
...
```

Increasing the hidden dimensions increases the number of trainable parameters.

A larger model has greater capacity to represent the task.

However:

> More parameters do not automatically imply a better model.

Important trade-offs include:

- convergence speed
- memory
- compute
- overfitting
- generalization
- training cost

---

# Capacity vs Convergence

The experiment shows that very small models may not have enough capacity to
reach perfect validation accuracy within the training budget.

As model capacity increases, convergence can become significantly faster.

After sufficient capacity is reached, however, adding many more parameters
does not necessarily continue reducing the required epochs.

This demonstrates diminishing returns.

---

# What Did the Network Learn?

After training, the weight matrices can be inspected rather than treating the
model as a complete black box.

The workshop visualizes learned weights as heatmaps.

Repeated patterns appear across the learned representations.

This is especially interesting because the task itself has periodic structure:

\[
(A+B)\bmod5
\]

Numbers separated by multiples of five belong to related modular classes.

The model therefore has an opportunity to learn **cyclic structure**.

---

# PCA and Learned Representations

High-dimensional learned representations can also be projected into two
dimensions using PCA.

The workshop shows that examples associated with the five modulo groups form
structured patterns in the lower-dimensional representation.

This gives a useful mental model:

```text
Raw numbers
    ↓
Neural network training
    ↓
Learned high-dimensional representation
    ↓
PCA visualization
    ↓
Visible cyclic structure
```

The neural network is not explicitly told to create a circle.

The pattern emerges from learning the underlying modulo relationship.

---

# Connection to Representation Learning

This small experiment illustrates a larger neural-network idea.

Models can learn internal geometric representations of structured concepts.

For example:

```text
Modulo arithmetic
→ cyclic representation
```

Similarly, larger models can learn structured representations for relationships
such as:

- days
- months
- years
- numerical patterns
- colors
- spatial relationships
- semantic relationships

The exact geometry is learned from the training objective and data.

---

# Connection to Previous Units

The progression now becomes:

```text
Perceptron
     ↓
Activation functions
     ↓
GPU execution
     ↓
MLP / FFN
     ↓
Loss functions
     ↓
Backpropagation
     ↓
Optimizer
     ↓
Training
```

The previous unit gave us:

```text
How wrong is the model?
```

This unit adds:

```text
How should the model parameters change?
```

---

# Connection to Large Language Models

The training loop for a language model is conceptually the same.

```text
Token batch
    ↓
Transformer
    ↓
Vocabulary logits
    ↓
Cross entropy
    ↓
Backpropagation
    ↓
Gradients
    ↓
Optimizer
    ↓
Updated model weights
```

The difference is scale.

Instead of a small MLP with thousands of parameters, LLMs may contain millions
or billions of parameters.

Instead of five output classes, the model predicts across an entire vocabulary.

But the core training loop remains recognizable.

---

# Training at LLM Scale

At larger scale, several details become increasingly important:

- optimizer choice
- learning-rate schedules
- warmup
- weight decay
- batch size
- gradient stability
- distributed training
- memory usage
- compute efficiency

The simple training loop therefore provides the conceptual foundation for much
larger systems.

---

# Key Takeaways

1. Loss tells us what should be minimized.

2. Derivatives measure how outputs change with inputs.

3. Gradients describe how loss changes with model parameters.

4. Gradient descent updates parameters in the direction that reduces loss.

5. Backpropagation applies the chain rule through the neural network.

6. PyTorch autograd automatically builds and differentiates the computational
   graph.

7. `.backward()` calculates gradients but does not update parameters.

8. `optimizer.step()` performs the parameter update.

9. Gradients must normally be cleared between batches using
   `optimizer.zero_grad()`.

10. The learning rate controls update magnitude.

11. Learning rates that are too small can make training extremely slow.

12. Learning rates that are too large can produce overshooting or unstable
    optimization.

13. Learning-rate schedules can combine warmup and decay.

14. Batch size controls how many examples contribute to one update.

15. Smaller batches produce more frequent and noisier gradient estimates.

16. Larger batches produce smoother gradient estimates but fewer updates per
    epoch.

17. Training data is used to update the model.

18. Validation data is used to evaluate performance without parameter updates.

19. Optimizers such as Momentum, AdaGrad, RMSProp, Adam, and AdamW extend the
    basic SGD idea.

20. Model capacity affects whether and how quickly a model can learn a task.

21. Learned weight matrices can reveal structure in the data.

22. Even a small neural network can learn geometric representations of cyclic
    relationships.

23. The training loop used for a small MLP has the same conceptual structure as
    the training loop used for large language models.

---

# Final Mental Model

Before this unit:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
```

After this unit:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated weights
  ↓
New prediction
```

Learning is therefore an iterative feedback process:

\[
\boxed{
\text{Predict}
\rightarrow
\text{Measure Error}
\rightarrow
\text{Differentiate}
\rightarrow
\text{Update}
\rightarrow
\text{Repeat}
}
\]

This is the core mechanism that transforms a randomly initialized neural
network into a trained model.
