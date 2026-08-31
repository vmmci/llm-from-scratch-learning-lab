# Experiment 08 — Training and Optimization Experiments

## Objective

Study how a neural network learns through backpropagation and investigate how
training behavior changes with:

- learning rate
- batch size
- model capacity

The practical task is to train an MLP to predict:

\[
(A+B)\bmod5
\]

for all ordered pairs:

\[
A,B\in\{0,\ldots,99\}
\]

The experiments use the same learning pipeline:

```text
Input batch
    ↓
Forward pass
    ↓
Cross-entropy loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
AdamW update
    ↓
New parameters
```

---

# Experimental Setup

## Dataset

All ordered pairs of integers from `0` to `99` were generated.

Therefore:

\[
100\times100=10,000
\]

examples were created.

The target is:

\[
y=(A+B)\bmod5
\]

The dataset is perfectly balanced:

| Class | Samples |
|---:|---:|
| 0 | 2,000 |
| 1 | 2,000 |
| 2 | 2,000 |
| 3 | 2,000 |
| 4 | 2,000 |

---

## Input Representation

Each pair `(A, B)` is converted into a 200-dimensional one-hot vector.

```text
dimensions   0–99  → A
dimensions 100–199 → B
```

Only two values in the vector are active for each example.

---

## Default Model

The default architecture is:

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

Compactly:

```text
200 → 32 → 16 → 5
```

The default network contains:

\[
\boxed{7,045\text{ trainable parameters}}
\]

---

## Data Split

The dataset was divided into:

```text
70% training
30% validation
```

giving approximately:

```text
Training:   7,000 examples
Validation: 3,000 examples
```

The training set is used for parameter updates.

The validation set is used only to evaluate learned behavior.

---

# Training Procedure

The optimizer is:

```python
torch.optim.AdamW
```

and the objective is:

```python
nn.CrossEntropyLoss()
```

The core training step is:

```python
optimizer.zero_grad()

logits = model(x)

loss = criterion(logits, y)

loss.backward()

optimizer.step()
```

This separates four important operations:

```text
clear previous gradients
        ↓
calculate predictions and loss
        ↓
calculate gradients
        ↓
update parameters
```

---

# Experiment 1 — Baseline Training

The baseline configuration used:

```text
Architecture: 200 → 32 → 16 → 5
Optimizer:    AdamW
Loss:         CrossEntropyLoss
Learning rate: 1e-3
Batch size:    32
```

The recorded training run was:

| Epoch | Loss | Train Accuracy | Validation Accuracy |
|---:|---:|---:|---:|
| 1 | 1.6118 | 20.2% | 19.6% |
| 2 | 1.6090 | 26.9% | 20.4% |
| 3 | 1.5923 | 50.0% | 40.8% |
| 4 | 1.3009 | 88.4% | 85.3% |
| 5 | 0.5677 | 100.0% | 99.9% |
| 6 | 0.1767 | 100.0% | 100.0% |

The model reached:

\[
\boxed{100\%\text{ validation accuracy at epoch 6}}
\]

### Observation

The beginning of training stayed close to random five-class performance:

\[
\frac{1}{5}=20\%
\]

After several epochs, accuracy increased rapidly while loss decreased.

This demonstrates the complete learning loop working end to end.

---

# Experiment 2 — Learning-Rate Sweep

A fresh model was trained for every learning rate.

The tested values were:

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

The maximum training budget was:

```text
25 epochs
```

## Recorded Results

| Learning Rate | Epochs to 100% Validation Accuracy |
|---:|---:|
| `1e-5` | Did not converge within 25 |
| `5e-5` | Did not converge within 25 |
| `1e-4` | Did not converge within 25 |
| `5e-4` | 12 |
| `1e-3` | 5 |
| `3e-3` | **3** |
| `1e-2` | **3** |
| `3e-2` | 13 |
| `1e-1` | Did not converge within 25 |
| `3e-1` | Did not converge within 25 |

---

## Interpretation

The results show a clear non-monotonic relationship.

Very small learning rates:

```text
1e-5
5e-5
1e-4
```

made extremely slow progress.

For example, with:

```text
LR = 1e-5
```

the validation accuracy remained near random performance after 25 epochs.

Increasing the learning rate improved convergence substantially.

The fastest recorded values were:

```text
3e-3 → 3 epochs
1e-2 → 3 epochs
```

However, increasing the learning rate further did not continue improving
training.

At:

```text
3e-2
```

convergence slowed to 13 epochs.

At:

```text
1e-1
3e-1
```

the model failed to converge within the training budget.

---

## Main Learning-Rate Finding

The experiment demonstrates:

```text
Too small
   ↓
updates are too weak
   ↓
slow learning

Intermediate
   ↓
effective parameter updates
   ↓
fast convergence

Too large
   ↓
unstable / ineffective optimization
   ↓
poor convergence
```

The best learning rate is therefore not simply the largest possible value.

---

# Experiment 3 — Batch-Size Sweep

The learning rate was fixed at:

```text
1e-3
```

while batch size was varied.

The tested values were:

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

## Recorded Results

| Batch Size | Epochs to 100% Validation Accuracy |
|---:|---:|
| 4 | **3** |
| 8 | 4 |
| 16 | 4 |
| 32 | 6 |
| 64 | 9 |
| 128 | 12 |
| 256 | 18 |
| 512 | Did not converge within 25 |
| 1024 | Did not converge within 25 |
| 2000 | Did not converge within 25 |

---

## Interpretation

For this experiment, smaller batches converged in fewer epochs.

The fastest recorded result was:

```text
batch size = 4
```

which reached perfect validation accuracy in:

```text
3 epochs
```

As batch size increased, the number of epochs required also increased.

For example:

```text
32  → 6 epochs
64  → 9 epochs
128 → 12 epochs
256 → 18 epochs
```

Very large batches did not converge within the 25-epoch budget.

---

## Why Batch Size Changes Training

With approximately 7,000 training samples:

a batch size of `4` gives many optimizer steps per epoch.

A batch size of `2000` gives only a few optimizer steps per epoch.

Therefore:

```text
Small batch
    ↓
many parameter updates per epoch
    ↓
noisier gradient estimate

Large batch
    ↓
fewer parameter updates per epoch
    ↓
smoother gradient estimate
```

---

## Important Benchmarking Caveat

This experiment measures:

```text
epochs to convergence
```

not:

```text
optimizer steps
```

or:

```text
wall-clock training time
```

A smaller batch performs more updates during one epoch.

Therefore:

> Fewer epochs does not automatically mean less computation or faster
> wall-clock training.

The batch-size experiment should be interpreted as a study of optimization
behavior rather than a hardware-performance benchmark.

---

# Experiment 4 — Model Capacity Sweep

The next experiment kept:

```text
Learning rate: 1e-3
Batch size:    32
```

while changing hidden-layer dimensions.

The tested architectures were:

```text
200 → 4    → 2    → 5
200 → 8    → 4    → 5
200 → 16   → 8    → 5
200 → 32   → 16   → 5
200 → 64   → 32   → 5
200 → 128  → 64   → 5
200 → 256  → 128  → 5
200 → 512  → 256  → 5
200 → 1024 → 512  → 5
200 → 2048 → 1024 → 5
```

## Recorded Results

| Hidden Dimensions | Parameters | Epochs to 100% Validation Accuracy |
|---|---:|---:|
| `4 / 2` | 829 | Did not converge within 25 |
| `8 / 4` | 1,669 | Did not converge within 25 |
| `16 / 8` | 3,397 | 9 |
| `32 / 16` | 7,045 | 6 |
| `64 / 32` | 15,109 | 4 |
| `128 / 64` | 34,309 | 4 |
| `256 / 128` | 84,997 | **3** |
| `512 / 256` | 235,525 | **3** |
| `1024 / 512` | 733,189 | **3** |
| `2048 / 1024` | 2,514,949 | **3** |

---

# Under-Capacity Models

The smallest models:

```text
4 / 2
8 / 4
```

failed to reach perfect validation accuracy within 25 epochs.

For example, the `4 / 2` model ended near:

```text
Train accuracy: 78.9%
Validation accuracy: 78.5%
```

after 25 epochs.

This suggests that the model had insufficient capacity, or at least
insufficient capacity under the chosen optimizer, learning rate, and training
budget, to fully represent the task.

---

# Increasing Capacity

Increasing hidden dimensions improved convergence substantially.

```text
16 / 8   → 9 epochs
32 / 16  → 6 epochs
64 / 32  → 4 epochs
256 /128 → 3 epochs
```

Larger representations gave the optimizer more expressive capacity for
learning the modulo structure.

---

# Diminishing Returns

After the network reached approximately:

```text
256 / 128
```

additional parameters did not reduce the recorded convergence below:

```text
3 epochs
```

Even increasing the model to more than:

\[
2.5\text{ million parameters}
\]

still required approximately 3 epochs.

This demonstrates diminishing returns:

```text
Increase capacity
      ↓
large initial benefit
      ↓
eventual plateau
```

---

# Experiment 5 — Minimal Training Loop

After the hyperparameter experiments, the full training logic was rewritten in
a compact form.

The selected configuration was:

```text
Learning rate: 1e-3
Batch size:    4
Optimizer:     AdamW
Loss:          CrossEntropyLoss
```

The essential training loop was:

```python
for epoch in range(epochs):

    model.train()

    for x, y in train_loader:

        optimizer.zero_grad()

        loss = criterion(model(x), y)

        loss.backward()

        optimizer.step()
```

This small block contains the core mechanics of neural-network learning.

---

# Backpropagation Interpretation

The statement:

```python
loss.backward()
```

causes PyTorch autograd to compute:

\[
\frac{\partial L}{\partial\theta}
\]

for every trainable parameter:

\[
\theta
\]

The gradients describe local sensitivity of the loss to each parameter.

Then:

```python
optimizer.step()
```

uses those gradients to update the model.

Therefore:

```text
backpropagation
≠
parameter update
```

Backpropagation computes gradients.

The optimizer uses those gradients to perform the update.

---

# Why `zero_grad()` Matters

PyTorch accumulates gradients.

Without:

```python
optimizer.zero_grad()
```

the new gradients would be added to gradients from previous batches.

The usual loop therefore explicitly clears them before the next backward pass.

---

# Train vs Validation Behavior

During training:

```python
model.train()
```

is used and gradients are calculated.

During validation:

```python
model.eval()
```

and:

```python
torch.no_grad()
```

are used.

No optimizer update occurs on validation samples.

This preserves the validation set as an independent measure of learned
behavior.

---

# Main Experimental Findings

The experiments reveal that convergence depends on an interaction between
several factors:

```text
Learning rate
      +
Batch size
      +
Model capacity
      +
Optimizer
      +
Training budget
      ↓
Observed convergence
```

No hyperparameter should be interpreted completely independently.

---

## Finding 1 — Learning Rate Has a Useful Range

The relationship was not:

```text
larger LR = faster training
```

Instead:

```text
too small
   ↓
slow

appropriate range
   ↓
fast

too large
   ↓
unstable / ineffective
```

---

## Finding 2 — Batch Size Changes Update Frequency

Smaller batches required fewer epochs in this experiment.

However, they also created many more optimizer steps per epoch.

Therefore epoch count alone is not sufficient for determining computational
efficiency.

---

## Finding 3 — Capacity Can Limit Learning

The smallest networks did not completely learn the task within the experiment's
training budget.

Increasing capacity substantially improved convergence.

---

## Finding 4 — More Parameters Eventually Stop Helping

Beyond sufficient capacity, increasing model size did not improve the recorded
three-epoch convergence time.

This is an example of diminishing returns.

---

# Connection to the Loss Unit

The previous unit studied:

\[
L(\theta)
\]

as a loss landscape.

This unit introduces movement through that landscape.

```text
Unit 07

Where is the model in the loss landscape?
```

becomes:

```text
Unit 08

How can the model move toward lower-loss regions?
```

The gradient provides the local direction.

The optimizer determines how that information is converted into a parameter
update.

---

# Connection to LLM Training

The modulo-5 network is small, but the training structure is directly related
to language-model training.

```text
Small experiment

one-hot inputs
    ↓
MLP
    ↓
5 logits
    ↓
Cross Entropy
    ↓
Backpropagation
    ↓
AdamW
```

becomes:

```text
Language model

token sequences
    ↓
Transformer
    ↓
vocabulary logits
    ↓
Cross Entropy
    ↓
Backpropagation
    ↓
large-scale optimizer
```

The same conceptual system remains:

\[
\boxed{
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Update}
}
\]

---

# Limitations

The experiments are intended for learning rather than formal hyperparameter
optimization.

Important limitations include:

- results come from individual training runs
- neural-network initialization introduces stochastic variation
- data-loader ordering can affect convergence
- convergence was defined specifically as 100% validation accuracy
- experiments were capped at 25 epochs
- batch-size comparisons used epochs rather than optimizer-step count
- wall-clock runtime was not compared
- only AdamW was used in the practical sweeps
- optimizer alternatives were discussed conceptually but not benchmarked
  against each other

Therefore the exact epoch counts should be treated as recorded experimental
results rather than universal constants.

---

# Conclusion

This experiment closes the complete neural-network training loop.

The model starts with randomly initialized parameters.

It performs a forward pass:

\[
x\rightarrow\hat{y}
\]

measures error:

\[
\hat{y}\rightarrow L
\]

propagates that error backward:

\[
L\rightarrow\nabla_\theta L
\]

and updates the parameters:

\[
\theta
\rightarrow
\theta'
\]

Repeated many times, this process turns an initially random network into a
model capable of learning structured behavior.

The central lesson is:

> Training performance is determined not only by the model architecture, but
> also by how gradients, learning rate, batch size, optimizer behavior, and
> model capacity interact.
