# Experiment 10 — Random Weight Initialization

## Objective

Investigate how weight initialization and activation-function choice affect optimization in a deep MLP.

The experiment compares:

- Uniform[-1,1] + ReLU
- He/Kaiming + ReLU
- Xavier/Glorot + ReLU
- Xavier/Glorot + Tanh

The core question is:

> If the architecture, data, optimizer, and random seed are held constant, how much does initialization change convergence behavior?

---

# Experimental Task

The network learns the arithmetic mapping:

[
y=(A	imes B)mod(C+1)
]

with:

[
A,Bin{0,ldots,99}
]

and the notebook generates:

```python
itertools.product(
    range(100),
    range(100),
    range(19)
)
```

Therefore:

[
100	imes100	imes19
=
oxed{190,000	ext{ examples}}
]

---

# Input Representation

Each variable is one-hot encoded.

The saved notebook uses:

```text
A → 100 dimensions
B → 100 dimensions
C →  19 dimensions
-------------------
    219 dimensions
```

The target contains 19 possible class indices.

The actual printed dataset configuration is:

```text
Input dim: 219
Output classes: 19
```

## Implementation Note

An earlier model-display cell instantiates the class defaults:

```text
220 → 512 → 512 → 256 → 256 → 128 → 20
```

but the training function later creates the model with:

```python
train(
    input_dim=219,
    output_dim=19,
    ...
)
```

All recorded initialization experiments use the **219-input / 19-output** configuration.

---

# Dataset Split

The notebook uses an 80/20 split:

| Split | Samples |
|---|---:|
| Training | 152,000 |
| Test | 38,000 |
| Total | 190,000 |

The label distribution is not uniform because different modulus values produce different output frequencies.

---

# Model Architecture

The training configuration is:

```text
219
 ↓
Linear 512
 ↓
Activation
 ↓
Linear 512
 ↓
Activation
 ↓
Linear 256
 ↓
Activation
 ↓
Linear 256
 ↓
Activation
 ↓
Linear 128
 ↓
Activation
 ↓
Linear 19
```

Compactly:

[
219ightarrow512ightarrow512ightarrow256ightarrow256ightarrow128ightarrow19
]

For this actual configuration, the network contains approximately:

[
oxed{607,763	ext{ trainable parameters}}
]

---

# Training Configuration

The recorded notebook uses:

```text
Optimizer:      AdamW
Learning rate:  1e-3
Weight decay:   0.1
Batch size:     256
Maximum epochs: 100
Loss:           CrossEntropyLoss
Seed:           42
```

The same seed is reset before each run.

The convergence criterion is strict:

```python
if test_acc == 1.00 and train_acc == 1.00:
    ...
```

So a run is labeled "converged" only when both train and test accuracy are exactly 100%.

---

# Initialization Implementations

## Uniform[-1,1]

```python
nn.init.uniform_(
    m.weight,
    -1,
    1
)
```

This uses a fixed range that does not account for fan-in or fan-out.

---

## He / Kaiming Normal

```python
nn.init.kaiming_normal_(
    m.weight,
    nonlinearity="relu"
)
```

The workshop associates this with:

[
sigma=sqrt{rac{2}{fan_{in}}}
]

and ReLU activation.

---

## Xavier / Glorot Normal

```python
nn.init.xavier_normal_(
    m.weight
)
```

The workshop gives:

[
sigma=
sqrt{
rac{2}
{fan_{in}+fan_{out}}
}
]

Xavier is tested with both Tanh and ReLU.

---

# Results

## Convergence Summary

| Run | Converged epoch | Final recorded outcome |
|---|---:|---|
| He + ReLU | **25** | 100% train / 100% test |
| Xavier + ReLU | **49** | 100% train / 100% test |
| Uniform[-1,1] + ReLU | **85** | 100% train / 100% test |
| Xavier + Tanh | Did not reach 100% | 99.0% train / 98.3% test at epoch 100 |

The ranking by the notebook's strict convergence criterion is:

```text
He + ReLU
   ↓ fastest
Xavier + ReLU
   ↓
Uniform + ReLU
   ↓
Xavier + Tanh
   ↓ did not hit 100% within 100 epochs
```

---

# Experiment 1 — Uniform + ReLU

The workshop slide labels this combination as a configuration that can fail to converge.

The saved notebook run initially stayed near a plateau:

```text
Epoch 1  test accuracy ≈ 33.2%
Epoch 10 test accuracy ≈ 33.6%
Epoch 20 test accuracy ≈ 45.3%
```

Training eventually escaped the plateau and reached:

[
oxed{100%	ext{ test accuracy at epoch 85}}
]

### Observation

This configuration learned much more slowly than the architecture-aware initializations.

The important empirical conclusion from this saved run is therefore:

> Uniform[-1,1] did not permanently collapse here, but it required substantially more epochs.

---

# Experiment 2 — He + ReLU

He/Kaiming initialization was paired with ReLU.

The model improved rapidly:

```text
Epoch 1  test accuracy ≈ 53.4%
Epoch 8  test accuracy ≈ 80.6%
Epoch 17 test accuracy ≈ 96.8%
Epoch 21 test accuracy = 100%
```

The notebook's strict criterion required train and test accuracy to be simultaneously 100%, which occurred at:

[
oxed{	ext{epoch 25}}
]

### Observation

This was the fastest configuration in the recorded experiment.

It provides a concrete demonstration of why initialization and activation choice should be considered together.

---

# Experiment 3 — Xavier + Tanh

Xavier initialization was paired with Tanh.

The run improved gradually but did not satisfy the notebook's exact 100% convergence criterion.

At epoch 100:

```text
Train accuracy: 99.0%
Test accuracy:  98.3%
```

### Observation

The slide summarizes Xavier + Tanh as "converges but slowly."

In this specific saved run it was indeed slow, but it did **not** reach 100% within the 100-epoch budget.

This distinction is important when documenting experiments:

```text
theoretical / qualitative expectation
             ≠
guaranteed result of one finite training run
```

---

# Experiment 4 — Xavier + ReLU

Xavier was also tested with ReLU.

The model reached 100% at:

[
oxed{	ext{epoch 49}}
]

This was substantially faster than Uniform + ReLU but slower than He + ReLU.

### Observation

Xavier can still support successful optimization with ReLU, but the initialization designed around ReLU produced faster convergence in this run.

---

# Comparison

The most direct comparison is convergence speed:

| Method | Epochs to strict 100% |
|---|---:|
| He + ReLU | **25** |
| Xavier + ReLU | **49** |
| Uniform + ReLU | **85** |
| Xavier + Tanh | >100 / not reached |

Relative to Uniform + ReLU, He + ReLU reached the strict criterion in roughly:

[
rac{25}{85}approx0.29
]

of the epochs.

This is a large practical difference even though all other major training settings were held constant.

---

# Why the Results Matter

Initialization changes the statistical scale of the signal entering every layer.

A deep MLP repeatedly performs:

```text
matrix multiply
      ↓
activation
      ↓
matrix multiply
      ↓
activation
      ↓
...
```

If initialization and activation are poorly matched, useful signal and gradient scales can deteriorate across depth.

If they are better matched:

```text
stable signal scale
      ↓
more useful gradients
      ↓
faster optimization
      ↓
fewer epochs
      ↓
less compute
```

This connects a statistical design decision directly to systems cost.

---

# Additional Diagnostic Experiment

A useful extension is to inspect activation and gradient standard deviations **before long training**.

For each configuration:

1. initialize a fresh model
2. run one batch forward
3. record the standard deviation after each Linear layer
4. backpropagate one loss
5. record gradient standard deviation for each weight tensor

This can reveal the mechanism behind the convergence curves:

```text
initialization
     ↓
activation scale by depth
     ↓
gradient scale by depth
     ↓
optimization behavior
```

The cleaned Colab notebook includes this diagnostic as an additional experiment.

---

# Key Findings

1. Initialization choice materially changed convergence speed.
2. He/Kaiming + ReLU was the fastest combination in the recorded run.
3. Xavier + ReLU learned successfully but required almost twice as many epochs as He + ReLU under the notebook's strict criterion.
4. Uniform[-1,1] + ReLU spent many early epochs near a plateau and required 85 epochs.
5. Xavier + Tanh reached high accuracy but did not reach the exact 100% criterion within 100 epochs.
6. The recorded run should be distinguished from simplified qualitative labels in the workshop slides.
7. Matching initialization scale to the activation and layer dimensions can save substantial training compute.
8. Weight initialization is not a cosmetic implementation detail; it directly influences optimization dynamics.
