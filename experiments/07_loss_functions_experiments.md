# Experiment 07 — Loss Functions and Loss Landscapes

## Objective

Explore how loss functions quantify prediction error and how model loss changes
as parameters move through parameter space.

The experiments progress from a single perceptron to a GPT-2 loss-landscape
visualization.

The main experiments include:

1. Residual error
2. Mean Squared Error
3. MSE loss landscape
4. Loss curves for different targets
5. Cross-entropy classification
6. Cross-entropy loss landscape
7. GPT-2 two-dimensional loss-landscape slice

---

## Experiment 1 — Residual Error

A simple perceptron was configured with:

```text
weight = 0.5
bias   = 0.2
input  = 1.0
```

The prediction was:

\[
\hat{y}
=
0.5(1.0)+0.2
=
0.7
\]

with expected value:

\[
y=0.5
\]

The residual was therefore:

\[
e
=
y-\hat{y}
=
0.5-0.7
=
-0.2
\]

Recorded result:

```text
Math residual: -0.2000
```

### Observation

The residual preserves direction.

A negative value indicates that the prediction is larger than the expected
value.

---

## Experiment 2 — Mean Squared Error

The squared residual is:

\[
(-0.2)^2=0.04
\]

The manual and PyTorch calculations matched:

```text
Math MSE:    0.0400
PyTorch MSE: 0.0400
```

### Observation

Squaring removes the sign of the residual and gives larger errors a stronger
penalty.

---

## Experiment 3 — MSE Loss Landscape

The perceptron's weight and bias were varied over a two-dimensional parameter
grid.

```text
weight: -2 → 3
bias:   -2 → 2
```

For each pair:

\[
(w,b)
\]

the model prediction was calculated and evaluated using MSE.

The result is a loss surface:

\[
L(w,b)
\]

Conceptually:

```text
Weight ──┐
         ├── prediction ── MSE ── loss
Bias ────┘
```

### Observation

Loss can be interpreted geometrically.

Different parameter combinations correspond to different heights on the loss
surface.

Training can later be understood as moving through this space toward regions
of lower loss.

---

## Experiment 4 — Loss for Different Targets

The perceptron was fixed at:

```text
weight = 0.5
bias   = 0.2
```

giving:

\[
\hat{y}=0.5x+0.2
\]

Squared loss was plotted for several expected values:

```text
-1.0
 0.0
 0.5
 1.0
 2.0
```

For each target:

\[
L(x)
=
\left[
y_{\text{expected}}
-
(0.5x+0.2)
\right]^2
\]

Each target therefore generates its own parabola.

### Minimum-Loss Locations

The minimum occurs when:

\[
y_{\text{expected}}
=
0.5x+0.2
\]

so:

\[
x
=
\frac{y_{\text{expected}}-0.2}{0.5}
\]

Examples:

| Target | Minimum-loss input |
|---:|---:|
| -1.0 | -2.4 |
| 0.0 | -0.4 |
| 0.5 | 0.6 |
| 1.0 | 1.6 |
| 2.0 | 3.6 |

### Observation

The visualization shows directly that loss approaches zero when prediction and
target match.

---

# From Regression to Classification

The next experiments replace a continuous target with a discrete class target.

The model generates ten logits:

```text
class 0
class 1
...
class 9
```

The predicted class is:

```python
torch.argmax(logits)
```

while prediction quality is measured using cross entropy.

---

## Experiment 5 — Cross-Entropy Digit Classification

The educational classifier maps one scalar input to ten class logits.

Conceptually:

```text
Input
  ↓
Perceptron value
  ↓
10 logits
  ↓
Cross Entropy
```

For the original classification example, the target digit was class `1`.

The model correctly predicted the target class and produced a recorded
cross-entropy loss of approximately:

```text
0.8870
```

### Observation

Correct classification and low loss are not identical concepts.

`argmax` only identifies the largest logit.

Cross entropy also measures how strongly the correct class is preferred over
the alternatives.

---

## Experiment 6 — Cross-Entropy Loss Landscape

The classifier's weight and bias were varied using the same parameter-grid
idea as the MSE experiment.

For each:

\[
(w,b)
\]

the model generated logits and cross entropy was calculated.

This produces:

\[
L_{\text{CE}}(w,b)
\]

rather than an MSE surface.

### Observation

The loss-landscape idea is independent of the exact loss function.

The geometry changes because the objective changes, but the interpretation is
similar:

```text
Parameter location
       ↓
Model prediction
       ↓
Loss
```

---

# Experiment 7 — GPT-2 Loss Landscape

The second notebook extends the same idea to pretrained GPT-2.

The full GPT-2 parameter space is far too high-dimensional to visualize
directly.

Instead, the experiment constructs a two-dimensional slice.

Two random directions are created:

\[
R_1,\;R_2
\]

and the model parameters are perturbed according to:

\[
\Theta(\alpha,\beta)
=
\Theta_0
+
\alpha R_1
+
\beta R_2
\]

where:

- \(\Theta_0\) is the pretrained model
- \(\alpha\) controls movement along the first direction
- \(\beta\) controls movement along the second direction

---

## Direction Normalization

The experiment uses filter-normalized random directions.

For a weight tensor \(W\), a random direction \(d\) is rescaled so that:

\[
\lVert d\rVert_F
=
\lVert W\rVert_F
\]

This prevents parameter tensors with different scales from affecting the
visualization purely because of their magnitude.

---

## Perturbed GPT-2 Parameters

Recorded experiment configuration:

```text
GPT-2 total parameters: 124,439,808
Perturbed tensors:      26
Perturbed parameters:   81,851,136
```

The experiment perturbs selected embeddings and weight matrices rather than
every model parameter.

---

## Loss at the Pretrained Model

Before perturbation, GPT-2 was evaluated at its pretrained parameters.

Recorded result:

```text
Loss at origin: 3.9607
```

This provides the reference location:

\[
(\alpha,\beta)=(0,0)
\]

---

## Landscape Sampling

The full experiment used:

```text
GRID_SIZE = 100
RANGE     = 1.0
```

Therefore:

\[
100\times100
=
10{,}000
\]

different parameter configurations were evaluated.

The sampled directions covered:

\[
\alpha,\beta\in[-1,1]
\]

Recorded full-sweep runtime:

```text
approximately 25 min 27 s
```

Recorded loss range:

```text
minimum = 4.045
maximum = 84.741
```

The recorded center-grid value was:

```text
4.083
```

---

## Origin vs Grid Center

The separately evaluated pretrained loss was:

```text
3.9607
```

while the grid-center entry was:

```text
4.083
```

These values are not expected to be identical because the grid contained an
even number of points.

A `100 × 100` grid across `[-1,1]` does not contain an exact sample at:

\[
\alpha=0,\qquad\beta=0
\]

The separately evaluated origin is therefore the direct measurement of the
pretrained model.

---

## GPT-2 Visualizations

Two views were generated from the sampled landscape.

### 3D Surface

The three axes represent:

```text
x → α
y → β
z → log cross-entropy loss
```

This shows how rapidly loss increases as the parameters move away from the
pretrained solution.

### 2D Contour

The contour representation gives a top-down view of the same parameter-space
slice.

The pretrained solution is marked at:

\[
(0,0)
\]

This makes the local geometry around the pretrained model easier to inspect.

---

# Connection Between the Small and Large Experiments

The perceptron experiment uses:

\[
L(w,b)
\]

while the GPT-2 experiment studies a slice of:

\[
L(\Theta)
\]

The fundamental idea is the same.

```text
Small perceptron

2 parameters
    ↓
easy to visualize directly
```

versus:

```text
GPT-2

millions of parameters
    ↓
visualize only a 2D slice
```

The GPT-2 experiment is therefore a scaled-up version of the same conceptual
idea introduced using weight and bias.

---

# Connection to Language Modeling

GPT-style language models perform next-token classification.

At each token position:

```text
Hidden representation
        ↓
Vocabulary logits
        ↓
Cross Entropy
        ↓
Actual next token
```

The digit-classification example used ten possible classes.

A language model applies the same general idea to an entire vocabulary.

This provides a direct connection between the small classification experiment
and the objective used to train language models.

---

# Main Findings

1. Residual error provides directional prediction error.

2. MSE turns residuals into a non-negative optimization objective.

3. Loss can be visualized as a surface over model parameters.

4. Different targets generate different loss curves even for the same model.

5. Classification models generate logits rather than a single scalar
   prediction.

6. Cross entropy measures more than whether `argmax` selected the correct
   class.

7. The loss-landscape concept applies to both regression and classification.

8. Large-model loss landscapes are too high-dimensional to visualize directly.

9. Two-dimensional parameter perturbations provide a useful local view of a
   large-model objective.

10. The GPT-2 experiment evaluated 10,000 parameter configurations.

11. Moving away from the pretrained GPT-2 parameters produced much larger
    cross-entropy loss across the sampled directions.

12. Loss defines what should be minimized but does not determine how parameters
    are updated.

---

# Limitations

The GPT-2 landscape experiment visualizes only one two-dimensional slice
through a very high-dimensional parameter space.

Therefore:

- it is not a complete representation of the full loss landscape
- different random directions can produce different slices
- only selected parameter tensors were perturbed
- the measured surface depends on the evaluation data
- the grid provides sampled rather than continuous geometry

The visualization should therefore be interpreted as an intuition-building
experiment rather than a complete geometric description of GPT-2's objective.

---

# Conclusion

The experiments move from a scalar residual to the geometry of a large language
model's loss function.

The progression is:

```text
Prediction error
      ↓
Loss function
      ↓
Loss as a function of parameters
      ↓
Loss landscape
      ↓
High-dimensional neural-network objective
```

This unit provides the quantity that the next training concepts will attempt to
minimize.

The model already knows how to produce predictions.

Loss functions tell us how to judge those predictions.
