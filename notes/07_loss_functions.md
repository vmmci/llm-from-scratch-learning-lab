# 07 — Loss Functions

## Overview

A neural network can produce an output immediately after initialization, but
that output is not necessarily useful.

The model needs a way to quantify how far its prediction is from the desired
result.

This is the role of a **loss function**.

A loss function converts the difference between prediction and target into a
numerical value that can later be minimized during training.

A useful mental model is:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Compare with target
  ↓
Loss
```

The lower the loss, the better the model is performing according to the chosen
objective.

---

# Loss Is a Measure, Not an Update Rule

A loss function tells us how poor a prediction is.

It does not, by itself, tell us how to change the model parameters.

Conceptually:

```text
Model prediction
      ↓
Loss function
      ↓
How wrong are we?
```

Later optimization methods will use this information to change weights and
biases.

This distinction is important:

> The loss defines what should be minimized.  
> The optimizer determines how the parameters move toward lower loss.

---

# Prediction Error

Suppose the expected value is:

\[
y
\]

and the model prediction is:

\[
\hat{y}
\]

A simple residual error is:

\[
e = y - \hat{y}
\]

The sign of the residual contains directional information.

For example, in the notebook:

```text
weight = 0.5
bias   = 0.2
input  = 1.0
```

The perceptron predicts:

\[
\hat{y}
=
0.5(1.0)+0.2
=
0.7
\]

while:

\[
y=0.5
\]

Therefore:

\[
e
=
0.5-0.7
=
-0.2
\]

The negative sign indicates that the prediction is above the expected value.

---

# Residual Error vs Absolute Error

The mathematical residual in the experiment was:

```text
-0.2000
```

PyTorch's:

```python
nn.L1Loss()
```

returned:

```text
0.2000
```

These are not contradictory.

The residual is:

\[
y-\hat{y}
\]

while L1 loss uses the absolute distance:

\[
|y-\hat{y}|
\]

Therefore:

```text
Residual: -0.2
L1 loss:   0.2
```

The residual preserves direction.

The absolute loss preserves magnitude but removes the sign.

---

# Common Regression Error Measures

The unit introduces several related ways of measuring prediction error.

## Mean Error

For \(n\) predictions:

\[
\text{Mean Error}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)
\]

Because positive and negative errors can cancel each other, mean error is not
always a good measure of total prediction quality.

---

## Mean Squared Error

Mean Squared Error squares each residual before averaging:

\[
\text{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

Squaring has two important effects:

- negative and positive residuals no longer cancel
- larger errors receive a stronger penalty

For the notebook example:

\[
e=-0.2
\]

so:

\[
\text{MSE}
=
(-0.2)^2
=
0.04
\]

The manual calculation and PyTorch both produced:

```text
Math MSE:    0.0400
PyTorch MSE: 0.0400
```

---

## Root Mean Squared Error

RMSE takes the square root of MSE:

\[
\text{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

RMSE returns the error to the same units as the original target variable.

A useful progression is:

```text
Residual
   ↓
Square
   ↓
Mean
   ↓
Square root
```

---

# Loss Landscape

A model usually contains many adjustable parameters.

For a simple perceptron:

\[
\hat{y}=wx+b
\]

the loss depends on both:

- weight \(w\)
- bias \(b\)

Therefore the loss can be written as:

\[
L(w,b)
\]

If we evaluate many combinations of \(w\) and \(b\), we can visualize the
result as a surface.

Conceptually:

```text
Weight
   ↘
     Loss
   ↗
Bias
```

The height of the surface represents how poorly the model performs at each
parameter combination.

---

# MSE Loss Landscape Experiment

The notebook evaluates a perceptron across a grid of weights and biases.

The grid used:

```text
weight: -2 → 3
bias:   -2 → 2
```

For each point:

```text
1. Set weight
2. Set bias
3. Run the perceptron
4. Compare prediction with target
5. Calculate MSE
6. Store the loss
```

This produces a three-dimensional loss surface.

For the simple perceptron:

\[
\hat{y}=wx+b
\]

and squared error:

\[
L(w,b)
=
(wx+b-y)^2
\]

The loss surface therefore describes how prediction quality changes as the
parameters change.

---

# Visualizing Loss for Different Targets

The exercise also fixed:

```text
weight = 0.5
bias   = 0.2
```

and varied the input while plotting squared error for several expected values:

```text
y_expected = -1.0
y_expected =  0.0
y_expected =  0.5
y_expected =  1.0
y_expected =  2.0
```

The prediction is:

\[
\hat{y}=0.5x+0.2
\]

so the loss for a given expected value is:

\[
L(x)
=
\left[
y_{\text{expected}}
-
(0.5x+0.2)
\right]^2
\]

Each expected value therefore generates its own loss curve.

The minimum occurs when:

\[
y_{\text{expected}}
=
\hat{y}
\]

because then:

\[
L=0
\]

This visualization makes loss easier to interpret geometrically.

---

# From Regression to Classification

Squared error is useful when the target is a continuous numerical value.

Classification introduces a different problem.

Instead of predicting one continuous number, a model may produce one score for
each possible class.

For digit recognition:

```text
Class 0
Class 1
Class 2
...
Class 9
```

the model produces ten outputs.

These raw class scores are called **logits**.

---

# A Simple 1 → 10 Digit Recognizer

The notebook constructs a small educational model that receives one scalar
value and generates ten logits.

Conceptually:

```text
One input
   ↓
Internal scalar value
   ↓
10 logits
   ↓
Predicted digit
```

The model computes:

\[
z=wx+b
\]

and then constructs one logit for each digit.

The largest logit determines the predicted class:

```python
torch.argmax(logits)
```

For the example:

```text
input        = 1.3
weight       = 1.0
bias         = 0.0
target class = 1
```

the largest logit occurred at class:

```text
1
```

so the model predicted the correct digit.

---

# Logits Are Not Probabilities

The outputs before probability normalization are called logits.

A larger logit means a class is preferred relative to other classes, but the
values are not themselves probabilities.

Conceptually:

```text
Model
  ↓
Logits
  ↓
Relative class scores
```

For classification loss, these logits can be passed directly to PyTorch's
cross-entropy implementation.

---

# Cross-Entropy Loss

For multi-class classification, the unit uses **Cross Entropy**.

A simplified categorical form is:

\[
L
=
-
\sum_{k}
y_k
\log(p_k)
\]

where:

- \(y_k\) identifies the correct class
- \(p_k\) is the predicted probability assigned to that class

The idea is simple:

```text
High probability on correct class
            ↓
         Low loss

Low probability on correct class
            ↓
         High loss
```

In PyTorch:

```python
F.cross_entropy(logits, target_class)
```

combines the class-score normalization and negative log-likelihood calculation.

For the digit example, the notebook produced:

```text
Cross-entropy loss = 0.8870
```

---

# Classification Loss Is More Than Argmax

Classification accuracy and cross-entropy measure different things.

`argmax` answers:

> Which class has the largest score?

Cross entropy asks something stronger:

> How strongly does the model support the correct class relative to the
> alternatives?

Two models can predict the same class while having different cross-entropy
losses.

A model that assigns much stronger support to the correct class can have lower
loss even when both models produce the same `argmax`.

---

# Cross-Entropy Loss Landscape

The notebook also constructs a loss landscape for the digit classifier.

Again, the model parameters are varied over a grid:

```text
weight: -2 → 3
bias:   -2 → 2
```

At each point:

```text
set weight
   ↓
set bias
   ↓
calculate logits
   ↓
cross entropy
   ↓
store loss
```

This gives:

\[
L(w,b)
\]

for the classification problem.

The resulting surface shows how the classification objective changes across
parameter space.

---

# Choosing a Loss Function

Different tasks require different loss functions.

A useful distinction from the unit is:

```text
Regression
    ↓
continuous target
    ↓
Residual / MAE / MSE / RMSE

Classification
    ↓
discrete class target
    ↓
Cross Entropy
```

There are many other possible loss functions.

The important point is that the loss should match the prediction task.

---

# Loss Landscapes and Optimization

Loss landscapes can become much more complicated as neural networks become
larger.

A simple perceptron may produce a smooth and easily visualized surface.

A large neural network contains millions or billions of parameters.

Its full loss function therefore exists in a parameter space with an enormous
number of dimensions.

We cannot visualize the complete space directly.

Instead, researchers often visualize two-dimensional slices through the
high-dimensional parameter space.

---

# From Simple Loss Surfaces to GPT-2

The second notebook applies the loss-landscape idea to pretrained GPT-2.

The goal is not to visualize all GPT-2 parameters simultaneously.

Instead, it constructs a two-dimensional slice through parameter space.

A simplified view is:

```text
Pretrained parameters Θ₀
          ↓
Choose direction R₁
Choose direction R₂
          ↓
Move by α along R₁
Move by β along R₂
          ↓
Evaluate loss
          ↓
Build 2D loss surface
```

The perturbed parameters are:

\[
\Theta(\alpha,\beta)
=
\Theta_0
+
\alpha R_1
+
\beta R_2
\]

Each pair:

\[
(\alpha,\beta)
\]

therefore corresponds to a modified version of the model.

---

# Filter-Normalized Random Directions

The GPT-2 notebook follows the filter-normalized random-direction approach
described in *Visualizing the Loss Landscape of Neural Nets*.

Two random directions are generated:

\[
R_1,\ R_2
\]

For each weight tensor \(W\), the random direction \(d\) is rescaled so that:

\[
\lVert d\rVert_F
=
\lVert W\rVert_F
\]

where:

\[
\lVert\cdot\rVert_F
\]

is the Frobenius norm.

The purpose is to keep perturbation scale meaningful across parameters with
different magnitudes.

Without normalization, large parameter tensors or layers could dominate the
visualization simply because of scale.

---

# Partial GPT-2 Perturbation

The notebook does not perturb every GPT-2 parameter tensor.

It perturbs:

- token embeddings
- positional embeddings
- weight matrices from the first half of the transformer blocks

The notebook reported:

```text
26 parameter tensors
81,851,136 perturbed parameters
```

The pretrained GPT-2 model itself is the 124M-parameter GPT-2 configuration.

This experiment therefore visualizes a controlled slice of a large model rather
than modifying every parameter simultaneously.

---

# Evaluating GPT-2 Loss

For each modified parameter configuration:

\[
\Theta_0+\alpha R_1+\beta R_2
\]

the model is evaluated on text batches.

The notebook computes the model's average cross-entropy loss:

```python
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
```

This directly connects the simple classification cross-entropy examples to the
loss used by a language model.

For a causal language model, the objective measures how well the model predicts
the next token.

---

# GPT-2 Loss at the Pretrained Parameters

Before applying perturbations, the notebook restores the pretrained model and
evaluates its loss.

The reported value was:

```text
Loss at origin: 3.9607
```

This corresponds to the unperturbed pretrained parameters:

\[
\Theta_0
\]

and provides a reference point for the landscape.

---

# Sampling the GPT-2 Loss Landscape

The final run used:

```text
GRID_SIZE = 100
RANGE     = 1.0
```

which produced:

\[
100\times100
=
10{,}000
\]

model evaluations.

The sweep explores:

\[
\alpha,\beta\in[-1,1]
\]

For every grid point:

```text
Perturb GPT-2 weights
        ↓
Run evaluation batches
        ↓
Calculate cross-entropy loss
        ↓
Store result
```

The complete 10,000-point sweep took approximately:

```text
25 minutes 27 seconds
```

on the CUDA environment used for the notebook.

The recorded loss range was:

```text
minimum: 4.045
maximum: 84.741
```

The notebook also reported the grid's center entry as:

```text
4.083
```

---

## Sampling Note

The pretrained loss was evaluated separately and produced:

```text
3.9607
```

while the 100×100 sampled grid reported a center entry of:

```text
4.083
```

Because the sweep uses an even number of samples across `[-1, 1]`, the grid
does not sample exactly \(\alpha=0,\beta=0\).

Therefore the separately evaluated origin loss is the direct measurement at
the pretrained parameters.

---

# Visualizing the GPT-2 Landscape

The notebook produces two views.

## 3D Surface

The axes represent:

```text
x-axis → α
y-axis → β
z-axis → log(Cross-Entropy Loss)
```

The loss is log-scaled for better visual contrast.

This produces a three-dimensional view of how quickly loss changes as the
weights move away from the pretrained model.

---

## 2D Contour

The same data is also displayed from above as a contour plot.

The pretrained parameter location is marked at:

\[
(\alpha,\beta)=(0,0)
\]

This view makes local geometry easier to inspect.

---

# Why Loss Landscapes Matter

Training can be interpreted as searching parameter space for configurations
with lower loss.

Conceptually:

```text
Current parameters
       ↓
Current loss
       ↓
Change parameters
       ↓
New loss
       ↓
Repeat
```

The loss surface gives geometric intuition for this optimization problem.

However, the loss function itself does not provide the movement rule.

Later units introduce the mechanisms that determine how parameters are updated.

---

# GPT-2 Style Architecture Decision

For GPT-style language modeling, the unit chooses:

```text
Cross-Entropy Loss
```

because the model performs next-token prediction.

For every token position, the model outputs logits over the vocabulary.

The target is the correct next token.

Conceptually:

```text
Current context
      ↓
GPT
      ↓
Vocabulary logits
      ↓
Cross Entropy
      ↓
Compare against actual next token
```

This connects the simple 10-class digit example directly to language modeling.

The number of classes is simply much larger:

```text
Digit classifier:
10 possible classes

Language model:
entire vocabulary of possible next tokens
```

---

# Connection to Previous Units

The roadmap now becomes:

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
```

The earlier units explained how the model produces an output.

This unit introduces the quantity that tells us whether that output is good.

That creates the missing link required for training:

```text
Model parameters
       ↓
Forward computation
       ↓
Prediction
       ↓
Loss
       ↓
Training signal
```

The next optimization-related concepts can now use this loss to change the
model parameters.

---

# Key Takeaways

1. A randomly initialized model can produce outputs before it has learned
   anything useful.

2. A loss function quantifies disagreement between prediction and target.

3. Residual error preserves the sign of the prediction error.

4. L1 loss measures absolute distance and therefore removes that sign.

5. MSE squares the residual and penalizes larger errors more strongly.

6. RMSE returns squared error to the original target scale.

7. A loss landscape describes loss as a function of model parameters.

8. Even a single perceptron can be visualized as a surface over weight and bias.

9. Classification models produce logits over possible classes.

10. Cross entropy measures how well the model supports the correct class.

11. Correct classification alone does not imply minimal cross-entropy loss.

12. Different prediction tasks require different loss functions.

13. Large neural-network loss landscapes exist in extremely high-dimensional
    parameter spaces.

14. Two-dimensional parameter-space slices can provide intuition about the
    geometry of these high-dimensional objectives.

15. The GPT-2 experiment perturbed the pretrained model along two
    filter-normalized random directions.

16. The GPT-2 experiment sampled a 100×100 grid, corresponding to 10,000 model
    evaluations.

17. Cross entropy is the loss used in this workshop's GPT-style next-token
    prediction architecture.

18. A loss function defines what the model should minimize, but does not itself
    specify how the weights should be updated.

---

# Final Mental Model

A useful way to think about this unit is:

```text
Prediction
   ↓
Compare with reality
   ↓
Loss
   ↓
Scalar measure of error
```

For a small model:

```text
weight + bias
      ↓
prediction
      ↓
MSE
      ↓
simple loss surface
```

For a language model:

```text
millions of parameters
       ↓
token logits
       ↓
cross entropy
       ↓
high-dimensional loss landscape
```

Loss therefore provides the bridge between **forward computation** and
**learning**.

The model can already calculate predictions.

The loss tells us how those predictions should be judged.
