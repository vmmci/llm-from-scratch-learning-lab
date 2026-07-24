# 03-04 - Neuron and Activation Experiments

## Experiment Goal

Compare the output of a basic perceptron before and after applying activation functions.

## Functions Tested

| Function | Formula | Purpose |
|---|---|---|
| Perceptron | `wx + b` | Linear transformation |
| ReLU | `max(0, x)` | Removes negative values |
| ReLU² | `max(0, x)^2` | Amplifies positive values |
| Leaky ReLU | `x if x >= 0 else alpha*x` | Keeps a small negative signal |
| GeLU | smooth activation | Smoothly gates values |
| Tanh | range from -1 to 1 | Squashes values symmetrically |
| Sigmoid | range from 0 to 1 | Squashes values into probabilities-like scale |
| Swish | `x * sigmoid(x)` | Smooth gated activation |
| SwiGLU | gated activation | Used in modern LLM-style architectures |
| Combined ReLU | `ReLU(x) + ReLU(-x)` | Preserves magnitude from both directions |

## Perceptron Observation

The perceptron output is linear.

Changing the weight changes the slope.

Changing the bias shifts the line up or down.

## Activation Function Observations

| Activation | Main Behavior | Observation |
|---|---|---|
| ReLU | Cuts negative values to zero | Simple and sparse, but loses negative information |
| ReLU² | Squares positive ReLU values | Positive values grow faster than standard ReLU |
| Leaky ReLU | Keeps small negative values | Prevents negative inputs from completely disappearing |
| GeLU | Smoothly gates values | Smoother than ReLU and historically common in transformer models |
| Tanh | Outputs between -1 and 1 | Preserves sign but saturates for large values |
| Sigmoid | Outputs between 0 and 1 | Useful for gating but saturates at extremes |
| Swish | Smooth self-gated function | Allows small negative values and smooth growth |
| SwiGLU | Gated activation | More common in modern LLM architectures |
| Combined ReLU | Combines ReLU(x) and ReLU(-x) | Behaves like magnitude preservation |

## Exercise Answers

### Where do activation functions agree?

They agree most clearly near `x = 0` and often show increasing behavior for positive inputs.

ReLU and Leaky ReLU match exactly for positive values.

### Where do activation functions diverge?

They diverge most strongly for negative values.

- ReLU turns negative values into zero.
- Leaky ReLU keeps a small negative signal.
- Tanh keeps negative outputs.
- Sigmoid never becomes negative.
- GeLU and Swish can produce small negative values.

They also diverge for large positive values because ReLU² and SwiGLU grow faster than ReLU.

### Which region is most interesting?

The region near zero is the most interesting.

Small positive and small negative values show how each activation function decides whether the signal should pass, shrink, disappear, or amplify.

### Why would combining activation functions be useful?

Combining activation functions can preserve different kinds of information.

For example, ReLU alone removes negative information, but combining `ReLU(x)` and `ReLU(-x)` can preserve the magnitude of both positive and negative signals.

## Main Takeaway

A perceptron alone is linear.

Activation functions introduce non-linearity, which allows neural networks to represent more complex patterns.
