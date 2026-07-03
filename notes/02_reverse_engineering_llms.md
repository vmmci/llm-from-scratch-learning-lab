# 02 - Reverse Engineering LLMs

## Goal

Inspect existing GPT-style language models to understand their internal architecture, parameters, modules, and functions.

## Main Idea

Instead of treating an LLM as a black box, this unit examines the model directly in memory.

The goal is to identify the components needed to build a GPT-style LLM from scratch.

## Slides Covered

- Reverse Engineering LLMs
- Model architecture and parameters
- Modules
- Functions
- Reverse engineering a roadmap
- Full roadmap
- Code roadmap
- Learning more about LLM architecture

## What I Did

- Loaded GPT-2 using Hugging Face Transformers.
- Counted the total number of parameters.
- Printed the GPT-2 architecture using `print(model)`.
- Used `torchinfo.summary()` to inspect model structure and shapes.
- Listed the unique module/component types inside the model.
- Visualized the GPT-2 hierarchy.
- Mapped model components to the workshop roadmap.
- Inspected functional operations used inside the model.
- Loaded and inspected another Hugging Face model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## Key Findings

### GPT-2 Parameter Count

GPT-2 contains:

```text
124,439,808 parameters This means the model has over 124 million learnable values.

GPT-2 Main Components
The model inspection showed components such as:

Embedding
GPT2Block
GPT2Attention
GPT2MLP
LayerNorm
Dropout
Conv1D
Linear
ModuleList
NewGELUActivation
My Understanding

A GPT-style LLM is not a single mysterious object.

It is a neural network made of repeated and organized components:

Token embeddings
Positional embeddings
Transformer blocks
Attention layers
MLP / feed-forward layers
Normalization layers
Dropout
Output language modeling head

Reverse engineering helps convert the model from a black box into smaller components that can be studied one by one.

Roadmap Connection

The model inspection explains why the rest of the workshop studies:

Perceptrons
Activation functions
MLP / FFN
Loss functions
Backpropagation
Initialization
Residual connections
Normalization
Regularization
Softmax
Tokenizers
Embeddings
Attention
Transformers
Pretraining
Evaluation
Instruct tuning
Reinforcement learning
