# 01 - Intro to LLMs and Sampling

## Goal

Understand how a GPT-style language model generates text one token at a time and how sampling strategies affect the next-token selection.

## Slides Covered

- What are LLMs?
- Why sample?
- Temperature
- Top-K sampling
- Top-P sampling
- Excel demo
- Colab code demo

## Key Concepts

### Autoregressive Generation

An LLM generates text one token at a time.  
After generating a token, the token is added back to the input, and the model predicts the next token again.

### Logits

Logits are the raw output scores from the model before converting them into probabilities.

### Softmax

Softmax converts logits into probabilities.

### Temperature

Temperature controls how sharp or flat the probability distribution becomes.

- Low temperature: more deterministic.
- High temperature: more random and creative.

### Top-K

Top-K keeps only the K most probable tokens and samples from them.

### Top-P

Top-P keeps the smallest set of tokens whose cumulative probability reaches P.

## My Understanding

The model does not generate a full answer at once.  
It repeatedly predicts the next token based on the previous context.

Sampling does not change the model weights.  
It only changes how we choose the next token from the probability distribution.

## Colab Work

Notebook: `notebooks/01_intro_sampling.ipynb`

## Experiment Plan

I will test the same prompt using different decoding settings:

- greedy decoding
- temperature = 0.3
- temperature = 0.8
- temperature = 1.2
- top_k = 20
- top_p = 0.9

## Reference

Based on Justin Angel's "Building LLMs to Develop Intuition" workshop.
