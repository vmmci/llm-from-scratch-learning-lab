# 02 - Model Inspection

## Models Used

| Model | Purpose |
|---|---|
| `gpt2` | Main model inspected for GPT-style architecture |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Additional Hugging Face model used for architecture inspection |

## GPT-2 Inspection Results

| Inspection Step | Output / Finding | Observation |
|---|---|---|
| Load model | `GPT2LMHeadModel.from_pretrained("gpt2")` | Loaded a pretrained GPT-2 language model from Hugging Face. |
| Count parameters | `124,439,808` parameters | GPT-2 small has over 124M learnable parameters. |
| Print architecture | `GPT2LMHeadModel(...)` | The model contains a transformer backbone and a language modeling head. |
| Token embeddings | `wte: Embedding(50257, 768)` | GPT-2 maps token IDs into 768-dimensional vectors. |
| Positional embeddings | `wpe: Embedding(1024, 768)` | GPT-2 supports a context length of 1024 positions. |
| Transformer blocks | `12 x GPT2Block` | GPT-2 small uses 12 repeated transformer blocks. |
| Attention | `GPT2Attention` | Attention is one of the core components inside each transformer block. |
| MLP / FFN | `GPT2MLP` | Each transformer block also contains a feed-forward network. |
| Normalization | `LayerNorm` | Layer normalization appears inside the transformer blocks. |
| Regularization | `Dropout(p=0.1)` | GPT-2 uses dropout in several places. |
| Activation | `NewGELUActivation` | GPT-2 uses GELU-style activation in the MLP. |

## Unique Components Found

```text
Conv1D
Dropout
Embedding
GPT2Attention
GPT2Block
GPT2LMHeadModel
GPT2MLP
GPT2Model
LayerNorm
Linear
ModuleList
NewGELUActivation

TinyLlama Inspection

The notebook also inspected:

TinyLlama/TinyLlama-1.1B-Chat-v1.0

This helped compare GPT-2-style architecture with a newer causal language model architecture.

Main Observation

Reverse engineering shows that LLMs are built from organized repeated modules, not from one single block.

The repeated transformer block is the main structure that combines attention, MLP layers, normalization, and residual-style processing.

Takeaway

Before building an LLM from scratch, inspecting an existing model helps identify the exact components that need to be studied and implemented.

