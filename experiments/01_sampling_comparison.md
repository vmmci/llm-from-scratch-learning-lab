 # 01 - Sampling Comparison

## Prompts Used

```text
You should eat more
The capital of Texas is
The currency of the United States is the
The capital of France is
the meaning of life is

| Setting                  | Prompt                                     | Output Summary                                                         | Observation                                                                                                                  |
| ------------------------ | ------------------------------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| One-token generation     | `You should eat more`                      | `You should eat more fruits`                                           | The model predicted one next token and completed the phrase with a plausible word.                                           |
| Two-token generation     | `The capital of Texas is`                  | `Austin`                                                               | The answer required more than one token internally, but the decoded output is the expected city name.                        |
| Next-token probabilities | `The currency of the United States is the` | Top token: `' dollar'` with probability `49.1811%`                     | Before generating, the model assigns probabilities to many possible next tokens.                                             |
| Temperature = 0.8        | `The capital of France is`                 | `Paris. It's where the French people meet...`                          | Temperature sampling made the output less deterministic and allowed a more open-ended continuation.                          |
| Temperature = 1          | `The currency of the United States is the` | `The currency of the United States is the dollar`                      | The sampled token was still the most natural completion.                                                                     |
| Top-K = 50               | `The currency of the United States is the` | `The currency of the United States is the dollar`                      | Top-K limited sampling to the 50 most likely next tokens.                                                                    |
| Top-P = 0.9              | `The currency of the United States is the` | `The currency of the United States is the "`                           | Top-P sampled from a cumulative probability set; the output was less stable in this run.                                     |
| Greedy decoding          | `the meaning of life is`                   | `the meaning of life is to be free from fear, to be free from pain...` | With `do_sample=False`, the model selected the most likely next token at every step. The result was coherent but repetitive. |
| Exercise answer          | `The meaning of life is`                   | `The meaning of life is to be happy. To be happy is to be free...`     | Greedy generation produced a meaningful continuation, but repetition appeared again.                                         |

Main Observation
LLMs generate text one token at a time.

The model first produces logits, then these logits are converted into probabilities.
Sampling strategies control how the next token is selected from those probabilities.

Greedy decoding is deterministic because it always selects the most likely token.
Temperature, Top-K, and Top-P introduce controlled randomness without changing the model weights.

Takeaway

Sampling does not retrain the model.
It only changes the decoding strategy used to choose the next token.
