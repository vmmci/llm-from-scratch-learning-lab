# Stanford CS336 — Lecture 01: Language Models From Scratch

> Source note: the uploaded executable lecture identifies itself as **CS336: Language Models From Scratch (Spring 2024)**. It is being used here as a complementary conceptual source alongside the newer Stanford CS336 material already tracked in this repository.

## Why This Lecture Matters

The lecture's central philosophy is **understanding via building**.

Rather than treating language models as APIs that we only prompt, the course asks us to reconstruct the pipeline and understand the engineering decisions underneath it.

The main question is not:

> How do I use an existing LLM?

It is:

> If I had to build and train a language model myself, what would I need to understand and what decisions would I need to make?

This makes CS336 a useful complement to the Justin Angel workshop used as the primary numbered implementation track in this repository.

---

# 1. Executable Lectures

The source itself is an **executable lecture**: lecture content is represented as Python code.

This structure allows the lecture to combine:

- explanations,
- executable examples,
- references,
- code navigation,
- and hierarchical organization.

The lecture therefore treats code not only as an implementation artifact, but also as a medium for explaining a technical system.

---

# 2. Why Build Language Models From Scratch?

The lecture argues that abstraction has progressively increased in AI research.

A simplified progression is:

```text
Earlier research
    ↓
Implement and train the model directly
    ↓
Download pretrained models and fine-tune them
    ↓
Prompt large hosted models
```

Higher-level abstractions improve productivity, but they can hide important lower-level behavior.

The lecture therefore focuses on rebuilding research-engineering knowledge underneath those abstractions.

## Three Types of Knowledge

The lecture separates knowledge into three categories:

### Mechanics

How systems work.

Examples:

- what a Transformer is,
- how distributed training mechanisms work,
- how tokenization works,
- how optimization is implemented.

### Mindset

How to reason about systems and scale.

Examples:

- squeezing performance from hardware,
- thinking in terms of resource constraints,
- using scaling laws,
- evaluating engineering trade-offs.

### Intuition

Empirical judgment about what settings are likely to work well.

Examples:

- hyperparameter choices,
- data-processing decisions,
- architecture choices,
- training configurations.

A key point is that **mechanics and mindset transfer more reliably across scales than intuition**.

A hyperparameter choice that works for a small experiment may not transfer directly to a much larger model.

---

# 3. Learning at Small Scale vs Building at Large Scale

The course is not attempting to reproduce a frontier-scale model.

Instead, it asks:

> What can we learn at small scale that still generalizes to large-scale language-model engineering?

This distinction is important.

Small models are useful for learning:

- the architecture,
- the training pipeline,
- performance bottlenecks,
- system behavior,
- scaling methodology,
- and debugging practices.

But some large-scale design decisions remain empirical.

This is why experiments remain important even when the mathematical story appears convincing.

---

# 4. A Compressed History of Language Modeling

The lecture places modern LLMs inside a longer progression.

The major conceptual stages include:

```text
Statistical language modeling
        ↓
n-gram models
        ↓
neural language modeling
        ↓
sequence-to-sequence models
        ↓
attention
        ↓
Transformer
        ↓
large-scale autoregressive models
        ↓
scaling laws
        ↓
compute-optimal training
        ↓
modern open and frontier models
```

The examples referenced in the lecture include milestones such as:

- Shannon-style language entropy work,
- neural language modeling,
- sequence-to-sequence modeling,
- attention,
- the Transformer,
- GPT-2,
- T5,
- GPT-3,
- scaling laws,
- The Pile / GPT-J,
- OPT,
- BLOOM,
- PaLM,
- Chinchilla,
- LLaMA,
- Mistral,
- mixture-of-experts models,
- and modern frontier systems.

The historical trend is not only "more parameters."

The emphasis evolved roughly from:

```text
Parameter count
      ↓
Compute-optimal training
      ↓
Training smaller models for more tokens / better efficiency
```

This becomes important later when studying scaling laws and compute budgets.

---

# 5. Efficiency as the Unifying Perspective

The most important systems idea introduced in this lecture is that language-model development is fundamentally constrained by resources.

The lecture frames the central engineering question as:

> Given fixed resources, how do we train the best model possible?

The relevant resources include:

```text
Data
+
Compute
+
Memory
+
Communication
```

A concrete way to think about this is:

```text
Given a dataset
+
Given a fixed hardware cluster
+
Given a fixed training time
        ↓
What design choices maximize model quality?
```

This means architecture decisions cannot be separated completely from systems constraints.

---

# 6. The End-to-End Language-Model Pipeline

The lecture presents a stylized pipeline with three major stages:

```text
Data
  ↓
Pretraining
  ↓
Alignment
```

Expanded:

```text
Raw data
   ↓
Data processing
   ↓
Pretraining dataset
   ↓
Tokenizer training
   ↓
Transformer language model
   ↓
Pretraining
   ↓
Instruction data
   ↓
Instruction tuning
   ↓
Preference data
   ↓
Preference optimization
```

This pipeline is extremely useful because it gives us a map for the entire LLM lifecycle.

---

# 7. Data Does Not "Just Exist"

The lecture emphasizes that raw training data must be acquired and transformed.

Possible sources include:

- web pages,
- books,
- research papers,
- code repositories,
- and other document collections.

Raw data can arrive as:

```text
HTML
PDF
Directories
Structured files
```

not necessarily clean text.

## Data Processing

The lecture highlights three important operations:

### Filtering

Keep useful data and remove undesirable or low-quality content.

### Deduplication

Avoid spending compute repeatedly training on duplicated content and reduce unwanted memorization effects.

### Conversion

Convert source formats such as HTML into text while preserving useful structure and content.

A useful systems interpretation is:

```text
Bad data
   ↓
wasted training compute
```

Therefore data quality is also a compute-efficiency problem.

---

# 8. Tokenization as a Compute Trade-off

The lecture defines tokenization as converting text into integer token sequences.

The course uses **Byte-Pair Encoding (BPE)** as its tokenizer approach.

A central trade-off is between:

```text
Vocabulary size
      ↕
Compression ratio / sequence length
```

A very small representation may create long token sequences.

A larger vocabulary can compress text more aggressively, but introduces other costs.

The lecture explicitly connects tokenization to hardware efficiency: processing raw bytes may be conceptually elegant, but it can be inefficient for today's architectures because it increases the amount of sequence computation required.

So tokenization is not merely a text-preprocessing choice.

It is also part of the compute budget.

---

# 9. Transformer Architecture Is a Design Space

The lecture introduces a Transformer language model but emphasizes that "Transformer" does not refer to one permanently fixed architecture.

Modern variants can change components such as:

- normalization placement,
- activation / gated FFNs,
- RMSNorm,
- Rotary Position Embeddings (RoPE),
- grouped-query attention,
- and parallelized layer designs.

This produces an important engineering mindset:

```text
Transformer
≠
one immutable implementation
```

Instead:

```text
Transformer family
    ↓
multiple architecture decisions
    ↓
trade-offs in quality, stability, memory, and throughput
```

---

# 10. Pretraining Is More Than Calling `train()`

The lecture's pretraining stub highlights several major decisions:

- optimizer,
- learning-rate schedule,
- batch size,
- number of attention heads,
- hidden dimension,
- and other hyperparameters.

Conceptually:

```text
Tokenizer
+
Model architecture
+
Training data
+
Optimizer
+
Learning-rate schedule
+
Batching
+
Compute budget
        ↓
Pretrained language model
```

This connects model design and training-system design directly.

---

# 11. Instruction Tuning

After pretraining, the lecture introduces instruction data as pairs of:

```text
(prompt, response)
```

The idea is that a pretrained model may already contain many useful capabilities, but instruction tuning helps surface those capabilities in a form that follows user requests.

The supervised objective can be expressed conceptually as maximizing:

\[
p(\text{response}\mid\text{prompt})
\]

This creates the transition from a base language model to an instruction-following model.

---

# 12. Preference Data and Alignment

The next stage generates multiple candidate responses for a prompt and collects preference information.

A simplified preference example is:

```text
Prompt
  ↓
Response A
Response B
  ↓
Human / preference signal
  ↓
A preferred over B
```

The lecture mentions two alignment approaches:

### PPO

A reinforcement-learning-based approach used historically in instruction-following systems.

### DPO

A simpler direct preference-optimization approach that learns from preference pairs without reproducing the entire PPO pipeline.

The pipeline therefore becomes:

```text
Pretraining
   ↓
Instruction tuning
   ↓
Preference optimization
```

---

# 13. Hardware-Bound Thinking

The lecture repeatedly returns to hardware efficiency.

It frames several design decisions through this lens:

### Data processing

Do not waste accelerator compute on poor data.

### Tokenization

Avoid unnecessarily long representations that increase sequence computation.

### Architecture

Choose implementations that keep accelerators effectively utilized.

### Training

Use the available data and compute efficiently rather than thinking only in terms of more epochs.

### Scaling laws

Run smaller experiments to reason about settings before spending the target compute budget.

### Alignment

Better alignment can make a smaller model more useful for the target behavior.

This creates one unified perspective:

\[
\boxed{\text{Model quality is constrained by how efficiently resources are converted into learning.}}
\]

---

# 14. Connection to the Justin Angel Track

The two resources approach the same system from different directions.

```text
Justin Angel
    ↓
Build individual mechanisms for intuition

Stanford CS336
    ↓
Connect those mechanisms into an end-to-end research-engineering system
```

## Cross-Reference Map

| Stanford CS336 Lecture 01 Concept | Justin Angel Track | Connection |
|---|---|---|
| Understanding via building | Unit 02 — Reverse Engineering LLMs | Both reject treating the model as a black box |
| Hardware / compute efficiency | Unit 05 — GPU Performance | GPU utilization, kernels, memory movement, workload efficiency |
| Transformer architecture variants | Units 06 and later architecture units | FFNs become one component of the larger Transformer design space |
| Pretraining objective | Unit 07 — Loss Functions | Cross entropy becomes the language-model training objective |
| Optimizer and LR schedule | Unit 08 — Backpropagation / Training | AdamW, batch size, learning rate, optimization loop |
| Model persistence | Unit 09 — Saving & Loading | Training produces artifacts/checkpoints that must survive the runtime |
| BPE tokenization | Future Unit 15 — Tokenizers | Stanford gives the systems/data framing; Justin builds the mechanism |
| Transformer model | Future Unit 18 — Transformers | Stanford gives the pipeline context; Justin develops the architecture progressively |
| Pretraining | Future Unit 19 — Pretraining | CS336 frames resource allocation and end-to-end training decisions |
| Instruction tuning | Future Unit 21 — Instruct Fine-Tuning | Same post-training stage |
| PPO / DPO | Future Unit 22 — Reinforcement Learning | Preference optimization and alignment |

---

# 15. Connection to What We Have Already Built

Our repository currently contains the following progression:

```text
Perceptron
    ↓
Activation functions
    ↓
GPU performance
    ↓
MLP / FFN
    ↓
Loss functions
    ↓
Backpropagation
    ↓
Saving & loading
```

Lecture 01 provides the larger map around those pieces:

```text
Raw data
    ↓
Tokenization
    ↓
Transformer
    ↓
Pretraining
    ↓
Checkpointed model
    ↓
Instruction tuning
    ↓
Preference optimization
```

This means the Justin track is currently teaching us many of the internal components that will eventually occupy the middle of the Stanford pipeline.

---

# 16. Assignment 1 Connection

The executable lecture ends by pointing to the first CS336 assignment on the basics.

We will treat the assignments as a separate applied track in this repository rather than copying Stanford starter material into our own work.

The intended workflow will be:

```text
Lecture understanding
      ↓
Read assignment specification
      ↓
Implement independently
      ↓
Run experiments / tests
      ↓
Document our implementation and findings
```

Assignment work will be documented separately from lecture notes so that source material and our own implementation remain clearly distinguishable.

---

# Key Takeaways

1. CS336's philosophy is understanding language models by building the system yourself.
2. Mechanics and systems mindset transfer across scales more reliably than small-scale hyperparameter intuition.
3. Large language models should be understood as an end-to-end pipeline rather than only as a Transformer architecture.
4. Data quality, tokenization, architecture, optimization, and alignment are all constrained by resources.
5. Efficiency is the unifying systems perspective of the lecture.
6. Tokenization is both a linguistic representation problem and a compute-efficiency problem.
7. Transformer architecture is a design space rather than a single fixed implementation.
8. Pretraining requires coordinated decisions about model, optimizer, learning rate, batch size, data, and compute.
9. Instruction tuning and preference optimization transform a base model into a more useful instruction-following system.
10. The Justin Angel workshop and Stanford CS336 are complementary: one develops component-level intuition while the other provides the full research-engineering pipeline.

---

# Final Mental Model

The most useful mental model from Lecture 01 is:

```text
Resources
(data + compute + memory + communication)
                 ↓
           Design decisions
                 ↓
Data → Tokenizer → Transformer → Pretraining
                              ↓
                         Base model
                              ↓
                    Instruction tuning
                              ↓
                    Preference tuning
                              ↓
                     Useful LLM system
```

The central engineering problem is therefore not simply:

> How do we build a Transformer?

It is:

> How do we convert limited data and hardware resources into the best language model we can build?

That systems perspective will be used to connect future CS336 lectures to the numbered Justin Angel implementation units in this repository.

---

## Source

- Stanford CS336 executable `lecture_01.py` supplied for this learning lab.
- The source identifies itself as **CS336: Language Models From Scratch (Spring 2024)**.
