# Stanford CS336 — Language Modeling from Scratch (Spring 2026)

## Why this resource is included

Stanford CS336 is being used as a complementary academic resource alongside the implementation-first workshop already documented in this repository.

The course follows the full language-model development pipeline: data preparation, tokenization, Transformer construction, training, systems optimization, scaling, evaluation, and alignment. It is especially useful for strengthening the mathematical and systems-level reasoning behind the components implemented in the main numbered learning units.

Official course page: https://cs336.stanford.edu/

Lecture recordings are available through Stanford Online's CS336 Spring 2026 playlist on YouTube.

---

## Relationship to this repository

The numbered `notes/`, `notebooks/`, and `experiments/` folders remain the primary hands-on learning sequence.

CS336 is treated as a **parallel reference track**, not as a replacement for that sequence.

```text
Primary workshop / implementation path
        ↓
Build each LLM component from scratch
        ↓
Document notes + notebooks + experiments

                +

Stanford CS336
        ↓
Deepen theory, systems reasoning, scaling,
data, and end-to-end language-model training
```

When a CS336 lecture directly strengthens a numbered unit, the relevant concept can be cross-referenced in that unit's notes rather than duplicating the whole lecture.

---

## Spring 2026 course scope

The official course describes an end-to-end implementation-heavy path through language modeling, including:

- tokenization
- PyTorch and resource accounting
- architectures and hyperparameters
- attention alternatives and mixture of experts
- GPUs and TPUs
- kernels and Triton
- parallelism and distributed training
- scaling laws
- pretraining data collection and cleaning
- evaluation
- alignment and reasoning with supervised fine-tuning / reinforcement learning

The assignments progress from building a minimal Transformer language model to systems optimization, scaling, data processing, and alignment.

---

## Initial lecture connection

### Lecture 1 — Overview, Tokenization

The first lecture introduces the overall language-modeling pipeline and frames several systems and scaling questions that will become important later in this repository.

One useful conceptual idea is to think about scaling not as selecting one fixed configuration, but as learning a **scaling recipe** that maps available compute to appropriate hyperparameters.

```text
Available compute (FLOPs)
        ↓
Scaling recipe
        ↓
Model/data/training hyperparameters
```

At large scale, exhaustive hyperparameter tuning is too expensive. Instead, smaller-scale experiments can be used to estimate how loss and training behavior scale, then extrapolate toward a larger target training budget.

This connects naturally to topics already studied here:

- GPU performance and resource accounting
- training hyperparameters
- optimizer and learning-rate decisions
- model capacity
- checkpointing and long-running training

and it prepares for later topics such as:

- scaling laws
- distributed training
- training-system design
- pretraining compute allocation

---

## How this resource will be documented

Rather than copying lecture material, this repository will record:

1. concepts that materially improve understanding of the current LLM-from-scratch implementation,
2. connections between CS336 and the numbered workshop units,
3. selected experiments or derivations reproduced independently,
4. systems/scaling insights that are not covered deeply in the primary workshop.

This keeps the repository focused on original learning artifacts rather than course-material duplication.
