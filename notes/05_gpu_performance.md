# 05 — GPU Performance

## Overview

This unit explores how the same mathematical computation can have very different
performance characteristics depending on how it is executed.

The experiments use the GELU activation function and compare several execution
levels:

1. Pure Python on CPU
2. Custom PyTorch operations on GPU
3. PyTorch with `torch.compile`
4. PyTorch built-in GELU
5. Custom CUDA C++ kernel
6. Triton kernel

The goal is not simply to determine which implementation is fastest, but to
understand why performance changes across different software and hardware
execution strategies.

---

## Core Idea

Moving a computation to the GPU does not automatically make it optimal.

Performance depends on factors such as:

- degree of parallelism
- workload size
- kernel launch overhead
- kernel fusion
- memory access patterns
- intermediate memory traffic
- synchronization
- implementation quality
- hardware architecture

The same mathematical function can therefore have significantly different
execution times even when all implementations run on the same GPU.



---

## From Chip Design to GPU Performance

Understanding GPU performance becomes much clearer when viewed from the bottom up.

Modern AI workloads eventually reduce to large numbers of relatively simple
numerical operations executed repeatedly and in parallel.

A simplified hierarchy is:

```text
Transistors
    ↓
Logic gates
    ↓
Adders / Multipliers
    ↓
Multiply-Accumulate operations (MACs)
    ↓
Vector and matrix operations
    ↓
Neural network layers
    ↓
GPU / accelerator execution
    ↓
Large language models

Logic and Arithmetic

At the hardware level, computation is built from basic logic gates such as:

AND
OR
NOT
NAND

These gates can be combined to construct arithmetic circuits such as adders
and multipliers.

A particularly important operation for machine learning is the
Multiply-Accumulate operation:

a×b+c

Large matrix multiplications consist of enormous numbers of these operations.

For example, matrix multiplication can conceptually be represented as:

for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i][j] += A[i][k] * B[k][j]

The mathematical operation is simple, but neural networks require this type of
computation at massive scale.

Parallelism

One major reason GPUs are effective for machine learning is that many
independent operations can be executed simultaneously.

Instead of processing every value sequentially, GPU hardware contains many
parallel execution resources that can operate on different pieces of data at
the same time.

This makes workloads such as:

matrix multiplication
vector operations
activation functions
attention operations

well suited to GPU execution.

However, having parallel hardware available does not automatically guarantee
high performance.

The software must expose enough parallel work and organize it efficiently.

Clock Cycles and Throughput

Digital hardware executes according to clock cycles.

Between clock edges, combinational logic performs work, and the resulting
values can then be stored in registers.

A simplified view of performance is:

Throughput≈Work per clock cycle×Clock cycles per second

This means that a processor with more useful parallel work per cycle can
achieve higher throughput even if clock frequency alone does not increase.

The slowest register-to-register combinational path is called the
critical path.

It constrains the minimum clock period:

T
clock
	​

≥T
critical path
	​


and therefore limits the maximum possible clock frequency.

This is a hardware timing concept and should not be confused with software
kernel execution or kernel launch overhead.

Compute Is Only Part of Performance

Arithmetic units are not the only important part of a processor.

Data must also move through a memory hierarchy.

A simplified view is:

Compute units
      ↕
Registers / local storage
      ↕
Caches / shared memory
      ↕
Device memory
      ↕
Host memory

Moving data can sometimes cost as much as, or more than, performing the
arithmetic itself.

For this reason, accelerator performance depends on both:

how quickly computation can be performed
how efficiently data can be supplied to the computation

This becomes especially important in GPU programming, where unnecessary
intermediate tensors or repeated memory accesses can reduce performance.

Repeated Hardware and Specialized Accelerators

Modern accelerators often contain repeated computational structures.

GPUs use many parallel execution units organized into larger processing
groups, while specialized AI accelerators can dedicate significant hardware
to matrix operations.

This reflects an important architecture principle:

Hardware architecture is designed around the workloads it is expected to run.

Different workloads may therefore benefit from different architectural
choices.

FPGA vs ASIC

Two useful hardware design approaches are:

FPGA

reconfigurable
useful for experimentation and prototyping
flexible
generally less efficient than a custom chip

ASIC

designed for a specific purpose
fixed after fabrication
expensive to design
can achieve significantly better efficiency and performance

This illustrates a general trade-off between:

flexibility↔specialization

The same trade-off also appears at the software level.

High-level frameworks provide flexibility and convenience, while custom GPU
kernels allow more control over execution.

Connection to This Unit

The GPU performance experiment in this unit applies these hardware ideas at
the software execution level.

The mathematical operation remains the same:

GELU(x)

but its execution changes across several implementations:

Python CPU
    ↓
PyTorch GPU
    ↓
torch.compile
    ↓
Optimized PyTorch kernel
    ↓
Custom CUDA kernel
    ↓
Triton kernel

The experiment therefore asks a more useful question than simply:

"Is the GPU faster?"

Instead, it asks:

"How does the way we organize the same computation affect how efficiently
the hardware can execute it?"
