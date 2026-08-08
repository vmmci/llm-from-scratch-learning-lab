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
execution times depending on how it is represented and executed.

---

# From Chip Design to GPU Performance

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
```

## Logic and Arithmetic

At the hardware level, computation is built from basic logic gates such as:

- AND
- OR
- NOT
- NAND

These gates can be combined to construct arithmetic circuits such as adders
and multipliers.

A particularly important operation for machine learning is the
Multiply-Accumulate operation:

$$
a \times b + c
$$

Large matrix multiplications consist of enormous numbers of these operations.

For example, matrix multiplication can conceptually be represented as:

```python
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i][j] += A[i][k] * B[k][j]
```

The mathematical operation is simple, but neural networks require this type of
computation at massive scale.

---

## Parallelism

One major reason GPUs are effective for machine learning is that many
independent operations can be executed simultaneously.

Instead of processing every value sequentially, GPU hardware contains many
parallel execution resources that can operate on different pieces of data at
the same time.

This makes workloads such as:

- matrix multiplication
- vector operations
- activation functions
- attention operations

well suited to GPU execution.

However, having parallel hardware available does not automatically guarantee
high performance.

The software must expose enough parallel work and organize it efficiently.

---

## Clock Cycles and Throughput

Digital hardware executes according to clock cycles.

Between clock edges, combinational logic performs work, and the resulting
values can then be stored in registers.

A simplified view of performance is:

$$
\text{Throughput}
\approx
\text{Work per clock cycle}
\times
\text{Clock cycles per second}
$$

This means that a processor with more useful parallel work per cycle can
achieve higher throughput even if clock frequency alone does not increase.

The slowest register-to-register combinational path is called the
**critical path**.

It constrains the minimum clock period:

$$
T_{\text{clock}}
\ge
T_{\text{critical path}}
$$

and therefore limits the maximum possible clock frequency.

This is a hardware timing concept and should not be confused with software
kernel execution or kernel launch overhead.

---

## Compute Is Only Part of Performance

Arithmetic units are not the only important part of a processor.

Data must also move through a memory hierarchy.

A simplified view is:

```text
Compute units
      ↕
Registers / local storage
      ↕
Caches / shared memory
      ↕
Device memory
      ↕
Host memory
```

Moving data can sometimes cost as much as, or more than, performing the
arithmetic itself.

For this reason, accelerator performance depends on both:

- how quickly computation can be performed
- how efficiently data can be supplied to the computation

This becomes especially important in GPU programming, where unnecessary
intermediate tensors or repeated memory accesses can reduce performance.

---

## Repeated Hardware and Specialized Accelerators

Modern accelerators often contain repeated computational structures.

GPUs use many parallel execution units organized into larger processing
groups, while specialized AI accelerators can dedicate significant hardware
to matrix operations.

This reflects an important architecture principle:

> Hardware architecture is designed around the workloads it is expected to run.

Different workloads may therefore benefit from different architectural
choices.

---

## FPGA vs ASIC

Two useful hardware design approaches are:

### FPGA

- reconfigurable
- useful for experimentation and prototyping
- flexible
- generally less efficient than a custom chip

### ASIC

- designed for a specific purpose
- fixed after fabrication
- expensive to design
- can achieve significantly better efficiency and performance

This illustrates a general trade-off between:

$$
\text{flexibility}
\quad \leftrightarrow \quad
\text{specialization}
$$

The same trade-off also appears at the software level.

High-level frameworks provide flexibility and convenience, while custom GPU
kernels allow more control over execution.

---

## Connection to This Unit

The GPU performance experiment in this unit applies these hardware ideas at
the software execution level.

The mathematical operation is based on GELU, while its implementation changes
across several execution strategies:

```text
Python CPU
    ↓
PyTorch GPU
    ↓
torch.compile
    ↓
Optimized PyTorch built-in
    ↓
Custom CUDA kernel
    ↓
Triton kernel
```

The experiment therefore asks a more useful question than simply:

> Is the GPU faster?

Instead, it asks:

> How does the way we organize a computation affect how efficiently the
> hardware can execute it?

---

# GELU as the Benchmark Operation

The custom implementations use the tanh approximation of GELU:

$$
\mathrm{GELU}(x)
=
0.5x
\left(
1+
\tanh
\left[
\sqrt{\frac{2}{\pi}}
\left(
x+0.044715x^3
\right)
\right]
\right)
$$

GELU is useful for this experiment because it contains several element-wise
operations:

- multiplication
- addition
- exponentiation
- `tanh`

When expressed directly using multiple PyTorch operations, these steps can
result in multiple GPU kernel launches.

This makes GELU a useful small example for studying:

- GPU parallelism
- kernel launch overhead
- kernel fusion
- intermediate memory traffic
- optimized library kernels
- custom GPU kernels

---

# Execution Levels

## Level 1 — Pure Python on CPU

The first implementation evaluates GELU element by element using a Python loop.

Conceptually:

```text
element 1 → GELU
element 2 → GELU
element 3 → GELU
...
```

This provides a simple CPU baseline.

For approximately 8 million elements, the measured execution time was:

```text
Python CPU: 3.541025 seconds
```

This implementation is intentionally simple.

It does not use:

- vectorized tensor operations
- GPU execution
- GPU kernel fusion
- large-scale hardware parallelism

Its purpose is to establish a baseline rather than represent an optimized CPU
implementation.

---

## Level 2 — Custom PyTorch Operations on GPU

The same GELU equation was then implemented using PyTorch tensors placed on a
Tesla T4 GPU.

This exposes the computation to GPU parallelism.

However, the equation is still represented as several independent PyTorch
operations.

Conceptually:

```text
power
  ↓
multiply
  ↓
add
  ↓
multiply
  ↓
tanh
  ↓
add
  ↓
multiply
```

Although these operations all execute on the GPU, they do not necessarily
execute as one GPU kernel.

This leads to an important distinction:

> Running on a GPU is not the same as running optimally on a GPU.

---

## Level 3 — PyTorch with `torch.compile`

The custom PyTorch implementation runs on the GPU, but the GELU equation is
still represented as several PyTorch operations.

Using:

```python
compiled_gelu = torch.compile(gelu_torch)
```

allows PyTorch to analyze the computation graph and generate a more optimized
execution strategy.

One of the most important optimizations observed in this experiment was
**kernel fusion**.

Instead of executing several element-wise operations through separate GPU
kernels, the compiler can combine them into fewer kernels.

Conceptually:

```text
Eager PyTorch

pow
 ↓
mul
 ↓
add
 ↓
mul
 ↓
tanh
 ↓
add
 ↓
mul
```

can become closer to:

```text
Compiled PyTorch

pow + mul + add + tanh + ...
            ↓
       fused kernel
```

### Profiler Evidence

The profiler confirmed this behavior.

The eager PyTorch implementation showed separate operations such as:

```text
aten::mul
aten::add
aten::pow
aten::tanh
```

Across ten GELU executions, multiple calls were made for each operation.

The compiled implementation instead showed a fused GPU kernel named:

```text
triton_poi_fused_add_mul_pow_tanh_0
```

with approximately one fused kernel execution per GELU call.

For an 8M-element tensor:

```text
Custom PyTorch median: 2.5660 ms
torch.compile median:  0.2793 ms
```

This corresponds to approximately:

$$
\frac{2.5660}{0.2793}
\approx
9.19\times
$$

faster execution in this benchmark.

The hardware did not change.

The mathematical expression did not change.

What changed was the way the computation was organized and executed.

This demonstrates an important performance principle:

> Software execution strategy can significantly change how efficiently the
> same hardware is used.

---

# Kernel Fusion and Memory Traffic

Kernel fusion is useful for more than reducing kernel-launch overhead.

Without fusion, intermediate results may need to be written to and read from
GPU memory between operations.

A simplified eager execution might look like:

```text
Input
  ↓
Kernel A
  ↓
Intermediate tensor
  ↓
Kernel B
  ↓
Intermediate tensor
  ↓
Kernel C
  ↓
Output
```

A fused implementation can instead perform several operations during one
kernel execution:

```text
Input
  ↓
Fused Kernel
  ↓
Output
```

This can reduce:

- kernel launches
- intermediate tensor creation
- memory reads
- memory writes
- synchronization overhead

For element-wise operations such as GELU, reducing unnecessary memory traffic
can be particularly important.

---

## Level 4 — PyTorch Built-in GELU

The next implementation used PyTorch's built-in GELU operation.

High-level libraries such as PyTorch contain kernels that have already been
optimized for common operations.

A custom implementation may be mathematically correct and run on the GPU, but
a library implementation can still be significantly faster because it has
been optimized specifically for that operation.

In this experiment, PyTorch's built-in GELU performed especially well for
small and medium tensor sizes.

For example:

```text
32K elements

Custom PyTorch:   0.1085 ms
torch.compile:    0.0914 ms
PyTorch built-in: 0.0206 ms
```

and:

```text
1M elements

Custom PyTorch:   0.3329 ms
torch.compile:    0.0939 ms
PyTorch built-in: 0.0403 ms
```

This reinforces another important principle:

> Lower-level code is not automatically faster than a mature optimized
> library implementation.

### Benchmarking Caveat

The built-in benchmark above used:

```python
F.gelu(x)
```

with PyTorch's default GELU mode.

The custom PyTorch, compiled, CUDA, and Triton implementations used the tanh
approximation.

Therefore, the built-in row in the recorded benchmark is not a perfectly
identical mathematical workload.

The numerical difference is small, but a strictly controlled benchmark should
instead use:

```python
F.gelu(x, approximate="tanh")
```

and rerun the built-in timing.

This distinction is important when interpreting small performance differences
between optimized implementations.

---

## Level 5 — Custom CUDA C++ Kernel

The next level implemented GELU directly as a CUDA C++ kernel.

CUDA exposes lower-level control over GPU execution.

Each GPU thread can process a different element of the input tensor.

The kernel maps a thread to an element using concepts such as:

```text
blockIdx.x
threadIdx.x
blockDim.x
```

A simplified index calculation is:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This allows many tensor elements to be processed concurrently.

The experiment used a fused CUDA kernel in which the full GELU approximation
was calculated inside a single kernel.

This avoids expressing GELU as several separate PyTorch operations.

For larger workloads, the custom CUDA kernel produced the fastest measured
median time in the recorded benchmark:

```text
8M elements:  0.2725 ms
32M elements: 1.0725 ms
```

However, the advantage over other optimized implementations was relatively
small.

For 32M elements:

```text
CUDA C++:          1.0725 ms
torch.compile:     1.1032 ms
Triton:            1.1424 ms
PyTorch built-in:  1.1448 ms
```

This is important because it shows that writing CUDA manually does not
automatically create a dramatically faster implementation.

Performance still depends on factors such as:

- block size
- occupancy
- memory access patterns
- instruction selection
- workload size
- compiler optimization
- kernel design

A production GPU kernel normally requires substantially more tuning than the
simple kernel used in this experiment.

---

## Level 6 — Triton

Triton provides another way to write custom GPU kernels.

It allows GPU programs to be expressed using Python-like syntax while still
providing explicit control over how blocks of data are processed.

Instead of directly working with CUDA concepts such as:

```text
blockIdx.x
threadIdx.x
```

the Triton implementation used concepts such as:

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

The general idea remains similar:

> Divide a large workload into independent blocks that can be processed in
> parallel.

Triton therefore occupies an interesting position between high-level PyTorch
code and lower-level CUDA programming.

It provides more control than ordinary PyTorch operations while requiring less
CUDA-specific boilerplate than a native CUDA C++ extension.

In this experiment, Triton achieved performance close to the optimized
implementations for large tensors:

```text
8M elements:  0.2888 ms
32M elements: 1.1424 ms
```

At 32M elements, Triton, PyTorch built-in GELU, `torch.compile`, and CUDA C++
were all within a relatively small performance range compared with the much
slower eager custom PyTorch implementation.

---

# Abstraction vs Control

The six execution levels illustrate a recurring engineering trade-off:

```text
Higher-level abstraction
        ↑
Python
PyTorch
torch.compile
Built-in kernels
Triton
CUDA C++
        ↓
More explicit control
```

Moving downward can provide greater control over execution, but it also
increases implementation complexity.

More control does not guarantee better performance.

An optimized high-level or built-in implementation can outperform a simple
low-level kernel.

The useful question is therefore not:

> What is the lowest-level implementation?

but:

> What implementation provides the best balance of performance, control,
> maintainability, and development effort for this workload?

---

# Benchmarking GPU Code Correctly

GPU benchmarking requires more care than ordinary CPU timing.

CUDA operations are asynchronous, which means the CPU can continue running
before the GPU has finished its work.

Because of this, a naive timer can measure only the time required to launch a
GPU operation rather than the time required to complete it.

To obtain more reliable measurements, the experiment used:

- GPU warm-up iterations
- `torch.cuda.synchronize()`
- CUDA events
- repeated measurements
- median execution time

Warm-up is important because the first execution may include initialization,
compilation, or caching overhead that should not be mixed with steady-state
execution time.

For `torch.compile`, compilation time was intentionally excluded from the
steady-state benchmark.

The experiment also measured GPU execution after the input tensor was already
located in device memory.

Therefore, host-to-device transfer time was not included in the main GPU
kernel benchmark.

This distinction matters because:

$$
\text{End-to-end time}
=
\text{data transfer}
+
\text{computation}
+
\text{synchronization}
+
\text{other overhead}
$$

while the main benchmark in this unit focused primarily on GPU computation.

---

# Correctness Before Performance

A faster implementation is useful only if it computes the intended function.

The custom implementations used the tanh approximation of GELU.

Initially, they were compared with PyTorch's default GELU implementation and
showed a maximum absolute difference of approximately:

```text
1.53e-04
```

This was not caused by an incorrect CUDA or Triton kernel.

The difference came from comparing the tanh approximation against a different
GELU calculation mode.

After using the same approximation as the reference:

```python
ref = F.gelu(inputs, approximate="tanh")
```

the observed maximum absolute errors were:

```text
Custom PyTorch: 0.00e+00
torch.compile:  0.00e+00
CUDA C++:       0.00e+00
Triton:         0.00e+00
```

This demonstrates an important benchmarking rule:

> Numerical equivalence should be verified before performance results are
> interpreted.

Note that this correctness test verified the custom implementations against
the tanh reference.

The previously recorded built-in performance benchmark itself still used
PyTorch's default GELU mode, as noted earlier.

---

# Experimental Results

The final benchmark was executed on a **Tesla T4 GPU**.

Median GPU execution times were:

| Tensor Size | Custom PyTorch | `torch.compile` | PyTorch Built-in* | CUDA C++ | Triton |
|---|---:|---:|---:|---:|---:|
| 32K | 0.1085 ms | 0.0914 ms | **0.0206 ms** | 0.0207 ms | 0.0440 ms |
| 256K | 0.1024 ms | 0.0903 ms | **0.0205 ms** | 0.0216 ms | 0.0469 ms |
| 1M | 0.3329 ms | 0.0939 ms | **0.0403 ms** | 0.0465 ms | 0.0529 ms |
| 8M | 2.5660 ms | 0.2793 ms | 0.2906 ms | **0.2725 ms** | 0.2888 ms |
| 32M | 10.2689 ms | 1.1032 ms | 1.1448 ms | **1.0725 ms** | 1.1424 ms |

\* The recorded PyTorch built-in benchmark used the default `F.gelu(x)`
implementation, not `approximate="tanh"`.

The results show that there is no universal ranking that applies to every
tensor size.

For smaller inputs, the PyTorch built-in GELU kernel performed extremely
well.

For the largest tested inputs, the simple custom CUDA kernel produced the
lowest median execution time in the recorded benchmark.

However, the difference between the optimized implementations was relatively
small compared with the difference between those implementations and eager
custom PyTorch.

At 32M elements:

```text
Custom PyTorch:   10.2689 ms
torch.compile:     1.1032 ms
PyTorch built-in:  1.1448 ms
CUDA C++:          1.0725 ms
Triton:            1.1424 ms
```

The eager custom PyTorch implementation was approximately:

$$
\frac{10.2689}{1.0725}
\approx
9.6\times
$$

slower than the fastest recorded implementation at that tensor size.

Because the built-in row used a slightly different GELU mode, very small
differences among the optimized implementations should not be overinterpreted.

---

# CPU vs GPU Baseline

For approximately 8 million elements, the pure Python CPU implementation took:

```text
3.541025 seconds
```

The fastest measured GPU implementation at the same scale was the CUDA kernel:

```text
0.2725 milliseconds
```

For this specific benchmark:

$$
\frac{3.541025}{0.0002725}
\approx
12{,}994.6\times
$$

This result should **not** be interpreted as:

> A GPU is always about 13,000 times faster than a CPU.

The comparison is specifically between:

```text
Pure Python scalar loop on CPU
            vs
Fused CUDA kernel on a Tesla T4
```

The two implementations differ not only in hardware, but also in:

- execution model
- parallelism
- Python overhead
- kernel organization
- memory behavior

The result therefore demonstrates the combined impact of hardware acceleration
and implementation strategy.

---

# What the Profiler Revealed

Performance measurements showed that `torch.compile` was dramatically faster
than the eager custom PyTorch implementation for large tensors.

The profiler helped explain why.

The eager implementation exposed several separate operations:

```text
aten::mul
aten::add
aten::pow
aten::tanh
```

For ten GELU executions, these operations resulted in multiple GPU kernel
calls.

The compiled implementation instead produced a fused kernel:

```text
triton_poi_fused_add_mul_pow_tanh_0
```

The compiled profile showed approximately one fused kernel execution for each
GELU call.

For ten executions on an 8M-element tensor:

```text
Eager PyTorch self CUDA time: 25.165 ms
Compiled self CUDA time:       2.478 ms
```

The profiler is not used as the primary benchmark because profiling itself adds
measurement overhead.

Instead, it provides evidence explaining the benchmark result.

The key observation is:

```text
Same mathematical expression
Same GPU
Same tensor size
        ↓
Different execution graph
        ↓
Different number and organization of kernels
        ↓
Different performance
```

---

# Connection to Large Language Models

The GELU experiment is small, but the underlying performance principles scale
directly to large language models.

LLMs repeatedly execute operations such as:

```text
matrix multiplication
        ↓
normalization
        ↓
activation functions
        ↓
attention
        ↓
more matrix multiplication
```

These operations are applied across:

- large tensors
- many layers
- many tokens
- many training or inference steps
- sometimes many GPUs

At that scale, seemingly small implementation decisions can accumulate across
billions of operations.

Important optimization ideas include:

- exposing enough parallel work
- reducing unnecessary kernel launches
- fusing compatible operations
- minimizing intermediate memory traffic
- using optimized kernels
- selecting appropriate tensor layouts
- choosing suitable numerical precision
- reducing communication and synchronization overhead

This means that model architecture alone does not determine LLM performance.

The software stack that maps the model onto the hardware is also a major part
of the system.

A useful abstraction is:

```text
Model mathematics
      ↓
Framework graph
      ↓
Compiler
      ↓
GPU kernels
      ↓
Memory hierarchy
      ↓
Hardware execution
```

Performance emerges from the interaction between all of these layers.

---

# Final Mental Model

The main lesson from this unit is that GPU performance is a **systems problem**.

It cannot be explained by one number such as core count or clock frequency.

A more complete view is:

```text
Performance
   =
Hardware capability
   +
Available parallelism
   +
Kernel implementation
   +
Compiler optimization
   +
Memory behavior
   +
Workload size
   +
Execution overhead
```

The same mathematical operation can therefore produce very different execution
times depending on how it moves through this stack.

---

# Key Takeaways

1. A GPU provides massive parallel execution capability, but software must
   expose and organize that parallelism effectively.

2. Moving a computation to the GPU does not automatically make it efficient.

3. Kernel-launch overhead is especially important for small or fragmented
   operations.

4. Kernel fusion can reduce both launch overhead and intermediate memory
   traffic.

5. `torch.compile` can significantly improve performance without changing the
   mathematical computation or the hardware.

6. Optimized PyTorch built-in kernels can outperform simple custom low-level
   implementations.

7. CUDA provides explicit low-level control, but writing CUDA does not
   automatically guarantee the fastest kernel.

8. Triton provides a useful middle ground between ordinary PyTorch and native
   CUDA development.

9. Performance rankings can change with workload size.

10. Correctness must be verified before comparing performance.

11. GPU timing requires synchronization, warm-up, repeated measurements, and
    appropriate timing tools.

12. Data movement and memory behavior can be as important as arithmetic
    throughput.

13. Benchmark methodology matters. Even a small mathematical difference between
    implementations should be documented before interpreting small timing gaps.

14. Hardware architecture and software execution strategy must be considered
    together when reasoning about LLM performance.

---

# Reflection

Before this unit, it is easy to think about GPU acceleration mainly as moving
a computation from the CPU to a faster parallel device.

The experiments show a more precise picture.

Even after a computation is already running on the GPU, performance can still
change by almost an order of magnitude depending on how the computation is
represented, compiled, fused, and executed.

This creates a direct connection between low-level chip architecture and
high-level machine learning software:

```text
Transistors
   ↓
Arithmetic units
   ↓
Parallel processors
   ↓
GPU kernels
   ↓
Frameworks and compilers
   ↓
Neural network operations
   ↓
Large language models
```

Understanding LLM performance therefore requires understanding not only what
the model computes, but also how those computations are mapped onto hardware.
