# Experiment 05 — GPU Performance Benchmark

## Objective

Benchmark the same GELU-style computation across multiple execution strategies
and investigate why implementations running on the same GPU can have very
different performance characteristics.

The experiment compares:

1. Pure Python on CPU
2. Custom PyTorch operations on GPU
3. PyTorch with `torch.compile`
4. PyTorch built-in GELU
5. Custom CUDA C++ kernel
6. Triton kernel

---

## Environment

- GPU: NVIDIA Tesla T4
- Framework: PyTorch
- GPU programming approaches:
  - PyTorch CUDA tensors
  - `torch.compile`
  - CUDA C++ extension
  - Triton
- Primary datatype: `float32`

The GPU benchmark measures execution after tensors are already located in GPU
memory.

Host-to-device transfer time is therefore excluded from the main GPU timing.

---

## Benchmark Operation

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

This operation is useful for studying GPU execution because it contains several
element-wise operations that can either execute separately or be fused into a
single kernel.

---

## Methodology

GPU timing used:

- warm-up iterations
- repeated executions
- CUDA events
- `torch.cuda.synchronize()`
- median execution time

The benchmark used the following tensor sizes:

| Label | Elements |
|---|---:|
| 32K | 32,768 |
| 256K | 262,144 |
| 1M | 1,048,576 |
| 8M | 8,388,608 |
| 32M | 33,554,432 |

Compilation and initialization overhead were excluded from steady-state timing.

---

## Correctness Check

Before interpreting performance, the custom implementations were compared
against:

```python
F.gelu(inputs, approximate="tanh")
```

Maximum absolute error:

```text
Custom PyTorch: 0.00e+00
torch.compile:  0.00e+00
CUDA C++:       0.00e+00
Triton:         0.00e+00
```

This confirms that the custom implementations matched the same GELU
approximation for the tested values.

### Built-in Benchmark Caveat

The recorded PyTorch built-in performance measurements used:

```python
F.gelu(x)
```

rather than:

```python
F.gelu(x, approximate="tanh")
```

Therefore, small timing differences between the built-in implementation and the
other optimized implementations should not be interpreted as a perfectly
controlled same-function comparison.

---

## Final Results

Median GPU execution time:

| Tensor Size | Custom PyTorch | `torch.compile` | PyTorch Built-in* | CUDA C++ | Triton |
|---|---:|---:|---:|---:|---:|
| 32K | 0.1085 ms | 0.0914 ms | **0.0206 ms** | 0.0207 ms | 0.0440 ms |
| 256K | 0.1024 ms | 0.0903 ms | **0.0205 ms** | 0.0216 ms | 0.0469 ms |
| 1M | 0.3329 ms | 0.0939 ms | **0.0403 ms** | 0.0465 ms | 0.0529 ms |
| 8M | 2.5660 ms | 0.2793 ms | 0.2906 ms | **0.2725 ms** | 0.2888 ms |
| 32M | 10.2689 ms | 1.1032 ms | 1.1448 ms | **1.0725 ms** | 1.1424 ms |

\* Built-in timing used PyTorch's default GELU mode.

---

## Result 1 — GPU Execution Alone Is Not Enough

The custom PyTorch implementation already executed on the Tesla T4, but it was
substantially slower than the optimized alternatives.

At 32M elements:

```text
Custom PyTorch: 10.2689 ms
CUDA C++:        1.0725 ms
```

The difference is approximately:

$$
\frac{10.2689}{1.0725}
\approx 9.6\times
$$

This demonstrates that placing tensors on the GPU does not automatically result
in efficient GPU execution.

---

## Result 2 — `torch.compile` Produced a Major Improvement

At 8M elements:

```text
Custom PyTorch: 2.5660 ms
torch.compile:  0.2793 ms
```

Speedup:

$$
\frac{2.5660}{0.2793}
\approx 9.19\times
$$

No change was made to:

- the GPU
- the input size
- the mathematical expression

The main change was the execution strategy.

---

## Profiler Investigation

The eager PyTorch profile exposed several separate operations:

```text
aten::mul
aten::add
aten::pow
aten::tanh
```

For ten executions of GELU, these operations resulted in multiple GPU kernel
calls.

The compiled version instead showed a fused kernel:

```text
triton_poi_fused_add_mul_pow_tanh_0
```

Profiler totals for ten executions on an 8M-element tensor:

```text
Eager PyTorch self CUDA time: 25.165 ms
Compiled self CUDA time:       2.478 ms
```

The profiler therefore supports the interpretation that kernel fusion was a
major reason for the observed speedup.

---

## Kernel Fusion Interpretation

The eager execution can be simplified as:

```text
Input
  ↓
Kernel
  ↓
Intermediate tensor
  ↓
Kernel
  ↓
Intermediate tensor
  ↓
Kernel
  ↓
Output
```

The compiled execution is closer to:

```text
Input
  ↓
Fused Kernel
  ↓
Output
```

Fusion can reduce:

- kernel launch overhead
- intermediate tensor creation
- global-memory reads
- global-memory writes

---

## Result 3 — Lower Level Did Not Always Mean Faster

For smaller tensors, PyTorch's optimized built-in GELU performed extremely
well.

At 32K:

```text
PyTorch built-in: 0.0206 ms
CUDA C++:         0.0207 ms
Triton:           0.0440 ms
```

At 1M:

```text
PyTorch built-in: 0.0403 ms
CUDA C++:         0.0465 ms
Triton:           0.0529 ms
```

The custom CUDA kernel became the fastest recorded implementation for the
largest workloads, but its advantage over the other optimized implementations
was small.

This shows that:

> More control does not automatically produce more performance.

---

## Result 4 — Workload Size Changes the Performance Ranking

For small workloads, fixed overhead such as kernel launch cost becomes
important relative to the amount of useful computation.

For larger workloads, the GPU has enough parallel work for differences in
kernel implementation and memory behavior to become more visible.

The fastest implementation therefore depended on tensor size.

There was no universal ranking across every workload.

---

## CPU vs GPU Baseline

A pure Python scalar loop was also tested with approximately 8 million
elements.

```text
Python CPU: 3.541025 seconds
CUDA GPU:   0.0002725 seconds
```

For this specific experiment:

$$
\frac{3.541025}{0.0002725}
\approx
12{,}994.6\times
$$

This should not be interpreted as a general CPU-versus-GPU hardware ratio.

The comparison is specifically between a pure Python scalar implementation and
a fused CUDA kernel running on a Tesla T4.

The result combines differences in:

- hardware
- parallel execution
- Python interpreter overhead
- kernel fusion
- implementation strategy

---

## Hardware Connection

The results connect directly to lower-level hardware concepts.

GPU performance depends not only on arithmetic capability but also on how
effectively software uses the hardware.

A useful mental model is:

```text
Parallel hardware
      +
Kernel organization
      +
Memory movement
      +
Compiler optimization
      +
Workload size
      ↓
Observed performance
```

This explains why several implementations using the same Tesla T4 produced
very different execution times.

---

## Limitations

This experiment is intentionally small and educational.

Important limitations include:

- only one GPU model was tested
- only one activation-style workload was studied
- the CUDA kernel was not extensively tuned
- Triton block-size tuning was not explored
- host-to-device transfer was excluded
- compilation time was excluded from steady-state measurements
- the recorded built-in GELU benchmark used a different approximation mode
- results should not be generalized directly to all GPU workloads

A production performance study would require more extensive benchmarking,
profiling, hardware coverage, and kernel tuning.

---

## Conclusion

The main result is not simply that GPUs are fast.

The experiment demonstrates that **how a computation is mapped onto the GPU
matters substantially**.

The eager PyTorch implementation, compiled PyTorch implementation, optimized
built-in kernel, custom CUDA kernel, and Triton kernel all used the same GPU,
yet their performance differed significantly.

The strongest observed optimization was the transition from fragmented eager
PyTorch execution to fused execution.

This provides a practical example of a broader principle in LLM systems:

> Model performance depends on the interaction between mathematics, software,
> compilers, kernels, memory, and hardware.
