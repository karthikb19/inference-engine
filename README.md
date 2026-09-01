# Inference Engine

This project is currently focused on implementing inference for `Qwen/Qwen3-0.6B` in C++/CUDA.

The model snapshot is stored locally under `models/Qwen3-0.6B/` and is intentionally excluded from git.


```

  1. RMSNorm — test against a CPU FP32 reference, including epsilon.
  2. Linear/GEMM infrastructure — needed for Q/K/V and MLP projections.
  3. RoPE, attention, residual adds.
  4. SwiGLU MLP.
  5. Final RMSNorm and LM-head projection.

```