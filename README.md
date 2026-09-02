# Inference Engine

This project is currently focused on implementing inference for `Qwen/Qwen3-0.6B` in C++/CUDA.

The model snapshot is stored locally under `models/Qwen3-0.6B/` and is intentionally excluded from git.


```

  1. RMSNorm — test against a CPU FP32 reference, including epsilon.
  2. Linear/GEMM infrastructure — needed for Q/K/V and MLP projections.
  3. RoPE, causal GQA attention, residual adds.
  4. SwiGLU MLP.
  5. Final RMSNorm and LM-head projection.

```

`Qwen3-0.6B` uses grouped-query attention: 16 query heads, 8 key/value
heads, and head dimension 128 (two query heads per KV head). The attention
kernel exercise lives in `causal_attention_kernel` in `src/kernels.cu`; its
contract and independent FP32 reference are in `tests/attention_test.cu`.
Run `make attention-test` while implementing it. The test is deliberately not
included in `make test` until the kernel body is completed.
