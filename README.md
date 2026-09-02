# Inference Engine

This project is currently focused on implementing inference for `Qwen/Qwen3-0.6B` in C++/CUDA.

The model snapshot is stored locally under `models/Qwen3-0.6B/` and is intentionally excluded from git.


```

  1. RMSNorm — test against a CPU FP32 reference, including epsilon.
  2. Linear/GEMM infrastructure — needed for Q/K/V and MLP projections.
  3. RoPE, causal GQA attention, residual adds.
  4. SwiGLU MLP (the activation kernel is available; its gate/up/down
     projections await the GEMM infrastructure).
  5. Final RMSNorm and LM-head projection.

```

`Qwen3-0.6B` uses grouped-query attention: 16 query heads, 8 key/value
heads, and head dimension 128 (two query heads per KV head). The attention
kernel exercise lives in `causal_attention_kernel` in `src/kernels.cu`; its
contract and independent FP32 reference are in `tests/attention_test.cu`.
Run `make attention-test` while implementing it. The test is deliberately not
included in `make test` until the kernel body is completed.

`launch_softmax` provides stable FP32 row softmax for attention logits.
Qwen3-0.6B's MLP is SwiGLU rather than a standalone SiLU: after the gate and
up projections, invoke `launch_swiglu(gate, up, activated, tokens, 3072)`,
then pass `activated` to the down projection. Run `make softmax-test` and
`make swiglu-test` for the kernel-specific checks.

For a prompt-to-text demo, activate the local Python environment containing
`transformers` and run:

```
python python/chat_infer.py --max-new-tokens 32 "What is the capital of France?"
```

The helper prints the chat-template input token IDs, runs greedy repeated
prefill through `build/embedding_gpu`, and decodes the generated token IDs.
