"""Run greedy Qwen3 chat inference through the local C++ CUDA executable.

Example:
    python python/chat_infer.py "What is the capital of France?"
"""

import argparse
import re
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3-0.6B"
DEFAULT_EXECUTABLE = ROOT / "build" / "embedding_gpu"
GENERATED_ID = re.compile(r"^generated token \d+: id=(\d+),", re.MULTILINE)


def chat_token_ids(tokenizer: AutoTokenizer, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Transformers 5 returns a BatchEncoding; older versions return a list.
    if not isinstance(encoded, (list, tuple)):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt the local Qwen3 CUDA inference executable.")
    parser.add_argument("prompt", help="User message to place in Qwen's chat template")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not MODEL_DIR.is_dir():
        parser.error(f"Model files are missing: {MODEL_DIR}")
    if not args.executable.is_file():
        parser.error(f"Inference executable is missing: {args.executable}; run `make` first")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    input_ids = chat_token_ids(tokenizer, args.prompt)
    print("Input token IDs:")
    print(" ".join(str(token_id) for token_id in input_ids))

    command = [str(args.executable), "--max-new-tokens", str(args.max_new_tokens),
               "--tokens", *(str(token_id) for token_id in input_ids)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        raise SystemExit(result.stderr.rstrip() or f"inference exited with status {result.returncode}")

    generated_ids = [int(token_id) for token_id in GENERATED_ID.findall(result.stdout)]
    print("Generated token IDs:")
    print(" ".join(str(token_id) for token_id in generated_ids))
    print("Generated text:")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
