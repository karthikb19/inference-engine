"""Tokenize text with the local Qwen3-0.6B tokenizer and dump token IDs.

Usage:
    python python/tokenizer.py "The capital of France is"
    python python/tokenizer.py --decode --ids 791 6864 315 9822 374

The C++ engine exchanges token IDs as space-delimited plain text.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize and decode text with Qwen3-0.6B."
    )
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--decode", action="store_true", help="Decode IDs to text")
    parser.add_argument("--ids", nargs="+", type=int, help="Token IDs to decode")
    parser.add_argument("--output", "-o", type=Path, help="Write token IDs to this file")
    args = parser.parse_args()

    if not MODEL_DIR.is_dir():
        parser.error(f"Qwen3-0.6B model files are missing: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    if args.decode:
        if not args.ids:
            parser.error("--decode requires --ids")
        print(tokenizer.decode(args.ids))
        return

    if not args.text:
        parser.error("Provide text to tokenize")

    token_ids = tokenizer.encode(args.text)
    serialized_ids = " ".join(str(token_id) for token_id in token_ids)
    if args.output:
        args.output.write_text(serialized_ids + "\n", encoding="utf-8")
        print(f"Wrote {len(token_ids)} tokens to {args.output}")
    else:
        print(serialized_ids)


if __name__ == "__main__":
    main()
