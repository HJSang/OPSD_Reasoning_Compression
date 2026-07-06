"""Convert DAPO-Math-17k-dedup parquet into slime jsonl for CRISP training.

Each output row:
    {
      "prompt":   "<original DAPO user content>",   # student prompt (unchanged)
      "label":    "<ground truth>",                 # metrics-only verification
      "metadata": {"question": "<bare question>"}   # used to build the teacher prompt
    }

slime wraps ``prompt`` into a user message and applies the chat template
(``--apply-chat-template --input-key prompt --label-key label``); the CRISP
reward_func rebuilds the teacher prompt from ``metadata.question`` at rollout
time. Extraction logic mirrors workspace/src/data/prepare_length_prune_data.py
so the student/teacher prompt pair is identical to the verl pipeline.

Split parity: same fixed-seed pandas shuffle + first-80% train split as
workspace/src/data/prepare_length_prune_data.py::sample_and_split, so the
training set is row-for-row identical to the verl pipeline's.

Usage:
    python prepare_crisp_slime_data.py \
        --input-parquet ../data/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet \
        --output ../data/crisp_slime/dapo_math_crisp_train.jsonl
"""

import argparse
import json
import os

import pandas as pd

DEFAULT_SEED = 42

# The instruction header DAPO-Math prepends to every question; stripping it
# yields the bare question for teacher-prompt construction. Must match
# workspace/src/data/prepare_length_prune_data.py.
DAPO_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: "
    "$Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)


def extract_user_content(prompt) -> str:
    """Return the original DAPO user-message content (the student prompt)."""
    if isinstance(prompt, str):
        messages = json.loads(prompt)
    else:
        messages = list(prompt)
    return messages[0]["content"]


def extract_question(content: str) -> str:
    """Strip the DAPO instruction header to get the bare question text."""
    if content.startswith(DAPO_PREFIX):
        return content[len(DAPO_PREFIX) :].strip()
    return content.strip()


def extract_ground_truth(reward_model) -> str:
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output", required=True, help="Output train jsonl path")
    parser.add_argument(
        "--val-output", default=None, help="Optional val jsonl path (the held-out 1-train_frac)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train fraction (default 0.8, matching the verl pipeline's split)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional cap on TRAIN rows (smoke tests)"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} rows from {args.input_parquet}")

    # Same fixed-seed shuffle + first-80% split as the verl pipeline
    # (prepare_length_prune_data.sample_and_split) -> identical train set.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_train = int(len(df) * args.train_frac)
    train_df, val_df = df.iloc[:n_train], df.iloc[n_train:]
    if args.max_rows:
        train_df = train_df.iloc[: args.max_rows]

    def write_jsonl(split_df, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        n_written = n_skipped = 0
        with open(path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                content = extract_user_content(row["prompt"])
                question = extract_question(content)
                gt = extract_ground_truth(row["reward_model"])
                if not question or not gt:
                    n_skipped += 1
                    continue
                f.write(
                    json.dumps(
                        {"prompt": content, "label": gt, "metadata": {"question": question}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1
        print(f"Wrote {n_written} rows to {path} ({n_skipped} skipped: missing question/GT)")

    write_jsonl(train_df, args.output)
    if args.val_output:
        write_jsonl(val_df, args.val_output)


if __name__ == "__main__":
    main()
