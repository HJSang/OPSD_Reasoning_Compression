"""
Preprocess DAPO-Math-17k-dedup as the training set and prepare the
held-out validation datasets: AIME 2024, AIME 2025, MATH-500.

Output Files:
    Training Data:
        - train.parquet: full DAPO-Math-17k-dedup, RL format
        - train_example.json: example of training data format

    Validation Data:
        - val_aime24.parquet: AIME 2024 validation set
        - val_aime25.parquet: AIME 2025 validation set
        - val_math500.parquet: MATH-500 validation set
        - val_combined.parquet: all three validation sets concatenated
        - val_example.json: example of validation data format

Usage:
    python process_eval_data.py --data_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import json
import os
import re
from pathlib import Path

import datasets
import pandas as pd

# Load shared prompt config (workspace/config/prompts.json)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS = json.loads((_WORKSPACE_ROOT / "config" / "prompts.json").read_text())

# Default prompt template (student); can be overridden via --prompt_template
_PROMPT_TEMPLATE_KEY = "opsd_qwen3_student"
INSTRUCTION_PREFIX = _PROMPTS[_PROMPT_TEMPLATE_KEY]["prefix"]
INSTRUCTION_SUFFIX = _PROMPTS[_PROMPT_TEMPLATE_KEY].get("suffix", "")


def process_dapo(data_path):
    """Process DAPO-Math-17k-dedup as a single training split.

    The training pipeline validates on AIME24/AIME25/MATH-500 only, so the
    previous train/val split was producing a val_dapo.parquet that no
    downstream code consumed. Keep all examples in training instead.

    Args:
        data_path: Path to the DAPO parquet file.

    Returns:
        Training dataset.
    """
    print(f"Loading DAPO-Math-17k-dedup data from {data_path}...", flush=True)
    df = pd.read_parquet(data_path)
    dataset = datasets.Dataset.from_pandas(df)

    def process_fn(example, idx):
        if 'prompt' in example and isinstance(example['prompt'], list):
            prompt = example['prompt']
        else:
            prompt_content = example.get('prompt', '')
            prompt = [{'role': 'user', 'content': prompt_content}]

        if 'extra_info' not in example:
            example['extra_info'] = {}

        example['extra_info']['index'] = idx
        example['extra_info']['data_source'] = 'math_dapo'

        data = {
            'data_source': example.get('data_source', 'math_dapo'),
            'prompt': prompt,
            'ability': example.get('ability', 'math'),
            'reward_model': example.get('reward_model', {}),
            'extra_info': example['extra_info']
        }
        return data

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])

    print(f"Processed {len(dataset)} examples from DAPO-Math-17k-dedup (training only)")
    return dataset


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format"""
    try:
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1)
        return text
    except:
        return text


def process_aime24(data_path):
    """Process AIME 2024 validation data"""
    print(f"Loading AIME 2024 data from {data_path}...", flush=True)
    df = pd.read_parquet(data_path)
    dataset = datasets.Dataset.from_pandas(df)

    def process_fn(example, idx):
        question = INSTRUCTION_PREFIX + example['problem'] + INSTRUCTION_SUFFIX
        solution = extract_boxed_answer(example['solution'])

        data = {
            'data_source': 'aime24',
            'prompt': [{'role': 'user', 'content': question}],
            'ability': 'math',
            'reward_model': {'style': 'rule', 'ground_truth': solution},
            'extra_info': {
                'split': 'test',
                'index': idx,
            }
        }
        return data

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])

    print(f"Processed {len(dataset)} validation examples from AIME 2024")
    return dataset


def process_aime25(data_path):
    """Process AIME 2025 validation data"""
    print(f"Loading AIME 2025 data from {data_path}...", flush=True)

    data_list = []
    with open(data_path, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    dataset = datasets.Dataset.from_list(data_list)

    def process_fn(example, idx):
        question = INSTRUCTION_PREFIX + example['problem'] + INSTRUCTION_SUFFIX
        solution = str(example['answer'])

        data = {
            'data_source': 'aime25',
            'prompt': [{'role': 'user', 'content': question}],
            'ability': 'math',
            'reward_model': {'style': 'rule', 'ground_truth': solution},
            'extra_info': {
                'split': 'test',
                'index': idx,
            }
        }
        return data

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])

    print(f"Processed {len(dataset)} validation examples from AIME 2025")
    return dataset



def process_math500(data_path):
    """Process MATH-500 validation data"""
    print(f"Loading MATH-500 data from {data_path}...", flush=True)

    data_list = []
    with open(data_path, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    dataset = datasets.Dataset.from_list(data_list)

    def process_fn(example, idx):
        question = INSTRUCTION_PREFIX + example['problem'] + INSTRUCTION_SUFFIX
        solution = str(example['answer'])

        data = {
            'data_source': 'math',
            'prompt': [{'role': 'user', 'content': question}],
            'ability': 'math',
            'reward_model': {'style': 'rule', 'ground_truth': solution},
            'extra_info': {
                'split': 'test',
                'index': idx,
            }
        }
        return data

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])

    print(f"Processed {len(dataset)} validation examples from MATH-500")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Root directory containing all data folders"
    )
    parser.add_argument(
        "--output_dir",
        default="./data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--prompt_template",
        default=None,
        help="Prompt template key from prompts.json (e.g. 'opsd_qwen3_student', 'length_prune_teacher'). "
             "Overrides the default student template for validation data."
    )

    args = parser.parse_args()

    # Override prompt template if specified
    if args.prompt_template:
        if args.prompt_template not in _PROMPTS:
            raise ValueError(
                f"Unknown prompt template '{args.prompt_template}'. "
                f"Available: {list(_PROMPTS.keys())}"
            )
        tmpl = _PROMPTS[args.prompt_template]
        INSTRUCTION_PREFIX = tmpl["prefix"]
        INSTRUCTION_SUFFIX = tmpl.get("suffix", "")
        print(f"Using prompt template: {args.prompt_template}")

    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Process DAPO as the training set (no held-out val — validation runs
    # on AIME24 / AIME25 / MATH-500 only).
    print("\n" + "="*80)
    print("Processing DAPO-Math-17k-dedup (training set)")
    print("="*80)
    train_path = os.path.join(data_dir, "DAPO-Math-17k-dedup", "distinct-prompts-with-rewards.parquet")
    train_dataset = process_dapo(train_path)

    train_output_path = os.path.join(output_dir, "train.parquet")
    train_dataset.to_parquet(train_output_path)
    print(f"Saved training data to {train_output_path}")

    train_example = train_dataset[0]
    with open(os.path.join(output_dir, "train_example.json"), 'w') as f:
        json.dump(train_example, f, indent=2)

    # Process validation datasets
    print("\n" + "="*80)
    print("Processing Validation Data")
    print("="*80)

    # AIME 2024
    aime24_path = os.path.join(data_dir, "aime24", "test-00000-of-00001.parquet")
    aime24_dataset = process_aime24(aime24_path)
    aime24_output_path = os.path.join(output_dir, "val_aime24.parquet")
    aime24_dataset.to_parquet(aime24_output_path)
    print(f"Saved AIME 2024 validation data to {aime24_output_path}")

    # AIME 2025
    aime25_path = os.path.join(data_dir, "aime25", "test.jsonl")
    aime25_dataset = process_aime25(aime25_path)
    aime25_output_path = os.path.join(output_dir, "val_aime25.parquet")
    aime25_dataset.to_parquet(aime25_output_path)
    print(f"Saved AIME 2025 validation data to {aime25_output_path}")

    # MATH-500
    math500_path = os.path.join(data_dir, "MATH-500", "test.jsonl")
    math500_dataset = process_math500(math500_path)
    math500_output_path = os.path.join(output_dir, "val_math500.parquet")
    math500_dataset.to_parquet(math500_output_path)
    print(f"Saved MATH-500 validation data to {math500_output_path}")

    # Create combined validation dataset
    print("\n" + "="*80)
    print("Creating Combined Validation Dataset")
    print("="*80)

    combined_val_dataset = datasets.concatenate_datasets([
        aime24_dataset,
        aime25_dataset,
        math500_dataset
    ])
    combined_val_output_path = os.path.join(output_dir, "val_combined.parquet")
    combined_val_dataset.to_parquet(combined_val_output_path)
    print(f"Saved combined validation data to {combined_val_output_path}")
    print(f"Total validation examples: {len(combined_val_dataset)}")

    # Save validation example
    val_example = combined_val_dataset[0]
    with open(os.path.join(output_dir, "val_example.json"), 'w') as f:
        json.dump(val_example, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Training examples: {len(train_dataset)}")
    print(f"  - DAPO-Math-17k-dedup (train): {len(train_dataset)}")
    print(f"\nValidation examples: {len(combined_val_dataset)}")
    print(f"  - AIME 2024: {len(aime24_dataset)}")
    print(f"  - AIME 2025: {len(aime25_dataset)}")
    print(f"  - MATH-500: {len(math500_dataset)}")
    print(f"\nAll processed data saved to: {output_dir}")
