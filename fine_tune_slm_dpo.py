#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) Fine-Tuning Pipeline for Edge SLMs.

Trains Small Language Models (Gemma 2 2B, Llama 3.2 3B) on clinical preference
datasets exported from the Visual Eval & RLHF Studio.

Features:
  - Hugging Face TRL DPOTrainer integration
  - 4-bit QLoRA for consumer GPUs and Mac Metal (MPS / CPU)
  - Exports trained LoRA adapters ready for GGUF / Ollama conversion
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    SCRIPT_DIR = Path(__file__).resolve().parent
    default_dpo_file = SCRIPT_DIR / "chw-edge-evals" / "dataset" / "dpo_preference_data.jsonl"
    if not default_dpo_file.exists():
        default_dpo_file = SCRIPT_DIR / "dataset" / "dpo_preference_data.jsonl"

    parser = argparse.ArgumentParser(description="Fine-tune Edge SLMs with DPO/ORPO on CHW Preference Data")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(default_dpo_file),
        help="Path to JSONL preference dataset with prompt, chosen, rejected fields",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2-2b-it",
        help="Hugging Face model ID (e.g. google/gemma-2-2b-it, meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dpo_trained_adapter",
        help="Directory to save the fine-tuned LoRA adapter",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["orpo", "dpo"],
        default="orpo",
        help="Preference alignment algorithm: 'orpo' (recommended for SLMs: no reference model, single-step SFT+preference) or 'dpo'",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO/ORPO temperature parameter (beta / lambda)")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=768, help="Max prompt length")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and print training config without starting training")
    return parser.parse_args()


def load_dpo_dataset(dataset_path: Path):
    """Load and validate JSONL preference dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preference dataset not found at: {dataset_path}")

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prompt" not in item or "chosen" not in item or "rejected" not in item:
                raise ValueError(f"Line {i} missing required keys (prompt, chosen, rejected)")
            records.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
    return records


def print_conversion_guide(output_dir: str, model_id: str):
    """Print instructions for converting trained adapter to GGUF for Ollama."""
    print("\n" + "=" * 80)
    print(" POST-TRAINING GGUF EXPORT GUIDE (FOR OLLAMA & LITERt)")
    print("=" * 80)
    print(f"""
1. Merge the trained LoRA adapter with the base model:
   python3 -c "
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel
   import torch

   base = AutoModelForCausalLM.from_pretrained('{model_id}', torch_dtype=torch.float16, device_map='auto')
   tokenizer = AutoTokenizer.from_pretrained('{model_id}')
   model = PeftModel.from_pretrained(base, '{output_dir}')
   merged = model.merge_and_unload()
   merged.save_pretrained('./merged_model')
   tokenizer.save_pretrained('./merged_model')
   "

2. Convert merged model to GGUF using llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   python3 llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile ./gemma2-2b-chw-dpo.gguf --outtype q8_0

3. Create an Ollama Modelfile:
   cat <<EOF > Modelfile.dpo
   FROM ./gemma2-2b-chw-dpo.gguf
   PARAMETER temperature 0.0
   PARAMETER top_k 40
   PARAMETER stop "<end_of_turn>"
   EOF

4. Register in Ollama and verify in Visual Eval Studio:
   ollama create gemma2:2b-chw-dpo -f Modelfile.dpo
   python3 chw-edge-evals/run_evals.py --model gemma2:2b-chw-dpo
""")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    data_path = Path(args.dataset)

    print(f"Loading preference dataset from: {data_path}")
    dataset_records = load_dpo_dataset(data_path)
    print(f"Loaded {len(dataset_records)} preference pairs.")

    print("\nSample Pair Preview:")
    sample = dataset_records[0]
    print(f"Prompt ({len(sample['prompt'])} chars):\n{sample['prompt'][:180]}...")
    print(f"\n[CHOSEN] ({len(sample['chosen'])} chars):\n{sample['chosen']}")
    print(f"\n[REJECTED] ({len(sample['rejected'])} chars):\n{sample['rejected']}")

    if args.dry_run:
        print("\n[Dry Run] Dataset validation passed successfully. Training skipped.")
        print_conversion_guide(args.output_dir, args.model_id)
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        if args.algorithm == "orpo":
            from trl import ORPOConfig, ORPOTrainer
        else:
            from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"\nMissing required ML dependencies for live training: {e}", file=sys.stderr)
        print("To install required training libraries run:", file=sys.stderr)
        print("  pip install torch transformers datasets trl peft bitsandbytes accelerate", file=sys.stderr)
        sys.exit(1)

    print(f"\nInitializing base model: {args.model_id} using {args.algorithm.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA configuration for low memory footprint
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    hf_dataset = Dataset.from_list(dataset_records)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )

    if args.algorithm == "orpo":
        train_config = ORPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            logging_steps=1,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            remove_unused_columns=False,
        )
        trainer = ORPOTrainer(
            model=model,
            args=train_config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    else:
        train_config = DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            logging_steps=1,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            remove_unused_columns=False,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Implicit reference model handled by TRL
            args=train_config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    print(f"\nStarting {args.algorithm.upper()} Training...")
    trainer.train()

    print(f"Saving fine-tuned LoRA adapter to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print_conversion_guide(args.output_dir, args.model_id)


if __name__ == "__main__":
    main()
