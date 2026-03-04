"""Upload a trained model checkpoint to HuggingFace Hub.

Usage:
    python scripts/upload.py --config configs/train_config.yaml
    python scripts/upload.py --checkpoint data/checkpoints --repo-id user/repo
    python scripts/upload.py --checkpoint data/checkpoints --merge-16bit
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

import yaml
from unsloth import FastLanguageModel


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint directory")
    parser.add_argument("--repo-id", type=str, help="Override HF repo ID")
    parser.add_argument("--merge-16bit", action="store_true", help="Merge LoRA and upload as 16-bit")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    hub_cfg = config.get("hub", {})

    checkpoint_dir = args.checkpoint or config["training"]["output_dir"]
    repo_id = args.repo_id or hub_cfg.get("repo_id")
    merge_16bit = args.merge_16bit or hub_cfg.get("merge_16bit", False)
    private = args.private if args.private is not None else hub_cfg.get("private", True)

    if not repo_id:
        parser.error("No repo_id: set hub.repo_id in config or pass --repo-id")

    print(f"Loading model from: {checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    if merge_16bit:
        print(f"Merging LoRA weights and pushing as 16-bit to {repo_id}...")
        model.push_to_hub_merged(
            repo_id,
            tokenizer,
            save_method="merged_16bit",
            private=private,
        )
    else:
        print(f"Pushing LoRA adapter to {repo_id}...")
        model.push_to_hub(repo_id, private=private)
        tokenizer.push_to_hub(repo_id, private=private)

    print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
