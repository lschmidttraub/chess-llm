"""Fine-tune DeepSeek-R1-Distill-Qwen-7B on chess data using Unsloth.

Loads a preprocessed HuggingFace Dataset and trains with LoRA via SFTTrainer.
Configuration is loaded from configs/train_config.yaml.
"""

from unsloth import FastLanguageModel
import argparse
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on chess data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    wandb_cfg = config.get("wandb", {})

    # Wandb setup
    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-llm"),
            name=wandb_cfg.get("run_name"),
        )
        report_to = "wandb"
    else:
        report_to = "none"

    # Load model
    print(f"Loading model: {model_cfg['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    # Apply LoRA
    print("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
    )

    # Load dataset
    dataset_dir = data_cfg["dataset_dir"]
    print(f"Loading dataset from: {dataset_dir}")
    dataset = load_from_disk(dataset_dir)
    if data_cfg.get("max_samples"):
        dataset = dataset.shuffle(seed=train_cfg["seed"]).select(range(min(data_cfg["max_samples"], len(dataset))))
    print(f"Dataset size: {len(dataset)} examples")

    # Configure trainer
    output_dir = train_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        optim=train_cfg["optim"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        weight_decay=train_cfg["weight_decay"],
        report_to=report_to,
        max_length=model_cfg["max_seq_length"],
        dataset_text_field=data_cfg["text_column"],
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save locally
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")
    print(f"To upload, run: python scripts/upload.py --checkpoint {output_dir}")


if __name__ == "__main__":
    main()
