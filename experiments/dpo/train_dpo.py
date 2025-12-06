"""
DPO Training Script for gemma-2b-it.

Trains the model using Direct Preference Optimization to align with
target model preferences.
"""

import json
import torch
import wandb
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from huggingface_hub import HfApi, create_repo
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import argparse
from datetime import datetime
import os


class WandbMetricsCallback(TrainerCallback):
    """Callback to add custom metrics to wandb logging.

    Note: When report_to="wandb" is set, all metrics are automatically logged.
    This callback is useful for adding custom metrics or ensuring specific ones are logged.
    """

    def __init__(self, metrics_to_log=None, add_custom_metrics=None):
        """
        Args:
            metrics_to_log: List of metric names to ensure are logged (for verification).
                           Examples: ['loss', 'eval_loss', 'rewards/accuracies']
            add_custom_metrics: Function that takes logs dict and returns additional metrics to log.
        """
        self.metrics_to_log = metrics_to_log
        self.add_custom_metrics = add_custom_metrics

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are written."""
        if logs is not None and args.report_to and "wandb" in args.report_to:
            # Metrics are already logged by trainer, but we can add custom ones
            if self.add_custom_metrics:
                custom_metrics = self.add_custom_metrics(logs, state)
                if custom_metrics:
                    wandb.log(custom_metrics, step=state.global_step)

            # Verify requested metrics are present (for debugging)
            if self.metrics_to_log:
                missing = [m for m in self.metrics_to_log
                          if not any(m in k for k in logs.keys())]
                if missing:
                    print(f"Warning: Requested metrics not found in logs: {missing}")


def load_dpo_dataset(file_path: str) -> Dataset:
    """Load DPO dataset from JSONL file and format for DPO training."""
    import re

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            # Extract options from prompt (text between < >)
            options = re.findall(r'<([^>]+)>', item['prompt'])

            if len(options) >= 2:
                option_a = options[0]
                option_b = options[1]

                # Map A/B to actual option text
                chosen_text = option_a if item['chosen'] == 'A' else option_b
                rejected_text = option_a if item['rejected'] == 'A' else option_b

                data.append({
                    'prompt': item['prompt'],
                    'chosen': f"I prefer {chosen_text.lower()}",
                    'rejected': f"I prefer {rejected_text.lower()}"
                })

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    return dataset


def setup_wandb(project_name: str = "calibration-dpo", run_name: str = None):
    """Initialize wandb for experiment tracking."""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"dpo_gemma2b_{timestamp}"

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model": "google/gemma-2-2b-it",
            "method": "DPO",
            "beta": 0.1,
            "learning_rate": 2e-5,
            "batch_size": 64,
            "epochs": 3,
            "seed": 42,
        }
    )
    return run_name


def main():
    parser = argparse.ArgumentParser(description="Train gemma-2b-it with DPO")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model to train (default: google/gemma-2-2b-it)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,  # Reduced from 64 to fit in memory
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,  # To achieve effective batch size of 64
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to log to wandb (e.g., 'loss eval_loss rewards/accuracies'). "
             "If not specified, all metrics are logged automatically."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub instead of saving locally"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for Hugging Face Hub (default: gemma-2b-it-dpo-{timestamp})"
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the Hub repository private"
    )
    parser.add_argument(
        "--save_local",
        action="store_true",
        help="Also save model locally when pushing to hub"
    )
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable checkpoint saving during training (only save final model if not pushing to hub)"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("="*60)
    print("DPO Training Configuration")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Beta: {args.beta}")
    print(f"Seed: {args.seed}")
    print(f"Max length: {args.max_length}")
    print(f"Output directory: {args.output_dir}")
    if args.no_checkpoints:
        print("Checkpoint saving: DISABLED (--no_checkpoints flag set)")
    else:
        print("Checkpoint saving: ENABLED")
    print("="*60)

    # Initialize wandb if requested
    run_name = None
    if args.use_wandb:
        print("\nInitializing wandb...")
        run_name = setup_wandb()
        print(f"Wandb run: {run_name}")

    # Load datasets
    print("\nLoading datasets...")
    data_dir = Path(args.data_dir)
    train_dataset = load_dpo_dataset(data_dir / "train_dpo.jsonl")
    test_dataset = load_dpo_dataset(data_dir / "test_dpo.jsonl")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Load model and tokenizer
    print(f"\nLoading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create reference model (copy of original model)
    print("\nCreating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"dpo_{timestamp}"

    # Determine if we should save checkpoints locally
    if args.no_checkpoints:
        save_strategy = "no"
    elif args.push_to_hub and not args.save_local:
        save_strategy = "no"
    else:
        save_strategy = "steps"

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_strategy="epoch"
    )

    # Optional: Define custom metrics to compute during evaluation
    # These will be automatically logged to wandb if report_to="wandb" is set
    def compute_metrics(eval_pred):
        """
        Compute custom metrics during evaluation.
        The eval_pred contains predictions and labels.
        For DPO, metrics are automatically computed, but you can add custom ones here.
        Any metrics returned here will be logged to wandb automatically.
        """
        metrics = {}
        # Example: Add custom metrics
        # metrics["custom_metric"] = some_computation(eval_pred)
        return metrics

    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")

    # Set up callbacks for wandb logging
    callbacks = []
    if args.use_wandb:
        # All metrics are automatically logged when report_to="wandb" is set
        # The callback can be used to verify specific metrics or add custom ones
        if args.wandb_metrics:
            callbacks.append(WandbMetricsCallback(metrics_to_log=args.wandb_metrics))
            print(f"Will verify these metrics are logged to wandb: {args.wandb_metrics}")
        print("All metrics (loss, eval_loss, rewards/*, etc.) will be automatically logged to wandb")

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
        # compute_metrics=compute_metrics,  # Uncomment to use custom metrics
    )

    # Train
    print("\n" + "="*60)
    print("Starting DPO training...")
    print("="*60)

    dpo_trainer.train()

    # Handle model saving/pushing
    if args.push_to_hub:
        print(f"\nPushing model to Hugging Face Hub...")
        hub_model_id = args.hub_model_id or f"gemma-2b-it-dpo-{timestamp}"

        # Push model to hub
        dpo_trainer.model.push_to_hub(
            hub_model_id,
            private=args.hub_private,
            commit_message=f"DPO training completed - eval_loss: {dpo_trainer.state.best_metric:.4f}"
        )
        tokenizer.push_to_hub(
            hub_model_id,
            private=args.hub_private,
            commit_message="Add tokenizer"
        )
        print(f"Model pushed to: https://huggingface.co/{hub_model_id}")

        # Optionally save locally as well
        if args.save_local:
            final_model_dir = output_dir / "final_model"
            print(f"Also saving locally to {final_model_dir}...")
            dpo_trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
    else:
        # Default behavior - save locally (unless no_checkpoints is set)
        if not args.no_checkpoints:
            final_model_dir = output_dir / "final_model"
            print(f"\nSaving final model to {final_model_dir}...")
            dpo_trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
        else:
            print("\nSkipping final model save (--no_checkpoints flag set)")

    # Save training metrics
    # Extract metrics from trainer state
    training_metrics = {}
    if hasattr(dpo_trainer.state, 'log_history') and dpo_trainer.state.log_history:
        # Get the last logged metrics (final training step)
        last_log = dpo_trainer.state.log_history[-1]
        # Extract all relevant metrics
        for key in ['train_loss', 'eval_loss', 'loss',
                   'train_rewards/chosen', 'train_rewards/rejected', 'train_rewards/accuracies', 'train_rewards/margins',
                   'eval_rewards/chosen', 'eval_rewards/rejected', 'eval_rewards/accuracies', 'eval_rewards/margins']:
            if key in last_log:
                training_metrics[key] = last_log[key]

    metrics = {
        "model": args.model_name,
        "method": "DPO",
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.num_epochs,
        "seed": args.seed,
        "best_metric": dpo_trainer.state.best_metric,
        "best_metric_name": training_args.metric_for_best_model,
        **training_metrics,  # Include all training metrics
    }

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining metrics saved to {metrics_path}")
    if 'train_loss' in training_metrics:
        print(f"Final train loss: {training_metrics['train_loss']:.4f}")
    if 'eval_loss' in training_metrics:
        print(f"Final eval loss: {training_metrics['eval_loss']:.4f}")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

    print("\n" + "="*60)
    print("Training complete!")
    if args.push_to_hub:
        hub_model_id = args.hub_model_id or f"gemma-2b-it-dpo-{timestamp}"
        print(f"Model available at: https://huggingface.co/{hub_model_id}")
        if args.save_local:
            print(f"Also saved locally to: {output_dir / 'final_model'}")
    elif not args.no_checkpoints:
        print(f"Model saved to: {output_dir / 'final_model'}")
    else:
        print("No checkpoints or final model saved (--no_checkpoints flag set)")
    print("="*60)


if __name__ == "__main__":
    main()