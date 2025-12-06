"""
Evaluation script for DPO-trained model.

Evaluates the trained model on the test set and computes alignment metrics:
- Mean KL divergence
- Mode-match accuracy
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
from scipy.stats import entropy


def load_test_data(file_path: str) -> List[Dict]:
    """Load test dataset from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_model_preference(model, tokenizer, prompt: str, device: str = "cuda") -> str:
    """
    Get model's preference for a single prompt.
    Uses deterministic generation (temperature=0).
    Parses the response to extract A or B based on which option was chosen.
    """
    # Extract the two options from the prompt
    # Prompt format: "...\n\n<Option A>\n<Option B>"
    lines = prompt.strip().split('\n')
    option_a = None
    option_b = None

    # Find the last two lines that are in <> format
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('<') and line.endswith('>'):
            if option_b is None:
                option_b = line[1:-1]  # Remove < and >
            elif option_a is None:
                option_a = line[1:-1]
                break

    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate with deterministic settings - generate enough tokens to get the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # Generate enough tokens to capture "I prefer <option>"
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Parse response to extract A or B
    # First check if it's just a single letter
    if response.upper() in ['A', 'B']:
        return response.upper()

    # Check if response contains the option text
    # Convert to lowercase for case-insensitive matching
    response_lower = response.lower()
    print(f"Response: {response_lower}")
    print(f"Option A: {option_a}")
    print(f"Option B: {option_b}")

    if option_a and option_b:
        option_a_lower = option_a.lower()
        option_b_lower = option_b.lower()

        # Check which option appears in the response
        has_a = option_a_lower in response_lower
        has_b = option_b_lower in response_lower

        # Return the one that appears (if only one appears)
        if has_a and not has_b:
            return 'A'
        elif has_b and not has_a:
            return 'B'

    # If we can't determine, return None
    return None


def run_multiple_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 10,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Run multiple forward passes to estimate preference distribution.
    Returns (pA, pB) - probability of choosing A and B.
    """
    votes_a = 0
    votes_b = 0

    for _ in range(n_samples):
        response = get_model_preference(model, tokenizer, prompt, device)
        if response == 'A':
            votes_a += 1
        elif response == 'B':
            votes_b += 1

    total_valid = votes_a + votes_b
    if total_valid == 0:
        return None, None

    pA = votes_a / total_valid
    pB = votes_b / total_valid

    return pA, pB


def compute_kl_divergence(p_target: np.ndarray, p_calibration: np.ndarray) -> float:
    """
    Compute KL divergence between target and calibration distributions.
    KL(target || calibration)
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_target = np.clip(p_target, epsilon, 1 - epsilon)
    p_calibration = np.clip(p_calibration, epsilon, 1 - epsilon)

    return entropy(p_target, p_calibration)


def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    n_samples: int = 10,
    device: str = "cuda"
):
    """
    Evaluate DPO-trained model on test set.

    Args:
        model_path: Path to trained model
        test_data_path: Path to test dataset
        output_dir: Directory to save results
        n_samples: Number of samples per question for distribution estimation
        device: Device to run on
    """
    print("="*60)
    print("DPO Model Evaluation")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Samples per question: {n_samples}")
    print(f"Device: {device}")
    print("="*60)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": 0},  # Force all layers to cuda:0
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} test examples")

    # Evaluate each example
    predictions = []
    kl_divergences = []
    mode_matches = []

    print("\nEvaluating model...")
    for example in tqdm(test_data, desc="Processing examples"):
        # Get model's preference distribution
        if n_samples > 1:
            # Multiple samples for distribution estimation
            pA_model, pB_model = run_multiple_samples(
                model, tokenizer, example['prompt'], n_samples, device
            )
        else:
            # Single sample for quick evaluation
            response = get_model_preference(model, tokenizer, example['prompt'], device)

            print(f"Response: {response}")
            if response == 'A':
                pA_model, pB_model = 1.0, 0.0
            elif response == 'B':
                pA_model, pB_model = 0.0, 1.0
            else:
                pA_model, pB_model = None, None

        # Skip if invalid response
        if pA_model is None:
            continue

        # Target distribution
        pA_target = example['pA']
        pB_target = example['pB']

        # Compute KL divergence
        target_dist = np.array([pA_target, pB_target])
        model_dist = np.array([pA_model, pB_model])
        kl = compute_kl_divergence(target_dist, model_dist)
        kl_divergences.append(kl)

        # Check mode match
        target_mode = example["chosen"]
        if target_mode == "A" and pA_model > 0.8:
            mode_match = True
        elif target_mode == "B" and pB_model > 0.8:
            mode_match = True
        else:
            mode_match = False
        mode_matches.append(mode_match)
        print(f"Mode match: {mode_match}")

        # Store prediction
        prediction = {
            'qid': example['qid'],
            'domain': example['domain'],
            'prompt': example['prompt'],
            'pA_target': pA_target,
            'pB_target': pB_target,
            'pA_model': pA_model,
            'pB_model': pB_model,
            'kl_divergence': kl,
            'mode_match': mode_match,
            'target_mode': target_mode,
        }
        predictions.append(prediction)

    # Compute aggregate metrics
    mean_kl = np.mean(kl_divergences) if kl_divergences else float('inf')
    std_kl = np.std(kl_divergences) if kl_divergences else 0.0
    mode_match_accuracy = np.mean(mode_matches) if mode_matches else 0.0

    # Compute per-domain metrics
    domain_metrics = {}
    for pred in predictions:
        domain = pred['domain']
        if domain not in domain_metrics:
            domain_metrics[domain] = {'kl': [], 'mode_matches': []}
        domain_metrics[domain]['kl'].append(pred['kl_divergence'])
        domain_metrics[domain]['mode_matches'].append(pred['mode_match'])

    domain_results = {}
    for domain, metrics in domain_metrics.items():
        domain_results[domain] = {
            'mean_kl': np.mean(metrics['kl']),
            'mode_match_acc': np.mean(metrics['mode_matches']),
            'n_examples': len(metrics['kl']),
        }

    # Save results
    output_dir = Path(model_path) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_path = output_dir / "predictions_test.jsonl"
    with open(predictions_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    # Save aggregate metrics
    metrics = {
        "method": "dpo",
        "mean_KL": mean_kl,
        "std_KL": std_kl,
        "mode_match_acc": mode_match_accuracy,
        "n_valid_predictions": len(predictions),
        "n_total_examples": len(test_data),
        "domain_metrics": domain_results,
    }

    metrics_path = output_dir / "metrics_test.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Valid predictions: {len(predictions)}/{len(test_data)}")
    print(f"Mean KL divergence: {mean_kl:.4f} (±{std_kl:.4f})")
    print(f"Mode-match accuracy: {mode_match_accuracy:.2%}")

    print("\nPer-domain results:")
    for domain, results in sorted(domain_results.items()):
        print(f"  {domain}: KL={results['mean_kl']:.4f}, Acc={results['mode_match_acc']:.2%} (n={results['n_examples']})")

    print(f"\nResults saved to:")
    print(f"  Predictions: {predictions_path}")
    print(f"  Metrics: {metrics_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO-trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test_dpo.jsonl",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples per question (1 for quick eval, 10 for distribution)"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device=device
    )


if __name__ == "__main__":
    main()