"""
Collect preference distributions from target and calibration models.

For each question in the dataset, runs 10 forward passes with order randomization
to estimate empirical preference distributions P(A) and P(B).
"""

import json
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import time

# Load environment variables
load_dotenv()

# Model configurations
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CALIBRATION_MODEL = "google/gemma-2-2b-it"
GEMINI_MODEL = "google/gemini-2.5-flash"

# Inference parameters
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False
MAX_NEW_TOKENS = 50  # Increased to allow full response
NUM_SAMPLES = 10  # 10 forward passes per question

# Gemini-specific parameters
GEMINI_TEMPERATURE = 0.6
GEMINI_MAX_TOKENS = 50

# Initialize OpenRouter client for Gemini
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def load_model_and_tokenizer(model_name, device="cuda"):
    """Load a model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load all models on a single GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device}")
    return model, tokenizer


def create_prompt(option_1, option_2):
    """Create the preference prompt without domain context."""
    return f"""Choose which you prefer based on what you think is best between the two options provided. Only respond with "I prefer <>" where <> has the option. The order is given randomly, disregard the order.

<{option_1}>
<{option_2}>"""


def get_preference(model, tokenizer, prompt, option_1, option_2, device="cuda", logger=None):
    """Run a single forward pass and get the model's preference."""
    # Apply chat template if it exists (for instruct models)
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

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the generated response
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Log the raw response
    if logger:
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Raw response: '{response}'")
        logger.debug(f"Generated token IDs: {generated_tokens.tolist()}")

    # Parse response: lowercase and search for option strings
    response_lower = response.lower()
    option_1_lower = option_1.lower()
    option_2_lower = option_2.lower()

    # Check which option appears in the response
    has_option_1 = option_1_lower in response_lower
    has_option_2 = option_2_lower in response_lower

    if has_option_1 and not has_option_2:
        return "OPTION_1"
    elif has_option_2 and not has_option_1:
        return "OPTION_2"
    else:
        # Either both found or neither found - invalid
        if logger:
            logger.warning(f"Invalid response - could not determine preference: '{response}'")
        return None


async def get_preference_openrouter(prompt, option_1, option_2, logger=None, max_retries=3):
    """Run a single API call to OpenRouter and get Gemini's preference."""
    for attempt in range(max_retries):
        try:
            response = await openrouter_client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GEMINI_TEMPERATURE,
                max_tokens=GEMINI_MAX_TOKENS,
            )

            response_text = response.choices[0].message.content.strip()

            # Log the raw response
            if logger:
                logger.debug(f"Prompt: {prompt}")
                logger.debug(f"Raw response: '{response_text}'")

            # Parse response: lowercase and search for option strings
            response_lower = response_text.lower()
            option_1_lower = option_1.lower()
            option_2_lower = option_2.lower()

            # Check which option appears in the response
            has_option_1 = option_1_lower in response_lower
            has_option_2 = option_2_lower in response_lower

            if has_option_1 and not has_option_2:
                return "OPTION_1"
            elif has_option_2 and not has_option_1:
                return "OPTION_2"
            else:
                # Either both found or neither found - invalid
                if logger:
                    logger.warning(f"Invalid response - could not determine preference: '{response_text}'")
                return None

        except Exception as e:
            if logger:
                logger.warning(f"API call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Wait before retry
            else:
                if logger:
                    logger.error(f"All API call attempts failed for this prompt")
                return None

    return None


def collect_preferences_for_question(model, tokenizer, question, device="cuda", logger=None):
    """
    Collect 10 preference samples for a single question with order randomization.

    Runs 5 samples with A first, 5 samples with B first to avoid order bias.
    """
    votes_A = 0
    votes_B = 0

    option_a = question["A"]
    option_b = question["B"]

    if logger:
        logger.info(f"Question {question['qid']}: A='{option_a}' vs B='{option_b}'")

    # First 5 samples: A first, B second (normal order)
    for i in range(5):
        prompt = create_prompt(option_a, option_b)
        response = get_preference(model, tokenizer, prompt, option_a, option_b, device, logger)

        if response == "OPTION_1":  # Chose first option (A)
            votes_A += 1
        elif response == "OPTION_2":  # Chose second option (B)
            votes_B += 1
        # Invalid responses are discarded

    # Next 5 samples: B first, A second (swapped order)
    for i in range(5):
        prompt = create_prompt(option_b, option_a)  # Swap order
        response = get_preference(model, tokenizer, prompt, option_b, option_a, device, logger)

        # Swap back the interpretation since we swapped the order
        if response == "OPTION_1":  # Model chose first option (which is B)
            votes_B += 1
        elif response == "OPTION_2":  # Model chose second option (which is A)
            votes_A += 1
        # Invalid responses are discarded

    # Compute probabilities
    total_votes = votes_A + votes_B
    if total_votes > 0:
        pA = votes_A / total_votes
        pB = votes_B / total_votes
    else:
        # All responses were invalid
        pA = None
        pB = None

    if logger:
        logger.info(f"Results: votes_A={votes_A}, votes_B={votes_B}, pA={pA}, pB={pB}")

    return {
        "votes_A": votes_A,
        "votes_B": votes_B,
        "pA": pA,
        "pB": pB,
        "total_valid": total_votes
    }


async def collect_preferences_for_question_openrouter(question, logger=None, semaphore=None):
    """
    Collect 10 preference samples for a single question using OpenRouter API.

    Runs 5 samples with A first, 5 samples with B first to avoid order bias.
    Uses parallel execution for faster processing with semaphore for concurrency control.
    """
    votes_A = 0
    votes_B = 0

    option_a = question["A"]
    option_b = question["B"]

    if logger:
        logger.info(f"Question {question['qid']}: A='{option_a}' vs B='{option_b}'")

    # Create all 10 tasks (5 with A first, 5 with B first)
    tasks = []

    # First 5 samples: A first, B second (normal order)
    for i in range(5):
        prompt = create_prompt(option_a, option_b)
        tasks.append({
            'prompt': prompt,
            'option_1': option_a,
            'option_2': option_b,
            'order': 'normal'  # A first
        })

    # Next 5 samples: B first, A second (swapped order)
    for i in range(5):
        prompt = create_prompt(option_b, option_a)
        tasks.append({
            'prompt': prompt,
            'option_1': option_b,
            'option_2': option_a,
            'order': 'swapped'  # B first
        })

    # Create coroutines for all tasks
    async def run_task_with_semaphore(task):
        if semaphore:
            async with semaphore:
                response = await get_preference_openrouter(
                    task['prompt'],
                    task['option_1'],
                    task['option_2'],
                    logger
                )
        else:
            response = await get_preference_openrouter(
                task['prompt'],
                task['option_1'],
                task['option_2'],
                logger
            )
        return task, response

    # Execute all tasks in parallel
    results = await asyncio.gather(*[run_task_with_semaphore(task) for task in tasks], return_exceptions=True)

    # Process results
    for result in results:
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Task failed with exception: {result}")
            continue

        task, response = result

        # Interpret response based on order
        if task['order'] == 'normal':
            if response == "OPTION_1":  # Chose first option (A)
                votes_A += 1
            elif response == "OPTION_2":  # Chose second option (B)
                votes_B += 1
        else:  # swapped order
            if response == "OPTION_1":  # Chose first option (which is B)
                votes_B += 1
            elif response == "OPTION_2":  # Chose second option (which is A)
                votes_A += 1
        # Invalid responses are discarded

    # Compute probabilities
    total_votes = votes_A + votes_B
    if total_votes > 0:
        pA = votes_A / total_votes
        pB = votes_B / total_votes
    else:
        # All responses were invalid
        pA = None
        pB = None

    if logger:
        logger.info(f"Results: votes_A={votes_A}, votes_B={votes_B}, pA={pA}, pB={pB}")

    return {
        "votes_A": votes_A,
        "votes_B": votes_B,
        "pA": pA,
        "pB": pB,
        "total_valid": total_votes
    }


def load_dataset(dataset_path):
    """Load the dataset from JSONL file."""
    questions = []
    with open(dataset_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def save_checkpoint(questions, output_path):
    """Save current progress to output file."""
    with open(output_path, 'w') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')


def load_progress(output_path):
    """Load progress from existing output file if it exists."""
    if os.path.exists(output_path):
        return load_dataset(output_path)
    return None


async def collect_all_preferences_async(model_name, dataset_path, output_path, checkpoint_freq=100, log_dir=None, max_concurrent=300):
    """
    Collect preferences for all questions in the dataset using async execution.

    Args:
        model_name: Name of the model to use (gemini for OpenRouter)
        dataset_path: Path to input dataset (full_dataset.jsonl)
        output_path: Path to save results
        checkpoint_freq: Save checkpoint every N questions
        log_dir: Directory to save log files
        max_concurrent: Maximum number of concurrent API calls (default: 300)
    """
    print("Going!")
    # Set up logging
    logger = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = "gemini"
        log_file = log_dir / f"{model_short}_{timestamp}.log"

        logger = logging.getLogger(f"preference_collection_{model_short}")
        logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.info(f"Starting preference collection for {model_name}")
        print(f"Logging to: {log_file}")

    # Calculate batch size: 300 concurrent tasks / 10 tasks per question = 30 questions
    questions_per_batch = max_concurrent // 10
    print(f"Using OpenRouter API with model: {model_name}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Processing {questions_per_batch} questions at a time")

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    questions = load_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")

    # Check for existing progress
    existing_progress = load_progress(output_path)
    if existing_progress is not None:
        print(f"Found existing progress file with {len(existing_progress)} questions")
        print("Resuming from where we left off...")

        # Count how many are already completed
        completed = sum(1 for q in existing_progress if q.get('pA') is not None)
        print(f"Already completed: {completed}/{len(existing_progress)}")

        # Use existing progress
        questions = existing_progress
        start_idx = completed
    else:
        start_idx = 0

    # Process questions in batches
    print(f"\nCollecting preferences for {model_name}...")
    print(f"Starting from question {start_idx}")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process questions with progress bar
    total_questions = len(questions)
    pbar = tqdm(total=total_questions - start_idx, desc="Processing questions", initial=0)

    for batch_start in range(start_idx, total_questions, questions_per_batch):
        batch_end = min(batch_start + questions_per_batch, total_questions)
        batch = questions[batch_start:batch_end]

        # Filter out already processed questions
        batch_to_process = [(i + batch_start, q) for i, q in enumerate(batch) if q.get('pA') is None]

        if not batch_to_process:
            # All questions in this batch are already processed
            pbar.update(len(batch))
            continue

        # Process batch concurrently
        async def process_question_wrapper(idx, question):
            results = await collect_preferences_for_question_openrouter(question, logger, semaphore)
            return idx, results

        # Run all questions in the batch concurrently
        batch_results = await asyncio.gather(
            *[process_question_wrapper(idx, q) for idx, q in batch_to_process],
            return_exceptions=True
        )

        # Update questions with results
        for result in batch_results:
            if isinstance(result, Exception):
                if logger:
                    logger.error(f"Question processing failed with exception: {result}")
                continue

            idx, results = result
            question = questions[idx]
            question['votes_A'] = results['votes_A']
            question['votes_B'] = results['votes_B']
            question['pA'] = results['pA']
            question['pB'] = results['pB']

        # Update progress bar
        pbar.update(len(batch))

        # Save checkpoint after each batch
        save_checkpoint(questions, output_path)
        if logger:
            logger.info(f"Checkpoint saved at question {batch_end}/{total_questions}")

    pbar.close()

    # Final save
    save_checkpoint(questions, output_path)
    print(f"\nCompleted! Results saved to {output_path}")

    # Print summary statistics
    valid_questions = sum(1 for q in questions if q['pA'] is not None)
    invalid_questions = len(questions) - valid_questions

    print(f"\nSummary:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Valid responses: {valid_questions}")
    print(f"  Invalid responses: {invalid_questions}")

    if logger:
        logger.info(f"Completed! Valid: {valid_questions}, Invalid: {invalid_questions}")


def collect_all_preferences(model_name, dataset_path, output_path, checkpoint_freq=100, log_dir=None, use_openrouter=False, max_concurrent=300):
    """
    Collect preferences for all questions in the dataset.

    Args:
        model_name: Name of the model to use (target, calibration, or gemini)
        dataset_path: Path to input dataset (full_dataset.jsonl)
        output_path: Path to save results
        checkpoint_freq: Save checkpoint every N questions
        log_dir: Directory to save log files
        use_openrouter: If True, use OpenRouter API instead of local model
        max_concurrent: Maximum number of concurrent API calls (default: 300)
    """
    if use_openrouter:
        # Use async version for OpenRouter
        asyncio.run(collect_all_preferences_async(
            model_name, dataset_path, output_path,
            checkpoint_freq, log_dir, max_concurrent
        ))
        return

    # Original synchronous code for local models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up logging
    logger = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = "target" if "llama" in model_name.lower() else "calibration"
        log_file = log_dir / f"{model_short}_{timestamp}.log"

        logger = logging.getLogger(f"preference_collection_{model_short}")
        logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.info(f"Starting preference collection for {model_name}")
        print(f"Logging to: {log_file}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    questions = load_dataset(dataset_path)
    print(f"Loaded {len(questions)} questions")

    # Check for existing progress
    existing_progress = load_progress(output_path)
    if existing_progress is not None:
        print(f"Found existing progress file with {len(existing_progress)} questions")
        print("Resuming from where we left off...")

        # Count how many are already completed
        completed = sum(1 for q in existing_progress if q.get('pA') is not None)
        print(f"Already completed: {completed}/{len(existing_progress)}")

        # Use existing progress
        questions = existing_progress
        start_idx = completed
    else:
        start_idx = 0

    # Process each question
    print(f"\nCollecting preferences for {model_name}...")
    print(f"Starting from question {start_idx}")

    for i in tqdm(range(start_idx, len(questions)), desc="Processing questions"):
        question = questions[i]

        # Skip if already processed
        if question.get('pA') is not None:
            continue

        # Collect preferences
        results = collect_preferences_for_question(model, tokenizer, question, device, logger)

        # Update question with results
        question['votes_A'] = results['votes_A']
        question['votes_B'] = results['votes_B']
        question['pA'] = results['pA']
        question['pB'] = results['pB']

        # Save checkpoint
        if (i + 1) % checkpoint_freq == 0:
            save_checkpoint(questions, output_path)
            if logger:
                logger.info(f"Checkpoint saved at question {i + 1}/{len(questions)}")
            print(f"\nCheckpoint saved at question {i + 1}/{len(questions)}")

    # Final save
    save_checkpoint(questions, output_path)
    print(f"\nCompleted! Results saved to {output_path}")

    # Print summary statistics
    valid_questions = sum(1 for q in questions if q['pA'] is not None)
    invalid_questions = len(questions) - valid_questions

    print(f"\nSummary:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Valid responses: {valid_questions}")
    print(f"  Invalid responses: {invalid_questions}")

    if logger:
        logger.info(f"Completed! Valid: {valid_questions}, Invalid: {invalid_questions}")

    # Clean up
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Collect preference distributions from models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["target", "calibration", "gemini"],
        help="Which model to run (target, calibration, or gemini)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full_dataset.jsonl",
        help="Path to input dataset (default: full_dataset.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (default: {model}_preferences.jsonl)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="Save checkpoint every N questions (default: 100)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=".logs",
        help="Directory to save log files (default: .logs)"
    )

    args = parser.parse_args()

    # Determine model name and whether to use OpenRouter
    use_openrouter = False
    if args.model == "target":
        model_name = TARGET_MODEL
    elif args.model == "calibration":
        model_name = CALIBRATION_MODEL
    else:  # gemini
        model_name = GEMINI_MODEL
        use_openrouter = True

    # Determine output path
    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = script_dir / f"{args.model}_preferences.jsonl"

    # Determine dataset path
    if not os.path.isabs(args.dataset):
        script_dir = Path(__file__).parent
        args.dataset = script_dir / args.dataset

    # Determine log directory path
    if not os.path.isabs(args.log_dir):
        script_dir = Path(__file__).parent
        args.log_dir = script_dir / args.log_dir

    print("="*60)
    print("PREFERENCE COLLECTION")
    print("="*60)
    print(f"Model: {model_name}")
    if use_openrouter:
        print(f"API: OpenRouter")
        print(f"Temperature: {GEMINI_TEMPERATURE}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Log directory: {args.log_dir}")
    print(f"Checkpoint frequency: every {args.checkpoint_freq} questions")
    print("="*60)

    # Run collection
    collect_all_preferences(
        model_name=model_name,
        dataset_path=args.dataset,
        output_path=args.output,
        checkpoint_freq=args.checkpoint_freq,
        log_dir=args.log_dir,
        use_openrouter=use_openrouter
    )


if __name__ == "__main__":
    main()
