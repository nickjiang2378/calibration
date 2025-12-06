"""
Generate A/B comparison questions for each domain.
Processes 10 domains at a time, generating 100 question pairs per domain.
"""

import json
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "google/gemini-2.5-flash"

def create_question_prompt(domain, num_questions=100):
    """Create prompt to generate A/B comparison questions for a domain."""
    return f"""Generate exactly {num_questions} A/B comparison questions for the domain: "{domain}"

Each question should present two distinct options (A and B) that someone might have a preference between. The options should be:
- Specific and concrete (not vague)
- Comparable and of similar "level" (e.g., don't compare a category to a specific item)
- Culturally diverse when applicable
- Varied in difficulty and obscurity (mix popular and niche)
- Genuinely preference-based (no objectively "correct" answer)

Examples of good comparisons:
- food domain: "pizza" vs "tacos", "sushi" vs "pasta", "mangoes" vs "strawberries"
- sports domain: "basketball" vs "soccer", "tennis" vs "badminton"
- programming domain: "Python" vs "JavaScript", "vim" vs "emacs"

Return ONLY a valid JSON array of exactly {num_questions} objects with this format:
[
  {{"A": "option1", "B": "option2"}},
  {{"A": "option3", "B": "option4"}},
  ...
]

IMPORTANT:
- Return ONLY the JSON array, no explanations
- Each option should be a short phrase (1-5 words max)
- Ensure all {num_questions} questions are unique and non-overlapping
- Options within each pair should be distinct"""

def generate_questions_for_domain(domain, num_questions=100, max_retries=3):
    """Generate questions for a single domain with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": create_question_prompt(domain, num_questions)
                    }
                ],
                temperature=0.8,
                max_tokens=6000,
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                content = content.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            questions = json.loads(content)

            # Validate
            if not isinstance(questions, list):
                raise ValueError(f"Response is not a list: {type(questions)}")

            # Validate structure
            valid_questions = []
            for q in questions:
                if isinstance(q, dict) and "A" in q and "B" in q:
                    # Ensure options are strings and non-empty
                    if q["A"] and q["B"] and q["A"] != q["B"]:
                        valid_questions.append(q)

            if len(valid_questions) < num_questions * 0.9:  # Allow 10% tolerance
                print(f"  Warning: Only got {len(valid_questions)}/{num_questions} valid questions for {domain}")

            return valid_questions

        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed for {domain}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print(f"  Failed to generate questions for {domain} after {max_retries} attempts")
                return []

def generate_questions_batch(domains, start_idx=0):
    """Generate questions for a batch of domains."""
    results = {}

    for i, domain in enumerate(domains, start_idx):
        print(f"[{i+1}/200] Generating 100 questions for: {domain}")

        questions = generate_questions_for_domain(domain)

        if questions:
            results[domain] = questions
            print(f"  ✓ Generated {len(questions)} questions")
        else:
            print(f"  ✗ Failed to generate questions")

        # Rate limiting - small delay between requests
        time.sleep(0.5)

    return results

def save_questions(domain, questions, output_dir):
    """Save questions for a domain to a JSON file."""
    output_path = output_dir / f"{domain}_questions.json"
    with open(output_path, 'w') as f:
        json.dump({
            "domain": domain,
            "num_questions": len(questions),
            "questions": questions
        }, f, indent=2)

def load_progress(progress_file):
    """Load progress from checkpoint file."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed_domains": []}

def save_progress(progress_file, completed_domains):
    """Save progress to checkpoint file."""
    with open(progress_file, 'w') as f:
        json.dump({"completed_domains": completed_domains}, f, indent=2)

def main(batch_size=10):
    """Main function to generate questions for all domains."""
    script_dir = Path(__file__).parent
    domains_file = script_dir / "dataset" / "domains.json"
    output_dir = script_dir / "dataset" / "questions"
    progress_file = script_dir / "dataset" / "generation_progress.json"

    # Load domains
    with open(domains_file, 'r') as f:
        data = json.load(f)
        domains = data["domains"]

    print(f"Loaded {len(domains)} domains")

    # Load progress
    progress = load_progress(progress_file)
    completed = set(progress["completed_domains"])

    # Filter out already completed domains
    remaining_domains = [d for d in domains if d not in completed]

    if not remaining_domains:
        print("All domains already completed!")
        return

    print(f"Remaining domains: {len(remaining_domains)}")
    print(f"Already completed: {len(completed)}")

    # Process in batches
    for i in range(0, len(remaining_domains), batch_size):
        batch = remaining_domains[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining_domains) + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches}: Processing {len(batch)} domains")
        print(f"{'='*60}\n")

        results = generate_questions_batch(batch, start_idx=len(domains) - len(remaining_domains) + i)

        # Save results
        for domain, questions in results.items():
            save_questions(domain, questions, output_dir)
            completed.add(domain)

        # Save progress
        save_progress(progress_file, list(completed))

        print(f"\nBatch {batch_num} complete. Total progress: {len(completed)}/{len(domains)} domains")

    print(f"\n{'='*60}")
    print(f"Generation complete! {len(completed)}/{len(domains)} domains processed")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(batch_size=batch_size)
