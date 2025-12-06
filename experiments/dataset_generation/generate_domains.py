"""
Generate 200 diverse domains for calibration experiments.
Uses OpenRouter's Gemini 2.5 Flash to create a culturally diverse set of domains.
"""

import json
import os
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

DOMAIN_GENERATION_PROMPT = """Generate a list of exactly 200 diverse domains for preference questions. These domains should span human knowledge and culture globally, including but not limited to:

- Food & Cuisine (various global cuisines)
- Arts (visual arts, performing arts, crafts)
- Music (genres, instruments, traditions)
- Literature (genres, styles, traditions)
- Film & Television
- Sports & Athletics
- Games & Gaming
- Science (physics, chemistry, biology, astronomy, earth science)
- Technology (programming, hardware, software, platforms)
- Mathematics & Logic
- Philosophy & Ethics
- Religion & Spirituality
- Politics & Governance
- Economics & Business
- Law & Justice
- History (various periods and regions)
- Geography & Places
- Languages & Linguistics
- Education & Learning
- Health & Medicine
- Psychology & Mental Health
- Fashion & Style
- Architecture & Design
- Transportation & Vehicles
- Nature & Environment
- Animals & Pets
- Plants & Gardening
- Weather & Climate
- Hobbies & Crafts
- Travel & Tourism

IMPORTANT REQUIREMENTS:
1. Each domain should be specific enough to generate meaningful A/B comparisons
2. Domains should be culturally diverse (not Western-centric)
3. Domains should cover high and low culture, popular and niche interests
4. Use single-word or short 2-3 word domain names (e.g., "food", "classical_music", "programming_languages")
5. No duplicates or highly overlapping domains

Return ONLY a valid JSON array with exactly 200 domain strings, like:
["food", "sports", "classical_music", "programming_languages", ...]

Do not include any explanations or additional text - just the JSON array."""

def generate_domains():
    """Generate 200 diverse domains using Gemini 2.5 Flash."""
    print(f"Generating 200 diverse domains using {MODEL}...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": DOMAIN_GENERATION_PROMPT
                }
            ],
            temperature=0.9,  # Higher temperature for diversity
            max_tokens=4000,
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        domains = json.loads(content)

        # Validate
        if not isinstance(domains, list):
            raise ValueError("Response is not a list")

        if len(domains) != 200:
            print(f"Warning: Got {len(domains)} domains instead of 200")

        # Remove duplicates and normalize
        domains = list(dict.fromkeys([d.lower().strip().replace(" ", "_") for d in domains]))

        print(f"Successfully generated {len(domains)} unique domains")

        return domains

    except Exception as e:
        print(f"Error generating domains: {e}")
        raise

def save_domains(domains, output_path):
    """Save domains to JSON file."""
    with open(output_path, 'w') as f:
        json.dump({
            "total": len(domains),
            "domains": domains
        }, f, indent=2)
    print(f"Saved {len(domains)} domains to {output_path}")

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    output_path = script_dir / "dataset" / "domains.json"

    # Generate domains
    domains = generate_domains()

    # Save to file
    save_domains(domains, output_path)

    # Print sample
    print("\nSample domains:")
    for i, domain in enumerate(domains[:20], 1):
        print(f"  {i}. {domain}")
    print(f"  ... ({len(domains) - 20} more)")

if __name__ == "__main__":
    main()
