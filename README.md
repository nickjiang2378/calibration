## Calibrating LM judges

Reproducing experiments:
- In `experiments/dataset_generation`, run `generate_domains.py` and `generate_questions.py` to generate the initial comparisons. Run `collect_preferences.py` to collect preferences from Gemini 2.5 Flash. See `gemini/` for the generated questions and preference selections.
- In `experiments/dpo`, run `python3 train_dpo.py --data_dir ../dataset_generation/gemini` to use DPO on the small LM judge.