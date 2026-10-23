

pip install pandas openai tqdm

python run_experiment.py input.csv

- liest die prompts und ruft die modelle auf
- erzeugt ein results.jsonl

python parse-results.csv

- liest results.jsonl
- generiert parsed_results.csv
