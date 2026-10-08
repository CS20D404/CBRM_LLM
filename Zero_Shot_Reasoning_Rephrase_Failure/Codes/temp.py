import json
from pathlib import Path

dataset_name = "ProfessionalLaw"

predictions = Path(
    "Zero_Shot_Reasoning_Rephrase_Failure/Outputs/"+str(dataset_name)+"/"
    "Phi-3.5-mini-instruct/predictions_thinking_na.json"
)
output_dir = predictions.parents[2]  # .../Outputs


model_name = "Phi-3.5-mini-instruct"
tag = "na"

records = json.loads(predictions.read_text())

entries = []
for idx, record in enumerate(records):
    if record.get("summarization_status") != "ok":
        continue

    entries.append({
        "id": record["id"],
        "outcome": record["outcome"],
        "dataset": dataset_name,
        "model": model_name,
        "thinking": tag,
        "sample_index": idx,
        "problem_statement": record["problem_statement"],
        "options": record["options"],
        "ground_truth": record["ground_truth"],
        "model_prediction": record["prediction"],
        "model_output": record["llm_output"],
        "summary": record["summary"],
        "justification": record["justification"],
    })

groups = {
    "All": entries,
    "Success": [entry for entry in entries if entry["outcome"] == "success"],
    "Failures": [entry for entry in entries if entry["outcome"] == "failure"],
}

for kind, group in groups.items():
    if not group:
        continue
    corpus_file = (
        output_dir / dataset_name / model_name
        / f"Summarized_{kind}" / "corpus.json"
    )
    corpus_file.parent.mkdir(parents=True, exist_ok=True)
    corpus_file.write_text(json.dumps(group, indent=2, ensure_ascii=False))
    print(f"Wrote {len(group)} entries to {corpus_file}")