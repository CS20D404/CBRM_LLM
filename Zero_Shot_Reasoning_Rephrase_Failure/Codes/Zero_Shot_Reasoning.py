"""
source myenv/bin/activate
pip install mlx-lm mlx-vlm pyyaml tqdm psutil
python -u Zero_Shot_Reasoning_Rephrase_Failure/Codes/Zero_Shot_Reasoning.py > Zero_Shot_Reasoning_Rephrase_Failure/Codes/Zero_Shot_Reasoning.txt
"""

import json
import re
import time
import resource
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

# ==================================================

SEED = 42
TOP_N = -1                          # -1 => run on all instances, else first N per dataset
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_YAML = SCRIPT_DIR / "../../Models/Codes/Models.yaml"
PROMPT_FILE = SCRIPT_DIR / "Zero_Shot_Prompt.txt"
SUCCESS_PROMPT_FILE = SCRIPT_DIR / "Zero_Shot_Success_Summary_Prompt.txt"
FAILURE_PROMPT_FILE = SCRIPT_DIR / "Zero_Shot_Failure_Summary_Prompt.txt"
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
OUTPUT_DIR = SCRIPT_DIR / "../Outputs"
SAMPLE_QUERIES_FILE = OUTPUT_DIR / "Sample_LLM_Queries.json"
EVAL_RESULTS_FILE = OUTPUT_DIR / "Zero_Shot_Evaluation_Results.jsonl"

THINKING_ORDER = [False]      # thinking=false for all models first, then thinking=true for all models
MAX_TOKENS_THINKING = 1024
MAX_TOKENS_ANSWER = 20
MAX_TOKENS_SUMMARY = 256      # summary (<50 words) + justification (<75 words) ~ 170-190 tokens, plus labels and margin

# ==================================================

# Matches "Summary:" / "Justification:" (case-insensitive; colon, full-width colon, en/em dash).
# Markdown emphasis (**, __, `, #) is stripped before matching, so "**Summary:**" also works.
LABEL_RE = re.compile(r"\b(summary|justification)\b\s*[:：–—]\s*", flags=re.IGNORECASE)
GENERATION_ERROR_PREFIX = "[GENERATION ERROR]"
NO_ANSWER = "(no valid option given)"


def make_id(dataset_name, model_name, tag, idx):
    """Single source of truth for ids: used in predictions, checkpoints and all three corpora."""
    return f"{dataset_name}__{model_name}__thinking_{tag}__{idx}"


def corpus_path(dataset_name, model_name, kind):
    """kind: 'Success' | 'Failures' | 'All'  ->  OUTPUT_DIR/{dataset}/{llm}/Summarized_{kind}/corpus.json"""
    return OUTPUT_DIR / dataset_name / model_name / f"Summarized_{kind}" / "corpus.json"


def format_options(ex):
    return "\n".join(f"{k}. {v}" for k, v in ex["options"].items())


def format_prompt(template, ex):
    return template.format(problem_statement=ex["problem_statement"], options=format_options(ex))


def format_summary_prompt(template, ex, pred_letter):
    options = ex["options"]
    # The model's wrong answer is a letter; look its text up in the options dict.
    your_answer = options.get(pred_letter, NO_ANSWER) if pred_letter else NO_ANSWER
    # Correct answer comes straight from ground_truth_text (falls back to options[ground_truth] if missing).
    correct_answer = ex.get("ground_truth_text") or options.get(ex["ground_truth"], "")
    # The success template doesn't use {your_answer}; str.format ignores unused keys.
    return template.format(
        problem_statement=ex["problem_statement"],
        your_answer=your_answer,
        correct_answer=correct_answer,
    )


def extract_letter(text):
    # Ordered from most- to least-specific. Within each pattern, the LAST match
    # wins, since thinking output may mention other letters before the final answer.
    if text.startswith(GENERATION_ERROR_PREFIX):
        return None                                   # don't pull letters out of an error message
    patterns = [
        r"Option:\s*\(?([A-E])\)?",   # expected format: "Option: B"
        r"Answer:\s*\(?([A-E])\)?",   # common alternate phrasing
        r"\(([A-E])\)",               # "(B)"
        r"\b([A-E])[\).:]",           # "B)" / "B." / "B:"
        r"\b([A-E])\b",               # bare standalone letter, last resort
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def _tidy(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("\"'“”‘’ ").strip()


def parse_summarization(text):
    """Robustly parses 'Summary: ... Justification: ...'.

    Returns (summary, justification, status). status is 'ok' or a failure reason:
      empty_output, unterminated_think, no_labels, missing_summary, missing_justification,
      empty_summary, empty_justification, truncated.
    Tolerates: markdown bold/headers, quotes, <think> blocks, labels in any case,
    labels on the same or separate lines, and the two sections in either order.
    """
    if not text or not text.strip():
        return None, None, "empty_output"

    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"<think>", clean, flags=re.IGNORECASE):
        return None, None, "unterminated_think"       # model ran out of tokens while still thinking
    clean = re.sub(r"[*_`#]+", "", clean).strip()

    first = {}                                        # first occurrence of each label
    for m in LABEL_RE.finditer(clean):
        first.setdefault(m.group(1).lower(), m)
    if not first:
        return None, None, "no_labels"
    if "summary" not in first:
        return None, None, "missing_summary"
    if "justification" not in first:
        return None, None, "missing_justification"

    ordered = sorted(first.values(), key=lambda m: m.start())
    sections = {}
    for i, m in enumerate(ordered):
        end = ordered[i + 1].start() if i + 1 < len(ordered) else len(clean)
        sections[m.group(1).lower()] = _tidy(clean[m.end():end])

    summary, justification = sections["summary"], sections["justification"]
    if not summary:
        return None, justification or None, "empty_summary"
    if not justification:
        return summary, None, "empty_justification"
    if justification[-1] not in ".!?)":               # cut off by the token limit
        return summary, justification, "truncated"
    return summary, justification, "ok"


def save_sample_query(key, prompt, output):
    samples = {}
    if SAMPLE_QUERIES_FILE.exists():
        samples = json.loads(SAMPLE_QUERIES_FILE.read_text())
    samples[key] = {"prompt": prompt, "output": output}
    SAMPLE_QUERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_QUERIES_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False))


def log_eval_result(record):
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def make_corpus_entry(record, dataset_name, model_name, tag, idx):
    return {
        "id": record["id"],
        "outcome": record["outcome"],                 # "success" | "failure"
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
    }


def update_corpora(dataset_name, model_name, entries):
    """Upserts by id into this (dataset, model)'s Summarized_All / Success / Failures corpora."""
    groups = {
        "All": entries,
        "Success": [e for e in entries if e["outcome"] == "success"],
        "Failures": [e for e in entries if e["outcome"] == "failure"],
    }
    for kind, group in groups.items():
        if not group:
            continue
        path = corpus_path(dataset_name, model_name, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        if path.exists():
            merged = {e["id"]: e for e in json.loads(path.read_text())}
        merged.update({e["id"]: e for e in group})
        path.write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False))


def generate_text(loader, model, tok_or_proc, config, prompt, temperature, thinking, max_tokens):
    kwargs = {} if thinking is None else {"enable_thinking": thinking}
    if loader == "mlx-vlm":
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(tok_or_proc, config, prompt, num_images=0, **kwargs)
        out = vlm_generate(model, tok_or_proc, formatted, max_tokens=max_tokens, temperature=temperature, verbose=False)
        return out if isinstance(out, str) else out.text
    else:
        from mlx_lm import generate as lm_generate
        from mlx_lm.sample_utils import make_sampler
        formatted = tok_or_proc.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, **kwargs
        )
        sampler = make_sampler(temp=temperature)
        return lm_generate(model, tok_or_proc, prompt=formatted, sampler=sampler, max_tokens=max_tokens, verbose=False)


def run_single_model(model_name, cfg, prompt_template, success_template, failure_template, dataset_dirs, thinking):
    """Runs one model at ONE fixed thinking value, across all datasets.
    Timings/accuracy are computed and logged PER DATASET, not mushed together.
    Returns a Counter of summarization-parse failure reasons (for the end-of-run total)."""
    path = str(MODELS_DIR / model_name)
    run_failures = Counter()

    if cfg["loader"] == "mlx-vlm":
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, tok_or_proc = load(path)
        config = load_config(path)
    else:
        from mlx_lm import load
        model, tok_or_proc = load(path)
        config = None

    tag = "na" if thinking is None else str(thinking).lower()
    summary_thinking = None if thinking is None else False   # no reasoning needed for a short summary
    captured = set()                                         # "answer" / "success" / "failure" already saved + printed

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name.replace("_CLEANED", "")
        samples = json.loads((dataset_dir / "sampled.json").read_text())
        if TOP_N != -1:
            samples = samples[:TOP_N]

        out_dir = OUTPUT_DIR / dataset_name / model_name
        final_path = out_dir / f"predictions_thinking_{tag}.json"
        ckpt_path = out_dir / f"predictions_thinking_{tag}.checkpoint.jsonl"

        if final_path.exists():
            continue  # resumable: whole combo already done (already logged in a prior run)

        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        if ckpt_path.exists():
            results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
        start_idx = len(results)

        max_tokens = MAX_TOKENS_THINKING if thinking else MAX_TOKENS_ANSWER
        desc = f"{model_name} | {dataset_name} | thinking={tag}"
        failures, dataset_time, summary_time = 0, 0.0, 0.0

        with open(ckpt_path, "a") as ckpt_f:
            pbar = tqdm(range(start_idx, len(samples)), initial=start_idx, total=len(samples), desc=desc)
            for idx in pbar:
                ex = samples[idx]
                prompt = format_prompt(prompt_template, ex)

                # ---------------- pass 1: zero-shot answer (timed) ----------------
                t0 = time.time()
                try:
                    text = generate_text(cfg["loader"], model, tok_or_proc, config, prompt,
                                         cfg["temperature"], thinking, max_tokens)
                except Exception as e:
                    text = f"{GENERATION_ERROR_PREFIX} {e}"
                dataset_time += time.time() - t0

                if "answer" not in captured:
                    save_sample_query(f"{model_name}__thinking_{tag}", prompt, text)
                    captured.add("answer")

                pred_letter = extract_letter(text)
                if pred_letter is None:
                    failures += 1
                pbar.set_postfix(failures=failures)

                is_correct = pred_letter == ex["ground_truth"]
                outcome = "success" if is_correct else "failure"

                # ---------------- pass 2: summary + justification (not part of benchmark timing) ----------------
                s_template = success_template if is_correct else failure_template
                s_prompt = format_summary_prompt(s_template, ex, pred_letter)

                t1 = time.time()
                try:
                    raw = generate_text(cfg["loader"], model, tok_or_proc, config, s_prompt,
                                        cfg["temperature"], summary_thinking, MAX_TOKENS_SUMMARY).strip()
                    summary, justification, s_status = parse_summarization(raw)
                except Exception as e:
                    raw, summary, justification, s_status = f"{GENERATION_ERROR_PREFIX} {e}", None, None, "generation_error"
                summary_time += time.time() - t1

                # ---- sanity check: save + print the first success and the first failure summarization ----
                if outcome not in captured:
                    save_sample_query(f"{model_name}__thinking_{tag}__summary_{outcome}", s_prompt, raw)
                    captured.add(outcome)
                    tqdm.write(
                        f"\n===== SANITY CHECK: first {outcome.upper()} summarization "
                        f"[{model_name} | {dataset_name} | thinking={tag}] =====\n"
                        f"Predicted: {pred_letter} | Ground truth: {ex['ground_truth']}\n"
                        f"--- PROMPT ---\n{s_prompt}\n"
                        f"--- RAW OUTPUT ---\n{raw}\n"
                        f"--- PARSED (status={s_status}) ---\n"
                        f"Summary: {summary}\nJustification: {justification}\n"
                        f"{'=' * 70}\n"
                    )

                # ---------------- record ----------------
                record = {k: v for k, v in ex.items() if k != "metadata"}
                if "id" in record:                            # keep the dataset's own id, don't clash with ours
                    record["source_id"] = record.pop("id")
                record["id"] = make_id(dataset_name, model_name, tag, idx)
                record["llm_output"] = text
                record["prediction"] = pred_letter
                record["evaluation_correct"] = is_correct
                record["outcome"] = outcome
                record["summarization_raw"] = raw
                record["summary"] = summary
                record["justification"] = justification
                record["summarization_status"] = s_status
                results.append(record)
                ckpt_f.write(json.dumps(record) + "\n")
                ckpt_f.flush()

        final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        ckpt_path.unlink()

        # -------- corpora: built from the finished results (includes resumed records) --------
        entries = [
            make_corpus_entry(r, dataset_name, model_name, tag, i)
            for i, r in enumerate(results)
            if r["summarization_status"] == "ok"
        ]
        update_corpora(dataset_name, model_name, entries)

        # -------- summarization parse failures, counted over ALL records of this combo --------
        status_counts = Counter(r["summarization_status"] for r in results if r["summarization_status"] != "ok")
        n_sum_fail = sum(status_counts.values())
        run_failures.update(status_counts)

        # -------- per (model, dataset, thinking) summary: printed + logged immediately --------
        n_new = len(samples) - start_idx
        n_total = len(results)
        n_correct = sum(1 for r in results if r["evaluation_correct"])
        accuracy = n_correct / n_total if n_total else 0.0
        per_query = dataset_time / n_new if n_new else 0.0
        peak_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes

        print(f"[{model_name} | {dataset_name} | thinking={tag}] RAM={peak_ram_gb:.2f} GB "
              f"| total_time={dataset_time:.1f}s | queries={n_total} | per_query={per_query:.2f}s "
              f"| accuracy={accuracy:.2%} | failures={failures} "
              f"| summarization_time={summary_time:.1f}s | corpus_entries={len(entries)} "
              f"| summarization_failures={n_sum_fail}", flush=True)
        if n_sum_fail:
            breakdown = ", ".join(f"{k}={v}" for k, v in status_counts.most_common())
            print(f"    ! {n_sum_fail}/{n_total} summarizations excluded from corpora ({breakdown})", flush=True)

        log_eval_result({
            "model": model_name,
            "dataset": dataset_name,
            "thinking": tag,
            "temperature": cfg["temperature"],
            "ram_gb": round(peak_ram_gb, 2),
            "total_time_sec": round(dataset_time, 2),
            "per_query_time_sec": round(per_query, 4),
            "summarization_time_sec": round(summary_time, 2),
            "queries": n_total,
            "failures": failures,
            "accuracy": round(accuracy, 4),
            "corpus_entries": len(entries),
            "summarization_failures": n_sum_fail,
            "summarization_failure_reasons": dict(status_counts),
        })

    return run_failures


if __name__ == "__main__":
    models_cfg = yaml.safe_load(open(MODELS_YAML))
    prompt_template = PROMPT_FILE.read_text()
    success_template = SUCCESS_PROMPT_FILE.read_text()
    failure_template = FAILURE_PROMPT_FILE.read_text()
    dataset_dirs = sorted(d for d in SAMPLED_DIR.iterdir() if d.is_dir())

    total_failures = Counter()
    for thinking in THINKING_ORDER:                  # false for ALL models first, then true for ALL models
        for model_name, cfg in models_cfg.items():
            if cfg["thinking"] is None:
                if thinking is True:
                    continue                          # unsupported models only run once, during the False pass
                effective_thinking = None
            else:
                effective_thinking = thinking
            total_failures.update(run_single_model(model_name, cfg, prompt_template, success_template,
                                                   failure_template, dataset_dirs, effective_thinking))

    if total_failures:
        breakdown = ", ".join(f"{k}={v}" for k, v in total_failures.most_common())
        print(f"Total summarization failures this run: {sum(total_failures.values())} ({breakdown})")
    print("Done.")