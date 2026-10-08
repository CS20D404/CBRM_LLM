"""
source myenv/bin/activate
pip install mlx-lm mlx-vlm pyyaml tqdm psutil sentence-transformers numpy scipy
python3 CBR_CB_Rephrase/Codes/RAG.py all > CBR_CB_Rephrase/Codes/RAG_all.txt
python3 CBR_CB_Rephrase/Codes/RAG.py failure > CBR_CB_Rephrase/Codes/RAG_failure.txt
python3 CBR_CB_Rephrase/Codes/RAG.py success > CBR_CB_Rephrase/Codes/RAG_success.txt

python3 CBR_CB_Rephrase/Codes/RAG.py all
python3 CBR_CB_Rephrase/Codes/RAG.py failure
python3 CBR_CB_Rephrase/Codes/RAG.py success
"""

import os
 
# Force fully offline behaviour for anything that touches the HF hub (sentence-transformers).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
 
import json
import math
import re
import sys
import time
import resource
import warnings
from pathlib import Path
 
import numpy as np
import yaml
from tqdm import tqdm
from scipy.stats import ttest_rel
from sentence_transformers import SentenceTransformer
 
# ==================================================
 
# Which summarized corpus to retrieve from (train fold only; test queries are always ALL test instances):
#   "all"     -> Summarized_All/corpus.json
#   "success" -> Summarized_Success/corpus.json   (questions the model got RIGHT)
#   "failure" -> Summarized_Failures/corpus.json  (questions the model got WRONG)
VALID_CONFIGS = ("all", "success", "failure")
CORPUS_DIRNAME = {"all": "Summarized_All", "success": "Summarized_Success", "failure": "Summarized_Failures"}
CORPUS_CONFIG = sys.argv[1] if len(sys.argv) > 1 else "all"
if CORPUS_CONFIG not in VALID_CONFIGS:
    raise SystemExit(f"CORPUS_CONFIG must be one of {VALID_CONFIGS}, got {CORPUS_CONFIG!r}")
 
SEED = 42                           # same seed => identical folds for every model AND every corpus config
TOP_N = -1                          # -1 => run on all instances, else first N per dataset (applied BEFORE splitting)
N_FOLDS = 5                         # 5 folds => every fold is an 80% train / 20% test split
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_YAML = SCRIPT_DIR / "../../Models/Codes/Models.yaml"
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
ZERO_SHOT_OUTPUT_DIR = SCRIPT_DIR / "../../Zero_Shot_Reasoning/Outputs"   # zero-shot baseline for the paired t-test
SUMMARY_CORPUS_ROOT = SCRIPT_DIR / "../../Zero_Shot_Reasoning_Rephrase_Failure/Outputs"  # summarized corpora
 
RAG_PROMPT_FILE = SCRIPT_DIR / "RAG_Prompt.txt"
RAG_CASE_TEMPLATE_FILE = SCRIPT_DIR / "RAG_Case_Template.txt"
 
CV_ROOT = SCRIPT_DIR / "../Outputs" / "cross_val"
SPLITS_DIR = CV_ROOT / "Splits"                            # shared by every model and every corpus config
OUTPUT_DIR = CV_ROOT / f"{CORPUS_CONFIG}_summarized"
SAMPLE_QUERIES_FILE = OUTPUT_DIR / "Sample_LLM_Queries_cross_val.json"
EVAL_RESULTS_FILE = OUTPUT_DIR / "RAG_Evaluation_Results_cross_val.jsonl"
 
THINKING_ORDER = [False]            # RAG is only ever run with thinking=false
MAX_TOKENS_ANSWER = 20
 
RAG_K = 3                                                  # number of retrieved neighbors per query
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_LOCAL_PATH = MODELS_DIR / "all-mpnet-base-v2"   # preferred local snapshot, if present
EMBEDDING_BATCH_SIZE = 64
 
# ==================================================
 
np.random.seed(SEED)
 
 
def load_embedder():
    """Loads the embedding model strictly offline."""
    if EMBEDDING_MODEL_LOCAL_PATH.exists():
        return SentenceTransformer(str(EMBEDDING_MODEL_LOCAL_PATH))
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
 
 
def embedding_text(ex):
    """Query-side text: the raw question."""
    return ex["problem_statement"]
 
 
def corpus_key_text(entry):
    """Corpus-side text that is embedded for retrieval: summary and justification combined."""
    return f"{entry['summary']}\n{entry['justification']}"
 
 
def encode(embedder, texts):
    if not texts:
        return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    return embedder.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
 
 
# ---------------------------------------------------------------- splits ----
 
def get_splits(dataset_name, samples):
    """Returns a list of N_FOLDS dicts {"fold": f, "train_ids": [...], "test_ids": [...]}.
    The split depends ONLY on the dataset's instance ids, SEED, N_FOLDS and TOP_N -- never on the model
    or the corpus config -- and is persisted to disk so every model / config / re-run uses the same folds."""
    ids = [ex["id"] for ex in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset_name}: instance ids are not unique; cannot build a reliable split.")
 
    path = SPLITS_DIR / dataset_name / f"splits_cross_val_k{N_FOLDS}_seed{SEED}_top{TOP_N}.json"
    if path.exists():
        splits = json.loads(path.read_text())
        covered = sorted(i for s in splits for i in s["test_ids"])
        if covered != sorted(ids):
            raise ValueError(f"{path} does not match the current dataset ids; delete it to regenerate.")
        return splits
 
    rng = np.random.RandomState(SEED)
    sorted_ids = sorted(ids, key=str)                     # order-independent of file ordering
    perm = rng.permutation(len(sorted_ids))
    fold_indices = np.array_split(perm, N_FOLDS)
    splits = []
    for f, test_idx in enumerate(fold_indices):
        test_set = {sorted_ids[i] for i in test_idx}
        splits.append({
            "fold": f,
            "train_ids": [i for i in sorted_ids if i not in test_set],
            "test_ids": [i for i in sorted_ids if i in test_set],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    return splits
 
 
# ------------------------------------------------- zero-shot baseline ----
 
def load_baseline(dataset_name, model_name, tag):
    """Returns ({instance_id: evaluation_correct (bool)}, path). The dict is None if the file is missing.
    Used ONLY for the paired t-test (retrieval pools come from the summarized corpora)."""
    path = ZERO_SHOT_OUTPUT_DIR / dataset_name / model_name / f"predictions_thinking_{tag}.json"
    if not path.exists():
        return None, path
    return {r["id"]: bool(r["evaluation_correct"]) for r in json.loads(path.read_text())}, path
 
 
# ------------------------------------------------- summarized corpus ----
 
def _norm_thinking(v):
    return "na" if v is None else str(v).lower()
 
 
def load_summary_corpus(ds, dataset_name, model_name, tag, embedder):
    """Loads the summarized corpus for (dataset, model) per CORPUS_CONFIG. Entries are mapped to sampled
    instance ids by EXACT problem_statement match (corpus ids / sample_index are ignored), then each entry's
    `summary` + `justification` (combined) is embedded as the retrieval key.
 
    Returns (corp, path) with corp = {"entries": {sid: entry}, "pos": {sid: row}, "embs": (n, d),
    "stmts": {sid: problem_statement}}, or (None, path) if the file is missing."""
    path = SUMMARY_CORPUS_ROOT / dataset_name / model_name / CORPUS_DIRNAME[CORPUS_CONFIG] / "corpus.json"
    if not path.exists():
        return None, path
    raw = json.loads(path.read_text())
 
    # If the corpus mixes thinking settings, keep only the one being run.
    if len({_norm_thinking(e.get("thinking")) for e in raw}) > 1:
        raw = [e for e in raw if _norm_thinking(e.get("thinking")) == tag]
 
    entries, unmapped, skipped = {}, 0, 0
    for e in raw:
        if not e.get("summary") or not e.get("justification"):
            skipped += 1
            continue
        sid = ds["stmt_to_id"].get(e.get("problem_statement"))        # exact text match
        if sid is None:
            unmapped += 1
            continue
        entries.setdefault(sid, e)                                     # first entry wins on duplicates
    if unmapped or skipped:
        print(f"[{model_name} | {dataset_name}] corpus {CORPUS_DIRNAME[CORPUS_CONFIG]}: "
              f"{unmapped} entries not mappable by problem_statement, {skipped} skipped "
              f"(missing summary/justification).", flush=True)
 
    order = list(entries.keys())
    embs = encode(embedder, [corpus_key_text(entries[i]) for i in order])   # retrieval key = summary + justification
    return {"entries": entries,
            "pos": {sid: i for i, sid in enumerate(order)},
            "embs": embs,
            "stmts": {sid: ds["by_id"][sid]["problem_statement"] for sid in order}}, path
 
 
def pool_ids_for_train(corp, train_ids):
    """Retrieval pool = corpus entries whose instance lies in the TRAIN fold (preserves train_ids order)."""
    return [i for i in train_ids if i in corp["entries"]]
 
 
# ------------------------------------------------------------- retrieval ----
 
def _topk_rows(sims, corpus_ids, kk):
    out = []
    for row in sims:
        top = np.argpartition(-row, kk - 1)[:kk]
        top = top[np.argsort(-row[top])]
        # drop masked (-inf) candidates: a query may have fewer than k eligible neighbours
        out.append([(corpus_ids[i], float(row[i])) for i in top if np.isfinite(row[i])])
    return out
 
 
def _statement_mask(query_stmts, corpus_stmts):
    """(n_query, n_corpus) boolean matrix, True where the query's problem_statement is identical to the
    corpus entry's problem_statement, i.e. the entry IS the question (self-retrieval)."""
    codes = {}
    q = np.array([codes.setdefault(s, len(codes)) for s in query_stmts])
    c = np.array([codes.setdefault(s, len(codes)) for s in corpus_stmts])
    return q[:, None] == c[None, :]
 
 
def retrieve_neighbors(query_embs, query_stmts, corpus_embs, corpus_ids, corpus_stmts, k):
    """Top-k corpus neighbours per TEST query by cosine similarity (embeddings are pre-normalized, so
    dot product == cosine). Queries = raw-question embeddings; corpus = summary embeddings.
    Any corpus entry with the same problem_statement as the query is excluded. If the corpus has fewer
    than k eligible items, all eligible ones are returned."""
    if corpus_embs.shape[0] == 0:
        return [[] for _ in range(query_embs.shape[0])]
    sims = query_embs @ corpus_embs.T                      # (n_test, n_corpus)
    sims[_statement_mask(query_stmts, corpus_stmts)] = -np.inf
    kk = min(k, corpus_embs.shape[0])
    return _topk_rows(sims, corpus_ids, kk)
 
 
def build_rag_prompt(prompt_template, case_template, ex, neighbors, corpus_entries):
    options = "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
    cases = []
    # Present cases in ASCENDING similarity: least similar first, most similar last (closest to the question).
    # `neighbors` itself stays in descending order, so stored metadata and top-1 stats are unaffected.
    ordered = sorted(neighbors, key=lambda n: n[1])
    for i, (neighbor_id, _sim) in enumerate(ordered, start=1):
        neighbor = corpus_entries[neighbor_id]
        cases.append(case_template.format(i=i, summary=neighbor["summary"],
                                          justification=neighbor["justification"]))
    similar_cases = "\n".join(cases)
    return prompt_template.format(similar_cases=similar_cases,
                                  problem_statement=ex["problem_statement"], options=options)
 
 
# ----------------------------------------------------------------- utils ----
 
def safe_round(x, ndigits):
    """Rounds, but passes through None/NaN as None (so they serialize as JSON null instead of NaN)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, ndigits)
 
 
def none_to_nan(x):
    return float("nan") if x is None else float(x)
 
 
def paired_ttest(zero_shot_correct, rag_correct):
    """Paired t-test between per-instance zero-shot and RAG correctness (0/1 lists).
    Returns (t_stat, p_value); both NaN when <2 pairs or when differences have zero variance."""
    if len(zero_shot_correct) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = ttest_rel(np.asarray(rag_correct, dtype=float), np.asarray(zero_shot_correct, dtype=float))
    return float(t_stat), float(p_value)
 
 
def extract_letter(text):
    # Ordered from most- to least-specific. Within each pattern, the LAST match wins.
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
 
 
def acc(bits):
    return sum(bits) / len(bits) if bits else float("nan")
 
 
# ------------------------------------------------------- dataset prepping ----
 
def prepare_datasets(dataset_dirs, embedder):
    """Loads samples, builds the (model- and config-independent) folds and embeds every question ONCE per
    dataset (query-side embeddings). Reused across all models and folds."""
    prepared = []
    seen_names = set()
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name.replace("_CLEANED", "")
        if dataset_name in seen_names:
            raise ValueError(f"Two dataset folders map to the same name {dataset_name!r}; their outputs and "
                             f"splits would overwrite each other. Rename or remove one.")
        seen_names.add(dataset_name)
        samples = json.loads((dataset_dir / "sampled.json").read_text())
        if TOP_N != -1:
            samples = samples[:TOP_N]
        splits = get_splits(dataset_name, samples)
        embs = encode(embedder, [embedding_text(ex) for ex in samples])
        by_id = {ex["id"]: ex for ex in samples}
        pos_by_id = {ex["id"]: i for i, ex in enumerate(samples)}
 
        # problem_statement -> id, only for statements that occur exactly once in the sampled data
        counts = {}
        for ex in samples:
            counts[ex["problem_statement"]] = counts.get(ex["problem_statement"], 0) + 1
        stmt_to_id = {ex["problem_statement"]: ex["id"] for ex in samples if counts[ex["problem_statement"]] == 1}
        n_dup = sum(1 for c in counts.values() if c > 1)
        if n_dup:
            print(f"[{dataset_name}] WARNING: {n_dup} problem_statements occur more than once in the sampled data; "
                  f"corpus entries with those statements cannot be mapped and are dropped.", flush=True)
 
        prepared.append({"name": dataset_name, "samples": samples, "splits": splits, "embs": embs,
                         "by_id": by_id, "pos_by_id": pos_by_id, "stmt_to_id": stmt_to_id})
    return prepared
 
 
# ------------------------------------------------------------- one fold ----
 
def run_fold(model_name, cfg, model, tok_or_proc, config, prompt_template, case_template,
             ds, corp, split, thinking, tag, out_dir, sample_state):
    fold = split["fold"]
    fold_path = out_dir / f"predictions_thinking_{tag}_cross_val_fold{fold}.json"
    ckpt_path = out_dir / f"predictions_thinking_{tag}_cross_val_fold{fold}.checkpoint.jsonl"
 
    if fold_path.exists():
        return json.loads(fold_path.read_text())        # resumable: fold already done
 
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    assert not (set(train_ids) & set(test_ids)), "train/test overlap!"
 
    # 1) retrieval pool = summarized corpus entries belonging to the TRAIN fold (used as-is)
    corpus_ids = pool_ids_for_train(corp, train_ids)
    if not corpus_ids:
        raise RuntimeError(f"Retrieval pool is empty for {model_name}/{ds['name']}/fold {fold}.")
    print(f"[{model_name} | {ds['name']} | fold {fold}] pool={len(corpus_ids)}", flush=True)
 
    # 2) test queries against the full pool
    corpus_entries = {i: corp["entries"][i] for i in corpus_ids}
    corpus_embs = corp["embs"][[corp["pos"][i] for i in corpus_ids]]           # summary+justification embeddings
    corpus_stmts = [corp["stmts"][i] for i in corpus_ids]
    test_samples = [ds["by_id"][i] for i in test_ids]   # queries: ALL test instances, regardless of config
    test_embs = ds["embs"][[ds["pos_by_id"][i] for i in test_ids]]             # raw-question embeddings
    test_stmts = [ex["problem_statement"] for ex in test_samples]
 
    neighbors_per_query = retrieve_neighbors(test_embs, test_stmts, corpus_embs, corpus_ids, corpus_stmts, RAG_K)
 
    results = []
    if ckpt_path.exists():
        results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
    start_idx = len(results)
    failures = sum(1 for r in results if r["prediction"] is None)
    fold_time = 0.0
 
    desc = f"{model_name} | {ds['name']} | thinking={tag} | RAG={CORPUS_CONFIG} | fold {fold + 1}/{N_FOLDS}"
    with open(ckpt_path, "a") as ckpt_f:
        pbar = tqdm(range(start_idx, len(test_samples)), initial=start_idx, total=len(test_samples), desc=desc)
        for idx in pbar:
            ex = test_samples[idx]
            neighbors = neighbors_per_query[idx]
            prompt = build_rag_prompt(prompt_template, case_template, ex, neighbors, corpus_entries)
 
            t0 = time.time()
            try:
                text = generate_text(cfg["loader"], model, tok_or_proc, config, prompt,
                                     cfg["temperature"], thinking, MAX_TOKENS_ANSWER)
            except Exception as e:
                text = f"[GENERATION ERROR] {e}"
            fold_time += time.time() - t0
 
            sample_key = f"{model_name}__{ds['name']}__thinking_{tag}__cross_val"
            if sample_key not in sample_state["captured"]:
                save_sample_query(sample_key, prompt, text)
                sample_state["captured"].add(sample_key)
 
            pred_letter = extract_letter(text)
            if pred_letter is None:
                failures += 1
            pbar.set_postfix(failures=failures)
 
            record = {k: v for k, v in ex.items() if k != "metadata"}
            record["llm_output"] = text
            record["prediction"] = pred_letter
            record["evaluation_correct"] = pred_letter == ex["ground_truth"]
            record["fold_cross_val"] = fold
            record["retrieval_metadata_cross_val"] = neighbors   # [(corpus_instance_id, similarity), ...]
            results.append(record)
            ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()
 
    test_acc = acc([int(r["evaluation_correct"]) for r in results])
    test_error = 1.0 - test_acc if results else float("nan")
 
    fold_payload = {
        "fold": fold,
        "corpus_config": CORPUS_CONFIG,
        "n_train": len(train_ids),
        "n_corpus": len(corpus_ids),                     # pool size actually used for test retrieval
        "n_test": len(test_ids),
        "test_error": safe_round(test_error, 6),
        "time_sec": round(fold_time, 2),
        "failures": failures,
        "results": results,
    }
    fold_path.write_text(json.dumps(fold_payload, indent=2, ensure_ascii=False))
    ckpt_path.unlink()
    return fold_payload
 
 
# ------------------------------------------------------------ one model ----
 
def run_single_model(model_name, cfg, prompt_template, case_template, prepared, thinking, embedder):
    """Runs one model at ONE fixed thinking value across all datasets and all folds.
    Queries come from the TEST fold; retrieval is only over the summarized TRAIN-fold corpus."""
    path = str(MODELS_DIR / model_name)
 
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
    sample_state = {"captured": set()}   # one sample prompt saved per (model, dataset)
 
    for ds in prepared:
        dataset_name = ds["name"]
        tagline = f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CORPUS_CONFIG}+summary]"
        out_dir = OUTPUT_DIR / dataset_name / model_name
        final_path = out_dir / f"predictions_thinking_{tag}_cross_val.json"
        if final_path.exists():
            continue  # resumable: whole (model, dataset) already done and logged
 
        baseline, baseline_path = load_baseline(dataset_name, model_name, tag)
        if baseline is None:
            print(f"{tagline} WARNING: zero-shot baseline not found at {baseline_path}; skipping.", flush=True)
            continue
 
        corp, corp_path = load_summary_corpus(ds, dataset_name, model_name, tag, embedder)
        if corp is None:
            print(f"{tagline} WARNING: summarized corpus not found at {corp_path}; skipping.", flush=True)
            continue
 
        pool_sizes = [len(pool_ids_for_train(corp, s["train_ids"])) for s in ds["splits"]]
        if min(pool_sizes) == 0:
            print(f"{tagline} WARNING: empty '{CORPUS_CONFIG}' retrieval pool in at least one fold "
                  f"(pool sizes per fold: {pool_sizes}); skipping.", flush=True)
            continue
        if min(pool_sizes) < RAG_K:
            print(f"{tagline} WARNING: retrieval pool smaller than k={RAG_K} in some folds "
                  f"(pool sizes per fold: {pool_sizes}); fewer cases will be shown there.", flush=True)
 
        out_dir.mkdir(parents=True, exist_ok=True)
 
        fold_payloads = [run_fold(model_name, cfg, model, tok_or_proc, config, prompt_template,
                                  case_template, ds, corp, split, thinking, tag, out_dir, sample_state)
                         for split in ds["splits"]]
 
        # Out-of-fold predictions: every instance is tested exactly once across the folds.
        results = [r for fp in fold_payloads for r in fp["results"]]
 
        # -------- aggregate summary --------
        n_total = len(results)
        n_correct = sum(1 for r in results if r["evaluation_correct"])
        accuracy = n_correct / n_total if n_total else 0.0
        dataset_time = sum(fp["time_sec"] for fp in fold_payloads)
        failures = sum(fp["failures"] for fp in fold_payloads)
        per_query = dataset_time / n_total if n_total else 0.0
        peak_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes
 
        fold_accs = [acc([int(r["evaluation_correct"]) for r in fp["results"]]) for fp in fold_payloads]
        fold_acc_mean = float(np.mean(fold_accs))
        fold_acc_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else float("nan")
 
        test_errors = [none_to_nan(fp["test_error"]) for fp in fold_payloads]
        test_err_mean = float(np.nanmean(test_errors)) if not all(math.isnan(x) for x in test_errors) else float("nan")
 
        corpus_sizes = [fp["n_corpus"] for fp in fold_payloads]
 
        top1_sims = [r["retrieval_metadata_cross_val"][0][1] for r in results if r["retrieval_metadata_cross_val"]]
        avg_top1_sim = sum(top1_sims) / len(top1_sims) if top1_sims else 0.0
 
        print(f"{tagline} RAM={peak_ram_gb:.2f} GB | test_time={dataset_time:.1f}s "
              f"| queries={n_total} | per_query={per_query:.2f}s | accuracy={accuracy:.2%} "
              f"| failures={failures} | folds={len(fold_payloads)} "
              f"| k={RAG_K} | avg_top1_sim={avg_top1_sim:.4f}", flush=True)
        print(f"{tagline} pool sizes per fold={corpus_sizes}", flush=True)
        print(f"{tagline} test_error per fold: "
              + ", ".join(f"{e:.2%}" for e in test_errors) + f" | mean={test_err_mean:.2%}", flush=True)
        print(f"{tagline} per-fold accuracy: "
              + ", ".join(f"{a:.2%}" for a in fold_accs)
              + f" | mean={fold_acc_mean:.2%} std={fold_acc_std:.4f}", flush=True)
 
        # -------- overall paired t-test (RAG vs zero-shot), paired by instance id --------
        zs_all, rag_all = [], []
        for record in results:
            baseline_correct = baseline.get(record["id"])
            if baseline_correct is None:
                continue  # no matching zero-shot baseline instance; excluded from this analysis
            record["zero_shot_baseline_correct"] = baseline_correct
            zs_all.append(int(baseline_correct))
            rag_all.append(int(record["evaluation_correct"]))
 
        zs_overall_acc, rag_overall_acc = acc(zs_all), acc(rag_all)
        t_overall, p_overall = paired_ttest(zs_all, rag_all)
 
        print(f"{tagline} OVERALL (n={len(zs_all)}): ZS_acc={zs_overall_acc:.2%} RAG_acc={rag_overall_acc:.2%} "
              f"| paired t-test t={t_overall:.4f} p={p_overall:.4g}", flush=True)
 
        zs_stats = {
            "n_baseline_matched_cross_val": len(zs_all),
            "zero_shot_accuracy_overall_cross_val": safe_round(zs_overall_acc, 4),
            "rag_accuracy_overall_cross_val": safe_round(rag_overall_acc, 4),
            "paired_ttest_t_overall_cross_val": safe_round(t_overall, 4),
            "paired_ttest_p_overall_cross_val": safe_round(p_overall, 6),
        }
 
        final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
 
        log_eval_result({
            "model": model_name,
            "dataset": dataset_name,
            "thinking": tag,
            "config": CORPUS_CONFIG,
            "corpus": CORPUS_DIRNAME[CORPUS_CONFIG],
            "retrieval_key": "summary+justification",
            "retrieved_fields": ["summary", "justification"],
            "id_mapping": "problem_statement_exact_match",
            "temperature": cfg["temperature"],
            "ram_gb": round(peak_ram_gb, 2),
            "total_time_sec": round(dataset_time, 2),
            "per_query_time_sec": round(per_query, 4),
            "queries": n_total,
            "failures": failures,
            "accuracy_cross_val": round(accuracy, 4),
            "n_folds_cross_val": len(fold_payloads),
            "fold_accuracies_cross_val": [safe_round(a, 4) for a in fold_accs],
            "fold_accuracy_mean_cross_val": safe_round(fold_acc_mean, 4),
            "fold_accuracy_std_cross_val": safe_round(fold_acc_std, 4),
            "train_sizes_cross_val": [fp["n_train"] for fp in fold_payloads],
            "corpus_sizes_cross_val": corpus_sizes,
            "test_sizes_cross_val": [fp["n_test"] for fp in fold_payloads],
            "test_errors_cross_val": [safe_round(e, 4) for e in test_errors],
            "test_error_mean_cross_val": safe_round(test_err_mean, 4),
            "k": RAG_K,
            "avg_top1_similarity_cross_val": round(avg_top1_sim, 4),
            **zs_stats,
        })
 
 
if __name__ == "__main__":
    print(f"CORPUS_CONFIG={CORPUS_CONFIG} (summarized, no consolidation) | N_FOLDS={N_FOLDS} | SEED={SEED} | k={RAG_K}",
          flush=True)
    models_cfg = yaml.safe_load(open(MODELS_YAML))
    prompt_template = RAG_PROMPT_FILE.read_text()
    case_template = RAG_CASE_TEMPLATE_FILE.read_text()
    dataset_dirs = sorted(d for d in SAMPLED_DIR.iterdir() if d.is_dir())
    embedder = load_embedder()
 
    # Folds + query-side (raw question) embeddings are built once, up front, and shared by every model.
    prepared = prepare_datasets(dataset_dirs, embedder)
 
    for thinking in THINKING_ORDER:                  # RAG only ever runs with thinking=false
        for model_name, cfg in models_cfg.items():
            effective_thinking = None if cfg["thinking"] is None else thinking
            run_single_model(model_name, cfg, prompt_template, case_template, prepared, effective_thinking, embedder)
 
    print("Done.")