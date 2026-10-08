import json
import glob
import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = "."
PLOT_PATH = "accuracy_cross_val_by_dataset.png"
FILENAME = "RAG_Evaluation_Results_cross_val.jsonl"

SHOW_BAR_LABELS = True   # write the accuracy inside each bar, near the top, rotated 90 deg
LABEL_DECIMALS = 3
LABEL_FONTSIZE = 6.5

# If every accuracy found is <= 1.0 they are assumed to be fractions and are shown as %.
AUTO_SCALE_TO_PERCENT = True

# Keys used to read dataset / LLM from a record, in case they are NOT encoded in the path.
# (Names found in the folder path take priority over these.)
DATASET_KEYS = ("dataset", "dataset_name")
LLM_KEYS = ("llm", "model", "model_name", "llm_name")

# config name -> (top-level folder, subfolder under Outputs/cross_val/ or None, accuracy key)
CONFIGS = {
    "CBR_CB_All_Cross_Val":        ("CBR_CB_All_Cross_Val",        None,                  "accuracy_cross_val"),
    "CBR_CB_Success_Cross_Val":    ("CBR_CB_Success_Cross_Val",    "success",             "accuracy_cross_val"),
    "CBR_CB_Failure_Cross_Val":    ("CBR_CB_Failure_Cross_Val",    "failure",             "accuracy_cross_val"),
    "CBR_CB_Success_Reputation":   ("CBR_CB_Success_Reputation",   "success_consolidated", "fold_accuracy_mean_cross_val"),
    "CBR_CB_Failure_Reputation":   ("CBR_CB_Failure_Reputation",   "failure_consolidated", "fold_accuracy_mean_cross_val"),
}

LLMS = [
    "Qwen3.5-4B-Instruct", "Qwen3-8B-MLX-4bit", "Gemma-4-E4B-it", "Gemma-3-4B-it",
    "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Phi-4-mini-instruct",
    "Phi-3.5-mini-instruct",
]
DATASETS = ["CaseHOLD", "MedQA", "MedMCQA", "ProfessionalLaw"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def norm(s):
    """Lowercase and strip everything but letters/digits, for tolerant name matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match_name(value, candidates):
    """Return the candidate whose normalised form equals that of value, else None."""
    nv = norm(value)
    for c in candidates:
        if norm(c) == nv:
            return c
    return None


def to_accuracy(v):
    """Return a float, or None if the value is not populated (null / '' / NaN / non-numeric)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        v = v.strip()
        if v == "" or v.lower() in ("none", "null", "nan", "n/a"):
            return None
        try:
            v = float(v)
        except ValueError:
            return None
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def list_files(folder, sub):
    parts = [BASE_DIR, folder, "Outputs", "cross_val"]
    if sub is not None:
        parts.append(sub)
    root = os.path.join(*parts)
    return sorted(glob.glob(os.path.join(root, "**", FILENAME), recursive=True)), root


def identify(obj, path_parts, key_list, candidates):
    """Find which candidate (dataset or LLM) a record belongs to: path first, then record fields."""
    for part in path_parts:
        m = match_name(part, candidates)
        if m:
            return m
    for k in key_list:
        if k in obj:
            m = match_name(obj[k], candidates)
            if m:
                return m
    return None


def load_config(cfg, folder, sub, metric, log):
    """Return {(dataset, llm): accuracy}. Unpopulated records are skipped (and counted)."""
    files, root = list_files(folder, sub)
    if not files:
        log["no_files"].append((cfg, root))
        return {}
    out = {}
    for f in files:
        rel_parts = os.path.relpath(f, root).split(os.sep)[:-1]
        with open(f, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log["bad_lines"].append((cfg, f, line_no))
                    continue
                dataset = identify(obj, rel_parts, DATASET_KEYS, DATASETS)
                llm = identify(obj, rel_parts, LLM_KEYS, LLMS)
                if dataset is None or llm is None:
                    log["unidentified"].append((cfg, f, line_no, dataset, llm))
                    continue
                acc = to_accuracy(obj.get(metric))
                if acc is None:
                    log["unpopulated"].append((cfg, dataset, llm))
                    continue
                if (dataset, llm) in out:
                    log["duplicates"].append((cfg, dataset, llm))
                out[(dataset, llm)] = acc   # later record overwrites earlier one
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_accuracy(acc, names, path):
    """acc: {(dataset, llm, config): accuracy in %}. One subplot per dataset, stacked vertically."""
    n_cfg = len(names)
    width = 0.8 / n_cfg
    x = np.arange(len(LLMS))
    colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(15, 4.2 * len(DATASETS)))
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, DATASETS):
        for k, cfg in enumerate(names):
            vals = [acc.get((dataset, llm, cfg), np.nan) for llm in LLMS]
            offsets = x + (k - (n_cfg - 1) / 2) * width
            bars = ax.bar(offsets, vals, width, label=cfg, color=colors[k % len(colors)])
            if SHOW_BAR_LABELS:
                for b, v in zip(bars, vals):
                    if np.isnan(v):
                        continue
                    inside = v >= 15
                    ax.annotate(
                        f"{v:.{LABEL_DECIMALS}f}",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, -2 if inside else 2),
                        textcoords="offset points",
                        ha="center",
                        va="top" if inside else "bottom",
                        rotation=90,
                        fontsize=LABEL_FONTSIZE,
                        color="white" if inside else "black",
                        fontweight="bold" if inside else "normal",
                    )
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(LLMS, rotation=20, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(n_cfg, 3), frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot to {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    names = list(CONFIGS)
    log = {k: [] for k in ("no_files", "bad_lines", "unidentified", "unpopulated", "duplicates")}

    # ---- Load ----
    raw = {}  # cfg -> {(dataset, llm): accuracy as stored}
    for cfg, (folder, sub, metric) in CONFIGS.items():
        raw[cfg] = load_config(cfg, folder, sub, metric, log)

    all_vals = [v for d in raw.values() for v in d.values()]
    scale = 1.0
    if AUTO_SCALE_TO_PERCENT and all_vals and max(all_vals) <= 1.0:
        scale = 100.0
        print("All accuracies are <= 1.0 -> interpreted as fractions and multiplied by 100.")

    acc = {(d, l, cfg): v * scale for cfg, dd in raw.items() for (d, l), v in dd.items()}

    # ---- Pairwise comparison (ordered pairs) ----
    better = pd.DataFrame(0, index=names, columns=names)
    n_compared = pd.DataFrame(0, index=names, columns=names)
    for dataset, llm in itertools.product(DATASETS, LLMS):
        for ci, cj in itertools.permutations(names, 2):
            a, b = acc.get((dataset, llm, ci)), acc.get((dataset, llm, cj))
            if a is None or b is None:
                continue
            n_compared.loc[ci, cj] += 1
            if a > b:
                better.loc[ci, cj] += 1

    table = pd.DataFrame("-", index=names, columns=names)
    for ci, cj in itertools.permutations(names, 2):
        table.loc[ci, cj] = f"{better.loc[ci, cj]}/{n_compared.loc[ci, cj]}"

    print("\nCell (row i, col j): #(dataset, LLM) pairs where row config had strictly higher "
          "accuracy than column config / #pairs where both were available\n")
    print(table.to_string())
    print(f"\nMax (dataset, LLM) pairs per cell: {len(DATASETS) * len(LLMS)}")

    # ---- Coverage ----
    print("\n" + "=" * 70)
    print("COVERAGE: populated (dataset, LLM) values per configuration")
    print("=" * 70)
    cov = pd.DataFrame(
        {cfg: [int(acc.get((d, l, cfg)) is not None) for d, l in itertools.product(DATASETS, LLMS)]
         for cfg in names},
        index=pd.MultiIndex.from_product([DATASETS, LLMS], names=["dataset", "llm"]),
    )
    print(cov.to_string())
    print(f"\nTotal populated: {cov.sum().to_dict()}")

    # ---- Warnings ----
    if log["no_files"]:
        print("\nWARNING: no files found for:")
        for cfg, root in log["no_files"]:
            print(f"  {cfg}: {os.path.join(root, '**', FILENAME)}")
    if log["unidentified"]:
        print(f"\nWARNING: {len(log['unidentified'])} records skipped because dataset and/or LLM "
              f"could not be identified (from path or from {DATASET_KEYS} / {LLM_KEYS}). First 5:")
        for item in log["unidentified"][:5]:
            print("  ", item)
    if log["bad_lines"]:
        print(f"\nWARNING: {len(log['bad_lines'])} unparsable JSON lines, e.g. {log['bad_lines'][:3]}")
    if log["unpopulated"]:
        print(f"\nIgnored {len(log['unpopulated'])} records with an unpopulated accuracy value:")
        for cfg in names:
            cells = [(d, l) for c, d, l in log["unpopulated"] if c == cfg]
            if cells:
                print(f"  {cfg}: {len(cells)} (e.g. {cells[:3]})")
    if log["duplicates"]:
        print(f"\nWARNING: {len(log['duplicates'])} duplicate (config, dataset, LLM) records "
              f"(later ones overwrote earlier ones), e.g. {log['duplicates'][:3]}")

    # ---- Mean accuracy table + plot ----
    if acc:
        acc_df = pd.DataFrame(
            [{"dataset": d, "llm": l, "config": c, "accuracy": a} for (d, l, c), a in acc.items()]
        ).pivot_table(index=["dataset", "llm"], columns="config", values="accuracy")
        acc_df = acc_df.reindex(columns=[n for n in names if n in acc_df.columns])
        print("\n" + "=" * 70)
        print("ACCURACY (%)")
        print("=" * 70)
        print(acc_df.round(3).to_string())
        plot_accuracy(acc, [n for n in names if n in acc_df.columns], PLOT_PATH)
    else:
        print("\nNo results found, skipping plot.")


if __name__ == "__main__":
    main()