import json
import glob
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = "."
ALPHA = 0.05
PLOT_PATH = "accuracy_by_dataset.png"

# True : bars use only ids shared by all available configs for that (dataset, LLM),
#        so bars are directly comparable and consistent with the paired tests.
# False: each bar uses all of that config's own instances (raw fraction of true).
USE_COMMON_IDS_FOR_PLOT = True

SHOW_BAR_LABELS = True   # write the accuracy inside each bar, near the top, rotated 90 deg
LABEL_DECIMALS = 3
LABEL_FONTSIZE = 6.5

# configuration folder -> subfolder under Outputs/ (None = no subfolder)
CONFIGS = {
    "Zero_Shot_Reasoning": None,   # ./Zero_Shot_Reasoning/Outputs/{dataset}/{llm}/*.json
    "CBR_CB_All": "all",           # ./CBR_CB_All/Outputs/all/{dataset}/{llm}/*.json
    "CBR_CB_Success": "success",
    "CBR_CB_Failure": "failure",
}

# Configs where files whose *filename* contains EXCLUDE_TOKEN are ignored
EXCLUDE_TOKEN = "_true"
CONFIGS_WITH_EXCLUSION = {"Zero_Shot_Reasoning"}

LLMS = [
    "Qwen3.5-4B-Instruct", "Qwen3-8B-MLX-4bit", "Gemma-4-E4B-it", "Gemma-3-4B-it",
    "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Phi-4-mini-instruct",
    "Phi-3.5-mini-instruct",
]
DATASETS = ["CaseHOLD", "MedQA", "MedMCQA", "ProfessionalLaw"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def to_bool(v):
    """Strict conversion of evaluation_correct to bool (null/unrecognised -> False)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no", ""):
            return False
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    return False


def list_json_files(config, sub, dataset, llm):
    """Sorted json files for a (config, dataset, llm) folder.
    For configs in CONFIGS_WITH_EXCLUSION, files whose filename contains EXCLUDE_TOKEN are dropped."""
    parts = [BASE_DIR, config, "Outputs"]
    if sub is not None:
        parts.append(sub)
    parts += [dataset, llm, "*.json"]
    files = sorted(glob.glob(os.path.join(*parts)))
    if config in CONFIGS_WITH_EXCLUSION:
        files = [f for f in files if EXCLUDE_TOKEN not in os.path.basename(f)]
        if len(files) > 1:
            print(f"WARNING: {len(files)} candidate files for {config}/{dataset}/{llm}: "
                  f"{[os.path.basename(f) for f in files]} -> merging all")
    return files


def load_results(config, sub, dataset, llm):
    """Return ({id: 0/1}, n_records) merged over the selected json files, or (None, 0) if missing.
    n_records > len(results) means duplicate ids (later records overwrite earlier ones)."""
    files = list_json_files(config, sub, dataset, llm)
    if not files:
        return None, 0
    results, n_records = {}, 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for obj in data:
            if "id" not in obj or "evaluation_correct" not in obj:
                continue
            n_records += 1
            results[obj["id"]] = int(to_bool(obj["evaluation_correct"]))
    return results, n_records


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def paired_compare(res_a, res_b):
    """Align on shared ids. Returns (mean_a, mean_b, p_value, n_shared) or None."""
    ids = sorted(set(res_a) & set(res_b), key=str)
    if len(ids) < 2:
        return None
    a = np.array([res_a[i] for i in ids], dtype=float)
    b = np.array([res_b[i] for i in ids], dtype=float)
    if np.all(a == b):
        p = 1.0  # identical outputs: t-test would give NaN
    else:
        p = stats.ttest_rel(a, b).pvalue
    return a.mean(), b.mean(), p, len(ids)


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def verify_accuracy(acc, tol=1e-9):
    """Recompute accuracy = #(evaluation_correct is True) / #records straight from the raw
    JSON files and compare with the values used in the plot."""
    rows = []
    for (dataset, llm, cfg), plotted in sorted(acc.items()):
        n_true = n_total = n_nonbool = n_noid = 0
        ids = []
        for f in list_json_files(cfg, CONFIGS[cfg], dataset, llm):
            with open(f, "r", encoding="utf-8") as fh:
                for obj in json.load(fh):
                    n_total += 1
                    v = obj.get("evaluation_correct")
                    if not isinstance(v, bool):
                        n_nonbool += 1
                    if "id" not in obj:
                        n_noid += 1
                    else:
                        ids.append(obj["id"])
                    n_true += (v is True)
        raw = 100.0 * n_true / n_total if n_total else np.nan
        rows.append({
            "dataset": dataset, "llm": llm, "config": cfg,
            "raw_acc": raw, "plotted_acc": plotted, "diff": plotted - raw,
            "records": n_total, "unique_ids": len(set(ids)),
            "non_bool": n_nonbool, "no_id": n_noid,
            "match": abs(plotted - raw) < tol,
        })
    df = pd.DataFrame(rows)
    bad = df[~df["match"]]
    print("\n" + "=" * 70)
    print("ACCURACY VERIFICATION (plotted vs raw #true / #records)")
    print("=" * 70)
    print(f"{len(df) - len(bad)}/{len(df)} bars match the raw fraction exactly.")
    if len(bad):
        print("\nMismatches (likely causes: common-id filtering, duplicate ids, "
              "records without id, non-boolean values):\n")
        print(bad.round(4).to_string(index=False))
    flagged = df[(df["non_bool"] > 0) | (df["no_id"] > 0)]
    if len(flagged):
        print("\nRecords with non-boolean 'evaluation_correct' or missing 'id':\n")
        print(flagged[["dataset", "llm", "config", "records", "non_bool", "no_id"]]
              .to_string(index=False))
    return df


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
                    # Tall bar: white text inside, just below the top.
                    # Very short bar: text goes above the bar in black.
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
    fig.legend(handles, labels, loc="upper center", ncol=n_cfg, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot to {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    names = list(CONFIGS)
    better = pd.DataFrame(0, index=names, columns=names)
    signif = pd.DataFrame(0, index=names, columns=names)
    n_compared = pd.DataFrame(0, index=names, columns=names)
    min_n = pd.DataFrame(np.nan, index=names, columns=names)
    max_n = pd.DataFrame(np.nan, index=names, columns=names)

    missing = []
    size_rows = []      # per (dataset, llm, config): unique ids and duplicates
    mismatch_rows = []  # per (dataset, llm, config pair) where instances were dropped
    acc = {}            # (dataset, llm, config) -> mean accuracy in %

    for dataset, llm in itertools.product(DATASETS, LLMS):
        data = {}
        for cfg, sub in CONFIGS.items():
            r, n_rec = load_results(cfg, sub, dataset, llm)
            if r is None:
                missing.append((cfg, dataset, llm))
            data[cfg] = r
            size_rows.append({
                "dataset": dataset, "llm": llm, "config": cfg,
                "unique_ids": None if r is None else len(r),
                "duplicate_records": None if r is None else n_rec - len(r),
            })

        # Mean accuracies for plotting
        available = [c for c in names if data[c] is not None and len(data[c]) > 0]
        if available:
            common = set.intersection(*(set(data[c]) for c in available))
            for cfg in available:
                if USE_COMMON_IDS_FOR_PLOT and common:
                    vals = [data[cfg][i] for i in common]
                else:
                    vals = list(data[cfg].values())
                acc[(dataset, llm, cfg)] = 100.0 * float(np.mean(vals))

        # Instance alignment report (unordered pairs)
        for ci, cj in itertools.combinations(names, 2):
            if data[ci] is None or data[cj] is None:
                continue
            shared = set(data[ci]) & set(data[cj])
            drop_i, drop_j = len(data[ci]) - len(shared), len(data[cj]) - len(shared)
            if drop_i or drop_j:
                mismatch_rows.append({
                    "dataset": dataset, "llm": llm, "pair": f"{ci} vs {cj}",
                    "n_first": len(data[ci]), "n_second": len(data[cj]),
                    "n_used": len(shared),
                    "dropped_first": drop_i, "dropped_second": drop_j,
                })

        # Paired comparisons (ordered pairs)
        for ci, cj in itertools.permutations(names, 2):
            if data[ci] is None or data[cj] is None:
                continue
            out = paired_compare(data[ci], data[cj])
            if out is None:
                continue
            mi, mj, p, n = out
            n_compared.loc[ci, cj] += 1
            min_n.loc[ci, cj] = n if np.isnan(min_n.loc[ci, cj]) else min(min_n.loc[ci, cj], n)
            max_n.loc[ci, cj] = n if np.isnan(max_n.loc[ci, cj]) else max(max_n.loc[ci, cj], n)
            if mi > mj:
                better.loc[ci, cj] += 1
                if p < ALPHA:
                    signif.loc[ci, cj] += 1

    # ---- Main table ----
    table = pd.DataFrame("-", index=names, columns=names)
    for ci, cj in itertools.permutations(names, 2):
        table.loc[ci, cj] = f"{better.loc[ci, cj]} ({signif.loc[ci, cj]})"

    print(f"\nCell (row i, col j): #(dataset, LLM) pairs where row config had higher mean accuracy "
          f"than column config (in brackets: # significant at p < {ALPHA}, paired t-test)\n")
    print(table.to_string())
    print(f"\nMax (dataset, LLM) pairs per cell: {len(DATASETS) * len(LLMS)}")
    print(f"Actually compared per cell:\n{n_compared.to_string()}")

    # ---- Instance counts ----
    sizes = pd.DataFrame(size_rows)
    print("\n" + "=" * 70)
    print("INSTANCES PER CONFIGURATION (unique ids)")
    print("=" * 70)
    print(sizes.pivot_table(index=["dataset", "llm"], columns="config",
                            values="unique_ids", aggfunc="first")[names].to_string())

    dups = sizes[sizes["duplicate_records"].fillna(0) > 0]
    if len(dups):
        print(f"\nWARNING: {len(dups)} (dataset, llm, config) combos contain duplicate ids "
              f"(later records overwrote earlier ones):")
        print(dups.to_string(index=False))

    print("\n" + "=" * 70)
    print("INSTANCES USED IN COMPARISONS (shared ids), min / max across (dataset, LLM) pairs")
    print("=" * 70)
    summary = pd.DataFrame("-", index=names, columns=names)
    for ci, cj in itertools.permutations(names, 2):
        if not np.isnan(min_n.loc[ci, cj]):
            lo, hi = int(min_n.loc[ci, cj]), int(max_n.loc[ci, cj])
            summary.loc[ci, cj] = str(lo) if lo == hi else f"{lo}-{hi}"
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("INSTANCES DROPPED DUE TO ID MISMATCH")
    print("=" * 70)
    if mismatch_rows:
        mm = pd.DataFrame(mismatch_rows)
        print(f"{len(mm)} comparisons dropped at least one instance:\n")
        print(mm.to_string(index=False))
    else:
        print("None: every compared pair of configurations had identical id sets.")

    if missing:
        print(f"\nWarning: {len(missing)} missing (config, dataset, llm) combos, e.g. {missing[:3]}")

    # ---- Mean accuracy table, verification, plot ----
    if acc:
        acc_df = pd.DataFrame(
            [{"dataset": d, "llm": l, "config": c, "accuracy": a} for (d, l, c), a in acc.items()]
        ).pivot_table(index=["dataset", "llm"], columns="config", values="accuracy")[names]
        print("\n" + "=" * 70)
        print("MEAN ACCURACY (%)")
        print("=" * 70)
        print(acc_df.round(3).to_string())
        verify_accuracy(acc)
        plot_accuracy(acc, names, PLOT_PATH)
    else:
        print("\nNo results found, skipping plot.")


if __name__ == "__main__":
    main()