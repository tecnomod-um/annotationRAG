"""
Ablation study (Q2.2): 2x2 design = interpretation (off/on) x knowledge source (reduced/full ontology).

Conditions (annotation model GPT-4o-mini, k = 5, same prompts and concept-type routing):
    A : full ontology,    no interpretation
    B  : full ontology,    with interpretation (GPT-5-mini)
    C  : reduced ontology, no interpretation
    D  : reduced ontology, with interpretation (GPT-5-mini)

Outputs
    ablation_metrics.csv          precision / recall / F1 per condition and ontology (+ pooled "All")
    ablation_mcnemar.csv          paired McNemar tests (raw p and Benjamini-Hochberg q)
    ablation_mcnemar.png          diverging bars, one panel per comparison
    ablation_precision_recall.png precision | recall | F1 per ontology, four conditions

Every input CSV must contain the ID column and, for each ontology, the evaluation columns
'<ONTO>_equiv' and '<ONTO>_contrib' (same format as the previous merged_*_inference_contributions.csv).
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# ----------------------------------------------------------------------------- configuration
ID_COL = "Label"                          # key shared by the four files
SUFFIXES = ["CLO", "CL", "UBERON", "BTO"]

CONDITIONS = {                            # edit the paths
    "A": dict(path="experiments/full_ontology/gpt4omini_fullontology_contributions.csv",    source="Full",    interp=False),
    "B":  dict(path="experiments/full_ontology_interpretation/gpt5mini_fullontology_interpretation_contributions.csv",       source="Full",    interp=True),
    "C":  dict(path="experiments/reduced_ontology/gpt5mini_reduced_contributions.csv", source="Reduced", interp=False),
    "D":  dict(path="experiments/reduced_ontology_interpretation/gpt5mini_reduced_interpretation_contributions.csv",    source="Reduced", interp=True),
}

# (baseline, comparison, title).  b = only the baseline is correct, c = only the comparison is correct
COMPARISONS = [
    ("C",  "D", "Interpretation effect\nreduced ontology (C → D)"),
    ("A", "B", "Interpretation effect\nfull ontology (A → B)"),
    ("C",  "A", "Knowledge-source effect\nno interpretation (C → A)"),
    ("D",  "B", "Knowledge-source effect\nwith interpretation (D → B)"),
]

ALPHA = 0.05
SIGNIFICANCE_ON = "q"                     # "q" = Benjamini-Hochberg adjusted, "p" = raw
COLOR_BASE = "#56B4E9"                    # Okabe-Ito yellow    (baseline / reduced ontology)
COLOR_COMP = "#CC79A7"                    # Okabe-Ito pink  (comparison / full ontology)
MC_BASE = "#009E73"
MC_COMP = "#D55E00"
EDGE = "#2E2E2E"


# ----------------------------------------------------------------------------- classification
def classify(row, suf, mode):
    """True (correct), False (incorrect) or None (not applicable) for one label / ontology / metric.
    Same rule as in the paper: exact + validated + valid contributions are correct."""
    equiv = row.get(f"{suf}_equiv")
    contrib = row.get(f"{suf}_contrib")
    equiv = equiv if isinstance(equiv, str) else None
    contrib = contrib if isinstance(contrib, str) else None

    if mode == "precision":               # only rows where the model made a prediction
        if equiv is not None:
            if equiv == "missing":        # model predicted '-' while gold existed
                return None
            return equiv in ("exact", "validated")
        if contrib is not None:           # gold was '-', model predicted something
            return contrib == "valid"
        return None
    if mode == "recall":                  # only rows where the gold standard has a value
        if equiv is not None:
            return equiv in ("exact", "validated")
        return None


def load_conditions():
    data = {}
    for key, cfg in CONDITIONS.items():
        df = pd.read_csv(cfg["path"])
        if df[ID_COL].duplicated().any():
            raise SystemExit(f"{cfg['path']}: duplicated values in '{ID_COL}'")
        data[key] = df.set_index(ID_COL)
    common = set.intersection(*[set(d.index) for d in data.values()])
    for key, d in data.items():
        if len(d) != len(common):
            print(f"[warning] condition {key}: {len(d)} rows, {len(common)} shared by all four -> using shared rows")
    return {k: d.loc[sorted(common)] for k, d in data.items()}


def classify_all(data):
    """cls[condition][(suffix, mode)] -> pandas Series (True / False / None) indexed by ID."""
    return {key: {(s, m): df.apply(lambda r, s=s, m=m: classify(r, s, m), axis=1)
                  for s in SUFFIXES for m in ("precision", "recall")}
            for key, df in data.items()}


# ----------------------------------------------------------------------------- precision / recall / F1
def metrics_table(cls):
    rows = []
    for key, cfg in CONDITIONS.items():
        tot = dict(pc=0, pn=0, rc=0, rn=0)
        for suf in SUFFIXES + ["All"]:
            if suf == "All":                              # pooled (micro) over the four ontologies
                pc, pn, rc, rn = tot["pc"], tot["pn"], tot["rc"], tot["rn"]
            else:
                p = cls[key][(suf, "precision")].dropna()
                r = cls[key][(suf, "recall")].dropna()
                pc, pn, rc, rn = int(p.sum()), len(p), int(r.sum()), len(r)
                tot["pc"] += pc; tot["pn"] += pn; tot["rc"] += rc; tot["rn"] += rn
            prec = pc / pn if pn else float("nan")
            rec = rc / rn if rn else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
            rows.append(dict(condition=key, source=cfg["source"],
                             interpretation="With" if cfg["interp"] else "Without", ontology=suf,
                             n_pred=pn, n_gold=rn, precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows)


def print_2x2(metrics):
    """One 2x2 block (source x interpretation) per ontology, each cell 'P / R / F1'."""
    for suf in SUFFIXES + ["All"]:
        sub = metrics[metrics.ontology == suf]
        print(f"\n=== {suf}  (cell = Precision / Recall / F1) ===")
        print(f"{'':10}{'Without interpretation':>30}{'With interpretation':>30}")
        for source in ("Reduced", "Full"):
            cells = []
            for interp in ("Without", "With"):
                r = sub[(sub.source == source) & (sub.interpretation == interp)].iloc[0]
                cells.append(f"{r.precision:.3f} / {r.recall:.3f} / {r.f1:.3f}")
            print(f"{source:10}{cells[0]:>30}{cells[1]:>30}")


# ----------------------------------------------------------------------------- McNemar
def fmt_p(v):
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def fmt_eq(name, v):
    return f"{name} < 0.001" if v < 0.001 else f"{name} = {v:.3f}"


def run_mcnemar(cls):
    results = []
    for base, comp, title in COMPARISONS:
        for suf in SUFFIXES:
            for mode in ("precision", "recall"):
                a, b_ = cls[base][(suf, mode)], cls[comp][(suf, mode)]
                paired = pd.DataFrame({"a": a, "b": b_}).dropna()        # comparable in BOTH conditions
                n = len(paired)
                if n == 0:
                    results.append(dict(base=base, comp=comp, title=title, ontology=suf, metric=mode,
                                        n=0, b=None, c=None, p=None, method=None))
                    continue
                x, y = paired["a"].astype(bool), paired["b"].astype(bool)
                both = int((x & y).sum()); b = int((x & ~y).sum())       # b: only baseline correct
                c = int((~x & y).sum()); neither = int((~x & ~y).sum())  # c: only comparison correct
                if b + c == 0:
                    p, method = 1.0, "no discordant pairs"
                else:
                    exact = (b + c) < 25
                    p = float(mcnemar([[both, b], [c, neither]], exact=exact, correction=not exact).pvalue)
                    method = "exact binomial" if exact else "chi2 (continuity-corrected)"
                results.append(dict(base=base, comp=comp, title=title, ontology=suf, metric=mode,
                                    n=n, b=b, c=c, p=p, method=method))
    res = pd.DataFrame(results)
    ok = res["p"].notna()
    res["q"] = float("nan")
    res.loc[ok, "q"] = multipletests(res.loc[ok, "p"], method="fdr_bh")[1]   # BH over all tests
    return res


def print_mcnemar(res):
    print(f"\n{'Comparison':10}{'Onto':8}{'Metric':11}{'n':>4}{'b':>5}{'c':>5}{'p':>9}{'q(BH)':>9}  method")
    for _, r in res.iterrows():
        name = f"{r.base}→{r.comp}"
        if r.n == 0:
            print(f"{name:10}{r.ontology:8}{r.metric:11}{0:>4}  -- no applicable cases --")
            continue
        star = "*" if (r.q if SIGNIFICANCE_ON == "q" else r.p) < ALPHA else ""
        print(f"{name:10}{r.ontology:8}{r.metric:11}{r.n:>4}{int(r.b):>5}{int(r.c):>5}"
              f"{fmt_p(r.p):>9}{fmt_p(r.q):>9} {star:1} {r.method}")


# ----------------------------------------------------------------------------- figures
def plot_mcnemar(res, out_path="ablation_mcnemar.png"):
    """2x2 grid of diverging bars (one panel per comparison). Left = only the baseline condition is
    correct (b), right = only the comparison condition is correct (c). Legend outside the axes."""
    res = res[res.n > 0]
    xmax = int(res[["b", "c"]].max().max()) + 3
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (base, comp, title) in zip(axes.ravel(), COMPARISONS):
        sub = res[(res.base == base) & (res.comp == comp)].reset_index(drop=True)
        y = range(len(sub))
        bars_b = ax.barh(y, -sub["b"], color=MC_BASE)
        bars_c = ax.barh(y, sub["c"], color=MC_COMP)
        for i, r in sub.iterrows():
            sig = (r.q if SIGNIFICANCE_ON == "q" else r.p) < ALPHA
            if sig:
                for bar in (bars_b[i], bars_c[i]):
                    bar.set_edgecolor(EDGE); bar.set_linewidth(1.8)
            ax.text(xmax + 1, i, f"{fmt_eq('p', r.p)}, {fmt_eq('q', r.q)}{' *' if sig else ''}", va="center",
                    ha="left", fontsize=8, fontweight="bold" if sig else "normal",
                    color=EDGE if sig else "#666666")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{r.ontology} – {r.metric.capitalize()}" for _, r in sub.iterrows()])
        ax.invert_yaxis()
        ax.set_xlim(-xmax, xmax + 14)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Discordant pairs (McNemar)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MC_BASE), plt.Rectangle((0, 0), 1, 1, color=MC_COMP)]
    fig.legend(handles, ["Baseline condition correct only (b)", "Comparison condition correct only (c)"],
               loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fig.suptitle("Ablation study: McNemar's test (paired, per label)\n"
                 "* = significant after Benjamini-Hochberg correction" if SIGNIFICANCE_ON == "q"
                 else "Ablation study: McNemar's test (paired, per label)\n* = p < 0.05", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


def _tint(hex_color, amount):
    """Mix a color with white (amount = share of white)."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


def plot_precision_recall(metrics, out_path="ablation_precision_recall.png"):
    """Three panels (Precision | Recall | F1), groups = ontologies (+ pooled), 4 bars per group.
    Hue = knowledge source (bluish green = reduced, reddish purple = full); light fill + coloured outline = without,
    solid fill = with interpretation. No hatching and no extra hues. These hues are deliberately different from
    the blue/orange pair used in the model-comparison figures."""
    order = [("C", "Reduced, no interpretation"), ("D", "Reduced, with interpretation"),
             ("A", "Full, no interpretation"), ("B", "Full, with interpretation")]
    colors = {"C": _tint(COLOR_BASE, 0.55), "D": COLOR_BASE, "A": _tint(COLOR_COMP, 0.55), "B": COLOR_COMP}
    edges = {"C": COLOR_BASE, "D": COLOR_BASE, "A": COLOR_COMP, "B": COLOR_COMP}
    groups = SUFFIXES + ["All"]
    x = range(len(groups))
    width = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (col, title) in zip(axes, (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))):
        for j, (key, name) in enumerate(order):
            vals = [float(metrics[(metrics.condition == key) & (metrics.ontology == g)][col].iloc[0])
                    for g in groups]
            offset = (j - 1.5) * width
            ax.bar([xi + offset for xi in x], vals, width=width, color=colors[key], label=name,
                   edgecolor=edges[key], linewidth=1.2)
            for xi, v in zip(x, vals):
                if v == v:
                    ax.annotate(f"{v:.2f}", (xi + offset, v), ha="center", va="bottom",
                                xytext=(0, 2), textcoords="offset points", fontsize=7)
        ax.set_xticks(list(x)); ax.set_xticklabels(groups)
        ax.set_title(title); ax.set_ylim(0, 1.1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("Ablation study: precision, recall and F1 by ontology and condition", fontweight="bold")
    plt.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend at the bottom
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    data = load_conditions()
    cls = classify_all(data)

    metrics = metrics_table(cls)
    print_2x2(metrics)
    metrics.to_csv("ablation_metrics.csv", index=False)

    res = run_mcnemar(cls)
    print_mcnemar(res)
    res.drop(columns=["title"]).to_csv("ablation_mcnemar.csv", index=False)

    plot_mcnemar(res)
    plot_precision_recall(metrics)