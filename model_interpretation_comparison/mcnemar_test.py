import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

df_a = pd.read_csv('gpt4omini/merged_gpt4omini_inference_contributions.csv')  # GPT-4o-mini
df_b = pd.read_csv('gpt5mini/merged_gpt5mini_inference_contributions.csv')   # GPT-5-mini

suffixes = ['CLO', 'CL', 'UBERON', 'BTO']


def classify(row, suf, mode):
    """Returns True (correct), False (incorrect), or None (excluded) for a given row."""
    equiv = row.get(f'{suf}_equiv')
    contrib = row.get(f'{suf}_contrib')
    equiv = equiv if isinstance(equiv, str) else None
    contrib = contrib if isinstance(contrib, str) else None

    if mode == 'precision':
        # include only rows where the model made a prediction (M != '-')
        if equiv is not None:
            if equiv == 'missing':        # model predicted '-' while gold existed
                return None
            return equiv in ('exact', 'validated')
        if contrib is not None:           # gold was '-', model predicted something
            return contrib == 'valid'
        return None                       # model predicted '-' -> not part of precision

    if mode == 'recall':
        # include only rows where the gold standard has a true value (C != '-')
        if equiv is not None:
            return equiv in ('exact', 'validated')
        return None                       # gold == '-' -> not part of recall


def run_mcnemar_tests():
    results = []
    for suf in suffixes:
        merged = df_a[['Sample_ID']].merge(df_b[['Sample_ID']], on='Sample_ID')
        da = df_a.set_index('Sample_ID')
        db = df_b.set_index('Sample_ID')

        for mode in ['precision', 'recall']:
            rows = []
            for sid in merged['Sample_ID']:
                ra, rb = da.loc[sid], db.loc[sid]
                ca = classify(ra, suf, mode)
                cb = classify(rb, suf, mode)
                # McNemar requires the row to be comparable for BOTH models
                if ca is None or cb is None:
                    continue
                rows.append((ca, cb))

            n = len(rows)
            if n == 0:
                results.append((suf, mode.capitalize(), 0, None, None, None, None))
                continue

            both_correct = sum(1 for a, b in rows if a and b)       # both models correct
            a_yes_b_no = sum(1 for a, b in rows if a and not b)     # only GPT-4o-mini correct (b)
            a_no_b_yes = sum(1 for a, b in rows if not a and b)     # only GPT-5-mini correct (c)
            both_wrong = sum(1 for a, b in rows if not a and not b)  # both models wrong

            table = [[both_correct, a_yes_b_no],
                     [a_no_b_yes, both_wrong]]
            b, c = a_yes_b_no, a_no_b_yes

            # exact binomial test when b+c is small; chi-squared with Yates' correction otherwise
            exact = (b + c) < 25
            res = mcnemar(table, exact=exact, correction=not exact)
            results.append((suf, mode.capitalize(), n, b, c, res.pvalue,
                             'exact' if exact else 'chi2-corrected'))
    return results


def print_results(results):
    print(f"{'Ontology':8} {'Metric':10} {'n':>4} {'b(4o>5)':>8} {'c(5>4o)':>8} {'p-value':>9}  {'method'}")
    for suf, mode, n, b, c, p, method in results:
        if n == 0:
            print(f"{suf:8} {mode:10} {n:>4}  -- no applicable cases --")
            continue
        sig = '*' if p is not None and p < 0.05 else ''
        print(f"{suf:8} {mode:10} {n:>4} {b:>8} {c:>8} {p:>9.4f} {sig} {method}")


def plot_results(results, out_path='mcnemar_diverging.png'):
    """Diverging bar chart: b (GPT-4o-mini only correct) left, c (GPT-5-mini only correct) right.
    Colors come from the validated categorical palette (slot 1 blue / slot 2 orange),
    which passes the CVD and normal-vision separation checks -- not an arbitrary choice."""
    plotted = [r for r in results if r[2] > 0]  # skip empty cases
    labels = [f"{suf} \u2013 {metric}" for suf, metric, *_ in plotted]
    b_vals = [-b for _, _, n, b, c, p, m in plotted]
    c_vals = [c for _, _, n, b, c, p, m in plotted]
    p_vals = [p for _, _, n, b, c, p, m in plotted]
    sig_flags = [p is not None and p < 0.05 for p in p_vals]

    COLOR_4O = "#2a78d6"   # validated categorical slot 1 (blue)
    COLOR_5 = "#eb6834"    # validated categorical slot 2 (orange)
    SIG_EDGE = "#2E2E2E"

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(labels))

    bars_b = ax.barh(y_pos, b_vals, color=COLOR_4O, label="GPT-4o-mini correct only (b)")
    bars_c = ax.barh(y_pos, c_vals, color=COLOR_5, label="GPT-5-mini correct only (c)")

    for i, sig in enumerate(sig_flags):
        if sig:
            bars_b[i].set_edgecolor(SIG_EDGE)
            bars_b[i].set_linewidth(1.8)
            bars_c[i].set_edgecolor(SIG_EDGE)
            bars_c[i].set_linewidth(1.8)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Discordant pairs (McNemar)")
    ax.set_title("GPT-4o-mini vs GPT-5-mini: McNemar's test\nDiscordant pairs by ontology and metric")

    xmax = max(c_vals) + 3
    for i, (p, sig) in enumerate(zip(p_vals, sig_flags)):
        star = " *" if sig else ""
        ax.text(xmax, i, f"p = {p:.3f}{star}", va="center", ha="left",
                 fontsize=9, fontweight="bold" if sig else "normal",
                 color=SIG_EDGE if sig else "#666666")

    ax.set_xlim(-max(abs(min(b_vals)), 1) - 2, xmax + 7)
    fig.subplots_adjust(right=0.78)  # leave room for the external legend
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


def compute_precision_recall(df):
    """Precision/recall per ontology for a single model's dataframe, using the same
    exact/validated/valid classification as classify() -- NOT raw string equality.
    This keeps the precision/recall figures consistent with the paper's TP/FP/FN
    definition (Table S3: I/R/C = correct, E/IC = wrong) and with the McNemar test above."""
    precisions, recalls = {}, {}
    for suf in suffixes:
        pred_correct = pred_total = 0
        gold_correct = gold_total = 0
        for _, row in df.iterrows():
            p = classify(row, suf, 'precision')
            if p is not None:
                pred_total += 1
                pred_correct += int(p)
            r = classify(row, suf, 'recall')
            if r is not None:
                gold_total += 1
                gold_correct += int(r)
        precisions[suf] = pred_correct / pred_total if pred_total else None
        recalls[suf] = gold_correct / gold_total if gold_total else None
    return precisions, recalls


def plot_precision_recall(model_names=('GPT-4o-mini', 'GPT-5-mini'),
                           out_path='precision_recall_comparison.png'):
    """Two side-by-side panels (Precision | Recall), one bar group per ontology,
    one bar per model. A single color per model -- the SAME two validated colors
    used in the McNemar figure, so color consistently means "model" across both
    figures -- and no hatching, no extra hues: that combination is what the
    reviewer flagged as jarring in the original figure."""
    dfs = [df_a, df_b]
    prec_list, rec_list = [], []
    for df in dfs:
        p, r = compute_precision_recall(df)
        prec_list.append(p)
        rec_list.append(r)

    COLORS = ["#2a78d6", "#eb6834"]  # same validated pair as the McNemar figure
    x = range(len(suffixes))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, metric_list, title in zip(axes, (prec_list, rec_list), ("Precision", "Recall")):
        for m_idx, (name, color) in enumerate(zip(model_names, COLORS)):
            vals = [metric_list[m_idx][suf] or 0 for suf in suffixes]
            offset = (m_idx - 0.5) * width
            ax.bar([xi + offset for xi in x], vals, width=width,
                   color=color, label=name)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}", (xi + offset, v), ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(suffixes)
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Score")
    fig.suptitle("GPT-4o-mini vs GPT-5-mini: Precision and Recall by ontology",
                 fontweight="bold")

    # single shared legend, outside the plot area (top-right of the figure)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.95),
               frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    results = run_mcnemar_tests()
    print_results(results)
    plot_results(results)
    plot_precision_recall()