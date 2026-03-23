from operator import index
import pandas as pd
import matplotlib.pyplot as plt


def get_precision_recall(df):
    """
    Calculate precision and recall for each ontology in the given dataframe.

    Parameters:
        df (DataFrame): The dataframe containing predictions and true values
                        for each ontology. Columns are expected to include
                        pairs of columns for each ontology with the suffixes '_C'
                        (for true values) and '_M' (for model predictions).

    Returns:
        tuple: Two dictionaries with precision and recall per ontology suffix.
    """
    for column in df:
        df[column] = df[column].fillna('unknown')

    suffixes = ['CLO', 'CL', 'UBERON', 'BTO']

    precisions = {}
    recalls = {}

    for suffix in suffixes:
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        tp = 0
        fp = 0
        fn = 0

        if true_col in df.columns and pred_col in df.columns:
            for i in range(len(df)):
                true_val = df.iloc[i][true_col]
                pred_val = df.iloc[i][pred_col]

                # Skip rows where both true and predicted values are "-"
                if true_val == "-" and pred_val == "-":
                    continue
                elif true_val != "-" and pred_val == "-":
                    # Model missed a true annotation → False Negative
                    fn += 1
                elif true_val == pred_val:
                    # Correct prediction → True Positive
                    tp += 1
                else:
                    # Wrong prediction → False Positive
                    fp += 1

            # Precision: of all predicted positives, how many are correct
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # Recall: of all actual positives, how many were found
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions[suffix] = precision
            recalls[suffix] = recall
        else:
            precisions[suffix] = None
            recalls[suffix] = None

    return precisions, recalls


def plot_metrics(models_data, model_names):
    """
    Plot precision and recall of different models for each ontology.
    Each model gets two bars: one for precision, one for recall.
    """
    all_precisions = []
    all_recalls = []

    for df in models_data:
        prec, rec = get_precision_recall(df)
        all_precisions.append(prec)
        all_recalls.append(rec)

    ontologies = ['CLO', 'CL', 'UBERON', 'BTO']

    # Color palette: each ontology gets two shades (precision=darker, recall=lighter)
    base_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    light_colors = ['#90CAF9', '#A5D6A7', '#FFCC80', '#F48FB1']

    bar_width = 2.5
    pair_gap = 0.5       # gap between precision and recall bars of same ontology
    ont_gap = 1.5        # gap between ontology groups within a model
    model_gap = 6        # extra gap between models

    fig, ax = plt.subplots(figsize=(16, 7))

    # Compute the starting x position for each model
    n_ont = len(ontologies)
    group_width = n_ont * (bar_width * 2 + pair_gap) + (n_ont - 1) * ont_gap
    model_positions = []
    current_x = 0
    for _ in model_names:
        model_positions.append(current_x)
        current_x += group_width + model_gap

    xtick_positions = []
    xtick_labels = []

    for m_idx, (model_name, prec_dict, rec_dict) in enumerate(
            zip(model_names, all_precisions, all_recalls)):

        model_start = model_positions[m_idx]
        ont_centers = []

        for o_idx, ont in enumerate(ontologies):
            # x position for this ontology group within the model
            ont_start = model_start + o_idx * (bar_width * 2 + pair_gap + ont_gap)

            prec_x = ont_start
            rec_x = ont_start + bar_width + pair_gap

            prec_val = prec_dict[ont] if prec_dict[ont] is not None else 0
            rec_val = rec_dict[ont] if rec_dict[ont] is not None else 0

            # Precision bar
            ax.bar(prec_x, prec_val, width=bar_width,
                   color=base_colors[o_idx], label=f'{ont} Precision' if m_idx == 0 else "")
            ax.annotate(f'{prec_val:.3f}',
                        (prec_x, prec_val), ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=9, color='black')

            # Recall bar
            ax.bar(rec_x, rec_val, width=bar_width,
                   color=light_colors[o_idx], label=f'{ont} Recall' if m_idx == 0 else "",
                   hatch='//')
            ax.annotate(f'{rec_val:.3f}',
                        (rec_x, rec_val), ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=9, color='black')

            ont_centers.append((prec_x + rec_x) / 2)

        # Model label centered under its group
        model_center = (ont_centers[0] + ont_centers[-1]) / 2
        xtick_positions.append(model_center)
        xtick_labels.append(model_name)

    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Ontology Precision and Recall by Method', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

    # Custom legend: one entry per ontology with solid=Precision, hatch=Recall
    from matplotlib.patches import Patch
    legend_elements = []
    for o_idx, ont in enumerate(ontologies):
        legend_elements.append(
            Patch(facecolor=base_colors[o_idx], label=f'{ont} – Precision'))
        legend_elements.append(
            Patch(facecolor=light_colors[o_idx], hatch='//', label=f'{ont} – Recall'))

    ax.legend(handles=legend_elements, fontsize=10, title='Ontology / Metric',
              title_fontsize=11, loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, ncol=1)

    plt.tight_layout()
    plt.show()


def main():
    df_comparison_RAG_5mini_reduced_inference_index = pd.read_csv("RAG_reduced_info.csv", header=0)  # our RAG approach
    df_comparison_RAG_4o_Bioportal_RAG = pd.read_csv("RAG_with_Bioportal.csv", header=0)  # RAG with Biportal
    df_comparison_4o_mini = pd.read_csv("base_model.csv", header=0)  # base model
    df_comparison_4o_mini_ft = pd.read_csv("finetuned-model.csv", header=0)  # fine-tuned model
    df_RAG_full_ontologies = pd.read_csv("RAG_full_ontologies.csv", header=0)

    models_data = [df_comparison_4o_mini, df_comparison_4o_mini_ft, df_RAG_full_ontologies,
                   df_comparison_RAG_4o_Bioportal_RAG, df_comparison_RAG_5mini_reduced_inference_index]
    model_names = ['Base model', 'Fine-tuned model', 'RAG full ontologies', 'RAG+Bioportal', "RAG reduced information"]

    plot_metrics(models_data, model_names)


if __name__ == "__main__":
    main()