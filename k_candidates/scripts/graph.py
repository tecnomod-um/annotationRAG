import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

# ===================================================================================
# Datos actualizados
# ===================================================================================
cell_line = pd.DataFrame({
    'Ontology': ['CLO', 'CL', 'UBERON', 'BTO'],
    'Hits@1':[0.26,0.48,0.48,0.52],
    'Hits@5':[0.36,0.78,0.66,0.64],
    'Hits@10':[0.38,0.82,0.80,0.66],
    'pval5-10': [0.836, 0.617, 0.115, 0.834],
    'pval1-5': [0.280, 0.002, 0.069, 0.224]
})

cell_type = pd.DataFrame({
    'Ontology': ['CL', 'UBERON', 'BTO'],
    'Hits@1':[0.71,0.63,0.55],
    'Hits@5':[0.95,0.76,0.79],
    'Hits@10':[1.00,0.89,0.79],
    'pval5-10': [0.152, 0.128, 1.000],
    'pval1-5': [0.006, 0.212, 0.028]
})

anatomy = pd.DataFrame({
    'Ontology': ['UBERON', 'BTO'],
    'Hits@1':[0.67,0.75],
    'Hits@5':[1.00,1.00],
    'Hits@10':[1.00,1.00],
    'pval5-10': [1.000, 1.000],
    'pval1-5': [0.028, 0.064]
})

datasets = [
    (cell_line, 'Hits@ and p-values for Cell lines', 'hits_cell_lines.png'),
    (cell_type, 'Hits@ and p-values for Cell types', 'hits_cell_types.png'),
    (anatomy,  'Hits@ and p-values for Anatomical structures', 'hits_anatomy.png')
]

# ===================================================================================
# Función genérica para brackets
# ===================================================================================
def add_pvalue_bracket(ax, x1, x2, y, pvalue, text_offset=0.02):
    ax.plot([x1, x1, x2, x2],
            [y, y + 0.015, y + 0.015, y],
            lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + text_offset,
            f"p={pvalue:.3f}", ha='center', va='bottom', fontsize=9)

# ===================================================================================
# Estética
# ===================================================================================
plt.style.use('seaborn-v0_8-whitegrid')
cmap = cm.get_cmap('Reds')
colors_red = [cmap(0.4), cmap(0.6), cmap(0.8)]

saved_files = []

# ===================================================================================
# Graficado con dos brackets por término
# ===================================================================================
for df, title, path in datasets:

    fig, ax = plt.subplots(figsize=(7, 5))

    x = range(len(df))
    width = 0.25

    # Barras
    ax.bar([p - width for p in x], df['Hits@1'], width, label='Hits@1', color=colors_red[0])
    ax.bar(x, df['Hits@5'], width, label='Hits@5', color=colors_red[1])
    ax.bar([p + width for p in x], df['Hits@10'], width, label='Hits@10', color=colors_red[2])

    # Añadir brackets para cada término
    for i in x:
        hits1 = df.loc[i, 'Hits@1']
        hits5 = df.loc[i, 'Hits@5']
        hits10 = df.loc[i, 'Hits@10']

        # Alturas crecientes para evitar solapamiento
        y1 = max(hits1, hits5, hits10) + 0.04    # para pval1-5
        y2 = y1 + 0.06                           # para pval5-10

        # Posiciones horizontales de cada barra
        x1 = i - width      # Hits@1
        x5 = i              # Hits@5
        x10 = i + width     # Hits@10

        # Bracket 1: Hits@1 vs Hits@5
        add_pvalue_bracket(ax, x1, x5, y1, df.loc[i, 'pval1-5'])

        # Bracket 2: Hits@5 vs Hits@10
        add_pvalue_bracket(ax, x5, x10, y2, df.loc[i, 'pval5-10'])

    ax.set_xticks(x)
    ax.set_xticklabels(df['Ontology'])
    ax.set_ylim(0, 1.2)
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    saved_files.append(path)

print(saved_files)
