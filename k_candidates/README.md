# **k_candidates — Candidate Retrieval Experiments**

This directory contains the experiments and analysis used to determine the **optimal number of retrieved candidates (k)** in the RAG pipeline.

The goal is to evaluate how candidate pool size (e.g., 5K vs 10K) affects:

* Retrieval quality
* Annotation accuracy
* Candidate ranking behavior

---

## **Overview**

The workflow in this directory follows three main steps:

```text
1. Sample selection (stratified)
2. Dataset alignment/filtering
3. Candidate position analysis
```

These steps allow a controlled and fair comparison between different retrieval configurations.

---

## **Scripts (Execution Order)**

All [scripts](https://github.com/tecnomod-um/annotationRAG/tree/main/k_candidates/scripts) are located in:

```
scripts/
```

They should be executed **in the following order**:

---

## **1. [`get_Sample_labels_errors.py`](https://github.com/tecnomod-um/annotationRAG/blob/main/k_candidates/scripts/get_sample_labels_errors.py)**

### **Purpose**

Creates a **stratified sample of 100 labels** from the dataset to ensure balanced evaluation across biological entity types.

### **Sampling Strategy**

* 50 → Cell Lines (`CL`)
* 38 → Cell Types (`CT`)
* 12 → Anatomical Structures (`A`)

### **Key Features**

* Ensures reproducibility using a fixed random seed (`42`)
* Maintains class balance
* Adds a unique identifier (`Sample_ID`) for traceability


### **Output**

[100_df_5_mini_inference_index.csv](https://github.com/tecnomod-um/annotationRAG/blob/main/k_candidates/5K/100_df_5_mini_inference_index.csv)
[100_df_5_mini_inference_index_10k.csv](https://github.com/tecnomod-um/annotationRAG/blob/main/k_candidates/10K/100_df_5_mini_inference_index_10k.csv)


### **Usage**

```bash
python get_Sample_labels_errors.py
```

---

## **2. `count_positions.py`**

### **Purpose**

Analyzes **where the correct ontology candidate appears** in the retrieved list.

This is a key step to evaluate:

* Retrieval effectiveness
* Ranking quality
* Impact of candidate pool size

---

## **Manual Annotation File**

The script uses:
[candidate_positions.tsv](candidate_positions.tsv)


This file is **manually curated** and contains, for each label:

* The position of the correct candidate in each ontology list

### **Format Example**

```text
Label	Type	CLO	CL	UBERON	BTO
ovary adult pool1	A	-	-	1	1
Fetal_Kidney	A	-	-	0	1
Arteries_C	A	-	-	1	1
VentricleRight	A	-	-	1	1
small_intestine_1	A	-	-	1	1
UCSD_Aorta	A	-	-	1	1
```

### **Interpretation**

* `1-X` → position
* `-` → not applicable for that ontology

---

### **What the Script Does**

```python
df.groupby(["Type", "UBERON"]).size()
```

* Groups results by:

  * Biological type (`CL`, `CT`, `A`)
  * Candidate position
* Outputs frequency counts

---

### **Usage**

```bash
python count_positions.py
```

---

## **Experimental Insight**

This pipeline allows you to:

* Compare **5K vs 10K candidate pools**
* Measure how often the correct candidate appears.
* Understand whether increasing k improves retrieval or introduces noise.

---

## **Design Considerations**

* **Stratified sampling** ensures fair representation of biological entities
* **Dataset alignment** avoids biased comparisons
* **Manual validation** provides high-quality ground truth for ranking evaluation

---

## **Notes**

* Paths in scripts are relative → run scripts from the `scripts/` directory
* Ensure all input files exist before execution
* The manual annotation step (`candidate_positions.tsv`) is critical for evaluation quality

