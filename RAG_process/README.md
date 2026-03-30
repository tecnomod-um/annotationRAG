# **RAG_process — Two-Stage RAG Pipeline**

This directory contains the full implementation of the **two-stage Retrieval-Augmented Generation (RAG) pipeline** used for automated biological metadata annotation.

It integrates:

* **Semantic interpretation of labels** using an LLM
* **Ontology-based retrieval + annotation** using a reduced multi-ontology knowledge base

---

## **Overview of the Pipeline**

The pipeline is divided into two sequential stages:

### **1. Interpretation Stage (GPT-5-mini)**

Each input label is analyzed to extract structured biological meaning:

* Cell Line
* Cell Type
* Anatomical Structure
* Additional contextual interpretation

This step transforms raw labels into **structured semantic representations**, improving downstream retrieval.

---

### **2. Annotation Stage (GPT-4o-mini)**

Using:

* The interpreted label
* Retrieved ontology candidates

The model selects the most appropriate **ontology identifiers (IRIs)** from:

* CLO
* CL
* UBERON
* BTO

---

## **Main Script**

### **`rag_gpt_PC_index.py`**

This is the core script implementing the full RAG workflow.

### **Key Responsibilities**

* Load dataset of biological labels (TSV format)
* Perform **label interpretation using GPT-5-mini**
* Query ontology embeddings (ChromaDB)
* Retrieve candidate ontology terms
* Perform **LLM-based candidate selection**
* Store final annotations and intermediate outputs

---

## **Pipeline Workflow**

```
Input TSV → JSON conversion
        ↓
[Stage 1] Label Interpretation (GPT-5-mini)
        ↓
Structured semantic fields
        ↓
Candidate retrieval from ontology vector DBs
        ↓
[Stage 2] Ontology ID selection (GPT-4o-mini / similar)
        ↓
Final annotation output (CSV + JSON logs)
```

---

## **Ontology Configuration**

The system uses a **reduced ontology index**, stored locally:

| Ontology | Description            |
| -------- | ---------------------- |
| CLO      | Cell Line Ontology     |
| CL       | Cell Ontology          |
| UBERON   | Anatomical structures  |
| BTO      | Brenda Tissue Ontology |

Each ontology is loaded from a persistent ChromaDB directory:

```python
db_metadata = {
    "CLO": {"persist_directory": "..."},
    ...
}
```

---

## **Input Format**

A TSV file with the following columns:

```
Label | CLO | CL | UBERON | BTO | Type
```

Where:

* `Label`: raw biological sample label
* `Type`:

  * `CL` → Cell Line
  * `CT` → Cell Type
  * `A` → Anatomical structure

---

## **Execution**

Run the script from the command line:

```bash
python rag_gpt_PC_index.py \
  -o <output_folder> \
  -dp <dataset_path> \
  -m <model_name>
```

### **Arguments**

| Argument                 | Description                   |
| ------------------------ | ----------------------------- |
| `-o` / `--output_folder` | Directory to store outputs    |
| `-dp` / `--dataset_path` | Path to input TSV file        |
| `-m` / `--model_name`    | LLM used for annotation stage |

---

## **Outputs**

The script generates multiple outputs:

### **1. Final Annotations**


[all_gpt_5_mini_inference_index_ids.csv](https://github.com/tecnomod-um/annotationRAG/blob/main/RAG_process/RAG_reduced_info_results/all_gpt_5_mini_inference_index_ids.csv)


Contains:

* Label
* Type
* Selected ontology IDs (CLO, CL, UBERON, BTO)

---

### **2. Label Interpretations**

[label_inferences_gpt5_mini.json](https://github.com/tecnomod-um/annotationRAG/blob/main/RAG_process/RAG_reduced_info_results/label_inferences_gpt5_mini.json)


Includes:

* Label (Input label from the user)
* Extracted semantic fields (```Cell_Line, Cell_Type, Anatomical_Structure```)
* Additional inferred context (```Additional_Info```)


## **Retrieval Mechanism**

* Uses **ChromaDB vector stores**
* Embeddings are created (if not already present) via:

```python
execute_if_not_exists(...)
```

* Candidate extraction:

```python
extract_iris_and_descriptions(query, persist_directory)
```

This returns:

* Candidate IRIs
* Candidate descriptions

---

## **Prompting Strategy**

[Prompts](https://github.com/tecnomod-um/annotationRAG/tree/main/RAG_process/scripts/prompts)
 are stored externally:

```
prompts/
│── search_interpretation.txt
│── IRIsearch_instructions_restrictions.txt
│── system_prompt.txt
```

This allows:

* Easy experimentation
* Reproducibility
* Prompt tuning without modifying code

--- 
## **Notes**

* Requires a `.env` file with:

```
OPENAI_API_KEY=your_key_here
```

* Ensure ontology vector databases exist or can be generated.
