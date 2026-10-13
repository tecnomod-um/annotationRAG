# **error_analysis — Manual Error Inspection**

This directory contains the tools and results for performing **fine-grained manual error analysis** of the RAG-based annotation pipeline.

Unlike the quantitative evaluation in `evaluation_scripts/`, this module focuses on:

* Understanding **why errors occur**
* Distinguishing between **interpretation vs annotation failures**
* Providing **human-validated insights** into model behavior

---

## **Overview**

The error analysis is based on a **manual classification workflow**, where each prediction is inspected and categorized into:

* **Interpretation Error (`I`)**
  The model incorrectly understood the input label.

* **Annotation / Mapping Error (`A`)**
  The interpretation is correct, but the wrong ontology identifier was selected.

* **No Error**
  Either:

  * The prediction matches the reference
  * Or no annotation is applicable (`-`)

---

## **Main Script**

### **`manual_mapping_error_check.py`**

### **Purpose**

Provides an **interactive interface** to manually classify errors for a given:

* Biological entity type (`CL`, `CT`, `A`)
* Ontology (`CLO`, `CL`, `UBERON`, `BTO`)

---

## **How It Works**

### **1. Input Dataset**

The script expects a CSV file containing:

* `Sample_ID`
* `Label`
* `Type`
* Pairs of columns per ontology:

  * `{ONTOLOGY}_C` → Correct (reference) identifier
  * `{ONTOLOGY}_M` → Model-predicted identifier

---

### **2. Automatic Classification**

The script automatically labels cases as **"No Error"** when:

* Both values are `-`
* Both identifiers are identical

```text
Example:
CLO_C = CLO:0001234
CLO_M = CLO:0001234 → No Error
```

---

### **3. Manual Classification**

For all other cases, the user is prompted:

```text
→ Classification (I/A):
```

Where:

* `I` → Interpretation Error
* `A` → Annotation Error
* `Enter` → Skip (optional)

---

### **4. Interactive Output Example**

```text
Sample_ID: S012
Label: Fetal_Kidney
BTO_C: BTO:0000671
BTO_M: BTO:0001234

→ Classification (I/A):
```

---

### **5. Summary Generation**

At the end, the script automatically computes a summary:

```text
Error_Type        Count
-----------------------
I                 12
A                 8
No Error          72
```

This summary is appended to the output file.

---

## **Usage**

```bash
python manual_mapping_error_check.py
```

---

## **Output**

The script generates a CSV file for each concept type and ontology:

```text
A_BTO_manual_classification.csv
```

### **Columns**

| Column              | Description          |
| ------------------- | -------------------- |
| Sample_ID           | Unique identifier    |
| Label               | Original label       |
| Type                | Biological type      |
| ONTOLOGY_C          | Reference identifier |
| ONTOLOGY_M          | Model prediction     |
| User_Classification | I / A / No Error     |

---

## **Design Highlights**

* Human-in-the-loop evaluation
* Clear separation of error types
* Automatic handling of trivial cases
* Built-in summary generation

---

## **Notes**

* The process is **interactive** → requires user input
* Recommended to use a **representative sample** (e.g., from `k_candidates/`)
* Ensure consistent column naming (`*_C`, `*_M`)

