# Bio-Oracle: Neuro-Symbolic Agentic AI for High-Content Screening

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Apple Silicon](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)-orange)](https://developer.apple.com/metal/pytorch/)
[![Framework](https://img.shields.io/badge/Agent-PydanticAI-purple)](https://github.com/pydantic/pydantic-ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Container-Arm64-blue)](https://www.docker.com/)

**Bio-Oracle** is a **Neuro-Symbolic Agent** designed to automate phenotype discovery in high-throughput microscopy.

It addresses the core challenge of modern screening: **Scaling cellular analysis without losing biological context**. By combining Deep Learning perception with Agentic reasoning, Bio-Oracle automates the interpretation of complex cellular assays.

![Results Summary](assets/results_summary.png)
*Automated "Reasoning" Dashboard: (1) Raw Image Ingestion -> (2) Neural Perception -> (3) Symbolic Outlier Detection.*

---

## **The Neuro-Symbolic "Moat"**

Unlike standard pipelines that output raw CSVs, Bio-Oracle acts as a reasoning engine:

1. **Neural Perception (Vision)**: Utilizes **Cellpose** to segment cells in dense, noisy images where traditional watershed algorithms fail.
2. **Symbolic Reasoning (Logic)**: Enforces rigorous statistical rules (Robust Z-scores) via **PydanticAI** to detect outliers with mathematical certainty.
3. **Agentic Workflow**: A **Gemini 2.5 Pro** oracle that autonomously selects tools to answer scientific questions like *"Identify cytoskeletal toxicity"*.

---

## **Key Capabilities**

### **1. 🚀 High-Performance Vision**

* **Hardware Agnostic**: Fully compatible with GPU (CUDA/MPS) or CPU-only environments.
* **Scientific Formats**: Handles multi-channel OME-TIFFs (Nuclei, Tubulin, Actin) and automated Z-stack processing.

| Task | Device | Time (s) |
| :--- | :--- | :--- |
| **Segmentation (224 cells)** | MacBook Pro (MPS) | **~2.5s** |
| **Segmentation (224 cells)** | CPU | ~15.0s |

### **2. 🧪 Scientific Rigor & Validation**

* **Ingestion**: Verifiable data loading and metadata preservation using `AICSImageIO`.
*   **Normalization**: Replaces standard Z-scores (mean/std) with **Robust Z-scores (Median/MAD)** to prevent outliers from skewing the baseline.
* **Validation**: Benchmarked using the **BBBC021 human MCF-7 drug-screen dataset**, specifically verifying phenotypic shifts in Taxol-treated samples.
* **Efficiency**: Core modules are written in **pure Python** to ensure the pipeline remains portable and lightweight across diverse research clusters.

### **3. 🧠 Transparent Reasoning & Observability**

The Agent provides a full **Chain of Thought** trace for every conclusion.

> **Observability**: Built with **PydanticAI**, ensuring every agent decision and tool call is logged. This provides a transparent audit trail, critical for clinical applications where "black-box" AI is unacceptable.

---

## **Architecture**

```mermaid

graph LR

A[Microscopy Image] -->|Ingestion| B(Vision Engine)

B -->|Cellpose/MPS| C[Mask Generation]

C -->|Quantification| D[Feature Extraction]

D -->|Median/MAD| E[Robust Normalization]

E --> F[Parquet Database]

F --> G{Oracle Agent}

G -->|Tools: Outlier Detection| H[Scientific Insight]

```

---

## **Installation & Usage**

### **Quick Start**

1. **Clone & Setup**:
```bash
git clone https://github.com/HarshShroff/Bio-Oracle.git
cd Bio-Oracle
./setup_env.sh
source .venv/bin/activate
```

2. **Data Preparation (BBBC021)**:
```bash
python scripts/data_fetcher.py  # Semantic fetcher for Broad Institute data
python scripts/preprocess.py    # Standardize to OME-TIFF
python scripts/generate_visuals.py # Generate Results Dashboard
```

3. **Consult the Oracle**:
```bash
export GEMINI_API_KEY="your_key"
python -m src.main --ask "Are there any outliers in the Actin channel (Ch2)?"
```

---

## **Future Expansion**

To further bridge the gap between AI and Biology, the following modules are planned:

1.  **PubMed RAG Integration**: Retrieve mechanism of action (MoA) data for identified outliers (e.g., *"Why does Taxol cause Actin polymerization?"*).
2.  **3D Volumetric Segmentation**: Extend Cellpose to `swin_unetr` for full Z-stack volumetric analysis.
3.  **Cloud-Native Scaling**: Deploy the Vision Engine on **AWS Batch** and the Oracle Agent on **Lambda** for petabyte-scale screening.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.