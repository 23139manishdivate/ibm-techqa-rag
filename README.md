# IBM TechQA RAG System

> A complete Retrieval-Augmented Generation pipeline built on the [IBM TechQA dataset](https://github.com/IBM/techqa) — designed for technical support question answering over IBM technote documents.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: CDLA-Permissive-1.0](https://img.shields.io/badge/Data_License-CDLA--Permissive--1.0-green)](https://cdla.dev/permissive-1-0/)

---

## 📋 Project Overview

This project implements a multi-stage RAG system that answers IBM technical support questions by retrieving relevant passages from a corpus of ~802,000 IBM technotes. The pipeline was developed on Kaggle and achieves strong retrieval recall on the TechQA development set.

### Key Results

| Metric | Phase 1 (450 Qs) | Phase 2 — Three-Signal RRF (160 Qs) |
|--------|-------------------|--------------------------------------|
| Recall@5  | 57.33% | 63.75% |
| Recall@10 | 66.89% | 71.88% |
| Recall@20 | 74.44% | 83.12% |
| Recall@50 | 81.56% | 88.12% |
| Recall@100| 86.00% | 91.25% |

---

## 🗂️ Repository Structure

```
ibm-techqa-rag/
├── configs/                    # YAML config files for each pipeline stage
├── notebooks/
│   ├── original/               # Original Kaggle notebook (all phases)
│   └── analysis/               # Post-hoc analysis notebooks
├── src/techqa_rag/             # Modular Python source code
│   ├── data/                   # Data loading & preprocessing
│   ├── retrieval/              # BM25 + dense retrieval + RRF fusion
│   ├── reranking/              # Cross-encoder reranking
│   ├── context_selection/      # Passage selection for the LLM
│   ├── generation/             # LLM inference (answer generation)
│   ├── evaluation/             # Scoring & metrics
│   └── utils/                  # Shared helpers
├── data/
│   ├── raw/technote_corpus/    # License PDF (full corpus: local only)
│   └── processed/              # Q&A splits + validation files
├── artifacts/                  # Indexes, embeddings, rankings (local only)
├── results/                    # CSV/JSON evaluation outputs
├── reports/                    # Experiment summaries & analysis
├── backups/kaggle/             # ZIP snapshots from Kaggle sessions
└── docs/                       # Architecture, dataset, and setup docs
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ajinkya252005/ibm-techqa-rag.git
cd ibm-techqa-rag
```

### 2. Create the environment

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate techqa-rag
```

**Using pip:**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Restore large artefacts

The FAISS index, embeddings, and ranking files are **not tracked by Git** (> 100 MiB). Transfer them from your local machine and place them in the paths listed in [`artifacts/ARTIFACT_REGISTRY.md`](artifacts/ARTIFACT_REGISTRY.md).

### 4. Open the notebook
```bash
jupyter lab notebooks/original/__notebook_source__.ipynb
```

---

## 🧠 Pipeline Architecture

```
Question
   │
   ▼
Preprocessing  (tokenise, clean)
   │
   ├──► Dense retrieval     (BGE-M3, FAISS, 151 140 chunks)
   ├──► Full-text BM25      (BM25Okapi, corrected tokenisation)
   └──► Title-only BM25     (lightweight title-aware signal)
              │
              ▼
        RRF Fusion          (k=60, three signals)
              │
              ▼
        Chunk → Doc         (max-score aggregation per doc)
              │
              ▼
        Context Selection   (top-k passages)
              │
              ▼
        LLM Generation      (Qwen 1.5B / 7B 4-bit, extractive)
              │
              ▼
        Evaluation          (TechQA official scorer)
```

---

## 📊 Dataset

The **TechQA dataset** (Castelli et al., 2020) consists of:
- **600** labeled training Q&A pairs
- **310** labeled development Q&A pairs
- **282** validation questions (held-out labels)
- A technote corpus of **801,998** IBM support documents

**License:** [CDLA-Permissive v1.0](data/raw/technote_corpus/CDLA-Permissive-v1.0.pdf)  
**Source:** https://github.com/IBM/techqa

---

## 📁 Large Files (Local Only)

The following files are excluded from Git due to size. See [`artifacts/ARTIFACT_REGISTRY.md`](artifacts/ARTIFACT_REGISTRY.md) for details:

| File | Size | Notes |
|------|------|-------|
| `techqa_faiss.index` | 590 MB | Dense retrieval index (BGE-M3, 151k vectors) |
| `embeddings_backup.zip` | 469 MB | Raw embedding matrix backup |
| `embeddings_final.zip` | 76 MB | Final normalised embeddings |
| `chunks_df.pkl` / `chunks_df_final.pkl` | ~104 MB each | Chunked document store |
| `full_technote_collection.sections.json` | ~6 GB | Full 802k-doc corpus |
| `training_dev_technotes.json` | ~278 MB | Training subset corpus |

---

## 📝 Experiment History

See [`docs/experiment_history.md`](docs/experiment_history.md) for a chronological log of all experiments run during development.

---

## 🔮 Next Steps

See [`reports/next_steps.md`](reports/next_steps.md) for the full roadmap. Key priorities:

1. Modularise notebook code into `src/techqa_rag/`
2. Add cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
3. Fine-tune on TechQA training set (600 labelled examples)
4. Evaluate Qwen 7B vs 14B on the GPU at college
5. Implement confidence calibration for unanswerable detection

---

## 🤝 Citation

```bibtex
@inproceedings{castelli2020techqa,
  title={The TechQA Dataset},
  author={Castelli, Vittorio and Chakravarti, Rishav and Dana, Saswati and Ferritto, Anthony
          and Florian, Radu and Hassid, Michael and Jiang, Nan and Kurtz, Ranit and Lally, Adam
          and McCawley, Sean and Murdock, James and Nachman, Liat and Roukos, Salim
          and Ward, Todd and Zhang, Rong},
  booktitle={ACL},
  year={2020}
}
```

---

## 📄 License

- **Code:** MIT License  
- **Dataset:** [CDLA-Permissive v1.0](https://cdla.dev/permissive-1-0/)
