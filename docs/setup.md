# Setup Guide

## Prerequisites

- Python 3.11
- Conda (recommended) or pip + venv
- CUDA 12.1+ and an NVIDIA GPU with ≥ 8 GB VRAM (for Qwen 7B 4-bit generation)

---

## 1. Clone the Repository

```bash
git clone https://github.com/ajinkya252005/ibm-techqa-rag.git
cd ibm-techqa-rag
```

---

## 2. Set Up the Python Environment

### Option A — Conda (recommended for GPU machines)

```bash
conda env create -f environment.yml
conda activate techqa-rag
```

### Option B — pip + venv (CPU or quick setup)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** For GPU support, replace `faiss-cpu` in `requirements.txt` with `faiss-gpu`.

---

## 3. Transfer Large Artefacts

The following files are not tracked by Git. Copy them from your local machine (e.g., via USB, network share, or cloud storage) to the specified paths:

| File to copy | Destination path in repo |
|-------------|--------------------------|
| `techqa_faiss.index` | `artifacts/indexes/techqa_faiss.index` |
| `embeddings_final.zip` | `artifacts/embeddings/embeddings_final.zip` |
| `embeddings_backup.zip` | `artifacts/embeddings/embeddings_backup.zip` |
| `chunks_df_final.pkl` | `artifacts/chunks/chunks_df_final.pkl` |
| `chunks_df.pkl` | `artifacts/chunks/chunks_df.pkl` |
| `dev_fused_rankings.pkl` | `artifacts/rankings/dev_fused_rankings.pkl` |

For the full corpus files (optional, only needed to re-index):
| File | Destination |
|------|-------------|
| `full_technote_collection.sections.json` | `data/raw/technote_corpus/` |
| `full_technote_collection.txt.bz2` | `data/raw/technote_corpus/` |
| `technotes.id_title_text` | `data/raw/technote_corpus/` |

---

## 4. Open the Notebook

```bash
jupyter lab notebooks/original/__notebook_source__.ipynb
```

The notebook contains all phases (1–4). Use the table of contents to navigate to the phase you want to continue from.

---

## 5. GPU-Specific Notes (College Machine)

When running on a machine with a stronger GPU:

1. **Use `faiss-gpu`** instead of `faiss-cpu`:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

2. **Increase batch size** in the embedding generation cells (look for `BATCH_SIZE = ...`)

3. **Try Qwen 7B in FP16** instead of 4-bit for better quality:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen1.5-7B-Chat",
       torch_dtype=torch.float16,
       device_map="auto"
   )
   ```

4. **Consider Qwen 14B or 72B** if VRAM allows (≥ 24 GB for 14B in FP16)

---

## 6. Verify Installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import faiss; print('FAISS OK')"
```
