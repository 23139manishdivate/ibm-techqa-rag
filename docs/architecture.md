# Pipeline Architecture

## Overview

The TechQA RAG system is a multi-stage pipeline that takes a technical support question and extracts a verbatim answer span from IBM technote documents.

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT QUESTION                           │
│   QUESTION_TITLE + QUESTION_BODY  +  candidate DOC_IDS           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Preprocessing  │
                    │  (tokenise,     │
                    │   clean text)   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌───────▼───────┐  ┌──────▼──────┐
   │  Dense      │   │  Full-text    │  │  Title-only │
   │  Retrieval  │   │  BM25         │  │  BM25       │
   │  BGE-M3     │   │  (corrected   │  │  (title     │
   │  FAISS      │   │   tokeniser)  │  │   field)    │
   │  151k chunks│   │               │  │             │
   └──────┬──────┘   └───────┬───────┘  └──────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │   RRF Fusion    │
                    │   (k = 60)      │
                    │   Three signals │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Chunk → Doc    │
                    │  Aggregation    │
                    │  (max-score per │
                    │   document)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Context        │
                    │  Selection      │
                    │  (top-k chunks) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LLM Generation │
                    │  Qwen 1.5B /    │
                    │  Qwen 7B 4-bit  │
                    │  (extractive)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Evaluation     │
                    │  TechQA scorer  │
                    │  (F1, EM,       │
                    │   Recall@k)     │
                    └─────────────────┘
```

---

## Component Details

### 1. Preprocessing
- Concatenate `QUESTION_TITLE` and `QUESTION_BODY` into a single query string
- Lowercase + basic normalisation for BM25
- BGE-M3 tokeniser for dense retrieval

### 2. Dense Retrieval (BGE-M3 + FAISS)
- **Model:** `BAAI/bge-m3` (1024-dim embeddings)
- **Index:** Flat IP (inner product) FAISS index
- **Corpus:** 151,140 chunks from the training-dev technote subset
- Chunk size: empirically tuned; overlapping windows
- Query embeddings shape: `(n_queries, 1024)`

### 3. BM25 Retrieval
Two BM25 variants run in parallel:
- **Full-text BM25** — indexes `text` field of each chunk; uses corrected whitespace tokenisation (bug fix applied in Phase 2)
- **Title-only BM25** — indexes the `title` field; adds a fast, low-noise signal

### 4. Reciprocal Rank Fusion (RRF)
```
score(d) = Σ  1 / (k + rank_i(d))    for i in {dense, bm25_full, bm25_title}
```
- `k = 60` (standard setting)
- Combines all three ranked lists into a single fused ranking
- Chunk-level scores are then aggregated to document level via `max(chunk_scores)`

### 5. Context Selection
- Select top-k chunks from fused ranking
- Concatenate chunk text as context for the LLM

### 6. LLM Generation
- **Models tried:** Qwen/Qwen1.5-1.8B, Qwen/Qwen1.5-7B-Chat-GPTQ-Int4
- **Task:** extractive span prediction — find the verbatim answer substring in the provided context
- 4-bit quantisation (bitsandbytes) used for the 7B model on Kaggle T4

### 7. Evaluation
- Official TechQA evaluation script (`evaluation.py`)
- **Retrieval metrics:** Recall@k (k ∈ {5, 10, 20, 50, 100})
- **Generation metrics:** Token F1, Exact Match (EM)
