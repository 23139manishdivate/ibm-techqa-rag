# Experiment History

A chronological record of all experiments conducted during development.

---

## Phase 1 — Baseline Retrieval (450 Questions)

**Date:** ~2026-10-05  
**Notebook section:** Cells 1–40

### Setup
- Corpus: training-dev technote subset (subset of full 802k corpus)
- Retrieval: BGE-M3 dense only + BM25 full-text
- Fusion: Simple RRF (two signals)
- Evaluation questions: 450 (mix of answerable + unanswerable from training set)

### Results

| Recall@k | Score |
|----------|-------|
| @5  | 57.33% |
| @10 | 66.89% |
| @20 | 74.44% |
| @50 | 81.56% |
| @100| 86.00% |

### Notes
- BM25 tokeniser had a bug: whitespace not stripped correctly → over-counted token frequencies
- Title field not used at this stage

---

## Phase 2 — Three-Signal RRF (160 Answerable Questions)

**Date:** ~2026-10-08  
**Notebook section:** Cells 40–90

### Changes from Phase 1
1. **Fixed BM25 tokeniser** — corrected whitespace stripping
2. **Added title-only BM25** as a third retrieval signal
3. Restricted evaluation to **160 answerable** dev questions for cleaner signal
4. RRF k kept at 60

### Results

| Recall@k | Hits | Total | Score |
|----------|------|-------|-------|
| @5  | 102 | 160 | 63.75% |
| @10 | 115 | 160 | 71.88% |
| @20 | 133 | 160 | 83.12% |
| @50 | 141 | 160 | 88.12% |
| @100| 146 | 160 | 91.25% |

### Notes
- Three-signal RRF improved Recall@5 by +6.4 pp over Phase 1
- Title signal was surprisingly helpful for exact-match doc lookup
- Max-chunk-score aggregation outperformed mean aggregation

---

## Phase 3 — Context Selection Experiments

**Date:** ~2026-10-09  
**Notebook section:** Cells 90–130

### Setup
- Used Phase-2 fused rankings as input
- Varied k (top-k chunks passed to LLM): {3, 5, 10, 20}
- Recorded context length and GPU memory usage

### Notes
- k=5 was a good balance: context fits within Qwen 1.5B context window
- k=10 required context truncation on the 1.5B model
- Results saved in `results/context_selection/dev_all_questions.csv`

---

## Phase 4 — LLM Answer Generation

**Date:** ~2026-10-10 – 2026-10-12  
**Notebook section:** Cells 130–200

### Models
| Model | Params | Quantisation | Device |
|-------|--------|--------------|--------|
| Qwen/Qwen1.5-1.8B | 1.8B | FP16 | Kaggle T4 |
| Qwen/Qwen1.5-7B-Chat-GPTQ-Int4 | 7B | 4-bit | Kaggle T4 |

### Task
Extractive QA — given retrieved context + question, output the verbatim answer span.

### Results (from `techqa_phase4_evaluated.csv`)
- Phase-4 answers generated for all 310 dev questions
- Evaluation run with official `evaluation.py` scorer

### Notes
- Qwen 7B 4-bit produced noticeably better answers than 1.5B on long-context questions
- Some sessions required saving progress mid-run (`techqa_phase4_answers_progress.csv`) due to Kaggle session timeouts
- Unanswerable detection (predicting '-' for N questions) remains a known weak point

---

## Failed / Aborted Experiments

| Experiment | Reason Abandoned |
|------------|-----------------|
| BM25 on full 802k corpus | Kaggle OOM — corpus too large to index in RAM |
| Sentence-level chunking | Lower recall than paragraph-level |
| MiniLM cross-encoder reranking | Not implemented; deferred to next phase |
| Fine-tuning on 600 training examples | Deferred — needs stronger GPU |
