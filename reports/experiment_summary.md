# Experiment Summary

## Retrieval Performance

### Phase 1 — Dual-Signal RRF (BGE-M3 + Full-text BM25)
Evaluated on **450** questions from the training set.

| Recall@k | Score |
|----------|-------|
| @5  | 57.33% |
| @10 | 66.89% |
| @20 | 74.44% |
| @50 | 81.56% |
| @100| 86.00% |

### Phase 2 — Three-Signal RRF (BGE-M3 + Fixed BM25 + Title BM25)
Evaluated on **160 answerable** dev questions.

| Recall@k | Hits / 160 | Score |
|----------|-----------|-------|
| @5  | 102 | 63.75% |
| @10 | 115 | 71.88% |
| @20 | 133 | 83.12% |
| @50 | 141 | 88.12% |
| @100| 146 | 91.25% |

**Improvement from Phase 1 → Phase 2:**

| Metric | Phase 1 | Phase 2 | Δ |
|--------|---------|---------|---|
| Recall@5  | 57.33% | 63.75% | +6.42 pp |
| Recall@10 | 66.89% | 71.88% | +4.99 pp |
| Recall@20 | 74.44% | 83.12% | +8.68 pp |
| Recall@50 | 81.56% | 88.12% | +6.56 pp |
| Recall@100| 86.00% | 91.25% | +5.25 pp |

---

## Generation (Phase 4)

- **Models:** Qwen1.5-1.8B (FP16), Qwen1.5-7B-Chat-GPTQ-Int4
- **Input:** Top-5 fused-ranked chunks per question
- **Task:** Extractive span prediction
- **Output files:**  
  - `results/generation/techqa_phase4_answers.csv` — final generated answers  
  - `results/evaluation/techqa_phase4_evaluated.csv` — scored against ground truth

---

## Corpus Statistics

| Statistic | Value |
|-----------|-------|
| Total technote corpus | 801,998 documents |
| Training+dev subset | 28,482 documents |
| Total chunks indexed | 151,140 |
| Embedding dimensions | 1,024 (BGE-M3) |
| FAISS index size | 590 MB |

---

## Key Decisions & Trade-offs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | BGE-M3 | Strong multilingual dense retrieval baseline |
| FAISS index type | Flat IP | Exact search; corpus small enough to fit in RAM |
| BM25 implementation | rank-bm25 (BM25Okapi) | Simple, no external server required |
| Chunk aggregation | max-score per doc | Outperformed mean in Phase 2 ablation |
| LLM | Qwen 7B 4-bit | Best quality within Kaggle T4 VRAM budget |
| RRF k | 60 | Standard default from the original RRF paper |
