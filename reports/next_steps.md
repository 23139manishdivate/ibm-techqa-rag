# Next Steps & Roadmap

Based on the current state of the project (Phase 4 complete, Recall@100 = 91.25% on 160 dev questions).

---

## Priority 1 — Modularise the Notebook into `src/`

The entire pipeline currently lives in `notebooks/original/__notebook_source__.ipynb`. Before adding new features, refactor into clean Python modules:

```
src/techqa_rag/
├── data/          load_qa.py, load_corpus.py, chunking.py
├── retrieval/     dense.py, bm25.py, fusion.py
├── reranking/     cross_encoder.py
├── context_selection/  selector.py
├── generation/    generator.py, prompts.py
├── evaluation/    metrics.py  (wrap evaluation.py)
└── utils/         config.py, logging.py
```

---

## Priority 2 — Cross-Encoder Reranking

Add a **ms-marco-MiniLM-L-6-v2** cross-encoder between retrieval and context selection:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(question, chunk) for chunk in top_k_chunks])
```

Expected improvement: +3–8 pp Recall@5.

---

## Priority 3 — Fine-tune on 600 Training Examples

Use the 600 labelled training instances to fine-tune either:
- **Retriever:** Contrastive fine-tuning of BGE-M3 (positive = answer doc, negatives = other DOC_IDs)
- **Reader (LLM):** SFT on (question, context, answer_span) triples using QLoRA

---

## Priority 4 — Evaluate Larger LLMs

On the college GPU (likely ≥ 16 GB VRAM), evaluate:
| Model | Expected VRAM | Notes |
|-------|---------------|-------|
| Qwen1.5-7B FP16 | ~14 GB | Better quality than 4-bit |
| Qwen1.5-14B Int4 | ~8 GB | Higher capacity |
| Llama-3-8B Instruct | ~16 GB | Alternative architecture |

---

## Priority 5 — Unanswerable Detection

Currently the system always outputs an answer span. Add a confidence threshold:
- Predict '-' when max retrieval score < threshold
- Train a simple classifier on (question, retrieved_context) → {answerable, unanswerable}

---

## Priority 6 — BM25 on Full 802k Corpus

Phase 1/2 only indexed the training-dev subset (~28k docs). Index the full 802k corpus:
- Requires ~32 GB RAM to build BM25 index in memory
- Or use an on-disk BM25 implementation (e.g., Elasticsearch, BM25S)

---

## Priority 7 — Hybrid Chunking Strategy

Experiment with different chunking strategies:
- Sentence-level with semantic similarity grouping
- Section-aware splitting using `full_technote_collection.sections.json`
- Sliding window with 50% overlap

---

## Priority 8 — End-to-End Evaluation on Test Set

Submit to the official TechQA leaderboard (if still active) or run internal evaluation on the held-out validation set using the official `evaluation.py` script.
