# Artifact Registry

All large files that are **not tracked by Git**. Transfer manually between machines.

> Files excluded from Git via `.gitignore`. Do NOT commit these.

---

## Indexes

| File | Path | Size | Description | Regen Command |
|------|------|------|-------------|---------------|
| `techqa_faiss.index` | `artifacts/indexes/` | 590 MB | Flat IP FAISS index of 151,140 BGE-M3 chunks (1024-dim) | See notebook Phase 2, cell "Build FAISS Index" |

---

## Embeddings

| File | Path | Size | Description |
|------|------|------|-------------|
| `embeddings_final.zip` | `artifacts/embeddings/` | 76 MB | Final normalised BGE-M3 embeddings (numpy .npy inside zip) |
| `embeddings_backup.zip` | `artifacts/embeddings/` | 469 MB | Raw unnormalised embeddings backup |

---

## Chunks

| File | Path | Size | Description |
|------|------|------|-------------|
| `chunks_df_final.pkl` | `artifacts/chunks/` | ~104 MB | Final chunked DataFrame: chunk_id, doc_id, title, text, chunk_text |
| `chunks_df.pkl` | `artifacts/chunks/` | ~104 MB | Original chunked DataFrame (before final cleaning pass) |

---

## Rankings

| File | Path | Size | Description |
|------|------|------|-------------|
| `dev_fused_rankings.pkl` | `artifacts/rankings/` | 21.9 MB | Phase-2 RRF fused rankings for 160 dev questions |
| `dev_title_rankings.pkl` | `artifacts/rankings/` | 21.9 MB | Title-only BM25 rankings for 160 dev questions |
| `dev_semantic_rankings.pkl` | `artifacts/rankings/` | 2.9 MB | Dense BGE-M3 rankings for 160 dev questions |
| `dev_fulltext_rankings.pkl` | `artifacts/rankings/` | 3.0 MB | Full-text BM25 rankings for 160 dev questions |
| `fixed_bm25_rankings.pkl` | `artifacts/rankings/` | 7.9 MB | Phase-1 fixed BM25 rankings (450 Qs) |
| `max_doc_rankings.pkl` | `artifacts/rankings/` | 0.5 MB | Max-aggregated doc-level rankings |
| `semantic_bm25_title_rankings.pkl` | `artifacts/rankings/` | 12.6 MB | Three-signal combined (pre-RRF) |
| `semantic_rankings.pkl` | `artifacts/rankings/` | 7.6 MB | Phase-1 semantic-only rankings |
| `title_aware_rankings.pkl` | `artifacts/rankings/` | 0.3 MB | Title-aware weighted rankings |
| `title_only_rankings.pkl` | `artifacts/rankings/` | 0.4 MB | Title-only BM25 rankings (smaller subset) |

---

## Backups

| File | Path | Size | Description |
|------|------|------|-------------|
| `techqa_backup.zip` | `backups/kaggle/` | 1.1 MB | Early session snapshot |
| `techqa_transfer_bundle.zip` | `backups/kaggle/` | 2.4 MB | Transfer bundle between Kaggle sessions |
| `techqa_experiment_backup.zip` | `backups/kaggle/` | 15.1 MB | Phase-1 full experiment backup |
| `techqa_phase2_backup.zip` | `backups/kaggle/` | 25.3 MB | Phase-2 full experiment backup |
| `techqa_phase3_backup.zip` | `backups/kaggle/` | 12.3 MB | Phase-3/4 full experiment backup |

---

## Raw Dataset Files (Multi-GB, Local Only)

| File | Path | Size | Description |
|------|------|------|-------------|
| `full_technote_collection.sections.json` | `data/raw/technote_corpus/` | ~6 GB | Full 802k-doc corpus, section-split |
| `full_technote_collection.txt.bz2` | `data/raw/technote_corpus/` | ~795 MB | BZ2 compressed full corpus |
| `technotes.id_title_text` | `data/raw/technote_corpus/` | ~2.5 GB | Tab-separated: id, title, text |
| `technotes.text` | `data/raw/technote_corpus/` | ~2.2 GB | Plain text body only |
| `training_dev_technotes.json` | `data/processed/training_and_dev/` | ~278 MB | Training+dev subset corpus |
| `training_dev_technotes.sections.json` | `data/processed/training_and_dev/` | ~228 MB | Section-split version of above |
