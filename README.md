# IBM TechQA Retrieval-Augmented QA

A learning-focused RAG project for answering technical support questions from IBM technotes. The work covers dataset auditing, source-aligned preprocessing and chunking, hybrid retrieval, passage selection, answer generation, and multi-stage evaluation.

> **Project status:** The V2 experiments and analysis are complete for this learning phase. This repository is not a deployed service, and development-set scores below are not a general-purpose accuracy estimate or a state-of-the-art claim.

## At a glance

- **Knowledge base used in V2:** 28,482 training/development IBM technotes.
- **V2 chunks:** 194,929 source-aligned chunks, with stable IDs and character offsets.
- **Evaluation set:** 160 answerable development questions; 158 have non-empty reference answers.
- **Document retrieval:** the designated gold document was in the fixed top-20 candidate pool for 133/160 questions (83.1%). This is document recall, **not** end-to-end answer accuracy.
- **Main context-selection result:** BM25 passage selection plus adjacent-chunk expansion increased annotated answer-span coverage from 47/160 to 75/160 in the saved selector comparison. Expansion also increased average Qwen prompt size from about 1,126 to 1,912 tokens.
- **Generators evaluated:** Qwen2.5-1.5B-Instruct and Gemini 3.5 Flash Lite, using saved plain and expanded contexts.

## What the project does

```text
IBM technote corpus + TechQA questions
                 │
                 ├── Canonical text and metadata audit
                 ├── Conservative section parsing
                 └── Source-aligned V2 chunking (IDs + character offsets)
                              │
          Existing hybrid retrieval artifacts / fixed top-20 document candidates
                              │
             Passage ranking within candidate documents (BM25 or BGE-M3)
                              │
                 Select context, optionally add adjacent chunks
                              │
                    Generate with Qwen or Gemini
                              │
       Evaluate retrieval, evidence coverage, answer similarity, and support
```

The V2 passage-ranking and generation experiments use **saved, fixed document-level candidate rankings** to compare downstream passage selection and context assembly. They do not rerun full-corpus retrieval as part of each passage-selector experiment. In real use, a question has no gold `DOC_IDS`; a complete inference system must search the available corpus. TechQA gold document IDs and answer offsets are evaluation labels, not retrieval inputs for unseen questions.

## Current findings

### Retrieval and evidence selection

| Stage / measure | Result | Interpretation |
|---|---:|---|
| Gold document in fixed top-20 candidates | 133/160 (83.1%) | Candidate document recall on answerable development questions; not answer accuracy. |
| BM25 plain context contains annotated answer span | 47/160 | Passage/context evidence coverage in the saved selector comparison. |
| BM25 + adjacent expansion contains annotated answer span | 75/160 | Expansion recovered answer spans split across or near selected chunks. |
| Mean Qwen prompt tokens, plain → expanded | 1,126 → 1,912 | The evidence gain comes with about 70% more prompt tokens. |

The selector comparison reported normalized reference-answer presence as 44/160 plain and 70/160 expanded. The later saved context-generation/failure-attribution artifacts report 45/160 and 71/160 for their corresponding contexts. These one-question differences are retained as an artifact/version consistency caveat; do not combine the counts as if they came from one identical output file. The annotated source-span measure is reported separately from reference-string matching.

### Generated answers and evaluation

All values below compare saved plain-context answers with saved adjacent-expanded-context answers on the same development questions. Two empty references are excluded from reference-based metrics (n=158).

| Metric | Qwen plain | Qwen expanded | Gemini plain | Gemini expanded |
|---|---:|---:|---:|---:|
| Token F1 | 0.2073 | 0.2319 | 0.2676 | 0.3178 |
| ROUGE-L | 0.1517 | 0.1741 | 0.2158 | 0.2636 |
| BERTScore F1 (`roberta-large`) | 0.8378 | 0.8432 | 0.8473 | 0.8553 |
| Normalized exact match | 0/158 | 0/158 | 0/158 | 0/158 |

These are different evaluation signals, **not percentages of correct answers**: Token F1 and ROUGE-L measure lexical overlap, BERTScore estimates semantic similarity, and exact match requires a normalized full-string match. Expanded context improved the mean values of these metrics in these paired runs, but that alone does not establish factual correctness for every answer.

An automated evidence-support review decomposed answers into claims and checked whether the judge could find supporting evidence in the supplied context. The claim support rates were 61.21% → 74.51% for Qwen and 91.92% → 94.10% for Gemini (plain → expanded). These are claim-level judge results, not answer-level accuracy; Gemini 3.5 Flash Lite was also the judge for both generators, so judge bias and self-evaluation effects are limitations. A separate GPT-reviewed pilot covered only 10 selected questions and is illustrative, not a representative human evaluation.

### Reader diagnostic

With Qwen2.5-1.5B-Instruct and an annotated oracle answer passage, mean Token F1 was 0.3110, compared with about 0.2319 for retrieved expanded context. Oracle context uses the gold answer annotation and is **not available at inference**. This diagnostic indicates both evidence-selection headroom and remaining reader/output-format limitations; it is not a deployable system score.

## Data and evaluation caveats

- The original TechQA release describes a larger technote collection (about 802k documents); the V2 preprocessing and experiments documented here use a 28,482-document training/development subset. The full corpus and processed V2 corpus are not distributed by this repository.
- TechQA answer offsets refer to the source `text` field. Canonicalization retains that text and its metadata; raw HTML `content` is a separate representation and is not interchangeable with the offset source.
- The V2 chunker targets 300 tokens, caps chunks at 500 tokens, uses 50-token overlap only for long sentence windows, and records source offsets. Its audit found all 160 supplied answer offset spans covered by contiguous V2 chunk unions; 152 were contained in one chunk.
- Seven non-empty development references do not occur in the designated gold document after the recorded normalization, and two references are empty. This limits answer-string-based metrics.
- Retrieval recall, passage/span coverage, final-context evidence, answer similarity, and claim support are reported separately. Do not call Recall@20 or a text-overlap score “RAG accuracy.”
- Some saved result files include benchmark questions, passages, model answers, or evidence quotes. Check dataset terms and inspect outputs before sharing them publicly.

## Repository map

```text
src/techqa_rag/
├── data/                  # Canonical corpus, section parsing, and V2 chunking
├── context_selection/     # Adjacent-chunk expansion and passage rendering
└── evaluation/            # Context ablations, generation, BERTScore, factuality,
                           # oracle contexts, and source-span scoring
configs/                   # Data, retrieval, generation, and evaluation settings
notebooks/original/        # Original end-to-end Kaggle notebook
docs/                      # Dataset, setup, architecture, Gemini, and implementation guides
reports/                   # Experiment summaries and detailed comparison/study documents
artifacts/                  # Local indexes, embeddings, chunks, and saved retrieval rankings
data/                      # Raw and processed datasets (local; not all tracked)
results/                   # Generated answers and evaluation outputs (local artifacts)
```

Some earlier stages—especially full-corpus retrieval and embedding/index construction—are represented by the original notebook and saved artifacts. The newer Python modules focus on source-aligned corpus processing, context selection, generation, and evaluation. This is not yet a single turnkey CLI that rebuilds every stage from a clean clone.

## Setup

Use Python 3.11. From PowerShell at the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU generation, install a PyTorch/CUDA and FAISS combination appropriate to your machine; the pip requirements use `faiss-cpu`. The Conda file describes a CUDA-oriented environment and may need adjustment on CPU-only systems.

Get the TechQA dataset from [IBM/techqa](https://github.com/IBM/techqa) and place the files at paths required by the selected script. The corpus, generated answers, indexes, and ranking artifacts are deliberately not bundled in a clean clone. Several V2 scripts default to a sibling `TechQA/TechQA` dataset path; pass explicit paths using their command-line options when your files are elsewhere. Never commit API keys; Gemini reads an environment variable (`GEMINI_API_KEY` or `gemini_api_key`).

Set the source package path and inspect available options before running a stage:

```powershell
$env:PYTHONPATH = (Join-Path $PWD "src")
python .\src\techqa_rag\data\build_canonical_corpus.py --help
python .\src\techqa_rag\data\parse_sections.py --help
python .\src\techqa_rag\data\build_chunks_patched.py --help
python .\src\techqa_rag\evaluation\run_v2_context_ablation.py --help
python .\src\techqa_rag\evaluation\evaluate_v2_context_generation.py --help
```

The preprocessing scripts require explicit input/output paths and refuse to overwrite outputs. Generation and evaluation scripts consume saved V2 contexts, QA files, and/or prior generation artifacts; inspect each script's `--help` and input requirements before running. Gemini calls use a hosted API and can consume quota. The oracle evaluation uses development gold evidence and should only be treated as a diagnostic.

## Key implementation and result files

| Purpose | File |
|---|---|
| Canonical corpus audit | `src/techqa_rag/data/build_canonical_corpus.py` |
| Conservative section parser | `src/techqa_rag/data/parse_sections.py` |
| V2 source-aligned chunker | `src/techqa_rag/data/build_chunks_patched.py` |
| Adjacent-chunk context assembly | `src/techqa_rag/context_selection/adjacent_chunks.py` |
| Passage/context ablation | `src/techqa_rag/evaluation/run_v2_context_ablation.py` |
| Qwen/Gemini paired generation | `src/techqa_rag/evaluation/evaluate_v2_context_generation.py` |
| BERTScore | `src/techqa_rag/evaluation/evaluate_v2_bertscore.py` |
| Evidence-support judge | `src/techqa_rag/evaluation/evaluate_v2_factuality.py` |
| Oracle-context diagnostic | `src/techqa_rag/evaluation/evaluate_v2_oracle_contexts.py` |
| Detailed experiment chronology | `reports/implementation_progress.md` |
| Earlier implementation comparison | `reports/IBM_TechQA_Implementation_Comparison.docx` |
| Follow-up results and learning guide | `reports/IBM_TechQA_RAG_Progress_After_Comparison.docx` |
| Gemini generation setup | `docs/gemini_reader.md` |

## Prioritized future scope (not implemented)

The project is at a reasonable stopping point for its learning goal. If resumed, prefer one controlled experiment at a time:

1. **BM25 + BGE-M3 passage-level rank fusion (RRF):** test whether semantic and exact-term passage rankings complement one another inside the same candidate documents. Keep the candidate pool, context token budget, expansion policy, prompt, and generator fixed. This has not yet been tested in the V2 passage experiment.
2. **Structure-aware passage features:** use section headings such as `ANSWER` and `RESOLVING THE PROBLEM` as soft signals, while retaining unsectioned content.
3. **Reader and output audit:** compare answers using the same evidence; measure grounded correctness, identifiers, abstention, latency, and token cost—not only reference overlap.
4. **Reranker or GraphRAG only if justified:** require a reproducible, controlled improvement. A local reranker setup did not beat the hybrid baseline on its tested metric; a fine-tuned reranker and GraphRAG results are not reproducibly present in this checkout.

No future score is predicted. Keep changes only when they improve a predeclared answer/evidence metric at an acceptable cost on a controlled evaluation.

## Further reading

- [Dataset description](docs/dataset_description.md)
- [Architecture notes](docs/architecture.md)
- [Setup notes](docs/setup.md)
- [Experiment history](docs/experiment_history.md)
- [Detailed implementation guide](docs/implementation_guide.md)
- [Gemini reader guide](docs/gemini_reader.md)
- [IBM TechQA dataset](https://github.com/IBM/techqa)

## Citation

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

## License

- **Code:** `pyproject.toml` declares MIT; this checkout does not currently include a `LICENSE` file.
- **Dataset:** IBM TechQA dataset terms apply; obtain and use it according to its source license. Dataset files and generated benchmark content are not included in this README.
