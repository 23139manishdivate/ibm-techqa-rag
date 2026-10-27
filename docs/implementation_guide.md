# IBM TechQA RAG: Step-by-Step Implementation Guide

This is an implementation specification for building the next version of the project in notebook code cells. It deliberately describes **what each step must do and how to verify it**, without prescribing implementation code. Follow the steps in order. Do not add a reranker, graph, new embedding model, fine-tuning, or a full-corpus rebuild unless a later experiment in this guide provides evidence for it.

## 0. Goal and operating modes

Build and evaluate one pipeline with two candidate-document modes:

1. **TechQA benchmark mode:** For each supplied benchmark question, use that question's `DOC_IDS` as the allowed candidate documents. Fetch their content from the technote corpus, then rank passages within them. Treat `DOC_IDS` only as candidate documents, never as answer labels or a way to identify the gold document.
2. **Open-corpus mode:** For a new user question with no `DOC_IDS`, search the full available corpus to find candidate documents, then use the same passage ranking, context selection, and answer generation stages.

Report results for the two modes separately. Benchmark mode measures passage selection and answering when the benchmark candidate pool is supplied. Open-corpus mode measures document discovery as well as answering. Never present benchmark-mode results as proof of open-corpus retrieval quality.

The development question set is for iteration. Do not train on development answers, offsets, or gold-document identity. Keep a record of every configuration and output so runs can be compared.

## 1. Project paths and known data constraints

The repository is `ibm-techqa-rag`. Its config files currently point to data beneath `ibm-techqa-rag/data`, but the audit found that the large corpus and some generated artifacts are actually in the sibling `TechQA/TechQA` directory. Before building anything, resolve every configured path and write a path inventory showing whether it exists, its size, and which dataset split it contains. Do not silently assume the README's locations are correct.

Inputs expected by the audited setup:

- `ibm-techqa-rag/data/processed/training_and_dev/training_Q_A.json`
- `ibm-techqa-rag/data/processed/training_and_dev/dev_Q_A.json`
- Training/dev technote corpus, presently found under the sibling `TechQA/TechQA` tree rather than the configured repository data path.
- Existing chunks, retrieval rankings, contexts, and experiment outputs, where present, under the sibling `TechQA/TechQA` tree and `ibm-techqa-rag/results`.
- `ibm-techqa-rag/src/techqa_rag/evaluation/evaluation.py` is the supplied official-style span evaluator. Read its exact required input schema before using it.

The complete technote corpus is very large and its sections JSON has nested structure. Do not load the entire file into memory or rebuild its index as part of the initial diagnostic work. Start with the 28,482-document training/dev corpus that supports the existing experiments. If the corpus file is missing, stop that stage and clearly report the missing path; do not substitute the answer-bearing `DOCUMENT` field from QA records as corpus content.

## 2. Non-negotiable data rules

- The canonical document text for answer spans is the dataset's `text` field. TechQA answer offsets refer to this text. Do not replace it with HTML-stripped `content`.
- Keep the original `content`, `text`, `title`, document ID, metadata, and source ID available in the normalized document record. `content` is diagnostic/raw HTML; differences between it and `text` are not automatically corruption.
- Preserve each source document's text exactly for offsets. Search normalization may create a separate derived field, but it must never overwrite canonical text or offsets.
- If extracting HTML keyword lists, keep them as a separate field or retrieval-only text with an explicit provenance label. Do not silently append keywords to answer passages or use them as evidence in generated answers.
- Never use `ANSWER`, `DOCUMENT`, answer offsets, or gold document identity as retrieval inputs at evaluation time. They are labels for scoring/oracle diagnostics only.
- Preserve unanswerable questions. For generation and scoring, the expected no-answer convention must match the evaluator; do not drop these rows from the evaluation report.

## 3. Phase A — Reproducible audit and baseline capture

### Step A1: Inventory inputs and outputs

Create a machine-readable inventory with resolved absolute paths, existence, file size, row/document count when practical, and a role (QA, corpus, chunks, index, results, config). Include the source notebook and configs. Detect duplicate copies of files and explicitly choose one source of truth per input.

**Checkpoint:** All inputs used in a run are listed and can be loaded from their resolved paths. No path is inferred from a stale README.

### Step A2: Validate QA and corpus schemas

Load train and dev records and validate required fields, types, null/empty rates, unique question IDs, and `DOC_IDS` structure. Validate corpus document IDs, title, text, raw content, and metadata. Check that dictionary key, document `id`, and `metadata.sourceDocumentId` agree wherever present. Report missing candidate IDs and candidates absent from the corpus.

For the dev set, report answerable and unanswerable counts; empty answers; whether the gold document is in `DOC_IDS`; whether answer strings occur in canonical gold `text`; and whether supplied offsets reproduce the answer exactly. Use the official format and document any known edge cases rather than silently correcting labels.

**Checkpoint:** Produce a validation report before any retrieval. A schema or ID mismatch must be fixed or explicitly isolated before proceeding.

### Step A3: Freeze baseline artifacts

Do not overwrite the old chunk file, index, rankings, or CSVs. Copy or reference them as a frozen baseline, record hashes/paths, and record the model/config, corpus version, chunk version, candidate-pool mode, context budget, prompt, and code revision for each result. If an artifact's provenance is unknown, label it “legacy, provenance incomplete.”

Recompute baseline metrics only where necessary for an apples-to-apples comparison. Record document Recall@5/10/20/50/100, answer-containing chunk recall at the same cutoffs where possible, selected-context answer containment, official span F1/EM, and custom token F1 as a separately named diagnostic. Never call retrieval recall “RAG accuracy.”

**Checkpoint:** A baseline report can be regenerated from saved inputs and identifies exactly which pipeline produced each metric. Do not compare train and dev values as if they were a controlled experiment.

## 4. Phase B — Repair document and chunk representation

### Step B1: Build a canonical document table

For each technote, create one canonical record with:

- `document_id` (stable source ID)
- `title`
- exact canonical `text`
- original HTML `content` retained separately if available
- metadata/source ID
- any extracted keyword terms stored separately and marked as keywords
- a source-document hash/version

Do not strip hidden keyword elements out of canonical `text` if they are present there. The dataset's detagged `text` is the offset reference. You may make a cleaned retrieval field for search, while always retaining exact canonical text.

### Step B2: Parse sections conservatively

Build a parser that recognizes section headings by explicit known heading patterns (including `QUESTION`, `ANSWER`, `SYMPTOM`, `CAUSE`, `RESOLVING THE PROBLEM`, `PROBLEM SUMMARY`, `LOCAL FIX`, `TEMPORARY FIX`, `ERROR DESCRIPTION`, `ENVIRONMENT`, and other observed headings). Headings vary and may be inline, punctuated, or absent; therefore:

- Treat a heading as a section boundary only when the line/markup shape supports it, not merely because a phrase appears in prose.
- Keep unrecognized text in an `UNSECTIONED` section; never discard it.
- Store the literal heading, normalized heading, and its source character start/end.
- Keep the heading with the section content and with every chunk derived from that section.
- Preserve source order, including repeated headings.

If parsing confidence is low, do not force a section label. Retain document-level text as the fallback evidence.

**Checkpoint:** On a sampled and programmatically counted set, every source character belongs to exactly one section span, apart from explicitly documented separator handling. A re-joined section text must reconstruct canonical text exactly (or explain a measured, intentional separator transformation).

### Step B3: Replace ambiguous chunk ordering metadata

Generate new chunks with a new versioned output name. Every chunk must contain:

- `chunk_id` globally unique and deterministic
- `document_id`
- `section_id` and `section_order` (0-based order in the document)
- `chunk_index_in_section` (0-based within that section)
- `chunk_index` (0-based global order in the document)
- `source_char_start` and `source_char_end` offsets into canonical `text`
- `text` copied from that exact canonical span
- section heading and section type
- token count plus the tokenizer/version used to count it
- source-document hash and chunker version

The old `chunk_index` reset inside sections, so duplicates such as `(document_id, chunk_index)` could not be sorted into a document sequence. The new global `chunk_index` must be unique per document and must agree with ascending source character offsets. Never use the old duplicate index to reconstruct sequence.

### Step B4: Use answer-preserving, structure-aware chunking

Do not use a fixed rule that splits every 600-token window across the document. Use this deterministic procedure:

1. Split each parsed section into paragraphs using source offsets.
2. Keep short adjacent paragraphs together while the combined text fits a configurable token target.
3. Split an individual overlong paragraph into sentence-aware pieces; if a sentence itself exceeds the target, use a token window with overlap.
4. Do not cross a section boundary just to fill a chunk.
5. Preserve a small configurable overlap only when a long paragraph/window was split. Do not add overlap between ordinary adjacent paragraph chunks by default.
6. Keep heading text in the chunk text or a reliably prepended heading, and retain its source offset mapping separately.
7. Ensure each answerable dev reference span is fully contained in at least one generated chunk where its gold document is available. If source answer text exists but no chunk contains it, record a chunking failure with question ID, document ID, and answer span.

Treat the token target and overlap as parameters, not settled truths. Start with a moderate target (for example, 300 tokens) and a maximum around 500; run the preservation and retrieval diagnostics before selecting a final value. Compare at most a small controlled set, such as 200, 300, and 500 token targets, using identical questions, candidate pools, retriever, context budget, and scoring. Do not run a broad parameter sweep.

**Checkpoint:** Rebuild statistics: number of docs/chunks, empty chunks, token distribution, source coverage, per-document ordering validity, duplicate chunk IDs, and answer-span containment. All answer-span failures must be categorized (missing in raw source vs parsing loss vs chunk split/offset bug).

## 5. Phase C — Separate retrieval evaluation from answerability

### Step C1: Benchmark mode candidate construction

For each QA row, map its `DOC_IDS` to canonical corpus documents. Deduplicate IDs while preserving their provided order for logging only. Candidate retrieval is restricted to these documents in benchmark mode. The retriever still ranks candidates from question text and document/chunk content; it must not receive the gold `DOCUMENT`, `ANSWER`, gold offsets, or a “gold first” ordering.

For questions whose candidate IDs are missing from the corpus, record the coverage failure and continue with the remaining candidates only under a clearly labeled policy. Report how many questions were affected. Do not silently fall back to global corpus search for only those rows.

### Step C2: Open-corpus mode candidate construction

For an unseen user query, there is no `DOC_IDS`. Search all available corpus documents/chunks using the existing three signals first: BGE-M3 dense retrieval, full-text BM25, and title BM25, fused with the existing RRF method. Keep the existing model and signals fixed during the first diagnostic comparison.

Use `DOC_IDS`-restricted benchmark results to measure the downstream passage/reader stages; use global retrieval results to measure realistic document discovery. Report them separately.

### Step C3: Establish the answer-bearing evidence funnel

For every answerable dev question whose gold span exists in raw canonical text, measure sequentially:

1. Gold document is available in its candidate pool.
2. Gold document appears in top-k retrieved documents (document recall).
3. At least one answer-containing chunk is in the retrieved chunk candidates (answer-bearing passage recall).
4. At least one answer-containing chunk is selected into final context (context answer containment).
5. Generated answer matches the reference under exact normalization and official span scoring.

Also report these rates for all applicable questions, not only their conditional percentages. Include counts and denominators. Break down failures as: candidate/doc discovery, chunk preservation, passage ranking, context budget/selection, or generation/scoring.

Use exact answer span containment based on source offsets for chunk/context evidence when offsets are valid. Also report normalized answer-string containment as a secondary measure for offset anomalies. An answer-string occurrence alone can be ambiguous; preserve this distinction.

**Checkpoint:** The counts must form an auditable funnel. For example, selected-context containment cannot exceed available answer-bearing chunks unless a different evidence matching rule is explicitly stated.

## 6. Phase D — Improve passage ranking and context selection without adding a reranker

### Step D1: Produce a broad, fixed candidate pool

For each question, collect the same fixed retrieval pool before any selector comparison (for example, top 100 chunks per retrieval signal or the corresponding stable candidate union). Deduplicate by `chunk_id`, retain source metadata and each raw signal rank/score, and save this candidate table. In benchmark mode, never include chunks outside the provided `DOC_IDS`.

### Step D2: Compare simple selectors under an equal budget

Compare only these selectors first:

- Existing hybrid chunk score and per-document cap, reconstructed faithfully where possible.
- Highest dense similarity.
- Highest BM25 rank/score.
- A simple lexical answer-evidence heuristic as a diagnostic only (must not use reference answer; may use query-term overlap and section type, e.g. ANSWER/RESOLVING THE PROBLEM).
- A section-aware selector that prefers likely solution sections but can still select other sections when query evidence is stronger.

Use exactly the same candidate pool, prompts, model, and maximum context-token budget in all comparisons. A fixed number of chunks is not an equal context budget because chunk sizes vary. Remove exact duplicate/near-duplicate chunks before filling budget. Preserve selected order by relevance, but give the generator heading and document title for each passage.

Record answer-bearing context containment, gold-document inclusion, context token count, redundancy, official span metrics, and runtime. Do not select using gold answer labels.

**Checkpoint:** Keep a new selector only if it improves the predeclared end-to-end measure (official span F1/EM or the agreed answer metric) without unacceptable answer-context loss or cost. Improvements to document recall alone do not qualify it as a successful context selector.

### Step D3: Only then run a controlled reranker test

The current saved cross-encoder comparison is not conclusive because the context budgets differed and the model/candidate setup may not be matched; the BGE reranker-v2 pilot is too small to establish a result. Do not declare “rerankers never work” or add one to the default pipeline.

After the simple-selector baseline is fixed, evaluate one reranker at a time on the exact same saved candidates and equal token budget. Measure whether it moves answer-bearing chunks upward and whether the final answer metrics improve. Use development labels only for evaluation, not fine-tuning or threshold tuning. If testing a fine-tuned reranker later, train on training questions only and keep dev untouched.

**Reject the reranker** if it fails to improve answer-bearing passage recall/context containment and end-to-end answer metrics consistently, or if gains are smaller than run-to-run uncertainty while adding significant cost. Do not optimize Recall@5 alone.

## 7. Phase E — Oracle-context experiment and reader diagnosis

### Step E1: Fix the reader contract

Use the existing Qwen2.5-1.5B-Instruct setup for the first controlled reader experiment because it is the model used by the audited generation notebook. Freeze its exact model revision, tokenizer, prompt, decoding parameters, maximum input tokens, and output parsing. Keep the prompt extractive: answer only from supplied evidence, quote the shortest complete answer span, and emit the defined no-answer marker when absent. Save raw model output and the exact final prompt/context for every question.

Do not compare the legacy Qwen 7B result unless its prompt, model revision, context construction, quantization, and evaluator are recovered and matched. Treat it as an unverified historical result.

### Step E2: Run three paired context conditions

Run the same answerable dev questions through the same reader and prompt with:

1. **Oracle answer passage:** A fixed window around a valid gold answer span from canonical gold document text. The window must include the full reference span and enough surrounding text for meaning. Log this is an oracle diagnostic unavailable at inference.
2. **Oracle full gold document:** Full canonical gold document, truncated only by the same reader input limit; record truncation. This isolates document-length/context issues from discovery.
3. **Retrieved selected context:** The actual context produced by the current fixed pipeline.

Keep prompt template, model, decoding, normalization, and evaluation identical. Include a fourth no-context control only if useful to measure model prior/hallucination behavior. Save all per-question predictions and context metadata.

Score with the supplied official evaluator in its required format, and separately report normalized exact match and token F1. Report mean and median, answerable/unanswerable strata, counts, and confidence intervals or paired bootstrap intervals where feasible. Do not use EM alone because TechQA answers may permit span overlap or formatting variation.

Interpretation:

- If oracle answer passage is strong but retrieved context is weak, prioritize retrieval/passage/context stages.
- If oracle answer passage is weak, improve the reader/prompt or answer extraction before expensive retrieval changes.
- If full document is worse than answer passage, input length/truncation or distraction may be harming the reader.
- These results diagnose the current reader; they do not establish leaderboard performance by themselves.

The earlier saved Qwen2.5-1.5B oracle experiment had answer-passage token F1 around 0.31 and only 3/160 exact outputs despite the reference span being present. This is a reason to replicate it carefully and inspect per-question failures, not a reason to immediately switch models.

## 8. Phase F — Generation and official scoring

### Step F1: Preserve answers as evidence spans

For every generated row save question ID, candidate mode, retrieved document IDs and ranks, selected chunk IDs and offsets, exact context text, context token count, prompt version, model/revision, raw completion, parsed answer, and runtime. This lets failures be traced to the exact evidence supplied.

### Step F2: Validate answer extraction

Do not rely only on free-form string comparison. Determine whether the official evaluator expects a predicted document ID plus start/end character offsets (verify from `evaluation.py`). If so, map the generated answer back to the selected canonical passage and produce offsets only when the answer can be located unambiguously. Record unlocatable predictions as a separate failure. Do not fabricate offsets or use gold offsets to build predictions.

Run the official scorer with the exact required JSON schema. Also report custom token F1 and normalized exact match in clearly separate columns. Validate scorer behavior on a few synthetic cases (exact answer, partial span, no answer) before scoring dev; these checks validate evaluation plumbing and must not be counted as model results.

Evaluate all dev questions, including unanswerable ones. Report answerable answer quality, unanswerable false-answer rate, and overall official metric using the evaluator's definition.

**Checkpoint:** A final result is not “official” unless the supplied evaluator actually ran successfully on the generated predictions. Save its command/config, stdout/diagnostics, and output.

## 9. Phase G — Open-corpus user-query validation

Once benchmark mode is stable, evaluate global retrieval separately. Use the same dev questions but ignore their `DOC_IDS` during retrieval; use gold labels only afterward to score whether the gold document/passages were found. This gives an open-corpus simulation without leaking candidate IDs into retrieval. It is still a benchmark simulation, so label it accordingly.

For actual new user questions, the system flow is:

1. Accept the question; candidate IDs are absent.
2. Search the entire available corpus with dense and lexical/title retrieval.
3. Fuse and retain a broad candidate set.
4. Rank answer-bearing passages and fill a fixed token budget.
5. Generate an evidence-grounded answer and return citations to technote title/ID and source passage.
6. Abstain when no adequate evidence is found, using a threshold calibrated on training data or a separate validation set, never tuned on final test labels.

Do not claim open-corpus capability if the full corpus index has not been built. The existing 28,482-doc subset supports a subset-mode prototype only. Expanding to the full 802k corpus is a later infrastructure step after the small-corpus pipeline is validated.

## 10. Phase H — Complexity decisions

Keep the initial system small and measurable:

- **Retain for baseline:** canonical `text`, exact offsets, BGE-M3 dense retrieval, full-text BM25, title BM25, RRF, deterministic saved candidates, and the existing reader for controlled diagnosis.
- **Modify:** section parser and chunk metadata; context selection to use answer-aware evidence diagnostics and equal token budgets; generation output to preserve traceability; evaluation to run the official scorer plus separate stage metrics.
- **Defer:** cross-encoder reranker, fine-tuned reranker, graph retrieval, embedding fine-tuning, new embedding models, full 802k index, and larger LLM comparisons until a controlled result identifies the stage they could fix.
- **Remove from active path only after confirming no dependency:** duplicate/legacy rankings, stale indexes, ambiguous old chunk ordering, and unproven context-selection variants. Archive instead of overwriting so results remain auditable.

## 11. Experiment protocol and result table

For every run record: run ID, date, code revision, data/corpus hash, chunker version, candidate mode, retrieval config, selector, model and prompt revision, token budget, random seed/decoding, runtime/hardware, and output paths.

Predeclare a primary end-to-end metric before comparing variants (prefer official span F1; report official EM and unanswerable false-answer rate too). Use paired per-question comparisons on the same dev examples. Include counts and denominators. A metric is not an improvement if it rises only because the evaluated question subset changed.

Recommended result table columns:

| Run | Candidate mode | Variant | Doc R@20 | Answer-passage R@20 | Context contains answer | Official span F1 | Official EM | No-answer false-answer rate | Avg context tokens | Runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Save per-question rows as well as aggregate metrics. Preserve all prior outputs; never reuse a filename for a different method.

## 12. Implementation order and stop/go gates

1. Resolve paths, validate schemas/IDs, freeze old results.
2. Build canonical documents and answer-offset integrity report.
3. Implement source-offset sections and globally ordered chunks.
4. Prove reconstruction, order uniqueness, and answer-span preservation.
5. Build the per-question answer-evidence funnel using saved or newly produced candidates.
6. Compare simple selectors under identical candidate pools and token budgets.
7. Replicate paired oracle-passage/full-document/current-context reader evaluation.
8. Run the official evaluator on saved generated outputs and fix prediction-to-span conversion.
9. Only if evidence points to ranking, run one controlled reranker comparison.
10. Only after benchmark-mode diagnosis, evaluate global retrieval and plan full-corpus indexing.

Stop and repair the earlier stage if its checkpoint fails. Do not proceed to model shopping while answer spans are being dropped, chunk ordering is ambiguous, or evaluation inputs are inconsistent.

## 13. Instructions for a coding assistant generating notebook cells

When this guide is given to a coding assistant, it should:

- Generate notebook cells in the order above, with a short markdown heading and purpose before each logical code block.
- Reuse verified project paths and inspect actual schemas rather than inventing columns or paths.
- Keep code modular enough to rerun each phase, but do not refactor the entire project into a package as a prerequisite.
- Make each data-changing step write to a new versioned output path; never overwrite raw data or frozen baseline artifacts.
- Include assertions and concise diagnostic tables at the specified checkpoints.
- Keep development answers/offsets strictly inside evaluation and oracle-only diagnostics; never pass them to normal retrieval, selection, or generation.
- Use the same questions, candidates, prompts, model settings, and token budgets for paired comparisons.
- Ask for missing files or credentials only if a required stage cannot proceed; otherwise report the limitation and continue with independent diagnostics.
- Avoid adding techniques or changing model families unless the guide's stop/go evidence calls for that experiment.

The expected deliverable from the coding assistant is notebook code cells and concise explanations of each cell, not a redesign of this plan.
