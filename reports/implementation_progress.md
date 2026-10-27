# IBM TechQA RAG implementation progress

Updated: 2026-10-20

## Phase A metric reconciliation

The current local QA files, technote corpus, and legacy `chunks_df.pkl` were inspected directly. Counts below use the exact rules stated in the metric name; they should replace ambiguous mixed-rule statements in the Phase A handover.

| Check | Recomputed result | Definition |
|---|---:|---|
| Answerable dev rows | 160 | `ANSWERABLE == "Y"` |
| Non-empty reference answers | 158 | Whitespace-only answer strings excluded |
| Raw exact reference string in canonical gold text | 149/158 | Case-sensitive substring search |
| Raw exact offset slice equals reference | 149/160 | `text[START_OFFSET:END_OFFSET] == ANSWER`; empty-reference rows count as not matching |
| Normalized reference string in canonical gold text | 151/158 | Whitespace collapsed, then case-folded |
| Normalized reference string in legacy gold-document chunks | 144/158 | Same normalization; chunks from `chunks_df.pkl` only |

The 149/160 raw offset-match count includes two empty-reference questions as failures. Among non-empty references, 149/158 exact offset slices match. Nine non-empty questions fail both raw reference-string occurrence and raw offset equality; two additional references are empty. Normalized matching finds two further formatting/case-equivalent occurrences. This reconciles the 149/160 table with the 11 flagged cases; older 147/158 figures remain unreconciled and should not be repeated without recovering their exact matching code.

The legacy chunk file has no source character offsets. A reliable offset-span containment metric cannot be derived merely by sorting its ambiguous section-local `chunk_index` values. The `144/158` normalized string-in-chunks result is a distinct measure and does not establish that the original answer span is fully represented.

Historical selected-context results remain separate by artifact/selector. Values such as 53, 55, and 63 answer-containing contexts are not one comparable baseline until their exact context artifact, matching rule, and selector are named. No single value has been chosen as definitive.

## Phase B1 — canonical documents

Implemented in `src/techqa_rag/data/build_canonical_corpus.py` and run against:

`../TechQA/TechQA/training_and_dev/training_dev_technotes.json`

Outputs:

- `data/processed/v2/canonical_technotes_v1.jsonl`
- `data/processed/v2/canonical_technotes_v1_manifest.json`

Checkpoint results: 28,482 source documents became 28,482 records; zero invalid documents, duplicate document IDs, source-key/ID mismatches, metadata source-ID mismatches, missing required fields, or missing metadata. All 12,909 hidden keyword lists are stored separately. Canonical `text` and raw HTML `content` remain separate. A row-by-row comparison against the input verified exact equality of `text`, `content`, `title`, and metadata for all 28,482 records.

## Phase B2 — conservative section parsing

Implemented in `src/techqa_rag/data/parse_sections.py` and run against the B1 JSONL.

Outputs:

- `data/processed/v2/technote_sections_v1.jsonl`
- `data/processed/v2/technote_sections_v1_manifest.json`

The parser only treats an explicitly listed heading on its own line as a boundary. Unrecognized/prefix text is retained as `UNSECTIONED`. The output has 134,138 sections across 28,482 documents and covers 101,222,947 canonical text characters. A separate row-by-row reconstruction check found zero documents whose ordered section text differed from the canonical source text.

The heading dictionary is intentionally conservative and may be expanded only after inspecting missed heading patterns; no text was discarded to achieve these counts.

## Phase B3/B4 — updated chunk build and local verification completed

`src/techqa_rag/data/build_chunks_patched.py` implements section-aware paragraph grouping, sentence splitting for long paragraphs, tokenizer-offset windows for overlong sentences, unique deterministic chunk IDs, global document order, section-local order, exact source character offsets, and a dev answer-span preservation manifest. It skips zero-token spans, uses a tokenizer safety margin and hard assertion for the configured maximum, refuses to overwrite outputs, and reports both single-chunk and contiguous-chunk-union answer-span coverage.

The updated script was run on Kaggle and the resulting v2 corpus and manifest were copied into this workspace. The local manifest reports:

| Check | Result |
|---|---:|
| Documents processed | 28,482 |
| Chunks created | 194,929 |
| Empty chunk ranges skipped | 0 |
| Chunk token count | min 1; max 500 |
| Answerable dev rows checked | 160 |
| Rows where offset slice exactly matches reference | 149 |
| Offset spans contained in one chunk | 152/160 |
| Offset spans covered by contiguous chunk union | 160/160 |
| Exact reference spans contained in one chunk | 142 |
| Exact reference spans covered by contiguous chunk union | 149 |
| Normalized reference present in gold-document chunks | 144/158 |

An independent local scan verified all 194,929 chunk IDs are unique, chunk offsets point to exact canonical source substrings, and every chunk is within the configured token maximum. Across 1,701 documents, the chunker leaves 12,201 source characters uncovered; all uncovered characters are whitespace, so this is not substantive answer text loss. These checks establish that the chunk unions cover all supplied offset spans. Seven exact answer spans are split across adjacent chunks; one-chunk containment is therefore not the right preservation criterion for those rows. The 144/158 string-in-chunks result is a separate diagnostic and should not be conflated with offset-span coverage.

## Phase B5 — adjacent evidence selection implemented and evaluated locally

`src/techqa_rag/context_selection/adjacent_chunks.py` adds a context selector that takes ranked retrieval seeds, expands each seed to nearby chunks from the same document and section, stitches touching/overlapping source spans without repeating overlap text, and counts the rendered source labels against a caller-provided generation tokenizer budget. Each selected chunk records whether it was retrieved or added as an adjacent expansion.

The module passed Python compilation and a synthetic overlap smoke check. A real-dev selector experiment was then run on the same fixed top-20 document rankings with a 4,096-token prompt budget. The candidate pool contains the gold document and supplied answer span for 133/160 questions (83.1%). BM25 passage ranking selected the gold document for 97/160 questions; its plain top-five chunks contained the offset span in 47/160 and the normalized answer string in 44/160. Radius-1 adjacent expansion increased those counts to 75/160 and 70/160, respectively, with mean prompt size rising from about 1,126 to 1,912 tokens.

These are context evidence metrics, not generated-answer accuracy. They show the selector can recover useful neighboring evidence, but do not prove that a generator will use it correctly.

## Phase C1 — BGE-M3 passage-ranking comparison completed

The Kaggle embedding archive was extracted to `artifacts/embeddings/v2_candidate_documents_top20/`. Its manifest reports 2,830 candidate documents and 18,351 chunks. The 18,351 x 1,024 float32 matrix is finite and normalized; its ordered chunk IDs exactly match the locally reconstructed candidate order. The evaluation cache reader was updated to accept validated float16 or float32 caches irrespective of which device created them, avoiding an unnecessary local re-embedding pass.

The BGE-M3 passage ranker was evaluated against BM25 on exactly the same 160 questions, fixed top-20 document candidate pools, context budget, and radius-1 selector. Candidate-pool answer-span coverage stayed 133/160 for both. BGE-M3 selected the gold document on 97/160 questions, the same as BM25. BGE-M3's plain top-five context contained the offset span on 41/160 and normalized reference on 37/160; radius-1 expansion raised those to 68/160 and 63/160. BM25 scored higher at 47/160 and 44/160 plain, and 75/160 and 70/160 after expansion. Mean BGE-M3 prompt size was about 1,137 tokens plain and 2,018 tokens expanded; neither condition hit the 4,096-token budget.

On paired questions, BGE-M3's expanded context recovered the normalized reference in 15 questions where BM25 did not; BM25 did so in 22 where BGE-M3 did not. This pilot provides no evidence that BGE-M3 passage ranking improves this selector over the simpler BM25 ranker. Neighbor expansion improved span/reference containment for both rankers, but generated answer quality has not yet been measured on these actual selected contexts.

## Phase C2 — BM25 context generation comparison completed

`src/techqa_rag/evaluation/evaluate_v2_context_generation.py` generated answers from the saved BM25 top-five contexts and the same contexts expanded by one neighboring chunk on each side. The experiment used the same Qwen2.5-1.5B-Instruct reader, prompt, greedy decoding, and 180-token output cap for both conditions. Each of the 160 answerable development questions was evaluated in both conditions (320 answers total); there were no generation errors or prompt truncations. Two empty references were excluded from answer-reference metrics.

| Metric | Plain top-five | Radius-1 expansion |
|---|---:|---:|
| Mean token F1 vs. reference | 0.2073 | 0.2319 |
| Median token F1 | 0.1977 | 0.2142 |
| Normalized exact match | 0/158 | 0/158 |
| Mean prompt tokens | 1,126 | 1,912 |
| Offset answer span present in context | 47/160 | 75/160 |
| Normalized reference string present in context | 45/160 | 71/160 |

The paired mean token-F1 change was +0.0247 for expansion (89 wins, 3 ties, 66 losses; 158 nonempty-reference questions). A simple 20,000-resample paired bootstrap gives a 95% interval of approximately +0.0118 to +0.0379. This suggests an improvement on this development set under the chosen reference-overlap metric, but it is not a human correctness assessment. Exact match was zero in both conditions, and expansion used about 70% more prompt tokens. The reference string was present in expanded context for 26 questions where it was absent from plain context; those questions gained +0.054 mean token F1 on average. Expansion also helped many questions where neither context contained the reference string, so the gain is not explained solely by exact string inclusion.

Interpretation: adjacency expansion is a promising, simple context-selection improvement for this fixed candidate pool and reader, but the gain is modest and does not establish reliable answer correctness. Keep it as an evaluated option, not yet as a proven production default. The absence of exact matches and relatively low F1 indicate substantial remaining answer-generation/evidence-alignment headroom. The 133/160 candidate-pool ceiling also means passage selection cannot recover evidence outside the fixed top-20 document pool.

Results are saved in:

- `results/generation/v2_bm25_context_generation.jsonl` (320 condition records)
- `results/evaluation/v2_bm25_context_generation.csv` (latest record per question/condition)
- `results/evaluation/v2_bm25_context_generation_summary.json` (overall, evidence-stratified, and paired metrics)

## Phase C3 — oracle-context reader diagnostic completed

`src/techqa_rag/evaluation/evaluate_v2_oracle_contexts.py` supplied the same Qwen2.5-1.5B-Instruct reader with (1) a 512-token-window passage around each development answer annotation and (2) the full gold technote, truncated from the beginning only when needed to fit the same 4,096-token prompt limit. The prompt, greedy decoding, and 180-token output cap matched Phase C2. The oracle run produced 320 answers for 160 questions with no generation errors; 158 nonempty references were scored. These conditions use gold labels and are diagnostic only.

| Metric | Oracle answer passage | Full gold technote | Retrieved top-five + neighbors |
|---|---:|---:|---:|
| Mean token F1 | 0.3110 | 0.2989 | 0.2319 |
| Median token F1 | 0.2976 | 0.2737 | 0.2142 |
| Normalized exact match | 0/158 | 1/158 | 0/158 |
| Mean prompt tokens | 591 | 1,149 | 1,912 |
| Reference present in context | 153/158 | 150/158 | 71/158 |
| Full document truncated | 0 | 10/160 | 0 |

The answer passage exceeded the expanded retrieved context by +0.0791 mean F1 (paired bootstrap 95% interval +0.0565 to +0.1031; passage won 105 questions, tied 1, and lost 52). This is direct evidence that the current evidence discovery/selection path leaves recoverable answer quality on the table. However, the oracle reader still averaged only 0.311 F1 and produced no normalized exact matches. This leaves a separate reader, prompt, answer-formatting, or metric-alignment problem after evidence is supplied. Exact-match results also differ from a prior historical oracle result (reported as 3/160); treat that old value as non-comparable until its exact generation and scoring setup is recovered.

The oracle passage exceeded the full gold document by only +0.0121 mean F1; its paired bootstrap interval crossed zero (−0.0018 to +0.0255; 63 passage wins, 50 ties, 45 full-document wins). Thus these results do not establish that passage-only context is generally better than a full note. They do show that dumping more text is not a reliable fix: 10 full notes were truncated, and full-note F1 remained lower on average. For the oracle passage, 153 of 158 nonempty references were found in the rendered context, yet mean F1 was about 0.311; reference presence alone did not make the small reader reproduce the answer.

The best next step is a focused reader/output audit, not a new retrieval model: compare the saved generated answer against the reference and supplied evidence for a stratified sample of cases (high/low F1; answer present/absent; expansion win/loss). Check whether answers are factually supported but paraphrased, overly verbose, omit identifiers, or genuinely wrong. Then test one extractive prompt change on this fixed evidence set and score it with the same metrics. Preserve this oracle run as a diagnostic ceiling, never as deployable performance.

Results are saved in:

- `results/generation/v2_oracle_context_generation.jsonl` (320 rows with reconstructed exact contexts)
- `results/evaluation/v2_oracle_context_generation.csv`
- `results/evaluation/v2_oracle_context_generation_summary.json` (overall, evidence-stratified, and paired comparisons)
- `src/techqa_rag/evaluation/evaluate_v2_oracle_contexts.py`

## Phase C4 — stratified reader audit and prompt pilots

`results/evaluation/v2_reader_audit_sample.md` records 12 paired examples sampled from cases where only expansion contains the reference, both contexts contain it, or neither does. The examples show that the single-reference F1 is imperfect and that evidence quality varies by question:

- DEV_Q071: adjacent expansion surfaces the exact IP configuration instruction and materially improves the answer.
- DEV_Q253: neither retrieved context contains the relevant IBM Content Collector fact; retrieval answers incorrectly, while the oracle passage answers correctly.
- DEV_Q007: the generated gateway version answers the user's question plausibly, but the supplied reference is a release-availability sentence, so lexical F1 undervalues the answer.
- DEV_Q288: retrieved contexts and the oracle reader disagree on Java 8 support; the expansion answer aligns better with the supplied reference. This is a genuine reader/evidence-conflict case requiring source-level review.
- DEV_Q235: the question asks where to find a bulletin, while the reference contains long CVE details; a URL-focused generated answer can be useful yet score poorly against that reference.

These examples confirm three distinct sources of low scores: missing or misleading retrieved evidence, reader mistakes even with relevant evidence, and reference answers that do not directly match the question's requested form. Automated F1 alone cannot label an answer factually correct or incorrect.

A second audit sampled oracle passages with low F1 but a present reference string, high-F1 controls, and references absent from the oracle context. This exposed annotation alignment issues beyond simple string preservation. DEV_Q240 asks about a `RecordChangeSystem.msl` startup error, but its designated gold note concerns a different ITX Launcher defect; the reference “Obtain patch for APAR PI64816” occurs in that note but does not answer the question. DEV_Q048 asks for IIS performance counters, while its gold note/reference describe installing the IIS 6.0 management compatibility pack to fix missing website data. These are likely question/reference/gold-note mismatches, despite exact answer text being present in the gold note. DEV_Q258 is a reader failure: the gold passage explicitly says “Unknown” indicates baselining has not started or has not been calculated, but Qwen invents unrelated causes. By contrast, DEV_Q188's low F1 answer gives the IBM guide link requested, while the reference contains only the PDF filename; that appears semantically useful despite low lexical overlap. The high-F1 SPSS examples are also paraphrases of the same vendor-support instruction. This small audit is not a corpus-wide mismatch rate, but it proves that “reference string in gold note” is not sufficient evidence that the annotation is semantically aligned.

Three prompt variants were piloted on the same 12 stratified questions, with the same Qwen model and the same adjacent-retrieved and oracle-passage contexts. The first demanded the shortest exact span and an explicit `NO_ANSWER` marker; it collapsed to mean F1 0.009 on retrieved contexts and 0.065 on oracle passages, often abstaining when answer text was present. It is rejected. A milder “focused answer, use source wording” variant scored 0.283 on retrieved contexts and 0.394 on oracle passages, versus 0.307 and 0.381 for the existing prompt on those same questions. The retrieved result was slightly worse; the oracle gain was too small and sample too small to justify a full-set change. A third exact-quotation prompt scored 0.107 on retrieved contexts and 0.250 on oracle passages; only 3/12 completions in either condition were exact normalized substrings of their supplied contexts. No prompt variant is adopted.

Pilot outputs are retained in `results/generation/v2_extractive_prompt_pilot*.jsonl` and their summaries in `results/evaluation/v2_extractive_prompt_pilot*_summary.json`.

## Phase C5 — official span-mapping diagnostic

`src/techqa_rag/evaluation/score_v2_generated_spans.py` conservatively maps a generated answer only when its complete normalized text occurs in both the exact selected context and a candidate document's canonical source text. It never uses the gold document or answer offsets to choose retrieved predictions. On the existing free-form Qwen outputs, 0/160 expanded retrieved answers and 0/160 oracle-passage answers mapped to source spans. The official scorer therefore returned 0 HasAns F1 for each answerable-subset file because every prediction was unlocatable and encoded as no answer. This zero is a diagnostic of the output-to-span mapping failure, not a valid estimate of semantic answer quality or a full-dev official result.

The exact-quote prompt pilot also produced a source-contained completion in only 3/12 cases per condition, so simply instructing this reader to quote is not sufficient. The current model and prompt contract do not produce the document IDs and offsets required by the benchmark evaluator.

Adapter artifacts are in `results/evaluation/official_span_v2/`, including prediction JSON, the answerable-subset input, official evaluator output, and per-question mapping CSVs.

## Next action

Before further model work, quantify the label issue with a controlled adjudication set: use the saved 24 examples to separate (a) question/reference/gold-note mismatch, (b) semantically correct but paraphrased answers, and (c) genuinely unsupported or incorrect answers. Expand that audit only if mismatch is common. Then define the product output contract: free-form supported answer plus citation, or benchmark-compatible source span. Do not optimize the current single-reference F1 or run another full prompt/model experiment until that target and label audit are settled.
