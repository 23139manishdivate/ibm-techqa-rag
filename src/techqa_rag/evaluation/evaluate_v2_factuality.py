"""Judge whether V2 generated-answer claims are supported by their exact contexts.

The script joins saved answers to the exact plain/expanded context strings by
question ID, asks one fixed Gemini judge to assess both conditions separately,
and saves resumable raw judgments plus claim-, answer-, and aggregate-level
artifacts. It does not use reference answers in the judge prompt.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from techqa_rag.evaluation.gemini_backend import create_gemini_client


CONDITIONS = ("plain_top_chunks", "adjacent_expansion")
CONTEXT_FIELDS = {
    "plain_top_chunks": "plain_top_chunks_context",
    "adjacent_expansion": "adjacent_expansion_context",
}
VERDICTS = {"supported", "contradicted", "not_supported"}
JUDGE_MODEL = "gemini-3.5-flash-lite"
BOOTSTRAP_SEED = 20261026
BOOTSTRAP_SAMPLES = 10_000


def read_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid, condition = str(row.get("question_id", "")).strip(), str(row.get("condition", "")).strip()
            if not qid or condition not in CONDITIONS:
                raise ValueError(f"Invalid question_id/condition at {path}:{line_no}: {qid!r}/{condition!r}")
            key = (qid, condition)
            if key in rows:
                raise ValueError(f"Duplicate generation record: {key} in {path}")
            row["question_id"], row["condition"] = qid, condition
            row["generated_answer"] = str(row.get("generated_answer") or "")
            row["generation_error"] = str(row.get("generation_error") or "")
            rows[key] = row
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def normalize_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_contexts(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    contexts: dict[str, dict[str, Any]] = {}
    for row in data.get("questions", []):
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid in contexts:
            raise ValueError(f"Missing or duplicate question_id in context artifact: {qid!r}")
        contexts[qid] = row
    if not contexts:
        raise ValueError(f"No question contexts in {path}")
    return contexts


def build_prompt(question: str, cases: list[dict[str, str]]) -> str:
    payload = {"question": question, "cases": cases}
    return (
        "You are an evidence-grounding evaluator. Evaluate each case independently. "
        "For each answer, split it into the smallest meaningful factual claims. Do not use outside knowledge. "
        "Use only the context attached to that same case; do not use one case's context to judge another case. "
        "A claim is supported only if that context directly states it or clearly entails it. "
        "Use contradicted only when the context directly conflicts with the claim. "
        "Use not_supported when neither support nor direct contradiction is present. "
        "Do not treat a claim as supported just because it sounds plausible. Extract only factual claims "
        "about the technical subject matter. Ignore greetings, opinions, pure instructions, and statements "
        "that merely describe what the answer says or that the supplied evidence is insufficient. An answer "
        "that only abstains is response_type abstention with an empty claims list. Empty answer is empty. "
        "For supported or contradicted claims, provide one exact, contiguous evidence_quote copied from that "
        "case's context, with no ellipses or paraphrasing; for not_supported claims use an empty quote. "
        "Keep claims atomic and preserve technical "
        "identifiers exactly. Return JSON only in this schema: {\"cases\":[{\"condition\": string, "
        "\"response_type\": \"substantive\"|\"abstention\"|\"empty\", \"claims\":[{\"claim\": string, "
        "\"verdict\": \"supported\"|\"contradicted\"|\"not_supported\", "
        "\"evidence_quote\": string, \"reason\": string}]}]}. Include exactly one case for each supplied case, preserving its condition. "
        "Keep each reason brief.\n\nINPUT JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def call_judge(client: Any, model: str, question: str, cases: list[dict[str, str]], retries: int) -> dict[str, Any]:
    from google.genai import types

    prompt = build_prompt(question, cases)
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "cases": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "condition": {"type": "STRING", "enum": list(CONDITIONS)},
                        "response_type": {"type": "STRING", "enum": ["substantive", "abstention", "empty"]},
                        "claims": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "claim": {"type": "STRING"},
                                    "verdict": {"type": "STRING", "enum": sorted(VERDICTS)},
                                    "evidence_quote": {"type": "STRING"},
                                    "reason": {"type": "STRING"},
                                },
                                "required": ["claim", "verdict", "evidence_quote", "reason"],
                            },
                        },
                    },
                    "required": ["condition", "response_type", "claims"],
                },
            }
        },
        "required": ["cases"],
    }
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Judge returned an empty response")
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("cases"), list):
                raise ValueError("Judge response does not match the expected JSON schema")
            return parsed
        except Exception as exc:
            code = str(getattr(exc, "code", "") or getattr(exc, "status_code", "") or exc).upper()
            retryable = any(token in code for token in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not retryable or attempt >= retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError("Judge retries exhausted")


def validate_judgment(case: dict[str, Any], context: str) -> dict[str, Any]:
    condition = str(case.get("condition", "")).strip()
    response_type = str(case.get("response_type", "")).strip().lower()
    if response_type not in {"substantive", "abstention", "empty"}:
        raise ValueError(f"Invalid response_type for {condition}: {response_type!r}")
    claims = case.get("claims")
    if not isinstance(claims, list):
        raise ValueError(f"Invalid claims list for {condition}")
    normalized_claims = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError(f"Claim is not an object for {condition}")
        verdict = str(claim.get("verdict", "")).strip().lower()
        statement = str(claim.get("claim", "")).strip()
        quote = str(claim.get("evidence_quote", ""))
        if verdict not in VERDICTS or not statement:
            raise ValueError(f"Malformed claim for {condition}: {claim!r}")
        quote_valid = bool(quote) and normalize_quote(quote) in normalize_quote(context)
        effective_verdict = verdict
        if verdict in {"supported", "contradicted"} and not quote_valid:
            # Without a traceable context quote, this run cannot substantiate the judge's support label.
            effective_verdict = "not_supported"
        normalized_claims.append({
            "claim": statement,
            "judge_verdict": verdict,
            "verdict": effective_verdict,
            "evidence_quote": quote,
            "evidence_quote_valid": quote_valid if quote else None,
            "reason": str(claim.get("reason", "")).strip(),
        })
    if response_type == "empty" and claims:
        raise ValueError(f"Judge marked {condition} empty but emitted claims")
    if response_type == "substantive" and not claims:
        # Keep this explicit: a substantive answer with no extracted claims is a judge anomaly.
        response_type = "no_claims_extracted"
    return {"condition": condition, "response_type": response_type, "claims": normalized_claims}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_checkpoint(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["question_id"]), str(row["provider"]))
            if row.get("status") == "ok":
                done[key] = row
    return done


def paired_bootstrap(deltas: list[float], seed: int = BOOTSTRAP_SEED) -> dict[str, Any] | None:
    if not deltas:
        return None
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    batch = 256
    for start in range(0, BOOTSTRAP_SAMPLES, batch):
        n = min(batch, BOOTSTRAP_SAMPLES - start)
        indexes = rng.integers(0, len(values), size=(n, len(values)))
        means[start:start+n] = values[indexes].mean(axis=1)
    return {
        "confidence_level": 0.95,
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "method": "paired percentile bootstrap resampling question IDs",
        "resamples": BOOTSTRAP_SAMPLES,
        "seed": seed,
    }


def summarize(answer_rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"conditions": {}}
    for provider in sorted({r["provider"] for r in answer_rows}):
        summary["conditions"][provider] = {}
        for condition in CONDITIONS:
            answers = [r for r in answer_rows if r["provider"] == provider and r["condition"] == condition]
            claims = [r for r in claim_rows if r["provider"] == provider and r["condition"] == condition]
            labels = [r["verdict"] for r in claims]
            supported = labels.count("supported")
            contradicted = labels.count("contradicted")
            not_supported = labels.count("not_supported")
            answers_with_claims = [r for r in answers if r["claim_count"] > 0]
            summary["conditions"][provider][condition] = {
                "answer_records_judged": len(answers),
                "answers_with_factual_claims": len(answers_with_claims),
                "abstentions_with_claims": sum(r["response_type"] == "abstention" and r["claim_count"] > 0 for r in answers),
                "abstentions": sum(r["response_type"] == "abstention" for r in answers),
                "empty_answers": sum(r["response_type"] == "empty" for r in answers),
                "judge_no_claims_anomalies": sum(r["response_type"] == "no_claims_extracted" for r in answers),
                "factual_claims": len(claims),
                "supported_claims": supported,
                "contradicted_claims": contradicted,
                "not_supported_claims": not_supported,
                "claim_support_rate": supported / len(claims) if claims else None,
                "claim_contradiction_rate": contradicted / len(claims) if claims else None,
                "claim_unsupported_rate": (contradicted + not_supported) / len(claims) if claims else None,
                "answers_all_claims_supported_rate": (
                    sum(r["unsupported_claim_count"] == 0 for r in answers_with_claims) / len(answers_with_claims)
                    if answers_with_claims else None
                ),
                "invalid_or_nonverbatim_evidence_quote_count": sum(r["evidence_quote_valid"] is False for r in claims),
            }

        by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for row in answer_rows:
            if row["provider"] == provider:
                by_pair.setdefault((row["question_id"], row["condition"]), row)
        paired_qids = sorted({qid for qid, cond in by_pair if cond == CONDITIONS[0]} & {qid for qid, cond in by_pair if cond == CONDITIONS[1]})
        deltas: dict[str, list[float]] = {"claim_support_rate": [], "claim_unsupported_rate": []}
        paired_details = []
        for qid in paired_qids:
            plain, expanded = by_pair[(qid, CONDITIONS[0])], by_pair[(qid, CONDITIONS[1])]
            if plain["claim_count"] == 0 or expanded["claim_count"] == 0:
                continue
            ds = expanded["claim_support_rate"] - plain["claim_support_rate"]
            du = expanded["claim_unsupported_rate"] - plain["claim_unsupported_rate"]
            deltas["claim_support_rate"].append(ds)
            deltas["claim_unsupported_rate"].append(du)
            paired_details.append({"question_id": qid, "plain_support_rate": plain["claim_support_rate"], "expanded_support_rate": expanded["claim_support_rate"], "support_rate_delta": ds, "plain_unsupported_rate": plain["claim_unsupported_rate"], "expanded_unsupported_rate": expanded["claim_unsupported_rate"], "unsupported_rate_delta": du})
        summary["conditions"][provider]["paired_expansion_minus_plain"] = {
            "question_ids_with_claims_in_both_conditions": len(paired_details),
            "mean_support_rate_delta": statistics.mean(deltas["claim_support_rate"]) if deltas["claim_support_rate"] else None,
            "support_rate_delta_95pct_ci": paired_bootstrap(deltas["claim_support_rate"]),
            "mean_unsupported_rate_delta": statistics.mean(deltas["claim_unsupported_rate"]) if deltas["claim_unsupported_rate"] else None,
            "unsupported_rate_delta_95pct_ci": paired_bootstrap(deltas["claim_unsupported_rate"], BOOTSTRAP_SEED + 1),
            "positive_support_deltas": sum(v > 0 for v in deltas["claim_support_rate"]),
            "ties_support_delta": sum(v == 0 for v in deltas["claim_support_rate"]),
            "negative_support_deltas": sum(v < 0 for v in deltas["claim_support_rate"]),
            "interpretation": "Positive support delta favors adjacent expansion; negative unsupported-rate delta favors adjacent expansion.",
        }
        summary["conditions"][provider]["paired_question_details"] = paired_details
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cooked = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            writer.writerow(cooked)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=root / "results/evaluation/v2_context_ablation.json")
    parser.add_argument("--qwen-input", type=Path, default=root / "results/generation/v2_bm25_context_generation.jsonl")
    parser.add_argument("--gemini-input", type=Path, default=root / "results/generation/v2_bm25_context_generation_gemini_gemini-3.5-flash-lite.jsonl")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=root / "results/evaluation")
    parser.add_argument("--delay-seconds", type=float, default=4.2, help="Throttle requests; 4.2s is about 14 requests/minute.")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional question-count pilot; omitted means all matched questions.")
    parser.add_argument("--force", action="store_true", help="Ignore previous successful checkpoint records and start a fresh checkpoint.")
    args = parser.parse_args()
    if args.delay_seconds < 0 or args.retries < 0:
        parser.error("delay and retries must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "v2_factuality_judgments.jsonl"
    if args.force and checkpoint.exists():
        checkpoint.unlink()

    contexts = load_contexts(args.contexts)
    sources = {"qwen": args.qwen_input, "gemini": args.gemini_input}
    generations = {provider: read_jsonl(path) for provider, path in sources.items()}
    qid_sets = {p: {qid for qid, _ in rows} for p, rows in generations.items()}
    common_qids = sorted(set(contexts) & qid_sets["qwen"] & qid_sets["gemini"])
    for provider, rows in generations.items():
        expected = {(qid, cond) for qid in qid_sets[provider] for cond in CONDITIONS}
        missing = expected - set(rows)
        if missing:
            raise ValueError(f"{provider} generation is missing {len(missing)} expected (question,condition) rows")
    if not common_qids:
        raise ValueError("No question IDs are shared by both generation files and context artifact")
    if args.limit is not None:
        common_qids = common_qids[:args.limit]

    # Validate that the context-selection artifact is the one actually used for generation.
    for provider, rows in generations.items():
        for qid in common_qids:
            for condition in CONDITIONS:
                field = CONTEXT_FIELDS[condition]
                exact_context = str(contexts[qid].get(field, ""))
                row = rows[(qid, condition)]
                if not exact_context:
                    raise ValueError(f"Empty saved {condition} context for {qid}")
                if int(row.get("context_characters", -1)) != len(exact_context):
                    raise ValueError(f"Context length mismatch for {provider}/{qid}/{condition}; refusing to judge mismatched evidence")
                if row.get("generation_error"):
                    raise ValueError(f"Generation error exists for {provider}/{qid}/{condition}; handle it explicitly before judging")

    completed = {} if args.force else read_checkpoint(checkpoint)
    client = create_gemini_client()
    total = len(common_qids) * len(sources)
    request_no = 0
    for qid in common_qids:
        for provider, rows in generations.items():
            key = (qid, provider)
            if key in completed:
                continue
            question_values = {str(rows[(qid, c)].get("question", "")) for c in CONDITIONS}
            if len(question_values) != 1:
                raise ValueError(f"Question text differs across conditions for {provider}/{qid}")
            question = question_values.pop()
            cases = []
            empty_flags = {}
            for condition in CONDITIONS:
                answer = rows[(qid, condition)]["generated_answer"]
                context = str(contexts[qid][CONTEXT_FIELDS[condition]])
                empty_flags[condition] = not answer.strip()
                cases.append({"condition": condition, "answer": answer, "context": context})

            request_no += 1
            try:
                judgment = call_judge(client, args.model, question, cases, args.retries)
                received = {str(c.get("condition", "")): c for c in judgment["cases"] if isinstance(c, dict)}
                if set(received) != set(CONDITIONS):
                    raise ValueError(f"Judge returned wrong condition labels for {qid}/{provider}: {sorted(received)}")
                validated = []
                for condition in CONDITIONS:
                    record = validate_judgment(received[condition], str(contexts[qid][CONTEXT_FIELDS[condition]]))
                    if empty_flags[condition] and record["response_type"] != "empty":
                        raise ValueError(f"Judge did not mark empty answer as empty for {qid}/{provider}/{condition}")
                    validated.append(record)
                result = {
                    "question_id": qid,
                    "provider": provider,
                    "judge_model": args.model,
                    "status": "ok",
                    "judgments": validated,
                }
            except Exception as exc:
                result = {"question_id": qid, "provider": provider, "judge_model": args.model, "status": "error", "error": repr(exc)}
                append_jsonl(checkpoint, result)
                raise RuntimeError(f"Stopped at {provider}/{qid}; error checkpoint saved. Rerun to resume: {exc}") from exc
            append_jsonl(checkpoint, result)
            completed[key] = result
            print(f"[{len(completed)}/{total}] Judged {provider}/{qid}", flush=True)
            if len(completed) < total and args.delay_seconds:
                time.sleep(args.delay_seconds)

    answer_rows, claim_rows = [], []
    for qid in common_qids:
        for provider, rows in generations.items():
            judgment_row = completed.get((qid, provider))
            if not judgment_row:
                continue
            for judged in judgment_row["judgments"]:
                condition = judged["condition"]
                claims = judged["claims"]
                total_claims = len(claims)
                support_rate = sum(c["verdict"] == "supported" for c in claims) / total_claims if total_claims else None
                contradiction_rate = sum(c["verdict"] == "contradicted" for c in claims) / total_claims if total_claims else None
                unsupported_count = sum(c["verdict"] in {"contradicted", "not_supported"} for c in claims)
                answer_rows.append({
                    "question_id": qid,
                    "provider": provider,
                    "condition": condition,
                    "response_type": judged["response_type"],
                    "claim_count": total_claims,
                    "supported_claim_count": sum(c["verdict"] == "supported" for c in claims),
                    "contradicted_claim_count": sum(c["verdict"] == "contradicted" for c in claims),
                    "not_supported_claim_count": sum(c["verdict"] == "not_supported" for c in claims),
                    "claim_support_rate": support_rate,
                    "claim_contradiction_rate": contradiction_rate,
                    "claim_unsupported_rate": unsupported_count / total_claims if total_claims else None,
                    "unsupported_claim_count": unsupported_count,
                    "generation_model": rows[(qid, condition)].get("model", provider),
                    "judge_model": args.model,
                })
                for index, claim in enumerate(claims, 1):
                    claim_rows.append({
                        "question_id": qid,
                        "provider": provider,
                        "condition": condition,
                        "claim_index": index,
                        "claim": claim["claim"],
                    "judge_verdict": claim["judge_verdict"],
                        "verdict": claim["verdict"],
                        "evidence_quote": claim["evidence_quote"],
                        "evidence_quote_valid": claim["evidence_quote_valid"],
                        "judge_reason": claim["reason"],
                        "generation_model": rows[(qid, condition)].get("model", provider),
                        "judge_model": args.model,
                    })

    prefix = args.output_dir / f"v2_factuality_{args.model.replace('/', '_')}"
    answer_csv = prefix.with_name(prefix.name + "_answers.csv")
    claim_csv = prefix.with_name(prefix.name + "_claims.csv")
    paired_csv = prefix.with_name(prefix.name + "_paired_differences.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(answer_csv, answer_rows)
    write_csv(claim_csv, claim_rows)
    paired_rows = []
    for provider in sources:
        by_key = {(r["question_id"], r["condition"]): r for r in answer_rows if r["provider"] == provider}
        for qid in common_qids:
            plain, expanded = by_key.get((qid, CONDITIONS[0])), by_key.get((qid, CONDITIONS[1]))
            if plain and expanded:
                paired_rows.append({
                    "question_id": qid,
                    "provider": provider,
                    "plain_claim_count": plain["claim_count"],
                    "expanded_claim_count": expanded["claim_count"],
                    "plain_support_rate": plain["claim_support_rate"],
                    "expanded_support_rate": expanded["claim_support_rate"],
                    "support_rate_delta": (expanded["claim_support_rate"] - plain["claim_support_rate"]) if plain["claim_support_rate"] is not None and expanded["claim_support_rate"] is not None else None,
                    "plain_unsupported_rate": plain["claim_unsupported_rate"],
                    "expanded_unsupported_rate": expanded["claim_unsupported_rate"],
                    "unsupported_rate_delta": (expanded["claim_unsupported_rate"] - plain["claim_unsupported_rate"]) if plain["claim_unsupported_rate"] is not None and expanded["claim_unsupported_rate"] is not None else None,
                })
    write_csv(paired_csv, paired_rows)
    summary = summarize(answer_rows, claim_rows)
    empty_reference_qids = sorted({qid for rs in generations.values() for (qid, condition), row in rs.items() if condition == CONDITIONS[0] and qid in common_qids and not str(row.get("reference_answer", "")).strip()})
    empty_reference_rows = len(empty_reference_qids) * len(sources)
    summary.update({
        "experiment": "Claim-level evidence support for saved V2 plain and adjacent-expanded contexts",
        "judge_model": args.model,
        "judge_model_is_also_a_generation_model": any(any(str(row.get("model", "")) == args.model for row in rs.values()) for rs in generations.values()),
        "input_generation_files": {p: str(path.resolve()) for p, path in sources.items()},
        "context_input_file": str(args.contexts.resolve()),
        "output_files": {"raw_checkpoint_jsonl": str(checkpoint.resolve()), "answer_csv": str(answer_csv.resolve()), "claim_csv": str(claim_csv.resolve()), "paired_differences_csv": str(paired_csv.resolve()), "summary_json": str(summary_path.resolve())},
        "matched_question_ids": len(common_qids),
        "completed_judge_requests": len(completed),
        "generated_answer_rows_judged": len(answer_rows),
        "empty_reference_question_ids_present_but_reference_not_used": empty_reference_qids,
        "empty_reference_rows_present_but_reference_not_used": empty_reference_rows,
        "generation_errors": 0,
        "generation_empty_answer_policy": "A genuinely empty generated answer is recorded as empty with zero claims; it remains in answer-level counts but contributes no claim-rate denominator.",
        "abstention_policy": "A response that only says the evidence is insufficient is counted as abstention with zero subject-matter claims; it remains in answer-level counts but not claim-rate denominator.",
        "claim_metric_denominator": "All extracted factual claims; each claim has equal weight.",
        "evidence_quote_validation": "A judge label of supported or contradicted counts only when its evidence quote appears as a contiguous quote in that exact context after whitespace/case normalization; otherwise the effective verdict is not_supported and the original judge verdict is retained separately.",
        "verdict_definitions": {"supported": "Context directly states or clearly entails claim.", "contradicted": "Context directly conflicts with claim.", "not_supported": "Context neither supports nor directly contradicts claim."},
        "context_integrity_check": "Saved context character count exactly matched each generation record's context_characters for both conditions and both providers.",
        "judge_prompt_does_not_include_reference_answer": True,
        "judge_prompt_version": "v1.1-technical-claims-and-exact-context-quotes",
        "judge_generation_config": {"temperature": 0, "max_output_tokens": 4096, "response_mime_type": "application/json", "structured_response_schema": True},
        "paired_analysis": "Adjacent expansion minus plain context; paired by question ID. Bootstrap CIs resample question IDs, not individual claims.",
        "bootstrap": {"resamples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "method": "paired percentile bootstrap over question IDs"},
        "request_delay_seconds": args.delay_seconds,
        "google_genai_version": importlib.metadata.version("google-genai"),
        "interpretation_limitations": [
            "This evaluates support by retrieved context, not truth against the world or correctness against the reference answer.",
            "Claim extraction and support labels are generated by an LLM judge and may be wrong; inspect claim-level evidence and manually audit a sample before treating the aggregate as definitive.",
            "The same Gemini model judges all conditions, but it also generated the Gemini answers; possible judge self-preference is a limitation.",
            "Support rates weight claims equally and may vary with how the judge splits claims.",
            "Paired confidence intervals quantify uncertainty under question resampling; they do not eliminate judge bias or establish generalization beyond this development set.",
        ],
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Answer-level CSV: {answer_csv}")
    print(f"Claim-level CSV: {claim_csv}")
    print(f"Paired differences CSV: {paired_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
