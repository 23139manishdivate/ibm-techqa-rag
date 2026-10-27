# Run the TechQA reader with Gemini

The Gemini option changes only answer generation. It uses the already-saved BM25 plain and adjacent-expanded contexts; it does not rebuild chunks, embeddings, indexes, or retrieval results. The Qwen path and its saved outputs remain available as the baseline.

## One-time setup

1. Install the Gemini SDK in the project's active Python environment:

   ```powershell
   python -m pip install google-genai
   ```

2. Set the Windows user environment variable `gemini_api_key` (or `GEMINI_API_KEY`) to the key in Windows **Environment Variables**. Never paste the key into source code or commit it.
3. Fully close and reopen PowerShell, VS Code, or Codex after changing the variable. A process only receives environment variables when it starts.
4. The default Gemini model is `gemini-3.5-flash-lite`. Model access and free-tier quotas are controlled by the Google AI Studio project associated with the key.

## Smoke run first

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Join-Path $PWD "src")
python .\src\techqa_rag\evaluation\evaluate_v2_context_generation.py --provider gemini --limit 2
```

This makes up to four generation calls: two questions times the plain and adjacent-expanded context conditions. Check the saved JSONL rows and summary before starting the full run. Gemini outputs use separate filenames containing the provider and model, so they do not overwrite Qwen results.

## Full development evaluation

After confirming the smoke output and your active API quota, run:

```powershell
python .\src\techqa_rag\evaluation\evaluate_v2_context_generation.py --provider gemini
```

The script evaluates up to 160 answerable development questions under both context conditions. It checkpoints each response, resumes successful rows on rerun, pauses between API calls, and stops on a failed Gemini request rather than continuing to send failing requests. Check your active RPM, TPM, and daily limits in AI Studio; if the daily quota is below the full run, rerun on a later day to continue from the checkpoint.

To select another model available to your project, pass its exact model ID:

```powershell
python .\src\techqa_rag\evaluation\evaluate_v2_context_generation.py --provider gemini --model MODEL_ID
```

The separate oracle-context script can also use Gemini, but oracle contexts use development gold labels and are diagnostics only. It makes up to 320 calls and is not part of deployable retrieval evaluation:

```powershell
python .\src\techqa_rag\evaluation\evaluate_v2_oracle_contexts.py --provider gemini --limit 2
```

Gemini uses the same system instructions, question text, and selected context text as the Qwen comparison, but it has its own tokenizer and hosted serving behavior. Prompt-token counts come from Gemini's returned usage metadata when available. Treat provider comparisons as controlled model comparisons, not identical tokenization or deterministic decoding. Token F1 and exact match remain reference-overlap metrics, not human answer-correctness scores.
