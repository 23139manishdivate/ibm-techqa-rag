# Dataset Description

## TechQA Dataset

**Source:** https://github.com/IBM/techqa  
**Paper:** Castelli et al., ACL 2020 — *"The TechQA Dataset"*  
**License:** [CDLA-Permissive v1.0](../data/raw/technote_corpus/CDLA-Permissive-v1.0.pdf)

---

## Splits

| Split | File | Questions | Format |
|-------|------|-----------|--------|
| Training | `training_Q_A.json` | 600 | JSON list of maps |
| Development | `dev_Q_A.json` | 310 | JSON list of maps |
| Validation (held-out) | `validation_questions.json` | 282 | JSON list of maps |

### Instance Schema

Each question instance contains:

```json
{
  "QUESTION_ID":    "unique string id",
  "QUESTION_TITLE": "short summary / title of the question",
  "QUESTION_BODY":  "full question text",
  "DOC_IDS":        ["doc_id_1", "doc_id_2", ...],
  "ANSWERABLE":     "Y or N",
  "DOCUMENT":       "doc_id or -",
  "START_OFFSET":   "int or -",
  "END_OFFSET":     "int or -",
  "ANSWER":         "verbatim text span or -"
}
```

- `DOC_IDS`: candidate document pool for this question (~5–10 docs per question)
- `ANSWERABLE = 'Y'`: 160 of the 310 dev questions have a ground-truth answer

---

## Technote Document Schema

Each document in `training_dev_technotes.json` / `validation_technotes.json`:

```json
{
  "id":       "unique document id",
  "title":    "page title (detagged)",
  "text":     "body text (automatically detagged HTML)",
  "content":  "raw HTML",
  "metadata": {
    "sourceDocumentId": "...",
    "date":             "last modified date",
    "productName":      "IBM product name",
    "productId":        "product id",
    "canonicalUrl":     "canonical URL"
  }
}
```

---

## Technote Corpus (Full, Local Only)

The full corpus contains **801,998** IBM technotes as of April 4, 2019.

| File | Size | Description |
|------|------|-------------|
| `full_technote_collection.sections.json` | ~6 GB | Section-split JSON, one array per line |
| `full_technote_collection.txt.bz2` | ~795 MB | BZ2-compressed; one JSON array per line |
| `technotes.id_title_text` | ~2.5 GB | Tab-separated: id, title, text |
| `technotes.text` | ~2.2 GB | Plain text body, one doc per record |

> ⚠️ These files are **not tracked by Git** and must be transferred manually to the college machine.

### Reading the Corpus

```python
import bz2, json

with bz2.open("data/raw/technote_corpus/full_technote_collection.txt.bz2", "rt") as f:
    for line in f:
        docs = json.loads(line)   # list of doc dicts
        for doc in docs:
            doc_id = doc["id"]
            title  = doc["title"]
            text   = doc["text"]
```

---

## Data Splits Used in Experiments

| Phase | Questions | Source |
|-------|-----------|--------|
| Phase 1 retrieval | 450 mixed | training set sample |
| Phase 2 retrieval | 160 answerable | dev set (answerable only) |
| Phase 4 generation | 310 | full dev set |
