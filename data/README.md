# Data Directory

This directory holds all dataset files for the TechQA RAG project.

## Structure

```
data/
├── raw/
│   └── technote_corpus/          # Original IBM technotes corpus
│       ├── CDLA-Permissive-v1.0.pdf  ← tracked in Git
│       ├── full_technote_collection.sections.json  ← LOCAL ONLY (~6 GB)
│       ├── full_technote_collection.txt.bz2         ← LOCAL ONLY (~795 MB)
│       ├── technotes.id_title_text                  ← LOCAL ONLY (~2.5 GB)
│       └── technotes.text                           ← LOCAL ONLY (~2.2 GB)
│
└── processed/
    ├── training_and_dev/         # Q&A pairs and referenced technotes
    │   ├── training_Q_A.json         (600 labeled training instances)
    │   ├── dev_Q_A.json              (310 labeled dev instances)
    │   ├── training_Q_A.txt          (plain-text version)
    │   ├── training_dev_technotes.json          ← LOCAL ONLY (~278 MB)
    │   └── training_dev_technotes.sections.json ← LOCAL ONLY (~228 MB)
    │
    └── validation/
        ├── validation_questions.json      (282 validation questions)
        ├── validation_reference.json      (ground truth)
        ├── validation_technotes.json      (referenced technotes)
        └── validation_example_output.json (example submission format)
```

## Notes

- Files marked **LOCAL ONLY** are excluded from Git via `.gitignore`.
- See `../artifacts/ARTIFACT_REGISTRY.md` for a full list of large files.
- License: [CDLA-Permissive v1.0](raw/technote_corpus/CDLA-Permissive-v1.0.pdf)
