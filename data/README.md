# `data/` — not in version control

This directory is where the pipeline reads and writes. It is **empty in this repository by
design**: the full tree is ~127 GB of downloaded corpora, extracted JSON, trained models and
release artifacts, none of which belongs in git and all of which the pipeline reproduces.

`make all` creates what it needs. The layout it expects:

| path | what lands there | produced by |
|---|---|---|
| `antimicrobial_databases/` | DRAMP, dbAMP, DBAASP, CAMP, AntiCP2, APD3 release files | steps 1–10 |
| `literature_data/` | PubMed/Scopus metadata and downloaded PDFs | steps 1–5 |
| `literature_data_json/` | Per-document extraction JSON, under `pubmed/`, `scopus/`, `google_patents/` | steps 6–12 |
| `patent_data_json/` | Patent extraction JSON | steps 6–12 |
| `gene_ontology/` | `go-basic.obo` and the project's slice of it | steps 43–44 |
| `sample_final_release/` | `final_dataset.csv`, `activities.csv`, `models/`, the JSON records, the reports | steps 40–52 |
| `sample_final_release/models/` | Served model pickles, namespaced `<type>/<scope>/[target]/<assay>/<task>/<block>.pkl` | steps 45–48 |
| `sample_final_release/models_retired/` | Models withdrawn from service but kept for provenance | manual |
| `structures/` | AlphaFold2/ColabFold PDBs | step 30 |
| `mpepe/` | MPEPE's repo and ~130 MB of weights, cloned on first use | `make env-mpepe` |
| `zenodo_dump/` | The archival bundle | step 54 |

Two things worth knowing before you run anything:

- **Extraction JSON is read only through `config.iter_extraction_files`**, which allowlists the
  recognized source subdirectories. Never walk `data/literature_data_json/` with a bare glob: a
  backup directory sitting beside the real one gets picked up silently and double-counts the
  corpus.
- **`sample_final_release/final_dataset.csv` is rewritten in place** by several steps, including
  the anomaly detector (step 45), which writes its `is_anomaly`/`anomaly_score` columns back into
  the file it just read. Keep a copy before re-running a step if you need the previous labels.

See the root `README.md` for the step table and `KNOWN_ISSUES.md` for which parts of the tree are
currently incomplete and why.
