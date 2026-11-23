import os
# Prevent PyTorch/OpenMP from spawning too many threads when used in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import logging
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from sample.config import DATA_DIR, PROMPTS_DIR, PATENTS_DIR, LITERATURE_DIR, LOGS_DIR

logger = logging.getLogger("sample.cli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main() -> None:
    """
    Entry point for the ``sample-cli`` console script.

    Parses ``sys.argv``, then dispatches to the module implementing the requested subcommand. Each
    subcommand corresponds to one numbered pipeline step; see the README for the full reference.

    :returns: ``None``. Failures surface as a logged error or a non-zero exit from the callee.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="sAMPle: Unified Pipeline CLI for Antimicrobial Peptide Data Engineering")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline steps to execute")
    
    # 1. PubMed parser
    parser_pubmed = subparsers.add_parser("pubmed", help="Download PubMed publication metadata")
    parser_pubmed.add_argument("--query-file", default=str(DATA_DIR / 'queries/pubmed.txt'), help="Path to pubmed query text file")
    parser_pubmed.add_argument("--email", default=os.environ.get("EMAIL", "1_1@gmail.com"), help="NCBI registered email")
    parser_pubmed.add_argument("--output", default=str(LITERATURE_DIR / 'pubmed_metadata.csv'), help="Path to output metadata CSV")
    
    # 2. Scopus parser
    parser_scopus = subparsers.add_parser("scopus", help="Download Scopus publication metadata")
    parser_scopus.add_argument("--query-file", default=str(DATA_DIR / 'queries/scopus.txt'), help="Path to scopus query text file")
    parser_scopus.add_argument("--output", default=str(LITERATURE_DIR / 'scopus_metadata.csv'), help="Path to output metadata CSV")
    
    # 3. Download PDFs parser
    parser_pdf = subparsers.add_parser("download-pdfs", help="Download full-text paper PDFs from DOIs")
    parser_pdf.add_argument("--source", required=True, choices=["pubmed", "scopus"], help="Source metadata to download (pubmed or scopus)")
    parser_pdf.add_argument("--email", default=os.environ.get("EMAIL", "1_1@gmail.com"), help="Email to use for Unpaywall requests")
    parser_pdf.add_argument("--workers", type=int, default=4, help="Number of concurrent download processes")
    parser_pdf.add_argument(
        "--max-attempts", type=int, default=3,
        help="Stop retrying a DOI after this many recorded failures across runs (0 = retry "
             "everything). Failures are recorded in data/literature_data/<source>_failed.csv; "
             "without this every run repeats the whole pending set, including papers that have no "
             "open-access copy at all.")
    parser_pdf.add_argument(
        "--limit", type=int, default=0,
        help="Attempt at most N pending papers, for a bounded trial run before committing to the "
             "full pending set.")
    
    # 4. Google Patents parser
    parser_gpatents = subparsers.add_parser("google-patents", help="Download Google Patents PDFs")
    parser_gpatents.add_argument("--csv", default=str(PATENTS_DIR / 'google_patents.csv'), help="Path to patents CSV")
    parser_gpatents.add_argument("--output-dir", default=str(PATENTS_DIR / 'google_patents'), help="Directory to save patent PDFs")
    parser_gpatents.add_argument("--workers", type=int, default=8, help="Number of concurrent thread workers")

    parser_db = subparsers.add_parser("databases", help="Download and scrape antimicrobial peptide databases")
    parser_db.add_argument("--targets", nargs="+", choices=["adp", "dbaasp", "dramp", "dbamp", "camp"], default=["adp", "dbaasp", "dramp", "dbamp", "camp"], help="Peptide databases to download")
    parser_db.add_argument("--output-dir", default=str(DATA_DIR / 'antimicrobial_databases'), help="Base directory for database downloads")
    parser_db.add_argument("--download-papers", action="store_true", help="Download associated paper PDFs for the databases")
    parser_db.add_argument("--workers", type=int, default=10, help="Number of concurrent scraper workers")
    parser_db.add_argument("--refresh", action="store_true",
                           help="Re-download files that already exist, picking up a new upstream release. OFF by default, because skipping existing files is what makes an interrupted step-5 run resumable -- but that default is also why this host stayed on DRAMP 4.0 for seven months while the same URLs served 5.0: DRAMP has no version in its download path and updates the files in place, so a new release is indistinguishable from an up-to-date local copy. For DRAMP the existing tree is archived to dramp_<release>_<date>/ rather than overwritten, since the current database was built from it.")
    parser_db.add_argument("--dramp-status", action="store_true",
                           help="Print which DRAMP release is on disk, and the per-file evidence, without downloading anything. Reads the manifest written by the last download and re-derives the release from the data, since the download URL names a version it does not serve.")

    parser_acp = subparsers.add_parser("import-anticancer", help="Import and normalise anticancer peptide datasets (AntiCP2, DCTPep, CancerPPD; ApInAPDB's host does not respond)")
    parser_acp.add_argument("--download", action="store_true", help="Download the raw source files first (needs network)")
    parser_acp.add_argument("--data-dir", default=None, help="Directory holding the downloaded raw files; the packaged location when omitted")
    parser_acp.add_argument("--output", default=None, help="Where to write the combined table; data/anticancer_databases/anticancer_peptides.csv when omitted")
    parser_acp.add_argument("--scrape-cancerppd", action="store_true", help="Also scrape CancerPPD's per-entry detail pages for the measured IC50/cell line/assay (~3,100 requests, rate-limited and resumable; skipped by default)")
    parser_acp.add_argument("--scrape-limit", type=int, default=None, help="Stop the CancerPPD scrape after this many NEW entries (use a small value first)")
    parser_acp.add_argument("--scrape-delay", type=float, default=0.5, help="Seconds between CancerPPD detail requests, to stay polite to the host")

    parser_cyc = subparsers.add_parser("import-cycpeptmpdb", help="Index CycPeptMPDB as a source: ~8,500 cyclic peptides with membrane-permeability annotations (all tagged head-to-tail/circular)")
    parser_cyc.add_argument("--download", action="store_true", help="Download the raw CycPeptMPDB CSV first (needs network)")
    parser_cyc.add_argument("--csv", default=None, help="Raw CycPeptMPDB CSV; the packaged download location when omitted")
    parser_cyc.add_argument("--output", default=None, help="Where to write the normalised table; data/cycpeptmpdb/cycpeptmpdb_peptides.csv when omitted")
    parser_cyc.add_argument("--link", action="store_true", help="Also link entries to release peptides that are the same molecule (by InChIKey)")
    parser_cyc.add_argument("--db", default=None, help="Release database for --link; the project default when omitted")

    # 7. Merge parser
    parser_merge = subparsers.add_parser("merge", help="Merge database files into a consensus dataset")
    parser_merge.add_argument("--output", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to save merged consensus CSV")
    parser_merge.add_argument("--base-dir", default=str(DATA_DIR / 'antimicrobial_databases'), help="Base directory of database downloads")
    
    # 8. Extract parser
    # Runs the classical-NLP-first extraction engine (sample/extractor.py + sample/nlp.py +
    # sample/llm.py): Aho-Corasick/regex candidate discovery, LLM-verified activities,
    # LLM fallback only for unverified candidates. Writes one <output-dir>/<internal_id>.json
    # file per document, which is the contract `validate`, `external-enrich`,
    # `merge-extractions`, and the webserver's Extractions viewer all read from.
    parser_extract = subparsers.add_parser("extract", help="Extract peptide data from screened documents")
    parser_extract.add_argument("--sources", nargs="+", default=["all"], help="Sources to extract from: pubmed, scopus, patents, or all (default: all)")
    parser_extract.add_argument("--limit", type=int, default=0, help="Limit number of papers to process (0 = all)")
    parser_extract.add_argument("--resume", action="store_true", help="Shorthand for --resume-mode exists: skip documents that already have an output JSON file. Right for the screening pass; WRONG for the refined pass, which finds an output for every document already and would skip the corpus -- use --resume-mode refined there.")
    parser_extract.add_argument("--resume-mode", choices=("off", "exists", "refined"), default=None,
                                help="How to continue an interrupted extraction. 'off' re-processes every document. 'exists' skips any document that already has an output JSON (the screening pass, step 8). 'refined' skips only documents whose output records that the refined model pass already ran, read from metadata.langextract_backend (step 9) -- which is what lets step 9 resume at all, since every document already has a screening output. Overrides --resume when both are given.")
    parser_extract.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 4), help="Number of parallel worker threads (clamped to CPU core count; default: min(cores, 4)). Per-document time is dominated by Ollama round trips, so the useful thread count is bounded by how many requests the model server can actually run at once -- not by CPU. With the default gemma4 (9.6GB) on a 12GB GPU there is only ~2GB left for KV cache after the weights, which supports roughly 1-2 concurrent slots; threads beyond that queue and add memory pressure without adding throughput. Raise this together with OLLAMA_NUM_PARALLEL only if the model is small enough that its weights plus N KV caches still fit in VRAM (a 1.5B model comfortably supports 8-12).")
    parser_extract.add_argument("--chunk-size", type=int, default=None,
                                help="Process documents in groups of this many, resetting the Ollama runtime (unloading the model, which terminates its llama-server runner) between groups. Default 100, from SAMPLE_EXTRACT_CHUNK_SIZE; 0 disables chunking. This exists because throughput decays over a long run rather than failing: measured on this corpus it held near 1 doc/s and fell to ~0.04 docs/s after ~2,000 documents, with no error logged, and recovered fully on a manual server restart. A chunk bounds how long any state inside the runner can accumulate.")
    parser_extract.add_argument("--doc-timeout", type=float, default=None,
                                help="Wall-clock seconds one document may spend. Default 300, from SAMPLE_DOCUMENT_BUDGET_SECONDS; 0 disables it. It caps NEW expensive work rather than interrupting work in progress -- a thread cannot be killed in CPython, and abandoning one would leave it holding an Ollama slot -- so each LLM call is clamped to the remainder, calls are refused once it is gone, and the langextract subprocess is killed outright. A document that hits the cap is still written, flagged metadata.budget_exceeded, with fewer records than it would have had.")
    parser_extract.add_argument("--input-dir", help="Directory containing input .txt chunks (optional, overrides the per-source default)")
    parser_extract.add_argument("--output-dir", help="Directory to save extracted JSONs (optional, overrides the per-source default)")


    parser_fieldrev = subparsers.add_parser(
        "review-fields",
        help="Per-field LLM review of extracted data: one prompt, schema, context budget and model per field")
    parser_fieldrev.add_argument("--run", action="store_true",
                                 help="Review the corpus -- step 12. Selects the fields a rule cannot settle AND a cheap signal says matter (~10,619 across the refined output, about 14.7h at two Ollama slots), records a verdict for each into metadata.field_review, and with --apply writes corrections that survive the field's own rules. Run AFTER step 11, whose deterministic repairs remove everything a model does not need to see.")
    parser_fieldrev.add_argument("--apply", action="store_true",
                                 help="Write proposed corrections, not just record them. A correction the field's own deterministic check rejects is refused and counted, so this cannot introduce a defect step 11 would have to clean up.")
    parser_fieldrev.add_argument("--max-calls", type=int, default=0,
                                 help="Stop after this many model calls (0 = no cap). How the full pass is run in pieces.")
    parser_fieldrev.add_argument("--dry-run", action="store_true",
                                 help="With --run: select and count targets without calling the model or writing")
    parser_fieldrev.add_argument("--list", action="store_true",
                                 help="Print every field spec -- model, evidence budget, failure modes, and how many of them a rule settles without a model call -- and exit")
    parser_fieldrev.add_argument("--show-prompt", metavar="FIELD", default=None,
                                 help="Print the system prompt and JSON schema one field would send, and exit. Read this before trusting a run: the prompt is built from failure modes measured on this corpus, and a stale count is a stale prompt.")
    parser_fieldrev.add_argument("--measure", action="store_true",
                                 help="Score each installed model on labelled cases drawn from the corpus, instead of assuming the spec's model is best. Labels come from the deterministic rules (known-bad) and the trustworthy tier (known-good); the rule is suppressed during the call so the model has to reach the verdict itself. Every case is one model call -- keep --limit small while an extraction is running.")
    parser_fieldrev.add_argument("--field", action="append", default=None,
                                 help="Restrict to this field; repeatable. All fields when omitted.")
    parser_fieldrev.add_argument("--model", action="append", default=None,
                                 help="Restrict --measure to this model tag; repeatable. Uninstalled tags are skipped, never pulled.")
    parser_fieldrev.add_argument("--limit", type=int, default=20,
                                 help="Cases per field for --measure (default 20)")
    parser_fieldrev.add_argument("--base-dir", default=None,
                                 help="Extraction output to draw cases from; data/literature_data_json when omitted")
    parser_fieldrev.add_argument("--variant", action="append", default=None,
                                 help="Prompt variant to measure; repeatable. Pass twice to compare two on the same cases (e.g. --variant v1 --variant v2). The variant in force when omitted.")
    parser_fieldrev.add_argument("--json", action="store_true", help="Machine-readable output")

    parser_vapply = subparsers.add_parser(
        "apply-verdicts",
        help="Act on step 12's recorded verdicts, and apply the mechanical corrections that need no verdict")
    parser_vapply.add_argument("--run", action="store_true",
                               help="Correct the corpus -- the second half of step 12. Two layers: the mechanical one settles what the record plus a local authority decides (a metric name in the unit column, a case variant of a vocabulary term, an abbreviated genus the document spells out, a three-letter sequence, a cell line spelled two ways) across the whole corpus; the authorized one takes step 12's proposals and writes only those the field's own authority confirms -- NCBI taxonomy, the 29-unit vocabulary, the 38-term activity vocabulary, the amino-acid alphabet. Run AFTER steps 11 and 12. Costs no model calls.")
    parser_vapply.add_argument("--dry-run", action="store_true",
                               help="Count every correction and refusal without writing. Run this first: the summary names the authority behind each change and the reason behind each refusal.")
    parser_vapply.add_argument("--verdict-file", default=None,
                               help="JSONL snapshot of step 12's verdicts (one object per document, with 'path' and 'verdicts'). Omit inside the pipeline, where verdicts are read from each document's own metadata.field_review; pass one only to act on verdicts a later step-12 pass has since overwritten, since that stage has no resume check.")
    parser_vapply.add_argument("--limit", type=int, default=0,
                               help="Stop after this many documents (0 = all)")
    parser_vapply.add_argument("--base-dir", default=None,
                               help="Extraction output; data/literature_data_json when omitted")
    parser_vapply.add_argument("--json", action="store_true", help="Machine-readable output")

    parser_polish = subparsers.add_parser(
        "polish-extractions",
        help="Re-apply the current activity-field rules to extraction records already on disk -- step 11")
    parser_polish.add_argument("--base-dir", default=None,
                               help="Extraction output to polish; data/literature_data_json when omitted. Read through config.iter_extraction_files, so only recognized source subdirectories are touched and the manual backups that accumulate beside them are not.")
    parser_polish.add_argument("--dry-run", action="store_true",
                               help="Report what would change without writing. Run this first: the counts tell you how much of the corpus predates the current rules.")
    parser_polish.add_argument("--limit", type=int, default=0,
                               help="Stop after this many documents (0 = all), for a quick sample of a large corpus")

    parser_name_disc = subparsers.add_parser("train-sequence-discriminator",
        help="Fit the bigram table that tells a peptide sequence from a compound name spelled in residue letters (ANTIMYCINA, SANDOSTATIN, ENKEPHALIN). Trained from this project's own catalogue plus a system wordlist; the output is small and committed, so the gate is on by default on a fresh clone.")
    parser_name_disc.add_argument("--output", default=None, help="Model JSON (default: data/peptide_name_bigrams.json)")
    parser_name_disc.add_argument("--sequences", default="data/antimicrobial_databases/merged_source.csv", help="Consensus CSV supplying the positive class")
    parser_name_disc.add_argument("--wordlist", default="/usr/share/dict/american-english", help="One word per line, supplying the negative class")

    parser_rf_filter = subparsers.add_parser("train-rf-filter", help="Train Random Forest filter to discriminate peptide sequences from English words")
    parser_rf_filter.add_argument("--output", default=str(DATA_DIR / 'rf_filter_model.pkl'), help="Path to save trained RF filter pickle")

    parser_external = subparsers.add_parser("external-enrich", help="Enrich NO_SEQ peptides with external DBs")
    parser_external.add_argument("--sources", nargs="+", default=["all"], help="Sources to enrich")
    parser_external.add_argument("--limit", type=int, default=0, help="Limit number of peptides to process (0 = all)")
    parser_external.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")

    parser_extract_pdf = subparsers.add_parser("extract-from-pdf", help="Extract peptide data from a single PDF")
    parser_extract_pdf.add_argument("pdf_path", help="Path to PDF")
    parser_extract_pdf.add_argument("--output", default="-", help="Output path")
    parser_extract_pdf.add_argument("--prompt", default=str(PROMPTS_DIR / '01_extraction_prompt.txt'), help="Path to LLM extraction prompt")
    parser_extract_pdf.add_argument("--schema", default=str(PROMPTS_DIR / '01_extraction_prompt_response_format.json'), help="Path to LLM extraction schema")
    parser_extract_pdf.add_argument("--verify-prompt", default=str(PROMPTS_DIR / '02_verification_prompt.txt'), help="Path to final verification prompt")
    parser_extract_pdf.add_argument("--screen-text-prompt", default=str(PROMPTS_DIR / '00_screening_text_prompt.txt'), help="Path to text screening prompt")
    parser_extract_pdf.add_argument("--model", default="gemma4:latest", help="Specify complex LLM model for extraction")
    parser_extract_pdf.add_argument("--screen-text-model", default="gemini-flash", help="Specify simpler text-only LLM model for screening")
    parser_extract_pdf.add_argument("--skip-screening", action="store_true", help="Bypass screening")
    
    parser_compare = subparsers.add_parser("compare-chemical-spaces", help="Compare chemical spaces")
    parser_compare.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to dataset")
    parser_compare.add_argument("--output-dir", default=str(DATA_DIR / 'sample_final_release/eda'), help="Path to output")

    # Advanced Extraction: PDF Table Parser
    parser_pdf_tables = subparsers.add_parser("extract-tables", help="Extract 2D grid tables from PDF papers")
    parser_pdf_tables.add_argument("pdf_path", help="Path to PDF paper")
    parser_pdf_tables.add_argument("--output", default="-", help="Path to output JSON")
    parser_pdf_tables.add_argument(
        "--backend", choices=["auto", "docling", "pdfplumber"], default="auto",
        help="Table reader. 'docling' (TableFormer, in the isolated sample-docling env) "
             "reconstructs ragged/merged cells and two-row headers that pdfplumber cannot; "
             "'auto' uses it when that environment is built and falls back to pdfplumber.")
    parser_pdf_tables.add_argument(
        "--ocr", choices=["auto", "on", "off"], default="auto",
        help="OCR via the local tesseract binary, for PDFs with no usable text layer. "
             "'auto' decides per document; measured over 300 sampled PDFs, ~0.3% need it.")

    # Advanced Enrichment: ESM-2 & Descriptors
    parser_esm2 = subparsers.add_parser("enrich-esm2", help="Compute ESM-2 Protein Language Model sequence embeddings")
    parser_esm2.add_argument("--dataset", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to dataset CSV")
    parser_esm2.add_argument("--output-dir", default=str(DATA_DIR / 'sample_enriched_release/embeddings'), help="Directory to save embeddings")

    parser_desc = subparsers.add_parser("enrich-descriptors", help="Compute biophysical descriptors (uH, Net Charge, Boman Index)")
    parser_desc.add_argument("--dataset", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to dataset CSV")
    parser_desc.add_argument("--output", default=str(DATA_DIR / 'sample_enriched_release/descriptors.csv'), help="Path to output CSV")

    parser_validate = subparsers.add_parser("validate", help="Validate LLM extraction results and compare with baseline")
    parser_validate.add_argument("--lit-dir", default=str(DATA_DIR / 'literature_data_json_repaired'), help="Directory of literature extraction JSONs")
    parser_validate.add_argument("--patent-dir", default=str(DATA_DIR / 'patent_data_json_repaired'), help="Directory of patent extraction JSONs")
    parser_validate.add_argument("--baseline", default=str(DATA_DIR / 'antimicrobial_databases/merged_source.csv'), help="Baseline merged source CSV")
    parser_validate.add_argument("--report", default=str(DATA_DIR / 'validation_report.json'), help="Path to save validation report JSON")
    parser_validate.add_argument("--report-only", action="store_true", help="Recompute the report from the existing *_validated trees without re-running the LLM audit (use after fixing --baseline)")

    # 10. Validate Error Injection parser
    parser_err_inj = subparsers.add_parser("validate-error-injection", help="Validate extraction schemas using error injection")
    parser_err_inj.add_argument("--input-dir", default=str(DATA_DIR / 'literature_data_json_repaired'), help="Directory containing JSON files")
    parser_err_inj.add_argument("--report", default=str(DATA_DIR / 'error_injection_report.json'), help="Path to save validation results")
    parser_err_inj.add_argument("--samples", type=int, default=2000, help="Number of injection trials, spread evenly across corruption types (default 2000, ~60 per type)")

    # 12. Revision parser -- the strict-contract gate that produces the *_revised extraction set
    # every later step reads. See sample/revision.py.
    parser_curation = subparsers.add_parser("select-curation-set", help="Step 16: select the manual-curation review set into data/manual_curation (one directory each: metadata.json + extraction.json + paper.txt + paper.pdf), plus curation_review.xlsx with one row per extracted activity and a verdict column beside every field, mixing documents the extractor DID find peptides in with documents it found none in, for human or LLM review")
    parser_curation.add_argument("--papers", type=int, default=None, help="How many literature papers to select (default 800)")
    parser_curation.add_argument("--patents", type=int, default=None, help="How many patents to select (default 200; note only ~26 patents are referenced by a released peptide, so the rest are necessarily negatives)")
    parser_curation.add_argument("--sample-size", type=int, default=None, help="Same size for BOTH corpora; overridden by --papers/--patents when those are given")
    parser_curation.add_argument("--positive-fraction", type=float, default=None, help="Share of each corpus that should be documents with extracted peptides (default 0.5); the remainder are negatives, and a positive shortfall is absorbed by negatives")
    parser_curation.add_argument("--no-db-filter", action="store_true", help="Do NOT restrict positives to documents a released peptide was drawn from. Widens the sample to extractions that never reached the release")
    parser_curation.add_argument("--output", default=None, help="Where to write the set; data/manual_curation when omitted")
    parser_curation.add_argument("--seed", type=int, default=20260817, help="Selection seed, so a rerun reproduces the same set")
    parser_curation.add_argument("--overwrite", action="store_true", help="Accepted for compatibility and now the default; see --no-overwrite")
    parser_curation.add_argument("--no-overwrite", action="store_true", help="Refuse to replace an existing set. Use this when a curator has annotations in the directory -- rebuilding discards them")
    parser_curation.add_argument("--refresh", action="store_true", help="Do NOT re-select: attach any missing paper.pdf to the existing set and rewrite curation_review.xlsx in place. Use this on a set already under review -- it adds files and deletes nothing, where a normal run rebuilds the tree and discards annotations")

    parser_revise = subparsers.add_parser(
        "revise", help="Enforce the strict output contract and write the *_revised extraction set")
    parser_revise.add_argument("--sources", nargs="+", default=["all"],
                               help="all | pubmed | scopus | patents")
    parser_revise.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8),
                               help="Parallel file workers")
    parser_revise.add_argument("--no-llm", action="store_true",
                               help="Deterministic contract only; skip the LLM repair attempt")
    parser_revise.add_argument("--model", default=None, help="Override the LLM used for repair")
    parser_revise.add_argument("--limit", type=int, default=None,
                               help="Process at most N files per directory (for smoke tests)")
    parser_revise.add_argument("--report", default=str(DATA_DIR / 'revision_report.json'),
                               help="Where to write the revision statistics")

    # Kept so existing invocations do not break; it now delegates to `revise`.
    parser_curation = subparsers.add_parser("validate-manual-curation",
                                            help="Deprecated alias for `revise`")
    parser_curation.add_argument("--input-dir", default=None, help="Ignored; see `revise`")
    parser_curation.add_argument("--output-dir", default=None, help="Ignored; see `revise`")
    parser_curation.add_argument("--limit", type=int, default=None, help="Process at most N files")

    # 11c. Extract and Repair parser
    parser_repair = subparsers.add_parser("extract-repair", help="Validate and repair LLM extraction JSON records")
    parser_repair.add_argument("--input-dir", default=str(DATA_DIR / 'literature_data_json_revised'), help="Folder containing revised JSON extraction records (step 18 output)")
    # The PUBLISHED-record schema, not the extraction prompt's response format. Those are
    # different shapes -- see strict_schema.published_json_schema.
    parser_repair.add_argument("--schema", default=str(PROMPTS_DIR / 'extraction_record_schema.json'), help="JSON schema for a published extraction record")
    parser_repair.add_argument("--output-dir", default=None, help="Folder where repaired JSONs are saved")
    parser_repair.add_argument("--log-dir", default=str(LOGS_DIR), help="Folder where repair log files are saved")

    # 11b. Merge Extractions parser
    parser_merge_extractions = subparsers.add_parser("merge-extractions", help="Merge antimicrobial database peptides with extracted peptides")
    parser_merge_extractions.add_argument("--consensus", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to consensus CSV")
    parser_merge_extractions.add_argument("--literature-dir", default=str(DATA_DIR / 'literature_data_json_repaired'), help="Path to literature JSON extractions folder")
    parser_merge_extractions.add_argument("--patent-dir", default=str(DATA_DIR / 'patent_data_json_repaired'), help="Path to patent JSON extractions folder")
    parser_merge_extractions.add_argument("--output", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to output combined CSV")
    parser_merge_extractions.add_argument("--database-dir", default=str(DATA_DIR / 'antimicrobial_databases'), help="Raw source-database directory, read for structured activity + publication data")
    parser_merge_extractions.add_argument("--no-activity-table", action="store_true", help="Skip the long-form activities.csv (wide consensus CSV only)")

    # 12. Serialize parser
    parser_serialize = subparsers.add_parser("serialize", help="Serialize raw consensus datasets into multiple formats")
    parser_serialize.add_argument("--consensus", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to consensus CSV")
    parser_serialize.add_argument("--output-dir", default=str(DATA_DIR / 'sample_raw_release'), help="Directory to save serialized outputs")

    # 13. Enrich parser
    # Split of the monolithic `enrich` command's first phase into its own step: RDKit/peptides.py
    # descriptors + stability features from the raw consensus. Kept fast and dependency-free on
    # purpose (no external tool), so it is the one enrichment step that essentially never fails --
    # unlike the five predictor steps below it, it does not need `only_missing`/resumability.
    parser_enrich_base = subparsers.add_parser("enrich-base", help="Compute RDKit/peptides.py descriptors and stability features from the raw consensus, creating the enriched dataset (first step of the enrichment pipeline)")
    parser_enrich_base.add_argument("--input", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to raw consensus CSV")
    parser_enrich_base.add_argument("--output-dir", default=str(DATA_DIR / 'sample_enriched_release'), help="Directory to save the enriched dataset")

    parser_enrich = subparsers.add_parser("enrich", help="Enrich consensus dataset with descriptors, embeddings, and 3D structures")
    parser_enrich.add_argument("--input", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to raw consensus CSV")
    parser_enrich.add_argument("--output-dir", default=str(DATA_DIR / 'sample_enriched_release'), help="Directory to save enriched outputs")
    parser_enrich.add_argument("--skip-3d", action="store_true", help="Skip computation of 3D molecular structures")
    parser_enrich.add_argument("--skip-predictors", action="store_true", help="Skip peptides.py/BioPython descriptor+stability enrichment and any installed external predictors (Macrel, HemoPI2)")
    parser_enrich.add_argument("--skip-external-predictors", action="store_true", help="Still compute peptides.py/BioPython descriptors, but do not shell out to Macrel/HemoPI2 even if installed")
    parser_enrich.add_argument("--skip-folding", action="store_true", help="Skip ColabFold structure prediction and use distance-geometry embedding of the stored SMILES instead")
    parser_enrich.add_argument("--fold-limit", type=int, default=None, help="Predict at most this many new structures in this run (folding is resumable, so a bounded run is safe)")
    parser_enrich.add_argument("--fold-models", type=int, default=1, help="AlphaFold2 models per sequence (ColabFold's default is 5; 1 is a fifth of the runtime)")
    parser_enrich.add_argument("--fold-recycle", type=int, default=3, help="Recycles through the network per model")

    # 13b. Single-predictor re-enrichment parsers -- one per external tool in
    # sample.peptide_predictors.EXTERNAL_PREDICTORS, so a single tool that failed partway through
    # the enrichment stage can be redone on its own, over the already-enriched dataset, without
    # recomputing descriptors or any OTHER predictor's results. Also each its own numbered
    # pipeline step. Resumable by default (--all forces a full recompute).
    #
    # AMPlify has no entry here (and no pipeline step) -- see the comment on
    # sample.peptide_predictors.EXTERNAL_PREDICTORS for why it was retired 2026-08-26.
    _PREDICTOR_CLI_COMMANDS = {
        "enrich-macrel": "macrel", "enrich-anticp2": "anticp",
        "enrich-hemopi2": "hemopi2", "enrich-algpred2": "algpred2",
    }
    for _cli_name, _predictor_name in _PREDICTOR_CLI_COMMANDS.items():
        _p = subparsers.add_parser(
            _cli_name,
            help=f"Re-run only the {_predictor_name} predictor over an already-enriched dataset, "
                 f"filling in peptides missing a result (pass --all to recompute every peptide)")
        _p.add_argument("--dataset", default=str(DATA_DIR / 'sample_enriched_release/enriched.csv'),
                        help="Enriched dataset to read AND write in place -- every other column "
                             "(descriptors, other predictors, embeddings) is left untouched")
        _p.add_argument("--all", action="store_true",
                        help="Recompute every peptide, not just ones missing a result")

    # Cheap amino-acid composition/frequency embedding columns (20-dim group ratios) -- NOT the
    # ESM-2 protein-language-model embeddings, which are `enrich-esm2`/`sample.esm_enrichment` and
    # a much heavier, separate thing. Named explicitly to avoid the two being confused.
    parser_enrich_freq = subparsers.add_parser("enrich-frequency-embeddings", help="Add amino-acid composition/frequency embedding columns to the enriched dataset (not ESM-2 -- see enrich-esm2 for that)")
    parser_enrich_freq.add_argument("--dataset", default=str(DATA_DIR / 'sample_enriched_release/enriched.csv'), help="Enriched dataset to read AND write in place")

    # RDKit distance-geometry 3D structures for short peptides ColabFold did not reach -- part two
    # of the structures step. Run `fold` first; this only fills entries `fold` left without one.
    parser_struct_fallback = subparsers.add_parser("enrich-structures-fallback", help="RDKit distance-geometry 3D structures for short peptides ColabFold didn't reach (run `fold` first)")
    parser_struct_fallback.add_argument("--dataset", default=str(DATA_DIR / 'sample_enriched_release/enriched.csv'), help="Enriched dataset carrying entry_id and sequence")
    parser_struct_fallback.add_argument("--structure-dir", default=str(DATA_DIR / 'sample_enriched_release/structures'), help="Directory for predicted/embedded structures")

    # 14. Decoy generation parser
    parser_decoys = subparsers.add_parser("decoys", help="Generate decoy peptide datasets (negative samples)")
    parser_decoys.add_argument("--positives", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Path to positives baseline CSV")
    parser_decoys.add_argument("--output", default=str(DATA_DIR / 'sample_decoy_databases/decoys.csv'), help="Path to save decoy dataset CSV")
    parser_decoys.add_argument("--ratio", type=float, default=1.0, help="Decoy to positive sequence ratio")
    parser_decoys.add_argument("--method", default="combined", choices=["shuffling", "uniprot", "amp_like", "combined"], help="Decoy generation method. 'amp_like' generates only the adversarial hard negatives (non-AMP by UniProt annotation, AMP-like in charge/hydrophobicity); 'combined' mixes all three sources per sample.decoys.DECOY_METHOD_MIX")

    # 15. Assembly parser
    parser_assembly = subparsers.add_parser("assembly", help="Assemble final dataset combining positives, decoys, and descriptors")
    parser_assembly.add_argument("--positives", default=str(DATA_DIR / 'sample_enriched_release/enriched.csv'), help="Path to enriched positives CSV")
    parser_assembly.add_argument("--decoys", default=str(DATA_DIR / 'sample_decoy_databases/decoys.csv'), help="Path to decoys CSV")
    parser_assembly.add_argument("--output", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to output final dataset CSV")

    # 16. Update final dataset parser
    parser_update = subparsers.add_parser("update-dataset", help="Update the final assembled dataset with new records")
    parser_update.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to current master dataset CSV")
    parser_update.add_argument("--new-data", required=True, help="Path to new data CSV to merge")
    parser_update.add_argument("--output", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to output updated CSV")

    # 17. Load SQL database parser
    parser_sql = subparsers.add_parser("load-sql", help="Load final assembled dataset into SQLite database")
    parser_sql.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to final dataset CSV")
    parser_sql.add_argument("--db", default=str(DATA_DIR / 'sample.db'), help="Path to SQLite database file")

    # 17b. Bibliography parser
    parser_bib = subparsers.add_parser("build-references", help="Harvest and resolve the bibliography from every source database")
    parser_bib.add_argument("--output", default=str(DATA_DIR / 'references.json'), help="Path to write the resolved bibliography JSON")
    parser_bib.add_argument("--sources", nargs="*", default=None, help="Restrict to these harvesters (DBAASP DRAMP ADP CAMP SAMPLE)")
    parser_bib.add_argument("--offline", action="store_true", help="Harvest local metadata only; skip CrossRef and NCBI lookups")
    parser_bib.add_argument("--no-title-search", action="store_true", help="Skip the CrossRef title search for references carrying no identifier")
    parser_bib.add_argument("--resolve-unlinked", action="store_true", help="Also resolve references no peptide or activity cites. Adds the whole downloaded corpus -- 83,634 extra CrossRef lookups, hours of traffic -- for records no page can display")

    # 17d. Chemical modifications parser
    parser_mods = subparsers.add_parser("build-modifications", help="Harvest terminal and residue chemical modifications (amidation, acylation, unusual residues)")
    parser_mods.add_argument("--output", default=str(DATA_DIR / 'modifications.json'), help="Path to write the harvested modifications JSON")
    parser_mods.add_argument("--dbaasp-dir", default=None, help="DBAASP JSON directory; the packaged location when omitted")

    # 17e. Source organism parser
    parser_org = subparsers.add_parser("build-source-organisms", help="Harvest each peptide's source organism, or Synthetic for designed sequences")
    parser_org.add_argument("--output", default=str(DATA_DIR / 'source_organisms.json'), help="Path to write the harvested origins JSON")
    parser_org.add_argument("--dbaasp-dir", default=None, help="DBAASP JSON directory; the packaged location when omitted")

    # 17f. Cross-reference parser
    parser_xref = subparsers.add_parser("build-cross-references", help="Harvest UniProt/PDB accessions per peptide from the resolved activity table")
    parser_xref.add_argument("--output", default=str(DATA_DIR / 'cross_references.json'), help="Path to write the harvested cross-references JSON")
    parser_xref.add_argument("--activities-csv", default=None, help="Wide activity table; the packaged release location when omitted")

    # 18b. SMILES validation parser
    parser_smiles = subparsers.add_parser("validate-smiles", help="Replace the published SMILES column with sequence-derived structures where available, and null out unparseable values")
    parser_smiles.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Dataset CSV to validate")
    parser_smiles.add_argument("--output", default=None, help="Where to write the result; overwrites --dataset when omitted")
    parser_smiles.add_argument("--predicted-structures", default=None, help="predicted_structures.json; the packaged location when omitted")

    # 22b. Family HMM parser
    parser_famhmm = subparsers.add_parser("build-family-hmms", help="Align SAMPLE-derived family clusters with MAFFT and build a profile HMM per cluster with hmmbuild")
    parser_famhmm.add_argument("--families", default=None, help="families.json; the packaged location when omitted")
    parser_famhmm.add_argument("--dataset", default=None, help="Dataset carrying entry_id/sequence; final_dataset.csv when omitted")
    parser_famhmm.add_argument("--output-dir", default=None, help="Directory for FASTA/alignment/HMM files; data/families/hmms when omitted")
    parser_famhmm.add_argument("--min-size", type=int, default=5, help="Minimum cluster size to build a profile for")
    parser_famhmm.add_argument("--limit", type=int, default=None, help="Build at most this many new HMMs this run (resumable)")
    parser_famhmm.add_argument("--no-resume", action="store_true", help="Rebuild every qualifying cluster even if its .hmm already exists")
    parser_famhmm.add_argument("--press", action="store_true", help="Also combine every built HMM into one file and hmmpress it for hmmscan")

    # 16b. Embedding index
    parser_emb = subparsers.add_parser("build-embedding-index", help="Compute ESM-2 embeddings for every real peptide and index them in Chroma for embedding search")
    parser_emb.add_argument("--dataset", default=None, help="Dataset carrying entry_id/sequence/is_decoy; final_dataset.csv when omitted")
    parser_emb.add_argument("--output-dir", default=None, help="Directory for the persistent Chroma index; data/embeddings/chroma when omitted")
    parser_emb.add_argument("--batch-size", type=int, default=250, help="Peptides embedded and added to the index per batch")
    parser_emb.add_argument("--limit", type=int, default=None, help="Index at most this many peptides (development/testing only)")

    # 17g. Gene Ontology parser
    parser_go = subparsers.add_parser("build-gene-ontology", help="Harvest Gene Ontology terms from CAMP's own scraped annotations")
    parser_go.add_argument("--output", default=str(DATA_DIR / 'gene_ontology.json'), help="Path to write the harvested GO terms JSON")
    parser_go.add_argument("--camp-dir", default=None, help="CAMP CSV directory; the packaged location when omitted")

    # 17h. Gene Ontology .obo slice
    parser_goobo = subparsers.add_parser("build-go-obo", help="Download go-basic.obo and slice it down to the GO terms this project's peptides carry, plus their ancestors")
    parser_goobo.add_argument("--gene-ontology-json", default=None, help="Harvested GO terms; data/gene_ontology.json when omitted")
    parser_goobo.add_argument("--go-obo", default=None, help="Local go-basic.obo; downloaded to the packaged location when omitted")
    parser_goobo.add_argument("--output", default=None, help="Where to write the slice; data/gene_ontology/sample_go_slice.obo when omitted")
    parser_goobo.add_argument("--no-download", action="store_true", help="Fail rather than download go-basic.obo if it is not already present")
    parser_goobo.add_argument("--force-download", action="store_true", help="Re-download go-basic.obo even if it already exists")

    # 17c. Administrator accounts.
    #
    # Deliberately usable without the web application. An installation whose only admin forgot
    # their password, or which has no admin at all, cannot be repaired through a login form -- the
    # web interface refuses to delete or disable the last active account precisely so it cannot
    # create that state, which leaves the shell as the only way out of it.
    parser_admin = subparsers.add_parser("admin", help="Manage administrator accounts for the web interface")
    admin_sub = parser_admin.add_subparsers(dest="admin_command", required=True)

    admin_add = admin_sub.add_parser("create", help="Create an administrator")
    admin_add.add_argument("username")
    admin_add.add_argument("--password", default=None, help="Password; prompted for without echo when omitted, which also keeps it out of the shell history")

    admin_pw = admin_sub.add_parser("passwd", help="Reset an administrator's password and revoke their sessions")
    admin_pw.add_argument("username")
    admin_pw.add_argument("--password", default=None, help="New password; prompted for without echo when omitted")

    admin_sub.add_parser("list", help="List administrators")

    admin_rm = admin_sub.add_parser("remove", help="Delete an administrator")
    admin_rm.add_argument("username")

    admin_en = admin_sub.add_parser("enable", help="Re-enable a suspended administrator")
    admin_en.add_argument("username")

    admin_dis = admin_sub.add_parser("disable", help="Suspend an administrator without deleting them")
    admin_dis.add_argument("username")

    # 13b. Structure prediction parser
    parser_fold = subparsers.add_parser("fold", help="Predict peptide structures with ColabFold and derive SDF + SMILES from them")
    parser_fold.add_argument("--input", default=str(DATA_DIR / 'sample_enriched_release/enriched.csv'), help="CSV carrying entry_id and sequence")
    parser_fold.add_argument("--structure-dir", default=str(DATA_DIR / 'sample_enriched_release/structures'), help="Directory for predicted structures and their SDF")
    parser_fold.add_argument("--output", default=str(DATA_DIR / 'predicted_structures.json'), help="Path to write the per-peptide record")
    parser_fold.add_argument("--limit", type=int, default=None, help="Predict at most this many new structures (resumable, so a bounded run is safe)")
    parser_fold.add_argument("--models", type=int, default=1, help="AlphaFold2 models per sequence")
    parser_fold.add_argument("--recycle", type=int, default=3, help="Recycles through the network per model")
    parser_fold.add_argument("--batch-size", type=int, default=50, help="Sequences per ColabFold invocation")

    # 17b1. Peptide topology parser
    parser_topo = subparsers.add_parser("build-topology", help="Harvest peptide topology (cyclic backbones, disulfide bridges) from the source databases")
    parser_topo.add_argument("--output", default=str(DATA_DIR / 'peptide_topology.json'), help="Path to write the topology")

    parser_ann = subparsers.add_parser("build-annotations", help="Pre-compute per-peptide annotations for all release peptides: external predictors (macrel/hemopi2/anticp2/algpred2; toxinpred3 has no working adapter, amplify was retired 2026-08-26 for a sustained upstream crash rate), Pfam domains, SAMPLE family/cluster, near-identity hits and family-HMM hits. Circular peptides handled explicitly.")
    parser_ann.add_argument("--db", default=None, help="Release database to read peptides from; the project default when omitted")
    parser_ann.add_argument("--store", default=None, help="Annotation database to write; data/peptide_annotations.sqlite when omitted")
    parser_ann.add_argument("--limit", type=int, default=0, help="Annotate at most this many NEW peptides this run (0 = all remaining; resumable, so run in slices)")
    parser_ann.add_argument("--no-resume", action="store_true", help="Re-annotate every peptide, ignoring which are already stored")
    parser_ann.add_argument("--no-predictors", action="store_true", help="Skip the external-predictor batch (annotate from families/hits/HMM only)")

    parser_pdbh = subparsers.add_parser("build-pdb-homology", help="Search RCSB for the closest real, experimentally-solved PDB structure to each folded peptide and superpose the ColabFold prediction onto it (CA-RMSD). Resumable, one HTTP round-trip per peptide -- run in slices.")
    parser_pdbh.add_argument("--structures-dir", default=None, help="Where ColabFold PDBs live; the release default when omitted")
    parser_pdbh.add_argument("--db", default=None, help="Release database to read peptide sequences from; the project default when omitted")
    parser_pdbh.add_argument("--store", default=None, help="Homology results database to write; data/pdb_homology.sqlite when omitted")
    parser_pdbh.add_argument("--limit", type=int, default=0, help="Search at most this many NEW peptides this run (0 = all remaining; resumable, so run in slices)")
    parser_pdbh.add_argument("--no-resume", action="store_true", help="Re-search every folded peptide, ignoring which are already stored")
    parser_pdbh.add_argument("--min-identity", type=float, default=0.7, help="Identity floor (0-1) for an RCSB hit to be worth fetching and superposing at all")
    parser_pdbh.add_argument("--sleep", type=float, default=0.5, help="Delay (seconds) between peptides, to stay polite to RCSB's public API")

    # 17b2. Peptide names parser
    parser_names = subparsers.add_parser("build-names", help="Harvest peptide names and synonyms from every source database")
    parser_names.add_argument("--output", default=str(DATA_DIR / 'peptide_names.json'), help="Path to write the harvested names")
    parser_names.add_argument("--vocabulary-threshold", type=int, default=5, help="Parenthetical tokens attached to this many distinct peptides are classification vocabulary, not synonyms")

    # 17c. Families parser
    parser_fam = subparsers.add_parser("build-families", help="Assign peptide families: Pfam anchors, then PepMatch clusters for the remainder")
    parser_fam.add_argument("--fasta", default=str(DATA_DIR / 'pfam/peptides.fasta'), help="Peptide FASTA")
    parser_fam.add_argument("--domtbl", default=str(DATA_DIR / 'pfam/pfam_hits.domtbl'), help="hmmsearch --domtblout table against Pfam-A")
    parser_fam.add_argument("--output", default=str(DATA_DIR / 'families/families.json'), help="Path to write family assignments")
    parser_fam.add_argument("--identity", type=float, default=0.8, help="Identity floor for clustering the Pfam-unmatched remainder")

    parser_upfam = subparsers.add_parser("update-families", help="Incrementally place NEW peptides into the existing family map (inherit or mint a new cluster) without renumbering existing families")
    parser_upfam.add_argument("--new", required=True, help="FASTA (or CSV with entry_id/sequence columns) of the new peptides to place")
    parser_upfam.add_argument("--families", default=str(DATA_DIR / 'families/families.json'), help="families.json to read and update")
    parser_upfam.add_argument("--db", default=None, help="Release database to read the existing corpus from; the project default when omitted")
    parser_upfam.add_argument("--identity", type=float, default=0.8, help="Identity floor for inheriting an existing family")
    parser_upfam.add_argument("--dry-run", action="store_true", help="Report the assignment without writing families.json")

    parser_incupd = subparsers.add_parser("incremental-update", help="Acquisition driver's last step: diff step 20's consensus CSV against the live release database, and place whatever is new (see 'make incremental-update' for the full chain)")
    parser_incupd.add_argument("--consensus", default=str(DATA_DIR / 'sample_raw_release/merged_from_database.csv'), help="Step 20's consensus CSV")
    parser_incupd.add_argument("--families", default=str(DATA_DIR / 'families/families.json'), help="families.json to read and update")
    parser_incupd.add_argument("--db", default=None, help="Release database to diff against and read the existing corpus from; the project default when omitted")
    parser_incupd.add_argument("--dry-run", action="store_true", help="Report the new peptides and their assignment without writing families.json")

    parser_gctx = subparsers.add_parser("genomic-context", help="Search RefSeq genomes for a peptide's gene with tBLASTn (95%% cov+identity), organise hits as a Krona-style taxonomy plot. Downloads genomes: a large, gated run.")
    parser_gctx.add_argument("--peptide", required=True, help="The peptide sequence to search for (one-letter code)")
    parser_gctx.add_argument("--peptide-id", default="query", help="Label for the peptide in the output")
    parser_gctx.add_argument("--assembly-summary", required=True, help="Path to NCBI assembly_summary.txt (reference/representative genomes)")
    parser_gctx.add_argument("--limit", type=int, default=0, help="Search at most this many genomes (0 = all reference/representative; use a small value first)")
    parser_gctx.add_argument("--work-dir", default=None, help="Scratch directory for downloads/BLAST; a temp dir when omitted")
    parser_gctx.add_argument("--output", default=None, help="Path to write the hits + Krona JSON")
    parser_gctx.add_argument("--svg", default=None, help="Path to write the Krona-style sunburst SVG")

    parser_sweep = subparsers.add_parser("genomic-context-sweep", help="RefSeq genomic-context sweep for a peptide: one genome at a time -> tBLASTn -> write hits to the database -> delete the genome. Resumable. Downloads genomes: a large, gated run.")
    parser_sweep.add_argument("--peptide", required=True, help="The peptide sequence to search for (one-letter code)")
    parser_sweep.add_argument("--peptide-id", required=True, help="The peptide's id -- the key the hits are stored and read back under (use the release entry_id so the peptide page finds them)")
    parser_sweep.add_argument("--assembly-summary", required=True, help="Path to NCBI assembly_summary.txt (reference/representative genomes)")
    parser_sweep.add_argument("--limit", type=int, default=0, help="Search at most this many NEW genomes this run (0 = all remaining; run in slices)")
    parser_sweep.add_argument("--store", default=None, help="Results database; data/genomic_context.sqlite when omitted")
    parser_sweep.add_argument("--work-dir", default=None, help="Scratch directory (one genome + BLAST db live here transiently); a temp dir when omitted")
    parser_sweep.add_argument("--no-resume", action="store_true", help="Re-search every genome, ignoring which are already recorded")

    # 17d. Synergy parser
    parser_syn = subparsers.add_parser("train-synergy", help="Train per-pathogen peptide-antibiotic synergy models from DBAASP FICI data")
    parser_syn.add_argument("--model-dir", default=str(DATA_DIR / 'sample_final_release/models_retired'), help="Directory to write models into; defaults to the retired tree, which the web server does not serve")
    parser_syn.add_argument("--report", default=str(DATA_DIR / 'sample_final_release/synergy_report.json'), help="Path to write the metrics report")
    parser_syn.add_argument("--min-rows", type=int, default=80, help="Minimum measurements before a pathogen gets its own model")

    # 17e. Bioactivity linkage parser
    parser_bioact = subparsers.add_parser("link-bioactivity", help="Link peptides to ChEMBL, PubChem BioAssay and BindingDB")
    parser_bioact.add_argument("--output", default=str(DATA_DIR / 'bioactivity.json'), help="Path to write the linkage")
    parser_bioact.add_argument("--sources", nargs="*", default=None, help="Restrict to chembl, pubchem and/or bindingdb")
    parser_bioact.add_argument("--limit", type=int, default=None, help="Process at most this many identifiers per archive")

    # 18. Train ML models parser -- the per-task, per-feature-block matrix trainer
    # (sample.ml_models.train_all): clustering-based grouped splits, CV-based model selection,
    # holdout as the final word, y-randomization, applicability domain, and a normality-informed
    # significance test against the null. See sample/ml_models.py's module docstring.
    parser_train = subparsers.add_parser("train-ml", help="Train the per-task, per-feature-block model matrix on the final peptide dataset")
    parser_train.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to dataset CSV")
    parser_train.add_argument("--model-dir", default=str(DATA_DIR / 'sample_final_release/models'), help="Directory to write namespaced model pickles into")
    parser_train.add_argument("--report", default=str(DATA_DIR / 'sample_final_release/ml_report.json'), help="Path to save the aggregate report JSON")
    parser_train.add_argument("--activities", default=str(DATA_DIR / 'sample_raw_release/activities.csv'), help="Step 20 activity table supplying per-class/assay/organism/cell-line labels")
    parser_train.add_argument("--filter-anomalies", action="store_true", help="Filter out detected anomalies prior to training")
    parser_train.add_argument("--tasks-limit", type=int, default=None, help="Train only the first N tasks; for a smoke test rather than a full release run")
    parser_train.add_argument("--min-task-examples", type=int, default=None, help="Minimum positives for a general/activity-group/assay/cancer-type task; sample.ml_models.MIN_POSITIVES_PER_TASK (60) when omitted")
    parser_train.add_argument("--min-target-examples", type=int, default=None, help="Minimum positives for an organism- or cell-line-specific task; 2x --min-task-examples when omitted")
    parser_train.add_argument("--families-json", default=None, help="Curated family assignment for grouping the train/holdout split (Pfam family, or step 36's own near-identity cluster, falling back to a fresh near-identity pass for whatever it doesn't cover); data/families/families.json when omitted")
    parser_train.add_argument("--fixed-hyperparameters", action="store_true", help="Use the fixed-hyperparameter XGBoost baseline (sample.ml_models._make_model, 5-fold OOF calibration, never searched) instead of the FLAML-searched estimator this command uses by default. Fast: no per-cell search, so a full-matrix run stays on the order of the old runtime rather than up to 600s x 2 searches per (task, block) cell.")

    # 18c-e. FLAML-tuned XGBoost (sample.flaml_xgboost) on the combined descriptors+composition
    # block, for the tasks that already have a kept train-ml model, split into three commands
    # rather than one so antibacterial/anticancer -- the two tasks with a registered
    # external-tool comparison -- can be trained and benchmarked first, without waiting on the
    # full matrix. ESM-2 (both the frozen-embedding+XGBoost and a full end-to-end fine-tune, tried
    # earlier) is deliberately not used here: this project's own comparison found it ahead on
    # ROC-AUC/F1/MCC but consistently behind on recall. Namespaces are resolved directly from
    # build_tasks() at run time (peptide_type/scope/task filters below), NOT read from a prior
    # train-ml report -- the simplified pipeline (steps 46-47) has no full-matrix step producing
    # one any more. Both commands share one progress file (data/sample_final_release/
    # flaml_rollout_progress.json by default), so running them in any order or combination never
    # re-trains a namespace twice; see sample.flaml_xgboost.train_flaml_tasks's docstring for the
    # actual training behavior.
    def _add_finetune_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to dataset CSV")
        p.add_argument("--model-dir", default=str(DATA_DIR / 'sample_final_release/models'), help="Directory to write the 'flaml' model pickle into")
        p.add_argument("--activities", default=str(DATA_DIR / 'sample_raw_release/activities.csv'), help="Step 20 activity table")
        p.add_argument("--families-json", default=str(DATA_DIR / 'families/families.json'), help="Curated family assignment for the train/holdout grouping")
        p.add_argument("--progress", default=None, help="Shared progress file; data/sample_final_release/flaml_rollout_progress.json when omitted")
        p.add_argument("--resume", action="store_true", help="Skip a namespace already recorded as done in --progress instead of retraining it. Off by default: every requested namespace is retrained (overwritten), even one already trained in a prior run.")
        return p

    _add_finetune_parser("train-antibacterial-ml-model", "FLAML-tuned XGBoost for the general antibacterial/general/any/classification task -- step 46")
    _add_finetune_parser("train-organism-specific-ml-models", "FLAML-tuned XGBoost for every organism-specific antimicrobial classification task (general/organism_specific/<organism>/<assay>/classification) -- step 47")

    parser_admet = subparsers.add_parser("admet", help="ADMET-lite summary for a peptide (sequence-derived physicochemistry + rule flags; permeability/toxicity deferred)")
    parser_admet.add_argument("sequence", help="Peptide sequence, one-letter code")
    parser_admet.add_argument("--json", action="store_true", help="Print the full profile as JSON")

    # 19. Validate ML models parser -- train-ml already validates each model as part of training
    # (holdout, CV, y-randomization, applicability domain all live in ml_report.json); this reads
    # that report back and prints a pass/fail summary rather than re-running a separate validation.
    parser_validate_ml = subparsers.add_parser("validate-ml", help="Summarize the quality-gate outcome recorded in train-ml's report")
    parser_validate_ml.add_argument("--report", default=str(DATA_DIR / 'sample_final_release/ml_report.json'), help="Path to train-ml's report JSON")

    parser_benchmark = subparsers.add_parser("benchmark", help="Benchmark a trained model against an external reference tool (e.g. HemoPI2) on the model's own holdout")
    parser_benchmark.add_argument("--model-id", required=True, help="Registry model id, e.g. hemolytic/general/any/classification/descriptors")
    parser_benchmark.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Assembled dataset the model was trained on")
    parser_benchmark.add_argument("--activities", default=str(DATA_DIR / 'sample_raw_release/activities.csv'), help="Step-14 activity table supplying the true labels")
    parser_benchmark.add_argument("--model-dir", default=None, help="Model namespace root; the registry default when omitted")
    parser_benchmark.add_argument("--report", default=None, help="Optional path to write the comparison JSON")

    parser_benchmark_all = subparsers.add_parser("benchmark-all", help="Benchmark every trained model with a registered reference tool (Macrel/AMPlify for antibacterial, AntiCP2 for anticancer, HemoPI2 for hemolytic) on each one's own holdout")
    parser_benchmark_all.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Assembled dataset the models were trained on")
    parser_benchmark_all.add_argument("--activities", default=str(DATA_DIR / 'sample_raw_release/activities.csv'), help="Step-14 activity table supplying the true labels")
    parser_benchmark_all.add_argument("--model-dir", default=None, help="Model namespace root; the registry default when omitted")
    parser_benchmark_all.add_argument("--report", default=str(DATA_DIR / 'sample_final_release/benchmark_report.json'), help="Path to write the combined comparison JSON")

    # 20. Exploratory Data Analysis parser
    parser_eda = subparsers.add_parser("eda", help="Run exploratory data analysis on the peptide dataset")
    parser_eda.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to dataset CSV")
    parser_eda.add_argument("--output-dir", default="sample/static/img/eda", help="Directory to save EDA plots")
    parser_eda.add_argument("--sample-size", type=int, default=500, help="Sample size for UMAP dimension reductions")

    # 21. Anomaly Detection parser
    def _contamination(value: str) -> Any:
        """Accepts IsolationForest's own contamination type: the literal 'auto', or a float."""
        return value if value == "auto" else float(value)

    parser_revalidate = subparsers.add_parser(
        "revalidate-resolutions",
        help="Step 10: re-check every sequence that synonym resolution applied to an extraction "
             "record, and withdraw the ones that no longer qualify. Idempotent -- a record whose "
             "resolutions all still hold is not rewritten")
    parser_revalidate.add_argument(
        "--dirs", nargs="*", default=None,
        help="Extraction output directories; data/literature_data_json and data/patent_data_json "
             "when omitted")
    parser_revalidate.add_argument("--dry-run", action="store_true",
                                   help="Report what would be withdrawn without writing")
    parser_revalidate.add_argument("--limit", type=int, default=None,
                                   help="Stop after this many records (for a cheap trial)")

    parser_anomaly = subparsers.add_parser("anomaly-detection", help="Detect anomalous peptide records using unsupervised learning (Isolation Forest)")
    parser_anomaly.add_argument("--dataset", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to dataset CSV")
    parser_anomaly.add_argument("--output", default=str(DATA_DIR / 'sample_final_release/final_dataset.csv'), help="Path to save updated dataset CSV")
    parser_anomaly.add_argument("--report", default=str(DATA_DIR / 'sample_final_release/anomaly_report.json'), help="Path to save anomaly report JSON")
    parser_anomaly.add_argument("--contamination", type=_contamination, default="auto",
                               help="IsolationForest contamination: 'auto' (scikit-learn's own default) or a fraction in (0, 0.5]")
    parser_anomaly.add_argument("--esm-components", type=int, default=0, metavar="N",
                               help="Reduce the ESM-2 embedding block to N principal components "
                                    "before fitting, so its 320 raw dimensions stop taking ~90%% "
                                    "of the forest's random split choices. 0 (default) fits the "
                                    "raw dimensions, as every release to date did.")

    # 22. Webserver parser
    parser_webserver = subparsers.add_parser("webserver", help="Start the sAMPle platform web server dashboard")
    parser_webserver.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser_webserver.add_argument("--port", type=int, default=8000, help="Binding port number")

    # 23. ML Worker parser
    parser_worker = subparsers.add_parser("worker", help="Start the sAMPle ML background prediction worker")


    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    if args.command == "pubmed":
        from sample.metadata import download_pubmed_metadata
        download_pubmed_metadata(args.query_file, args.email, args.output)
        
    elif args.command == "scopus":
        from sample.metadata import download_scopus_metadata
        api_key = os.environ.get("X-ELS-APIKEY")
        inst_token = os.environ.get("X-ELS-INST")
        download_scopus_metadata(args.query_file, api_key, inst_token, args.output)
        
    elif args.command == "download-pdfs":
        from sample.pdf import download_papers_multiprocess
        routers_env = os.environ.get("ROUTER")
        routers = [routers_env] if routers_env else []
        download_papers_multiprocess(args.source, args.email, routers, args.workers,
                                     max_attempts=args.max_attempts, limit=args.limit)
        
    elif args.command == "google-patents":
        from sample.patents import download_google_patents
        download_google_patents(args.csv, args.output_dir, args.workers)

        
    elif args.command == "import-anticancer":
        from sample.anticancer import (build_anticancer_table, download_anticp2,
                                       download_cancerppd, download_dctpep,
                                       scrape_cancerppd_details)
        if args.download:
            # Each source downloads independently: one unreachable host must not cost the others,
            # the same tolerance build_anticancer_table applies when loading them.
            for name, fetch in (("AntiCP2", download_anticp2), ("DCTPep", download_dctpep),
                                ("CancerPPD", download_cancerppd)):
                try:
                    fetch()
                except Exception as e:  # noqa: BLE001
                    logger.warning("%s not downloaded: %s", name, e)
        if args.scrape_cancerppd:
            scraped = scrape_cancerppd_details(delay_seconds=args.scrape_delay,
                                               limit=args.scrape_limit)
            logger.info("CancerPPD details: %d new, %d already had, %d failed, %d still "
                        "outstanding, %d known in total -> %s",
                        scraped["scraped"], scraped["already_scraped"], scraped["failed"],
                        scraped["remaining"], scraped["total"], scraped["path"])
        summary = build_anticancer_table(data_dir=args.data_dir, output_csv=args.output)
        logger.info("import-anticancer: %d unique sequences (%d anticancer positives) from %s -> %s",
                    summary["rows"], summary["positives"], summary["by_source"], summary["path"])

    elif args.command == "import-cycpeptmpdb":
        from sample.cycpeptmpdb import build_table, download, link_to_release
        if args.download:
            download()
        summary = build_table(csv_path=args.csv, output_csv=args.output)
        logger.info("import-cycpeptmpdb: %d cyclic peptides (%d with permeability) -> %s",
                    summary["rows"], summary["with_permeability"], summary["path"])
        if args.link:
            link = link_to_release(csv_path=args.output, db_path=args.db)
            logger.info("import-cycpeptmpdb: %d structural link(s) to %d release peptides -> %s",
                        link["links"], link["release_matched"], link["path"])

    elif args.command == "databases":
        from sample.databases import (
            download_adp_data, download_dbaasp_data, download_dramp_data,
            download_dbamp_data, download_camp_data, download_database_papers,
            detect_dramp_release, read_dramp_manifest
        )

        import json

        if args.dramp_status:
            # Deliberately returns without downloading: this is the provenance question ("which
            # release is this database built from?"), and answering it must never be able to
            # change the answer.
            dramp_dir = os.path.join(args.output_dir, "dramp")
            detected = detect_dramp_release(dramp_dir)
            manifest = read_dramp_manifest(dramp_dir)
            print(json.dumps({"directory": dramp_dir, "detected": detected,
                              "manifest_present": manifest is not None,
                              "manifest_generated": (manifest or {}).get("generated"),
                              "manifest_release": (manifest or {}).get("release")},
                             indent=2, ensure_ascii=False))
            return

        for target in args.targets:
            target_path = os.path.join(args.output_dir, target)
            if target == "adp":
                download_adp_data(target_path, args.workers)
            elif target == "dbaasp":
                download_dbaasp_data(target_path, max_workers=args.workers)
            elif target == "dramp":
                download_dramp_data(target_path, refresh=args.refresh)
            elif target == "dbamp":
                download_dbamp_data(target_path, args.workers)
            elif target == "camp":
                download_camp_data(target_path, num_pages=1000, max_workers=args.workers)
                
        if args.download_papers:
            email = os.environ.get("EMAIL", "1_1@gmail.com")
            routers_env = os.environ.get("ROUTER")
            routers = [routers_env] if routers_env else []
            for target in args.targets:
                download_database_papers(target, email, routers, max_workers=4)
                
    elif args.command == "merge":
        from sample.merge import merge_databases
        merge_databases(args.output, args.base_dir)
        
    elif args.command == "extract":
        os.makedirs("data/repaired_logs", exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler("data/repaired_logs/step_08_screening_extract_peptide_data.log")
        file_handler.setFormatter(log_formatter)
        logging.root.addHandler(file_handler)
        # Also print to the console -- this used to only go to the log file, which made a long
        # extraction run look hung (nothing printed for minutes) even while it was working fine.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logging.root.addHandler(console_handler)

        raw_sources = args.sources
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]

        db_base = "data/antimicrobial_databases/papers"
        db_subdirs = []
        if os.path.exists(db_base):
            db_subdirs = [d for d in os.listdir(db_base) if os.path.isdir(os.path.join(db_base, d))]
        if not db_subdirs:
            db_subdirs = ["adp", "dramp", "dbamp", "dbaasp", "camp"]

        expanded = []
        for s in raw_sources:
            if s == "all":
                expanded.extend(["pubmed", "scopus", "patents"] + db_subdirs)
            elif s in ["databases", "antimicrobial_databases"]:
                expanded.extend(db_subdirs)
            else:
                expanded.append(s)

        seen = set()
        sources_list = [x for x in expanded if not (x in seen or seen.add(x))]

        from sample.pipeline_runner import process_all_extractions
        for src in sources_list:
            if src in ["patents", "google_patents"]:
                actual_src = "google_patents"
                default_out = "data/patent_data_json/google_patents"
                default_text_out = "data/patent_data_text/google_patents"
            elif os.path.exists(f"data/antimicrobial_databases/papers/{src}") or src in db_subdirs:
                actual_src = src
                default_out = f"data/antimicrobial_databases/papers_json/{src}"
                default_text_out = f"data/antimicrobial_databases/papers_text/{src}"
            else:
                actual_src = src
                default_out = f"data/literature_data_json/{src}"
                default_text_out = f"data/literature_data_text/{src}"

            in_dir = args.input_dir or default_text_out
            out_dir = args.output_dir or default_out

            process_all_extractions(
                source=actual_src,
                input_dir=in_dir,
                output_dir=out_dir,
                limit=args.limit,
                workers=args.workers,
                resume=(args.resume_mode if args.resume_mode is not None else args.resume),
                chunk_size=args.chunk_size,
                document_budget=args.doc_timeout,
            )


    elif args.command == "review-fields":
        import json as _json

        from sample.field_review import FIELD_SPECS, describe_specs, group_by_model
        if args.list:
            print(describe_specs())
            print()
            for model, fields in group_by_model(tuple(FIELD_SPECS)).items():
                print(f"{model}: {', '.join(fields)}")
            return
        if args.show_prompt:
            if args.show_prompt not in FIELD_SPECS:
                raise SystemExit(f"unknown field {args.show_prompt!r}; "
                                 f"choose from {', '.join(FIELD_SPECS)}")
            spec = FIELD_SPECS[args.show_prompt]
            print(spec.system_prompt())
            print()
            print("JSON schema:")
            print(_json.dumps(spec.schema(), indent=2))
            print()
            print(f"evidence budget: {spec.evidence_chars} chars | "
                  f"max output: {spec.max_output_tokens} tokens | model: {spec.model}")
            print(f"rationale: {spec.model_rationale}")
            return
        if args.run:
            from sample.field_review_run import review_extractions
            review_extractions(base_dir=args.base_dir, dry_run=args.dry_run,
                               apply=args.apply, max_calls=args.max_calls,
                               limit=args.limit if args.limit != 20 else 0,
                               model=(args.model or [None])[0],
                               fields=tuple(args.field) if args.field else None)
            return
        if args.measure:
            from sample.field_review_bench import format_results, measure_field
            fields = args.field or list(FIELD_SPECS)
            payload = {}
            for name in fields:
                results = measure_field(name, models=tuple(args.model or ()),
                                        limit=args.limit, base_dir=args.base_dir,
                                        variants=tuple(args.variant or ()))
                payload[name] = results
                if not args.json:
                    print(format_results(name, results))
                    print()
            if args.json:
                print(_json.dumps(payload, indent=2))
            return
        raise SystemExit("choose one of --run, --list, --show-prompt FIELD or --measure")

    elif args.command == "apply-verdicts":
        import json as _json

        from sample.verdict_apply import run as apply_verdicts
        if not args.run and not args.dry_run:
            raise SystemExit("choose --run (write) or --dry-run (count only)")
        summary = apply_verdicts(base_dir=args.base_dir, verdict_file=args.verdict_file,
                                 dry_run=args.dry_run, limit=args.limit)
        if args.json:
            print(_json.dumps(summary, indent=2))
            return
        mode = "would change" if summary["dry_run"] else "changed"
        print(f"{summary['documents']:,} documents | "
              f"{summary['documents_changed']:,} {mode} | "
              f"{summary['changes']:,} corrections | {summary['refusals']:,} refused")
        for label, key in (("by field", "by_field"), ("by layer", "by_layer"),
                           ("by authority", "by_authority"), ("refused because", "by_refusal")):
            rows = sorted(summary[key].items(), key=lambda row: -row[1])
            if not rows:
                continue
            print(f"\n{label}:")
            for name, count in rows[:20]:
                print(f"  {count:8,}  {name}")
        if summary["skipped_changed_underneath"]:
            print(f"\n{summary['skipped_changed_underneath']:,} documents changed underneath "
                  f"this pass and were left alone")
        return

    elif args.command == "polish-extractions":
        from sample.polish import polish_extractions
        polish_extractions(base_dir=args.base_dir, dry_run=args.dry_run, limit=args.limit)

    elif args.command == "train-sequence-discriminator":
        from sample.sequence import train_name_discriminator
        written = train_name_discriminator(output_path=args.output,
                                           sequence_csv=args.sequences,
                                           wordlist=args.wordlist)
        if not written:
            raise SystemExit(1)

    elif args.command == "validate":
        from sample.validation import validate_extractions
        validate_extractions(
            literature_dir=args.lit_dir,
            patent_dir=args.patent_dir,
            merged_source_csv=args.baseline,
            report_path=args.report,
            run_llm=not args.report_only,
        )

    elif args.command == "validate-error-injection":
        from sample.validation_extras import validate_error_injection
        validate_error_injection(args.input_dir, args.report, args.samples)

    elif args.command == "select-curation-set":
        from sample.curation import (DEFAULT_POSITIVE_FRACTION, DEFAULT_SAMPLE_SIZES,
                                     build_curation_set, refresh_curation_set)
        if args.refresh:
            # Deliberately returns before any selection: --refresh exists precisely so a set that
            # someone is part-way through reviewing can gain the PDFs and the workbook without the
            # rebuild that would throw their work away.
            refreshed = refresh_curation_set(output_dir=args.output)
            logger.info("select-curation-set --refresh: %d documents in %s; PDFs %d attached, "
                        "%d already present, %d unavailable. Workbook: %s",
                        refreshed["documents"], refreshed["path"], refreshed["pdfs_attached"],
                        refreshed["pdfs_already_present"], refreshed["pdfs_unavailable"],
                        refreshed["review_workbook"])
            return
        # Per-corpus sizes: --papers/--patents win, then --sample-size for both, then the module
        # defaults (800/200). Built as a mapping rather than one number because the two corpora
        # are deliberately different sizes.
        sizes = dict(DEFAULT_SAMPLE_SIZES)
        if args.sample_size is not None:
            sizes = {corpus: args.sample_size for corpus in sizes}
        if args.papers is not None:
            sizes["papers"] = args.papers
        if args.patents is not None:
            sizes["patents"] = args.patents
        summary = build_curation_set(
            sample_size=sizes, output_dir=args.output, seed=args.seed,
            overwrite=not args.no_overwrite,
            positive_fraction=(DEFAULT_POSITIVE_FRACTION if args.positive_fraction is None
                               else args.positive_fraction),
            db_referenced_only=not args.no_db_filter)
        logger.info("select-curation-set: %d documents -> %s %s (%d positive / %d negative; %s); "
                    "%d without primary text.",
                    summary["selected"], summary["path"], summary["by_corpus"],
                    summary["positives"], summary["negatives"], summary["by_stratum"],
                    summary["missing_text"])

    elif args.command in ("revise", "validate-manual-curation"):
        import json as _json
        from sample.revision import run_revision
        if args.command == "validate-manual-curation":
            logger.warning(
                "`validate-manual-curation` is deprecated and now runs `revise`. The old command "
                "stamped curated=True / curator='Expert Curator' on records no human had seen; "
                "see sample/revision.py for the provenance this stage records instead."
            )
        stats = run_revision(
            sources=getattr(args, "sources", ["all"]),
            workers=getattr(args, "workers", 8),
            use_llm=not getattr(args, "no_llm", False),
            model=getattr(args, "model", None),
            limit=getattr(args, "limit", None),
        )
        report_path = getattr(args, "report", None) or str(DATA_DIR / "revision_report.json")
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            _json.dump(dict(sorted(stats.items())), fh, indent=2)
        logger.info(f"Revision report written to {report_path}")

    elif args.command == "extract-repair":
        from sample.validation import extract_repair_extractions
        extract_repair_extractions(
            input_dir=args.input_dir,
            schema_path=args.schema,
            output_dir=args.output_dir,
            log_dir=args.log_dir
        )

    elif args.command == "merge-extractions":
        from sample.merge import merge_database_and_extractions
        merge_database_and_extractions(
            consensus_csv=args.consensus,
            literature_dir=args.literature_dir,
            patent_dir=args.patent_dir,
            output_path=args.output,
            database_dir=args.database_dir,
            build_activity_table=not args.no_activity_table,
        )

    elif args.command == "enrich-base":
        from sample.enrichment import enrich_with_descriptors
        from sample.peptide_predictors import compute_descriptors_and_stability

        df = pd.read_csv(args.input, low_memory=False)
        df_desc = enrich_with_descriptors(df)
        df_desc = compute_descriptors_and_stability(df_desc)
        os.makedirs(args.output_dir, exist_ok=True)
        out_csv = os.path.join(args.output_dir, "enriched.csv")
        df_desc.to_csv(out_csv, index=False)
        logger.info(f"enrich-base: wrote descriptors for {len(df_desc)} peptides to {out_csv}")

    elif args.command == "enrich-frequency-embeddings":
        from sample.enrichment import enrich_with_embeddings

        df = pd.read_csv(args.dataset, low_memory=False)
        df_out = enrich_with_embeddings(df)
        df_out.to_csv(args.dataset, index=False)
        logger.info(f"enrich-frequency-embeddings: wrote embedding columns back to {args.dataset}")

    elif args.command == "enrich-structures-fallback":
        from sample.enrichment import enrich_with_2D_and_3D_structures
        from sample.folding import folded_entry_ids

        df = pd.read_csv(args.dataset, low_memory=False)
        enrich_with_2D_and_3D_structures(
            df, args.structure_dir, skip_entry_ids=folded_entry_ids(args.structure_dir))

    elif args.command in _PREDICTOR_CLI_COMMANDS:
        from sample.peptide_predictors import enrich_one_predictor

        predictor_name = _PREDICTOR_CLI_COMMANDS[args.command]
        df = pd.read_csv(args.dataset, low_memory=False)
        df_out = enrich_one_predictor(df, predictor_name, only_missing=not args.all)
        df_out.to_csv(args.dataset, index=False)
        logger.info(f"{args.command}: wrote {predictor_name}'s columns back to {args.dataset}")

    elif args.command == "serialize":
        from sample.serialize import serialize_raw_datasets
        serialize_raw_datasets(args.consensus, args.output_dir)

    elif args.command == "enrich":
        from sample.enrichment import enrich_with_descriptors, enrich_with_embeddings, enrich_with_2D_and_3D_structures
        df = pd.read_csv(args.input)
        df_desc = enrich_with_descriptors(df)

        # `enrich_with_predictors` was written and tested but had no caller, so its ~135
        # pep_*/stab_*/hemopi2_* columns reached no released file. It is sequence-only (no
        # dependency on the SMILES column, consistent with the enrichment steps' contract
        # elsewhere) and degrades gracefully: peptides.py/BioPython descriptors always run, and
        # Macrel/HemoPI2 contribute nothing rather than raising when not installed.
        if not args.skip_predictors:
            from sample.peptide_predictors import enrich_with_predictors, predictor_availability

            df_desc = enrich_with_predictors(
                df_desc, run_external=not args.skip_external_predictors)
            available = predictor_availability()
            logger.info(f"Predictor columns added. External predictors probed: "
                       f"{available or 'none installed'}.")

        df_emb = enrich_with_embeddings(df_desc)

        os.makedirs(args.output_dir, exist_ok=True)
        out_csv = os.path.join(args.output_dir, "enriched.csv")
        df_emb.to_csv(out_csv, index=False)
        logger.info(f"Saved enriched consensus sequence properties dataset to {out_csv}")
        
        if not args.skip_3d:
            structure_dir = os.path.join(args.output_dir, "structures")
            folded = 0

            # Structure policy, in order:
            #
            #   1. ColabFold, for everything it can usefully fold. It predicts from the SEQUENCE,
            #      the field this dataset is confident about, and the SDF and SMILES are read off
            #      the prediction. Predictions that fail geometric validation are discarded.
            #   2. RDKit distance geometry with force-field minimisation, for anything still
            #      without a structure that is short enough for it to mean something (<= 15
            #      residues). At that length there is no tertiary structure to predict; the
            #      conformation is local geometry, which ETKDG plus MMFF94 models well.
            #   3. Skip. A long sequence with no predicted fold gets no structure rather than a
            #      distance-geometry conformer unrelated to its real one.
            #
            # `skip_entry_ids` carries step 1's output into step 2, so a prediction is never
            # overwritten by an embedding.
            if not args.skip_folding:
                from sample.folding import build_structures, colabfold_available

                if colabfold_available():
                    peptides = [(str(r["entry_id"]), str(r.get("sequence") or ""))
                                for _, r in df_emb.iterrows() if r.get("entry_id")]
                    summary = build_structures(
                        peptides, structure_dir=structure_dir,
                        limit=args.fold_limit, num_models=args.fold_models,
                        num_recycle=args.fold_recycle)
                    folded = summary.get("structures", 0)
                else:
                    logger.warning(
                        "ColabFold is not installed, so structures fall back to distance-geometry "
                        "embedding of the stored SMILES. Run `make env-fold` to enable folding.")

            from sample.folding import folded_entry_ids

            enrich_with_2D_and_3D_structures(
                df_emb, structure_dir, skip_entry_ids=folded_entry_ids(structure_dir))

    elif args.command == "decoys":
        from sample.decoys import generate_decoys
        generate_decoys(args.positives, args.output, args.ratio, args.method)

    elif args.command == "assembly":
        logger.info(f"Assembling final dataset from {args.positives} and {args.decoys}...")
        df_pos = pd.read_csv(args.positives)
        df_dec = pd.read_csv(args.decoys)
        
        # Ensure label values are set
        df_pos["is_decoy"] = 0
        df_dec["is_decoy"] = 1

        # `negative_type` carries WHICH negative source each decoy came from (shuffled /
        # uniprot_fragment / uniprot_amp_like / fallback_random). pd.concat below already unions
        # columns, so it survives assembly on its own -- but only as an incidental NaN for every
        # positive, which reads identically to "a decoy whose source was not recorded". Setting
        # both sides explicitly gives the column one defined meaning per row, which is what
        # reporting performance per negative source (Section 4 of the paper) depends on. A decoy
        # file predating the column gets "unspecified" rather than silently becoming a positive.
        df_pos["negative_type"] = "real_positive"
        if "negative_type" not in df_dec.columns:
            df_dec["negative_type"] = "unspecified"
        else:
            df_dec["negative_type"] = df_dec["negative_type"].fillna("unspecified")

        # Enrich decoys with descriptors/embeddings to match feature shapes. Deliberately NOT run
        # through the external predictors (macrel/anticp2/hemopi2/algpred2): decoys are synthetic
        # negatives that exist only to train the ML models, and `sample.ml_models.feature_blocks`
        # never reads a predictor column as a training feature in the first place (today it
        # recognises only `esm2_dim_*` -- see that function's docstring) -- so running four slow
        # external tools over the whole decoy set bought nothing. Predictor scores are a
        # release-database annotation for real peptides, not an ML feature; decoys are never shown
        # as release records, so they never need one.
        from sample.enrichment import enrich_with_descriptors, enrich_with_embeddings
        df_dec_desc = enrich_with_descriptors(df_dec)
        df_dec_emb = enrich_with_embeddings(df_dec_desc)
        
        # Align column orders
        df_final = pd.concat([df_pos, df_dec_emb], ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df_final.to_csv(args.output, index=False)
        logger.info(f"Assembled final model-ready dataset with {len(df_final)} records. Saved to {args.output}")

    elif args.command == "update-dataset":
        logger.info(f"Updating dataset {args.dataset} with new data from {args.new_data}...")
        df_master = pd.read_csv(args.dataset)
        df_new = pd.read_csv(args.new_data)
        
        df_updated = pd.concat([df_master, df_new], ignore_index=True).drop_duplicates(subset=["sequence"], keep="last")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df_updated.to_csv(args.output, index=False)
        logger.info(f"Updated dataset written to {args.output}. Total rows: {len(df_updated)}")

    elif args.command == "validate-smiles":
        from sample.smiles_validation import validate_smiles_file
        validate_smiles_file(args.dataset, output_csv=args.output,
                             predicted_structures_json=args.predicted_structures)

    elif args.command == "build-references":
        from sample.references import build_bibliography
        build_bibliography(output_path=args.output, sources=args.sources,
                           resolve=not args.offline,
                           resolve_titles=not args.no_title_search,
                           linked_only=not args.resolve_unlinked)

    elif args.command == "build-modifications":
        from sample.modifications import build_modifications
        build_modifications(output_path=args.output, directory=args.dbaasp_dir)

    elif args.command == "build-source-organisms":
        from sample.provenance import build_source_organisms
        build_source_organisms(output_path=args.output, directory=args.dbaasp_dir)

    elif args.command == "build-cross-references":
        from sample.identifiers import build_cross_references
        build_cross_references(output_path=args.output, activities_csv=args.activities_csv)

    elif args.command == "build-family-hmms":
        from sample.family_hmms import build_family_hmms, combined_hmm_database
        build_family_hmms(families_json=args.families, dataset_csv=args.dataset,
                          out_dir=args.output_dir, min_size=args.min_size,
                          limit=args.limit, resume=not args.no_resume)
        if args.press:
            combined_hmm_database(out_dir=args.output_dir)

    elif args.command == "build-embedding-index":
        from sample.embeddings import build_embedding_index
        build_embedding_index(dataset_csv=args.dataset, chroma_dir=args.output_dir,
                              batch_size=args.batch_size, limit=args.limit)

    elif args.command == "build-gene-ontology":
        from sample.gene_ontology import build_gene_ontology
        build_gene_ontology(output_path=args.output, camp_dir=args.camp_dir)

    elif args.command == "build-go-obo":
        from sample.gene_ontology import build_go_slice, download_go_obo
        go_obo_path = args.go_obo
        if args.force_download:
            go_obo_path = download_go_obo(output_path=go_obo_path, force=True)
        build_go_slice(gene_ontology_json=args.gene_ontology_json, go_obo_path=go_obo_path,
                       output_path=args.output, download_if_missing=not args.no_download)

    elif args.command == "admin":
        import getpass

        from sample import admin as admin_mod
        from sample.engine import create_table, session_scope
        from sample.models import Admin

        # The admin table is not created by the pipeline, so the first command run against a fresh
        # database has to make it.
        create_table(Admin)

        def _read_password(supplied: Optional[str], who: str) -> str:
            """Prompts twice when no password was given on the command line."""
            if supplied:
                return supplied
            first = getpass.getpass(f"New password for {who}: ")
            if first != getpass.getpass("Repeat: "):
                raise admin_mod.AdminError("Passwords did not match.")
            return first

        try:
            with session_scope() as session:
                if args.admin_command == "create":
                    admin_mod.create_admin(
                        session, args.username, _read_password(args.password, args.username))
                    print(f"Created admin {args.username.strip().lower()!r}.")

                elif args.admin_command == "passwd":
                    admin_mod.set_password(
                        session, args.username, _read_password(args.password, args.username))
                    print(f"Password updated; existing sessions for "
                          f"{args.username.strip().lower()!r} are now invalid.")

                elif args.admin_command == "list":
                    admins = admin_mod.list_admins(session)
                    if not admins:
                        print("No administrators. Create one with: sample-cli admin create <name>")
                    for a in admins:
                        print(f"{a.username:24s} {'active' if a.is_active else 'disabled':9s} "
                              f"created {a.created_at}")

                elif args.admin_command == "remove":
                    admin_mod.delete_admin(session, args.username)
                    print(f"Deleted admin {args.username.strip().lower()!r}.")

                elif args.admin_command in ("enable", "disable"):
                    admin_mod.set_active(session, args.username,
                                         args.admin_command == "enable")
                    print(f"Admin {args.username.strip().lower()!r} "
                          f"{'enabled' if args.admin_command == 'enable' else 'disabled'}.")
        except admin_mod.AdminError as e:
            # A refusal, not a crash: print it plainly and exit non-zero so scripts can react.
            logger.error(str(e))
            raise SystemExit(1) from e

    elif args.command == "fold":
        from sample.folding import build_structures
        df = pd.read_csv(args.input, low_memory=False)
        peptides = [(str(r["entry_id"]), str(r.get("sequence") or ""))
                    for _, r in df.iterrows() if r.get("entry_id")]
        build_structures(peptides, structure_dir=args.structure_dir,
                         output_path=args.output, limit=args.limit,
                         num_models=args.models, num_recycle=args.recycle,
                         batch_size=args.batch_size)

    elif args.command == "build-topology":
        from sample.cyclization import build_topologies
        build_topologies(output_path=args.output)

    elif args.command == "build-annotations":
        from sample.peptide_annotations import build_all
        summary = build_all(db_path_release=args.db, store_path=args.store,
                            limit=args.limit if args.limit and args.limit > 0 else None,
                            resume=not args.no_resume, run_predictors=not args.no_predictors)
        logger.info("build-annotations: annotated %d peptide(s) this run; store now holds %d.",
                    summary["annotated"], summary["total_stored"])

    elif args.command == "build-pdb-homology":
        from sample.pdb_homology import build_all as build_pdb_homology
        summary = build_pdb_homology(
            structures_dir=args.structures_dir, db_path_release=args.db, store_path=args.store,
            limit=args.limit if args.limit and args.limit > 0 else None,
            resume=not args.no_resume, min_identity=args.min_identity, sleep_seconds=args.sleep)
        logger.info("build-pdb-homology: searched %d peptide(s) this run (%d with a PDB match); "
                    "store now holds %d.", summary["processed"], summary["matched"],
                    summary["total_stored"])

    elif args.command == "build-names":
        from sample.names import build_names
        build_names(output_path=args.output,
                    vocabulary_threshold=args.vocabulary_threshold)

    elif args.command == "build-families":
        from sample.families import build_families
        build_families(peptides_fasta=args.fasta, domtbl_path=args.domtbl,
                       output_path=args.output, identity=args.identity)

    elif args.command == "update-families":
        from sample.families import read_fasta
        from sample.incremental import update_families

        if args.new.lower().endswith((".fasta", ".fa", ".faa")):
            new_sequences = read_fasta(args.new)
        else:
            frame = pd.read_csv(args.new, low_memory=False)
            columns = {c.lower(): c for c in frame.columns}
            id_col = columns.get("entry_id") or columns.get("id")
            seq_col = columns.get("sequence") or columns.get("seq")
            if not id_col or not seq_col:
                raise SystemExit(
                    "--new CSV needs an entry_id (or id) and a sequence (or seq) column.")
            new_sequences = {str(r[id_col]): str(r[seq_col]) for _, r in frame.iterrows()}

        summary = update_families(new_sequences, families_json=args.families, db_path=args.db,
                                  identity=args.identity, apply=not args.dry_run)
        logger.info(
            "update-families: %d placed (%d inherited an existing family, %d in %d new clusters); "
            "%d skipped as already known; %s.",
            summary["assigned"], summary["inherited"],
            summary["assigned"] - summary["inherited"], summary["new_clusters"],
            summary["skipped_existing"],
            "written" if summary["applied"] else "dry run, nothing written")

    elif args.command == "incremental-update":
        from sample.acquisition import run_incremental_update
        summary = run_incremental_update(consensus_csv=args.consensus, families_json=args.families,
                                         db_path=args.db, dry_run=args.dry_run)
        if not summary.get("new_peptides"):
            logger.info("incremental-update: no new peptides since the last release build.")
        else:
            logger.info(
                "incremental-update: %d new peptide(s) since the last release build; %d placed "
                "(%d inherited an existing family, %d in %d new clusters); %s. Loading them into "
                "the live database, descriptors, bibliography etc. (steps 21-43, 44) is a separate, "
                "deliberate follow-on, not run by this command.",
                summary["new_peptides"], summary["assigned"], summary["inherited"],
                summary["assigned"] - summary["inherited"], summary["new_clusters"],
                "written" if summary["applied"] else "dry run, nothing written")

    elif args.command == "genomic-context":
        import json as _json
        import tempfile

        from sample.genomic_context import (build_context, download_refseq_genome, krona_svg,
                                            load_domain_lookup, parse_assembly_summary)

        with open(args.assembly_summary, encoding="utf-8") as fh:
            genomes = parse_assembly_summary(fh.read())
        if args.limit and args.limit > 0:
            genomes = genomes[:args.limit]
        logger.info("genomic-context: searching %d genome(s) for %s. This downloads each genome, "
                    "runs tBLASTn, and deletes it -- a large, time-bound run.",
                    len(genomes), args.peptide_id)

        work_dir = args.work_dir or tempfile.mkdtemp(prefix="sample_gctx_")
        peptide_fasta = os.path.join(work_dir, "query.faa")
        os.makedirs(work_dir, exist_ok=True)
        with open(peptide_fasta, "w", encoding="utf-8") as fh:
            fh.write(f">{args.peptide_id}\n{args.peptide.strip()}\n")

        result = build_context(peptide_fasta, genomes, download_refseq_genome, work_dir,
                               domain_lookup=load_domain_lookup())
        logger.info("genomic-context: %d hit(s) across %d/%d genomes.",
                    len(result["hits"]), result["genomes_with_hits"], result["genomes_searched"])
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as fh:
                _json.dump(result, fh, indent=2)
            logger.info("  Wrote %s", args.output)
        if args.svg:
            os.makedirs(os.path.dirname(os.path.abspath(args.svg)), exist_ok=True)
            with open(args.svg, "w", encoding="utf-8") as fh:
                fh.write(krona_svg(result["krona"]))
            logger.info("  Wrote %s", args.svg)

    elif args.command == "genomic-context-sweep":
        import tempfile

        from sample.genomic_context import (download_refseq_genome, parse_assembly_summary,
                                            sweep_to_store)

        with open(args.assembly_summary, encoding="utf-8") as fh:
            genomes = parse_assembly_summary(fh.read())
        logger.info("genomic-context-sweep: %d genome(s) available for %s (resume=%s). Each is "
                    "downloaded, searched, written to the database, and deleted -- one at a time.",
                    len(genomes), args.peptide_id, not args.no_resume)

        work_dir = args.work_dir or tempfile.mkdtemp(prefix="sample_gctx_sweep_")
        result = sweep_to_store(
            args.peptide_id, args.peptide, genomes, download_refseq_genome, work_dir,
            store_path=args.store, resume=not args.no_resume,
            limit=args.limit if args.limit and args.limit > 0 else None)
        logger.info("genomic-context-sweep: store holds %d hit(s) across %d/%d genomes for %s.",
                    result["total_hits"], result["genomes_with_hits"], result["genomes_searched"],
                    args.peptide_id)

    elif args.command == "train-synergy":
        from sample.synergy import train_synergy_models
        train_synergy_models(model_dir=args.model_dir, report_path=args.report,
                             min_rows=args.min_rows)

    elif args.command == "link-bioactivity":
        from sample.bioactivity import link_all
        link_all(output_path=args.output, sources=args.sources, limit=args.limit)

    elif args.command == "load-sql":
        from sample.serialize import load_sql_database
        load_sql_database(args.dataset, args.db)

    elif args.command == "admet":
        import json as _json

        from sample.admet import admet_profile
        profile = admet_profile(args.sequence)
        if not profile:
            logger.error("Empty or unusable sequence.")
        elif args.json:
            print(_json.dumps(profile, indent=2))
        else:
            phys = profile["physicochemical"]
            logger.info("ADMET-lite for %s (%d residues, %d non-standard):",
                        profile["sequence"], profile["length"], profile["non_standard_residues"])
            logger.info("  MW=%s pI=%s net charge(pH7)=%s (%s)",
                        phys["molecular_weight"], phys["isoelectric_point"],
                        phys["net_charge_ph7"], profile["flags"]["net_charge"]["class"])
            logger.info("  GRAVY=%s instability=%s (%s) aliphatic=%s",
                        phys["gravy"], phys["instability_index"],
                        "stable" if profile["flags"]["instability"]["stable"] else "unstable",
                        phys["aliphatic_index"])
            hl = profile["flags"]["half_life"]
            logger.info("  N-terminus %s -> %s half-life; aggregation hotspot fraction %s",
                        hl["n_terminal"], hl["class"],
                        profile["flags"]["aggregation"]["hotspot_fraction"])
            for gap in profile["gaps"]:
                logger.info("  [gap] %s: %s", gap["property"], gap["status"])

    elif args.command == "train-ml":
        from sample.ml_models import MIN_POSITIVES_PER_TASK, train_all

        # FLAML-searched by default (not the fixed-hyperparameter baseline this command used
        # before) -- --fixed-hyperparameters opts back into the old, fast, single-model-family
        # path. A FLAML cell is a real up-to-600s search (calibration fit + final fit) plus
        # FLAML_N_Y_RANDOMIZATIONS cheap frozen refits, so a full task x block matrix run under
        # the new default is a multi-day undertaking, not the old run's runtime -- narrow with
        # --tasks-limit for anything short of a deliberate full release run.
        if args.fixed_hyperparameters:
            model_factory = None
            n_calibration_folds = None  # train_all's own N_SPLITS default
            n_y_randomizations = None  # train_all's own N_Y_RANDOMIZATIONS default
            calibrate_threshold = True
        else:
            from sample.flaml_xgboost import (FLAML_N_CALIBRATION_FOLDS,
                                              FLAML_N_Y_RANDOMIZATIONS, FlamlXGBoostEstimator)
            model_factory = FlamlXGBoostEstimator
            n_calibration_folds = FLAML_N_CALIBRATION_FOLDS
            n_y_randomizations = FLAML_N_Y_RANDOMIZATIONS
            # Not calibrated: F1-threshold calibration (OOF or otherwise) kept landing at unstable
            # extremes for a FLAML-searched model (0.055 one run, 0.751 the next, on runs that
            # otherwise looked similar on ROC-AUC) -- pinned at the conventional 0.5 instead, by
            # instruction, so the reported operating point is fixed and predictable rather than
            # something a training run's own calibration pass chose freshly each time.
            calibrate_threshold = False

        train_all_kwargs: Dict[str, Any] = {}
        if n_calibration_folds is not None:
            train_all_kwargs["n_calibration_folds"] = n_calibration_folds
        if n_y_randomizations is not None:
            train_all_kwargs["n_y_randomizations"] = n_y_randomizations

        report = train_all(args.dataset, args.model_dir, args.report,
                           calibrate_threshold=calibrate_threshold,
                           activities_csv=args.activities,
                           filter_anomalies=args.filter_anomalies,
                           tasks_limit=args.tasks_limit,
                           min_task_examples=(args.min_task_examples
                                              if args.min_task_examples is not None
                                              else MIN_POSITIVES_PER_TASK),
                           min_target_examples=args.min_target_examples,
                           families_json=args.families_json,
                           model_factory=model_factory,
                           **train_all_kwargs)
        logger.info(f"Kept {report['n_models_kept']}/{report['n_tasks'] * len(report['feature_blocks'])} "
                   f"(task, feature block) cell(s); {report['n_models_rejected']} failed the "
                   f"quality gate. Report: {args.report}")

    elif args.command in ("train-antibacterial-ml-model", "train-organism-specific-ml-models"):
        from sample.flaml_xgboost import train_flaml_tasks
        from sample.ml_models import build_tasks

        # Resolved directly from build_tasks(), not a prior train-ml report -- the simplified
        # pipeline (steps 46-47) has no full-matrix step to read one from any more.
        df = pd.read_csv(args.dataset, low_memory=False)
        if "is_anomaly" in df.columns:
            df = df[df["is_anomaly"] == 0]
        tasks = build_tasks(df, args.activities)

        if args.command == "train-antibacterial-ml-model":
            namespaces = [t.namespace for t in tasks
                         if t.namespace == "antibacterial/general/any/classification"]
        else:  # train-organism-specific-ml-models
            namespaces = [t.namespace for t in tasks
                         if t.scope == "organism_specific" and t.task == "classification"]

        if not namespaces:
            logger.error(f"No task in {args.dataset}/{args.activities} matches {args.command}.")
        else:
            logger.info(f"{args.command}: {len(namespaces)} namespace(s) to train.")
            result = train_flaml_tasks(
                namespaces, dataset_csv=args.dataset, activities_csv=args.activities,
                families_json=args.families_json, model_dir=args.model_dir,
                progress_path=args.progress, resume=args.resume)
            logger.info(f"{args.command}: {len(result['written'])} written, "
                       f"{len(result['gate_failed'])} did not clear the quality gate, "
                       f"{len(result['errors'])} errored.")

    elif args.command == "validate-ml":
        import json as _json
        from sample.ml_models import _primary_metric
        if not os.path.exists(args.report):
            logger.error(f"{args.report} not found. Run train-ml first.")
        else:
            with open(args.report, encoding="utf-8") as fh:
                report = _json.load(fh)
            logger.info(f"{report['n_models_kept']} model(s) passed the quality gate, "
                       f"{report['n_models_rejected']} did not, out of {report['n_tasks']} "
                       f"task(s) x {len(report['feature_blocks'])} feature block(s).")
            for record in report.get("models", []):
                is_regression = str(record.get("learning_task", "classification")).lower() == "regression"
                metric = _primary_metric(is_regression)
                holdout = record.get("holdout", {}).get(metric, float("nan"))
                ci = record.get("holdout_ci95", {}).get(metric, [float("nan"), float("nan")])
                null_p = record.get("null_comparison", {}).get(metric, {}).get("primary_pvalue", float("nan"))
                logger.info(f"  [kept] {record['namespace']}/{record['feature_block']}: "
                           f"holdout {metric}={holdout:.3f} 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}] "
                           f"vs-null p={null_p:.3g}")
            for record in report.get("rejected", []):
                logger.info(f"  [dropped] {record['namespace']}/{record['feature_block']}: "
                           f"{record.get('quality_failures', ['unknown reason'])[0]}")

    elif args.command == "benchmark":
        import json as _json
        from sample.benchmark import benchmark_holdout, reference_tools_for

        tools = reference_tools_for(args.model_id)
        if not tools:
            logger.error("No external reference tool is registered for %s; nothing to benchmark "
                         "against. See sample.benchmark.REFERENCE_TOOLS.", args.model_id)
        else:
            result = benchmark_holdout(args.model_id, dataset_csv=args.dataset,
                                       activities_csv=args.activities, model_dir=args.model_dir)
            logger.info("Benchmark on %s's holdout (%d peptides) vs %s:",
                        args.model_id, result.get("n_holdout", result["model"]["n"]),
                        ", ".join(c["reference_tool"] for c in result["comparisons"]))
            model = result["model"]
            logger.info("  sAMPle model: ROC-AUC=%s F1=%.3f MCC=%.3f",
                        f"{model['roc_auc']:.3f}" if model["roc_auc"] is not None else "n/a",
                        model["f1"], model["mcc"])
            for comparison in result["comparisons"]:
                if comparison["reference"] is not None:
                    ref = comparison["reference"]
                    logger.info("  %s:  ROC-AUC=%s F1=%.3f MCC=%.3f", comparison["reference_tool"],
                                f"{ref['roc_auc']:.3f}" if ref["roc_auc"] is not None else "n/a",
                                ref["f1"], ref["mcc"])
                    if comparison["delta"].get("roc_auc") is not None:
                        logger.info("    ROC-AUC advantage (model - %s): %+.3f",
                                    comparison["reference_tool"], comparison["delta"]["roc_auc"])
                else:
                    logger.warning("  %s unavailable; only the sAMPle model was scored against it.",
                                   comparison["reference_tool"])
            chance = result.get("model_vs_chance")
            if chance:
                logger.info("  vs. chance: holdout ROC-AUC=%s, y-randomized mean=%s, p=%s",
                            f"{chance.get('holdout_roc_auc'):.3f}" if chance.get("holdout_roc_auc") is not None else "n/a",
                            f"{chance.get('y_randomization_mean_roc_auc'):.3f}" if chance.get("y_randomization_mean_roc_auc") is not None else "n/a",
                            f"{chance.get('null_comparison', {}).get('primary_pvalue'):.3g}" if chance.get("null_comparison", {}).get("primary_pvalue") is not None else "n/a")
            if args.report:
                os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
                with open(args.report, "w", encoding="utf-8") as fh:
                    _json.dump(result, fh, indent=2)
                logger.info("  Wrote %s", args.report)

    elif args.command == "benchmark-all":
        from sample.benchmark import benchmark_all

        results = benchmark_all(model_dir=args.model_dir, dataset_csv=args.dataset,
                                activities_csv=args.activities, report_path=args.report)
        n_errors = sum(1 for r in results.values() if "error" in r)
        logger.info(f"Benchmarked {len(results)} model(s); {n_errors} could not be evaluated. "
                   f"Report: {args.report}")

    elif args.command == "eda":
        from sample.analysis import run_exploratory_data_analysis
        run_exploratory_data_analysis(args.dataset, args.output_dir, args.sample_size)

    elif args.command == "anomaly-detection":
        from sample.analysis import run_anomaly_detection
        run_anomaly_detection(args.dataset, args.output, args.report, args.contamination,
                              esm_components=args.esm_components)

    elif args.command == "train-rf-filter":
        from sample.rf_filter import train_rf_filter
        train_rf_filter(args.output)

    elif args.command == "external-enrich":
        from sample.external_db import run_external_enrichment
        run_external_enrichment(args.sources, args.workers)

    elif args.command == "extract-from-pdf":
        from sample.pipeline_runner import process_single_pdf_file
        process_single_pdf_file(args.pdf_path, args.output)

    elif args.command == "compare-chemical-spaces":
        from sample.analysis import compare_chemical_spaces
        compare_chemical_spaces(args.dataset, args.output_dir)

    elif args.command == "extract-tables":
        import json
        # docling first when asked for. Measured over 40 random PubMed PDFs: 68% hold a table and
        # 25% hold one carrying peptide data (sequence/MIC/strain), at ~1.14 s/page. The pdfplumber
        # path over-detects badly on the same documents -- on one 12-page paper it reported 26
        # "tables", several of them 2x2 grids with zero non-empty cells -- and drops any ragged row
        # outright, which is exactly the row TableFormer exists to rebuild.
        recs = []
        backend_used = None
        if args.backend in ("auto", "docling"):
            from sample.docling_tables import extract_tables_from_pdf as extract_with_docling
            use_ocr = None if args.ocr == "auto" else (args.ocr == "on")
            recs = extract_with_docling(args.pdf_path, use_ocr=use_ocr)
            backend_used = "docling"
        # Falling back on an EMPTY result, not just on an unavailable environment: a paper whose
        # tables docling declines is still worth showing pdfplumber, and the reverse costs nothing.
        if not recs and args.backend in ("auto", "pdfplumber"):
            from sample.pdf_tables import extract_tables_from_pdf
            recs = extract_tables_from_pdf(args.pdf_path)
            backend_used = "pdfplumber"
        logger.info(f"extract-tables: {len(recs)} record(s) via {backend_used}")
        out_str = json.dumps(recs, indent=4, ensure_ascii=False)
        if args.output == "-":
            print(out_str)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_str)
            logger.info(f"Extracted {len(recs)} table records to {args.output}")

    elif args.command == "enrich-esm2":
        from sample.esm_enrichment import compute_esm2_embeddings
        df = pd.read_csv(args.dataset)
        seqs = df["sequence"].dropna().tolist()
        embs = compute_esm2_embeddings(seqs)
        if embs is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, "esm2_embeddings.npy")
            # Imported here, not at module scope: `numpy` was never imported in this file at all,
            # so this line raised NameError -- after the embeddings had already been computed,
            # which is the expensive part. Local, because the rest of the CLI does not need numpy.
            import numpy as np

            np.save(out_file, embs)
            logger.info(f"Saved ESM-2 embeddings matrix of shape {embs.shape} to {out_file}")

    elif args.command == "enrich-descriptors":
        from sample.descriptors import compute_all_descriptors
        df = pd.read_csv(args.dataset)
        desc_list = [compute_all_descriptors(s) if pd.notna(s) else {} for s in df["sequence"]]
        df_desc = pd.DataFrame(desc_list)
        df_out = pd.concat([df, df_desc], axis=1)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df_out.to_csv(args.output, index=False)
        logger.info(f"Enriched dataset with biophysical descriptors. Saved to {args.output}")

    elif args.command == "revalidate-resolutions":
        from sample.synonyms import revalidate_extraction_dirs

        directories = args.dirs or ["data/literature_data_json", "data/patent_data_json"]
        totals = revalidate_extraction_dirs(directories, dry_run=args.dry_run, limit=args.limit)
        mode = "WOULD withdraw" if args.dry_run else "withdrew"
        logger.info(f"Revalidated {totals['files']:,} extraction records across "
                    f"{len(directories)} directory/ies.")
        logger.info(f"  {mode} {totals['withdrawn']:,} resolution(s) from "
                    f"{totals['files_changed']:,} record(s); {totals['kept']:,} still qualify; "
                    f"{totals['errors']} unreadable.")
        for reason, count in totals["top_reasons"]:
            logger.info(f"    {count:5d}x {reason}")
        if not args.dry_run and totals["withdrawn"]:
            logger.info("  Each withdrawn record keeps its sequence_resolution with a "
                        "withdrawn_reason, so what was removed and why stays legible.")

    elif args.command == "webserver":
        from sample.api.app import run_server
        run_server(args.host, args.port)

    elif args.command == "worker":
        from sample.worker import main
        main()


if __name__ == "__main__":
    main()
