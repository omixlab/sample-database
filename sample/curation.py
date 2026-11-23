"""
Select a manual-curation sample: papers with extracted peptide data, one directory each.

**Why this exists.** Every quality signal this pipeline has is either self-referential (the strict
contract in :mod:`sample.strict_schema` judging the extractor's own output) or synthetic
(:mod:`sample.validation_extras`' error injection, which measures whether a corruption is *caught*,
not whether an uncorrupted record is *right*). Neither answers the question that decides whether the
released database is trustworthy: for a real paper, did the extractor find what a human reading that
paper would find, and is what it found actually correct?

Answering that needs a bounded set of real papers, each packaged with the primary text and the
extraction side by side. This module builds that package, and nothing else -- it renders no verdict.
The judging is deliberately somewhere else (a human, or the Gemini script that ships alongside the
output), because a pipeline that grades its own homework is exactly the failure this is meant to
detect.

**What a curation set looks like.** Two corpora are sampled independently -- literature papers and
patents -- because they are extracted from very different prose and fail in different ways (a patent
enumerates dozens of speculative sequence variants it never tests; a paper reports a handful it did).
Reviewing only one would leave the other's error modes unmeasured::

    papers/<internal_id>/
        metadata.json     identity (PMID, DOI, title, link) + extraction summary
        extraction.json   exactly what the pipeline extracted for this document
        paper.txt         the primary text the extraction was run against
        paper.pdf         the source PDF, as published
    patents/<patent_id>/
        metadata.json     identity (patent number, title, assignee, dates, link) + summary
        extraction.json   as above
        paper.txt         as above
        paper.pdf         as above

plus a top-level ``manifest.json`` recording the selection parameters and every chosen document, so
a later run can be compared against this one rather than guessed at, and
``curation_review.xlsx`` -- one row per extracted activity, with an empty verdict column beside
every field -- which is what a reviewer actually fills in.

**Why the PDF ships alongside ``paper.txt``.** The text file is what the extractor read, so it is
the right artifact for judging whether extraction was faithful *to its input*. It is the wrong one
for judging whether the input was faithful to the paper: ``paper.txt`` has already lost the table
borders, column order, superscripts and figure captions that a reviewer needs to confirm that a
MIC of 128 belongs to the organism on that row and not the one above it. Both are therefore
present, and a disagreement between them is itself a finding -- it localises the fault to text
extraction rather than to the extractor.

**The sample is stratified, not the first 300 rows.** Papers are bucketed by how much the extractor
claimed (how many peptides, and whether any activity carries a real measured value) and drawn evenly
across those buckets. Taking the head of the directory listing would over-represent whatever the
extractor found easy, which is the opposite of what a review set is for -- the hard cases are the
informative ones. Selection is seeded, so the same corpus and seed reproduce the same set.
"""
import json
import logging
import math
import os
import random
import re
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("sample.curation")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Where the curation set is written. The Gemini evaluation script lives in this directory too (it
#: is committed; the papers themselves are generated data), which is why the papers go in a
#: ``papers/`` subdirectory rather than directly here -- so a rebuild never has to consider deleting
#: anything but its own output.
CURATION_DIR: str = os.path.join(DATA_DIR, "manual_curation")
PAPERS_DIRNAME: str = "papers"

#: How many papers to select by default. Large enough that a per-field accuracy estimate has a
#: usable confidence interval (a 300-paper sample bounds a proportion to roughly +/-6 points at 95%
#: confidence), small enough to actually be reviewed by a person.
DEFAULT_SAMPLE_SIZE: int = 300

#: Per-corpus defaults, which are NOT equal and should not be. The literature corpus is the one
#: the release mostly rests on and has the documents to support a larger sample; the patent corpus
#: is bounded by how few patents carry extracted peptides at all (measured: 44 with peptides, of
#: which 26 are referenced by a released peptide). Asking for 200 patents therefore asks for mostly
#: negatives by necessity, not by choice -- see :data:`DEFAULT_POSITIVE_FRACTION`.
DEFAULT_SAMPLE_SIZES: Dict[str, int] = {"papers": 800, "patents": 200}

#: Target share of each corpus that should be documents the extractor DID find peptides in. The
#: remainder are documents it found nothing in.
#:
#: Both halves are needed, and they answer different questions. A positive shows whether what was
#: extracted is *correct* (precision, field accuracy). A negative -- a document the pipeline
#: reported nothing for -- is the only way to see what it *missed entirely*, which no amount of
#: reviewing successful extractions can reveal. A set of positives alone measures the pipeline
#: only where it already succeeded.
#:
#: Capped by availability rather than enforced: when a corpus has fewer positives than its share
#: (patents), the shortfall goes to negatives and the realised split is recorded in the manifest
#: instead of being silently padded or the request silently under-filled.
DEFAULT_POSITIVE_FRACTION: float = 0.5

#: Stratum label for a document with no extracted peptides. Kept deliberately distinct from the
#: four positive strata so a negative can never be counted as a positive in ``by_stratum``, and so
#: the round-robin cannot starve the negative half.
NEGATIVE_STRATUM: str = "no_peptide_found"

#: The release database, and the column per corpus that records which documents a released peptide
#: was drawn from (``pubmed:<id>.json | scopus:<id>.json`` for literature,
#: ``google_patents:<id>.json`` for patents -- the internal document id IS the join key, so no
#: DOI/PMID resolution is involved). Used to restrict POSITIVES to documents that actually
#: contributed a peptide to the shipped release, which is the population a reader of the release
#: cares about; a peptide-bearing extraction that never reached the database is not evidence about
#: the release. Negatives are never filtered this way -- a document with no peptides cannot, by
#: construction, be referenced by one.
RELEASE_DB: str = "sample.db"
DB_REFERENCE_COLUMNS: Dict[str, str] = {"papers": "LITERATURE", "patents": "PATENT"}

#: The two corpora sampled, each with its own extraction directories (most-advanced first: the
#: *repaired* set is what step 19 actually merges into the release, so it is the honest thing to
#: review) and the directory its primary text lives in. Patents are kept separate from papers rather
#: than pooled: their extraction failure modes differ, and a pooled sample would let the larger
#: corpus hide the smaller one's errors.
CORPORA: Dict[str, Dict[str, Any]] = {
    "papers": {
        "extraction_dirs": ("literature_data_json_repaired", "literature_data_json_revised",
                            "literature_data_json"),
        "text_dirs": {"pubmed": "literature_data_text/pubmed",
                      "scopus": "literature_data_text/scopus"},
        # The PDF lives beside the download ledger, not beside the extracted text: the text dirs
        # hold only what the text pass produced, and a document can have one without the other.
        "pdf_dirs": {"pubmed": "literature_data/pubmed",
                     "scopus": "literature_data/scopus"},
        "metadata_tables": ("literature_data/pubmed_downloaded.csv",
                            "literature_data/scopus_downloaded.csv"),
    },
    "patents": {
        "extraction_dirs": ("patent_data_json_repaired", "patent_data_json_revised",
                            "patent_data_json"),
        "text_dirs": {"google_patents": "patent_data_text/google_patents"},
        "pdf_dirs": {"google_patents": "patent_data/google_patents"},
        "metadata_tables": ("patent_data/google_patents.csv",),
    },
}

#: Kept for callers that still name the literature directories directly.
EXTRACTION_DIRS: Tuple[str, ...] = CORPORA["papers"]["extraction_dirs"]
TEXT_DIRS: Dict[str, str] = CORPORA["papers"]["text_dirs"]


def _normalise_document_id(value: str, corpus: str = "papers") -> str:
    """
    :param value: An identifier from a metadata table.
    :param corpus: A key of :data:`CORPORA`.
    :returns: The identifier in the form the extraction/text filenames use.

    Dash handling is corpus-specific and must stay that way: Google Patents writes
    ``CN-111892646-B`` in its CSV but names the file ``CN111892646B``, so patent ids lose their
    dashes -- while a literature ``internal_id`` is a UUID whose dashes are part of the filename.
    Stripping dashes from both (an earlier version of this function did) silently matched zero
    papers to their metadata, which showed up as every paper having an empty ``bibliographic``.
    """
    text = str(value or "").strip()
    return text.replace("-", "") if corpus == "patents" else text


def _load_document_metadata(corpus: str, data_dir: str = DATA_DIR) -> Dict[str, Dict[str, str]]:
    """
    Reads bibliographic metadata for one corpus, keyed by the id its files are named with.

    :param corpus: A key of :data:`CORPORA`.
    :param data_dir: The project data directory.
    :returns: ``{document_id: {field: value}}``. Empty when the tables do not exist -- the set is
        then built without bibliographic detail rather than failing, since the extraction and the
        text are what the review actually needs.
    """
    import pandas as pd

    metadata: Dict[str, Dict[str, str]] = {}
    for relative in CORPORA[corpus]["metadata_tables"]:
        path = os.path.join(data_dir, relative)
        if not os.path.exists(path):
            continue
        try:
            # Google Patents' export puts its search URL on line 1 and the real header on line 2;
            # literature tables have their header on line 1. Detected rather than hardcoded per
            # file, so a re-export that drops the banner still parses.
            with open(path, encoding="utf-8", errors="ignore") as handle:
                first = handle.readline()
            skiprows = 1 if first.lower().startswith("search url") else 0
            frame = pd.read_csv(path, dtype=str, low_memory=False, skiprows=skiprows)
        except Exception as e:  # noqa: BLE001 -- a missing/unreadable table must not sink selection
            logger.warning("Could not read %s: %s", path, e)
            continue

        columns = {c.lower().strip(): c for c in frame.columns}
        id_column = columns.get("internal_id") or columns.get("id")
        if id_column is None:
            logger.warning("%s has no internal_id/id column; skipping.", path)
            continue
        for _, row in frame.iterrows():
            document_id = _normalise_document_id(row.get(id_column), corpus=corpus)
            if not document_id:
                continue
            record = {}
            for column in frame.columns:
                value = row.get(column)
                if value is None or str(value).strip().lower() in ("", "nan"):
                    continue
                record[str(column).strip()] = str(value).strip()
            metadata[document_id] = record
    return metadata


#: Kept as the literature-only alias the earlier API exposed.
def _load_paper_metadata(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, str]]:
    """:returns: :func:`_load_document_metadata` for the ``papers`` corpus."""
    return _load_document_metadata("papers", data_dir)


def _extraction_root(data_dir: str = DATA_DIR, corpus: str = "papers") -> Optional[str]:
    """
    :param data_dir: The project data directory.
    :param corpus: A key of :data:`CORPORA`.
    :returns: The most advanced extraction directory that exists for `corpus`, or ``None``. The
        repaired set is preferred, because that is what the release is built from and therefore
        what a review should judge.
    """
    from sample.config import iter_extraction_files

    for name in CORPORA[corpus]["extraction_dirs"]:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and iter_extraction_files(path):
            return path
    return None


def _pdf_source_path(corpus: str, source: str, document_id: str,
                     data_dir: str = DATA_DIR) -> Optional[str]:
    """
    Where this document's PDF should be, if it was ever downloaded.

    :param corpus: A key of :data:`CORPORA`.
    :param source: The document's source (``pubmed``, ``scopus``, ``google_patents``).
    :param document_id: The document's internal id, already normalised.
    :param data_dir: The project data directory.
    :returns: An existing path, or None when the corpus/source has no PDF directory configured or
        the file is simply absent. Absence is normal and never fatal: a paper can have a usable
        text layer harvested from a source that refused the PDF.
    """
    relative = (CORPORA.get(corpus, {}).get("pdf_dirs") or {}).get(source)
    if not relative:
        return None
    candidate = os.path.join(data_dir, relative, f"{document_id}.pdf")
    return candidate if os.path.exists(candidate) else None


def _attach_pdf(corpus: str, source: str, document_id: str, target: str,
                data_dir: str = DATA_DIR) -> bool:
    """
    Copies this document's PDF into its curation directory as ``paper.pdf``.

    Copied rather than symlinked, because the curation set is handed to a reviewer (and, for the
    Gemini pass, uploaded) as a self-contained directory tree; a symlink into ``data/`` breaks the
    moment the tree is moved or archived, which is exactly what happens to it.

    :param corpus: A key of :data:`CORPORA`.
    :param source: The document's source.
    :param document_id: The document's internal id.
    :param target: The document's curation directory.
    :param data_dir: The project data directory.
    :returns: True when ``paper.pdf`` is present afterwards.
    """
    destination = os.path.join(target, "paper.pdf")
    origin = _pdf_source_path(corpus, source, document_id, data_dir)
    if origin is None:
        return os.path.exists(destination)
    if os.path.exists(destination) and os.path.getsize(destination) == os.path.getsize(origin):
        return True
    shutil.copyfile(origin, destination)
    return True


def summarise_extraction(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Describes what the extractor claimed for one document.

    This drives the stratification, and is also written into each paper's ``metadata.json`` so a
    reviewer sees the shape of the claim before reading the detail.

    :param document: A parsed extraction JSON. Both shapes this pipeline produces are accepted --
        the raw stage writes ``{"metadata": ..., "peptides": [...]}`` while the repaired stage
        writes a bare list of peptide records.
    :returns: ``{peptide_count, activity_count, measured_activity_count, with_sequence,
        with_name_only}``.
    """
    peptides = document if isinstance(document, list) else (document.get("peptides") or [])
    peptides = [p for p in peptides if isinstance(p, dict)]
    activities = [a for p in peptides for a in (p.get("biological_activities") or [])]
    measured = [a for a in activities if a.get("value_lower") is not None]
    return {
        "peptide_count": len(peptides),
        "activity_count": len(activities),
        "measured_activity_count": len(measured),
        "with_sequence": sum(1 for p in peptides if (p.get("peptide_sequence") or "").strip()),
        "with_name_only": sum(1 for p in peptides
                              if not (p.get("peptide_sequence") or "").strip()
                              and (p.get("peptide_name") or "").strip()),
    }


def _stratum(summary: Dict[str, Any]) -> str:
    """
    Buckets a paper by how much the extractor claimed about it.

    The buckets exist so the review set is not dominated by whichever case is most common (papers
    with a single named peptide and no number). A reviewer learns most from the dense papers and
    from the ones where a sequence was found but no activity was, so those are guaranteed their own
    share rather than left to chance.

    :param summary: :func:`summarise_extraction`'s output.
    :returns: A bucket label.
    """
    if summary["measured_activity_count"] >= 5:
        return "many_measured"
    if summary["measured_activity_count"] >= 1:
        return "some_measured"
    if summary["with_sequence"] >= 1:
        return "sequence_no_measurement"
    return "name_only"


def _db_referenced_document_ids(corpus: str, data_dir: str = DATA_DIR) -> Optional[set]:
    """
    Reads the document ids that released peptides were actually drawn from.

    :param corpus: A key of :data:`CORPORA`.
    :param data_dir: The project data directory; the release database is ``sample.db`` inside it.
    :returns: The set of internal document ids referenced by at least one peptide in the release,
        or ``None`` when the database (or its reference column) is not there at all. ``None`` means
        "no filter available" and is deliberately distinct from an empty set, which would mean
        "the release references nothing" and should select nothing -- conflating the two would let
        a missing database silently widen the sample instead of narrowing it.
    """
    column = DB_REFERENCE_COLUMNS.get(corpus)
    db_path = os.path.join(data_dir, RELEASE_DB)
    if column is None or not os.path.exists(db_path):
        logger.warning(
            "No release database at %s (or no reference column for %s); positives will NOT be "
            "restricted to documents referenced by released peptides. The sample then audits "
            "extractions that may never have reached the release.", db_path, corpus)
        return None

    import sqlite3

    ids: set = set()
    try:
        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                f"SELECT {column} FROM peptides "
                f"WHERE {column} IS NOT NULL AND TRIM({column}) <> ''")
            for (value,) in rows:
                # One peptide can cite several documents, pipe-separated, each as
                # "<source>:<document_id>.json".
                for token in str(value).split("|"):
                    token = token.strip()
                    if not token:
                        continue
                    if ":" in token:
                        token = token.split(":", 1)[1]
                    ids.add(os.path.splitext(token)[0])
        finally:
            connection.close()
    except Exception as e:  # noqa: BLE001 -- an unreadable database must not sink selection
        logger.warning("Could not read %s from %s (%s); positives will not be restricted.",
                       column, db_path, e)
        return None
    logger.info("%s: %d distinct document id(s) referenced by released peptides.", corpus, len(ids))
    return ids


def select_papers(sample_size: int = DEFAULT_SAMPLE_SIZE, data_dir: str = DATA_DIR,
                  seed: int = 20260817, corpus: str = "papers",
                  positive_fraction: float = 1.0,
                  db_referenced_only: bool = False) -> List[Dict[str, Any]]:
    """
    Chooses the documents for the curation set, stratified and seeded.

    Two populations, mixed according to `positive_fraction`:

    * **positives** -- documents the extractor found at least one peptide in. These show whether
      what was extracted is *correct*.
    * **negatives** -- documents it found nothing in, bucketed under :data:`NEGATIVE_STRATUM`.
      These are the only way to see what the pipeline *missed entirely*; reviewing successful
      extractions cannot reveal a document it silently passed over.

    :param sample_size: How many documents to select in total.
    :param data_dir: The project data directory.
    :param seed: Selection seed, so a rerun reproduces the same set.
    :param corpus: Which corpus to draw from; a key of :data:`CORPORA`.
    :param positive_fraction: Share of `sample_size` to fill with positives, rounded UP so a very
        small request still gets one. Defaults to 1.0 -- positives only, this function's
        long-standing behaviour, in which case no negative is ever added. Below 1.0, a shortfall in
        either pool is absorbed by the other (the patent corpus is short on positives; a small
        corpus can be short on negatives), so the caller gets `sample_size` documents whenever the
        corpus holds that many at all.
    :param db_referenced_only: Restrict POSITIVES to documents a released peptide was drawn from
        (:func:`_db_referenced_document_ids`). Negatives are never restricted this way -- a
        document with no peptides cannot be referenced by one. Degrades to unrestricted, with a
        warning, when the release database is absent.
    :returns: ``[{internal_id, source, path, summary, stratum, is_negative}]``, at most
        `sample_size` long.
    :raises FileNotFoundError: When no extraction output exists to select from.
    """
    root = _extraction_root(data_dir, corpus=corpus)
    if root is None:
        raise FileNotFoundError(
            f"No {corpus} extraction output under {data_dir}. Run step 8 "
            f"(`sample-cli extract`) first.")
    logger.info("Selecting %s for curation from %s.", corpus, root)

    from sample.config import iter_extraction_files

    referenced = _db_referenced_document_ids(corpus, data_dir) if db_referenced_only else None

    eligible: Dict[str, List[Dict[str, Any]]] = {}
    negatives: List[Dict[str, Any]] = []
    scanned = skipped_unreferenced = 0
    #: Zero-peptide documents rejected as not-a-document (CAPTCHA pages, stubs), by reason. Counted
    #: and logged rather than dropped quietly: how much of the corpus is junk is itself a finding,
    #: and one this project only discovered by auditing.
    skipped_unusable: Dict[str, int] = {}
    # iter_extraction_files, never a raw glob: it reads only the recognised source subdirectories,
    # so a stale `scopus_old`/`.bak` sibling cannot enter the review set. DECISIONS.md records the
    # incident where walking this tree shipped 22.7% duplicate activity rows.
    for path in iter_extraction_files(root):
        scanned += 1
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception:  # noqa: BLE001 -- an unreadable extraction is simply not a candidate
            continue
        document_id = os.path.splitext(os.path.basename(path))[0]
        summary = summarise_extraction(document)
        record = {
            "internal_id": document_id,
            "source": os.path.basename(os.path.dirname(path)),
            "path": path,
            "summary": summary,
        }
        if summary["peptide_count"] == 0:
            # A document the pipeline found nothing in is only evidence about RECALL if there was
            # something there to find. ~2.3% of the literature corpus is Cloudflare/CAPTCHA
            # interstitial text saved as a paper: it yields no peptides, so it scores as a correct
            # empty extraction and silently inflates apparent recall. Measured directly -- two of
            # the first thirteen audited documents were bot-check pages. Excluded from the negative
            # pool rather than reviewed, since a reviewer can learn nothing from one.
            text_relative = CORPORA[corpus]["text_dirs"].get(
                record["source"], f"literature_data_text/{record['source']}")
            text_file = os.path.join(data_dir, text_relative, f"{document_id}.txt")
            try:
                with open(text_file, encoding="utf-8", errors="ignore") as handle:
                    head = handle.read(8000)
            except OSError:
                head = ""
            if head:
                from sample.document_quality import is_unusable_document

                unusable, why = is_unusable_document(head)
                if unusable:
                    skipped_unusable[why or "unusable"] = \
                        skipped_unusable.get(why or "unusable", 0) + 1
                    continue
            record["is_negative"] = True
            negatives.append(record)
            continue
        if referenced is not None and document_id not in referenced:
            skipped_unreferenced += 1
            continue
        record["is_negative"] = False
        eligible.setdefault(_stratum(summary), []).append(record)

    total_positive = sum(len(v) for v in eligible.values())
    logger.info("%s: %d/%d documents carry extracted peptides%s, across %d strata: %s; "
                "%d with none (negative pool).",
                corpus, total_positive, scanned,
                f" and are release-referenced ({skipped_unreferenced} peptide-bearing documents "
                f"excluded as unreferenced)" if referenced is not None else "",
                len(eligible), {k: len(v) for k, v in sorted(eligible.items())}, len(negatives))
    if skipped_unusable:
        logger.info("%s: %d zero-peptide document(s) excluded from the negative pool as not "
                    "usable documents: %s. A CAPTCHA page yields no peptides, so counting it as a "
                    "correct empty extraction would inflate apparent recall.",
                    corpus, sum(skipped_unusable.values()), skipped_unusable)
    if total_positive == 0 and not negatives:
        return []

    rng = random.Random(seed)
    for bucket in eligible.values():
        bucket.sort(key=lambda r: r["internal_id"])  # deterministic before shuffling
        rng.shuffle(bucket)
    negatives.sort(key=lambda r: r["internal_id"])
    rng.shuffle(negatives)

    # CEILING, not round(): round() banker-rounds 0.5 down, so a 1-document request at the default
    # 50 % yielded ZERO positives and the whole sample was negatives. A small set with no positives
    # is a worse failure than one with no negatives, so the positive share rounds up.
    positive_target = min(math.ceil(sample_size * positive_fraction), total_positive)
    # positive_fraction >= 1.0 means "positives only" -- this function's long-standing contract and
    # its default. Filling the remainder with negatives there would silently change what every
    # existing caller gets.
    negative_target = (0 if positive_fraction >= 1.0
                       else min(sample_size - positive_target, len(negatives)))

    # Round-robin across strata rather than a fixed quota each: a stratum with fewer papers than its
    # share should not leave the set short, and one with more should absorb the remainder.
    selected: List[Dict[str, Any]] = []
    order = sorted(eligible)

    def _take_positives(limit: int) -> None:
        while sum(1 for r in selected if not r.get("is_negative")) < limit \
                and any(eligible[s] for s in order):
            for stratum in order:
                if not eligible[stratum]:
                    continue
                record = eligible[stratum].pop()
                record["stratum"] = stratum
                selected.append(record)
                if sum(1 for r in selected if not r.get("is_negative")) >= limit:
                    break

    def _take_negatives(limit: int) -> None:
        taken = sum(1 for r in selected if r.get("is_negative"))
        for record in negatives[:max(0, limit - taken)]:
            record["stratum"] = NEGATIVE_STRATUM
            selected.append(record)
        del negatives[:max(0, limit - taken)]

    _take_positives(positive_target)
    _take_negatives(negative_target)

    # Either pool running short must be absorbed by the other, in BOTH directions, or the caller
    # silently gets fewer documents than it asked for. The patent corpus is short on positives
    # (26 against a 100 share); a small corpus can just as easily be short on negatives.
    if len(selected) < sample_size:
        _take_positives(positive_target + (sample_size - len(selected)))
    if len(selected) < sample_size and positive_fraction < 1.0:
        _take_negatives(negative_target + (sample_size - len(selected)))
    return selected


def _build_one_corpus(corpus: str, sample_size: int, data_dir: str, output_dir: str,
                      seed: int, overwrite: bool,
                      positive_fraction: float = DEFAULT_POSITIVE_FRACTION,
                      db_referenced_only: bool = True) -> Dict[str, Any]:
    """
    Writes one corpus's directories under ``<output_dir>/<corpus>/``.

    :param corpus: A key of :data:`CORPORA`.
    :param sample_size: How many documents to select.
    :param data_dir: The project data directory.
    :param output_dir: The curation directory.
    :param seed: Selection seed.
    :param overwrite: Replace this corpus's existing directory.
    :param positive_fraction: Forwarded to :func:`select_papers`.
    :param db_referenced_only: Forwarded to :func:`select_papers`.
    :returns: ``{selected, positives, negatives, by_stratum, missing_text, entries}``;
        ``selected`` is 0 with a logged warning when the corpus has no extraction output, so one
        missing corpus never blocks the other.
    :raises FileExistsError: When the corpus directory exists and `overwrite` is false.
    """
    corpus_dir = os.path.join(output_dir, corpus)
    if os.path.isdir(corpus_dir) and os.listdir(corpus_dir):
        if not overwrite:
            raise FileExistsError(
                f"{corpus_dir} already holds a curation set. Pass overwrite=True (or "
                f"`--overwrite`) to replace it -- any annotations made in place will be lost.")
        shutil.rmtree(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)

    try:
        selected = select_papers(sample_size=sample_size, data_dir=data_dir, seed=seed,
                                 corpus=corpus, positive_fraction=positive_fraction,
                                 db_referenced_only=db_referenced_only)
    except FileNotFoundError as e:
        logger.warning("%s not sampled: %s", corpus, e)
        return {"selected": 0, "positives": 0, "negatives": 0, "by_stratum": {},
                "missing_text": 0, "entries": []}

    document_metadata = _load_document_metadata(corpus, data_dir)
    text_dirs = CORPORA[corpus]["text_dirs"]

    missing_text = 0
    missing_pdf = 0
    entries: List[Dict[str, Any]] = []
    for record in selected:
        document_id = record["internal_id"]
        target = os.path.join(corpus_dir, document_id)
        os.makedirs(target, exist_ok=True)

        with open(record["path"], encoding="utf-8") as handle:
            document = json.load(handle)
        with open(os.path.join(target, "extraction.json"), "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)

        text_relative = text_dirs.get(record["source"], f"literature_data_text/{record['source']}")
        text_path = os.path.join(data_dir, text_relative, f"{document_id}.txt")
        has_text = os.path.exists(text_path)
        if has_text:
            shutil.copyfile(text_path, os.path.join(target, "paper.txt"))
        else:
            missing_text += 1

        has_pdf = _attach_pdf(corpus, record["source"], document_id, target, data_dir)
        if not has_pdf:
            missing_pdf += 1

        metadata = {
            "document_id": document_id,
            "internal_id": document_id,
            "corpus": corpus,
            "source": record["source"],
            "stratum": record["stratum"],
            # Explicit, not inferred from peptide_count == 0 by every downstream reader: a
            # negative is a deliberate member of this set ("the pipeline reported nothing here --
            # is that right?"), not a positive that happens to be empty.
            "is_negative": bool(record.get("is_negative")),
            "extraction_summary": record["summary"],
            "has_primary_text": has_text,
            "has_pdf": has_pdf,
            "bibliographic": document_metadata.get(document_id, {}),
            "files": {"extraction": "extraction.json",
                      "paper_text": "paper.txt" if has_text else None,
                      "paper_pdf": "paper.pdf" if has_pdf else None},
        }
        with open(os.path.join(target, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        entries.append({k: metadata[k] for k in
                        ("document_id", "corpus", "source", "stratum", "is_negative",
                         "has_primary_text", "has_pdf")})

    by_stratum: Dict[str, int] = {}
    for entry in entries:
        by_stratum[entry["stratum"]] = by_stratum.get(entry["stratum"], 0) + 1
    negatives = sum(1 for e in entries if e["is_negative"])
    logger.info("%s: %d documents in %s (%d positive / %d negative; %s); %d without primary "
                "text, %d without a PDF.",
                corpus, len(entries), corpus_dir, len(entries) - negatives, negatives,
                by_stratum, missing_text, missing_pdf)
    return {"selected": len(entries), "positives": len(entries) - negatives,
            "negatives": negatives, "by_stratum": by_stratum, "missing_text": missing_text,
            "missing_pdf": missing_pdf, "entries": entries}


def build_curation_set(sample_size: Optional[Any] = None, data_dir: str = DATA_DIR,
                       output_dir: Optional[str] = None, seed: int = 20260817,
                       overwrite: bool = True,
                       corpora: Optional[Tuple[str, ...]] = None,
                       positive_fraction: float = DEFAULT_POSITIVE_FRACTION,
                       db_referenced_only: bool = True) -> Dict[str, Any]:
    """
    Writes the curation set: one directory per document, per corpus, plus a manifest.

    `sample_size` is **per corpus**, never a shared budget -- splitting one between them would let
    the larger corpus crowd the smaller out, and each needs enough documents to estimate its own
    accuracy. Pass an int to use the same size for every corpus, a ``{corpus: size}`` mapping for
    different ones, or omit it for :data:`DEFAULT_SAMPLE_SIZES` (800 papers, 200 patents).

    Each corpus is a mix of positives and negatives (:data:`DEFAULT_POSITIVE_FRACTION`), and
    positives are restricted by default to documents a released peptide was actually drawn from.
    One consequence is worth stating rather than discovering later: those positives are, by
    construction, documents the pipeline *succeeded* on -- they are in the release because
    extraction worked. That makes the precision estimate clean but biases a recall estimate
    optimistically, and the negative half is what counters it. The manifest records the realised
    split so the two halves are never read as one homogeneous sample.

    :param sample_size: Documents per corpus -- int, ``{corpus: int}``, or ``None`` for
        :data:`DEFAULT_SAMPLE_SIZES`.
    :param data_dir: The project data directory.
    :param output_dir: Where to write; :data:`CURATION_DIR` when omitted.
    :param seed: Selection seed.
    :param overwrite: Replace existing corpus directories. **On by default**: this set is rebuilt
        far more often than it is annotated in place, and the old behaviour failed every rebuild
        with a FileExistsError. Pass ``False`` (``--no-overwrite``) to restore the guard when a
        curator has annotations in the directory. Only the corpus directories are ever removed,
        never the evaluation script, environment file or README beside them.
    :param corpora: Which corpora to build; every key of :data:`CORPORA` when omitted.
    :param positive_fraction: Forwarded to :func:`select_papers`.
    :param db_referenced_only: Forwarded to :func:`select_papers`.
    :returns: A summary (``path``, ``selected``, ``by_corpus``, ``by_stratum``, ``positives``,
        ``negatives``, ``missing_text``).
    :raises FileExistsError: When a corpus directory exists and `overwrite` is false.
    """
    output_dir = output_dir or CURATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    chosen = tuple(corpora) if corpora else tuple(CORPORA)

    def _size_for(corpus: str) -> int:
        if isinstance(sample_size, dict):
            return int(sample_size.get(corpus, DEFAULT_SAMPLE_SIZES.get(corpus,
                                                                        DEFAULT_SAMPLE_SIZE)))
        if sample_size is None:
            return int(DEFAULT_SAMPLE_SIZES.get(corpus, DEFAULT_SAMPLE_SIZE))
        return int(sample_size)

    results: Dict[str, Dict[str, Any]] = {}
    for corpus in chosen:
        if corpus not in CORPORA:
            raise ValueError(f"Unknown corpus {corpus!r}; choose from {sorted(CORPORA)}.")
        results[corpus] = _build_one_corpus(corpus, _size_for(corpus), data_dir, output_dir, seed,
                                            overwrite, positive_fraction=positive_fraction,
                                            db_referenced_only=db_referenced_only)

    all_entries = [e for r in results.values() for e in r["entries"]]
    by_stratum: Dict[str, int] = {}
    for entry in all_entries:
        by_stratum[entry["stratum"]] = by_stratum.get(entry["stratum"], 0) + 1

    manifest = {
        "generated_by": "sample.curation.build_curation_set",
        "sample_size_requested_per_corpus": {c: _size_for(c) for c in chosen},
        "positive_fraction_requested": positive_fraction,
        "positives_restricted_to_release_referenced": db_referenced_only,
        # Stated in the artifact itself, not only in the code: the positive half is drawn from
        # documents that contributed a peptide to the release, i.e. ones extraction succeeded on.
        # Precision read from it is sound; recall read from it alone is optimistic, which is what
        # the negative half exists to measure.
        "sampling_caveat": (
            "Positives are documents a released peptide was drawn from, so they are cases the "
            "pipeline succeeded on; they bound precision, not recall. Negatives (stratum "
            f"'{NEGATIVE_STRATUM}') are documents the pipeline reported no peptide for and are "
            "the only half that measures what it missed. Do not pool the two into a single "
            "accuracy figure."),
        "seed": seed,
        "corpora": {
            corpus: {
                "selected": result["selected"],
                "positives": result["positives"],
                "negatives": result["negatives"],
                "by_stratum": result["by_stratum"],
                "documents_without_primary_text": result["missing_text"],
                # "" when the corpus has no extraction tree at all -- os.path.relpath("")
                # raises ValueError rather than returning something empty, so the guard is here
                # and not inside the relpath call.
                "extraction_root": (os.path.relpath(root, data_dir)
                                    if (root := _extraction_root(data_dir, corpus=corpus))
                                    else ""),
            }
            for corpus, result in results.items()
        },
        "total_selected": len(all_entries),
        "by_stratum": by_stratum,
        "documents": all_entries,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    summary = {
        "path": output_dir,
        "selected": len(all_entries),
        "by_corpus": {c: r["selected"] for c, r in results.items()},
        "by_stratum": by_stratum,
        "positives": sum(r["positives"] for r in results.values()),
        "negatives": sum(r["negatives"] for r in results.values()),
        "missing_text": sum(r["missing_text"] for r in results.values()),
        "missing_pdf": sum(r.get("missing_pdf", 0) for r in results.values()),
    }
    # The workbook is the artifact a reviewer opens, so a built set that lacks one is not finished.
    # It is written last and never allowed to sink the build: the directories are the expensive,
    # non-reproducible part (selection is seeded, but re-running still costs the whole copy), and
    # the workbook can be regenerated at any time with `--refresh`.
    try:
        summary["review_workbook"] = write_review_workbook(output_dir, data_dir)
    except Exception as e:  # noqa: BLE001 -- reported, and recoverable with --refresh
        summary["review_workbook"] = None
        logger.warning("Curation set built, but the review workbook was not written (%s: %s). "
                       "Re-run with `--refresh` to produce it.", type(e).__name__, e)

    logger.info("Curation set: %d documents in %s %s (%d positive / %d negative); "
                "%d without primary text, %d without a PDF.",
                summary["selected"], output_dir, summary["by_corpus"], summary["positives"],
                summary["negatives"], summary["missing_text"], summary["missing_pdf"])
    return summary


def refresh_curation_set(output_dir: Optional[str] = None, data_dir: str = DATA_DIR
                         ) -> Dict[str, Any]:
    """
    Brings an already-built curation set up to the current layout, without re-selecting anything.

    Attaches any missing ``paper.pdf``, updates each ``metadata.json`` to record it, and rewrites
    the review workbook. Nothing is deleted and no document is re-chosen, which is the point: the
    selection is seeded and reproducible, but a rebuild still discards any review already written
    into the tree, so adding a file to an existing set must not go through
    :func:`build_curation_set`.

    Idempotent -- a PDF already in place with the right size is left alone.

    :param output_dir: A built curation directory; :data:`CURATION_DIR` when omitted.
    :param data_dir: The project data directory.
    :returns: ``{path, documents, pdfs_attached, pdfs_already_present, pdfs_unavailable,
        review_workbook}``.
    :raises FileNotFoundError: When no curation set exists at `output_dir`.
    """
    output_dir = output_dir or (CURATION_DIR if data_dir == DATA_DIR
                                else os.path.join(data_dir, "manual_curation"))
    if not any(os.path.isdir(os.path.join(output_dir, c)) for c in CORPORA):
        raise FileNotFoundError(
            f"No curation set under {output_dir}. Run `sample-cli select-curation-set` first.")

    documents = attached = already = unavailable = 0
    for corpus in CORPORA:
        corpus_dir = os.path.join(output_dir, corpus)
        if not os.path.isdir(corpus_dir):
            continue
        for document_id in sorted(os.listdir(corpus_dir)):
            target = os.path.join(corpus_dir, document_id)
            metadata_path = os.path.join(target, "metadata.json")
            if not os.path.isfile(metadata_path):
                continue
            documents += 1
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            existed = os.path.exists(os.path.join(target, "paper.pdf"))
            has_pdf = _attach_pdf(corpus, metadata.get("source", ""), document_id, target, data_dir)
            if has_pdf and existed:
                already += 1
            elif has_pdf:
                attached += 1
            else:
                unavailable += 1

            metadata["has_pdf"] = has_pdf
            files = metadata.setdefault("files", {})
            files["paper_pdf"] = "paper.pdf" if has_pdf else None
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=False)

    workbook = write_review_workbook(output_dir, data_dir)
    logger.info("Refreshed %s: %d documents; PDFs %d attached, %d already present, %d "
                "unavailable. Workbook: %s",
                output_dir, documents, attached, already, unavailable, workbook)
    return {"path": output_dir, "documents": documents, "pdfs_attached": attached,
            "pdfs_already_present": already, "pdfs_unavailable": unavailable,
            "review_workbook": workbook}


#: Filename of the reviewer's workbook, written beside ``manifest.json``.
REVIEW_WORKBOOK: str = "curation_review.xlsx"

#: Allowed verdicts for one extracted field.
#:
#: ``missing`` and ``spurious`` are deliberately separate from ``incorrect``, because they are
#: different failures with different fixes: ``missing`` means the document states the fact and the
#: extractor did not record it (a recall failure in the parser), ``spurious`` means the extractor
#: recorded something the document does not state at all (a precision failure, and the one that
#: matters most for a published resource). Collapsing them into "wrong" would make the resulting
#: per-field table unable to answer which of the two the pipeline actually does.
FIELD_VERDICTS: Tuple[str, ...] = ("correct", "incorrect", "partial", "missing", "spurious", "n/a")

#: Allowed verdicts for the document as a whole.
#:
#: This is the only column that means anything on a negative row. A document the pipeline reported
#: no peptides for has no fields to grade, so its single question is whether that silence was
#: right (``ok``) or whether the document does report a peptide the pipeline never saw
#: (``missed_peptides``) -- which is the one measurement the positive half of the set cannot make.
DOCUMENT_VERDICTS: Tuple[str, ...] = ("ok", "missed_peptides", "not_relevant", "unreadable")

#: ``(header, column width)`` in sheet order. A verdict column always follows the field it grades,
#: so a reviewer reads value-then-judgement across the row rather than hunting for the far side of
#: the sheet; the long free-text columns go last for the same reason.
_REVIEW_COLUMNS: Tuple[Tuple[str, int], ...] = (
    ("row_key", 30), ("id", 24), ("type", 9), ("source", 15), ("pipeline_found", 15),
    ("pdf", 32), ("title", 46), ("link", 32),
    ("peptide_name", 20), ("verdict_name", 14),
    ("peptide_sequence", 30), ("verdict_sequence", 16),
    ("target_organism", 24), ("target_cell_line", 16), ("verdict_organism", 16),
    ("activity_group", 15), ("activity_type", 13), ("activity_value", 13),
    ("activity_unit", 12), ("verdict_activity", 16),
    ("verdict_document", 17), ("notes", 38), ("evidence_sentence", 60),
)

#: Columns the reviewer fills in; everything else is pipeline output and is left alone.
_EDITABLE = {"verdict_name", "verdict_sequence", "verdict_organism", "verdict_activity",
             "verdict_document", "notes"}

#: Control characters are not legal in the shared-strings part of an xlsx, and openpyxl raises on
#: them rather than escaping. Extracted context sentences come straight out of PDF text layers and
#: do contain them, so they are replaced instead of being allowed to abort the whole workbook.
_ILLEGAL_SHEET_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

#: Longest text written into one cell. Excel's own ceiling is 32,767, but an evidence sentence that
#: long is unreadable in a row and the extractor's ``context`` field can run to tens of thousands of
#: characters; the sentence is a locator for the reviewer, not the document.
_MAX_CELL_CHARS = 1000


def _sheet_text(value: Any, limit: int = _MAX_CELL_CHARS) -> str:
    """
    :param value: Any extracted value.
    :param limit: Longest string to keep.
    :returns: The value as a string safe to write into a worksheet cell, truncated with an
        ellipsis when it exceeds `limit`.
    """
    text = "" if value is None else str(value)
    text = _ILLEGAL_SHEET_CHARS.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    return text


def _activity_value(activity: Dict[str, Any]) -> str:
    """
    :param activity: One ``biological_activities`` entry.
    :returns: The measurement as the document stated it, operator included (``>128``).

    ``value_raw`` and not ``value_lower``: the reviewer is checking this against the PDF, and the
    normalised figure is a different claim (it has had unit conversion applied, which is the
    pipeline's work rather than the document's). A disagreement with ``value_lower`` is a unit-
    conversion bug, which is worth finding separately and not by hiding the original here.
    """
    operator = _sheet_text(activity.get("value_operator"), 8)
    raw = _sheet_text(activity.get("value_raw"), 40)
    return f"{operator}{raw}".strip()


def _review_rows(output_dir: str, corpora: Sequence[str] = ("papers", "patents")
                 ) -> List[Dict[str, Any]]:
    """
    Flattens a built curation set into one row per extracted activity.

    One row per *activity*, not per peptide: a peptide tested against six organisms is six
    independent facts that can each be right or wrong, and grading them on one line would force a
    single verdict over six judgements. The peptide's own fields are repeated down those rows so
    each line stands alone when the sheet is filtered or sorted.

    A peptide with no activities, and a document with no peptides, each still get exactly one row
    -- the latter is a negative, whose whole purpose is to be reviewed for what is absent.

    :param output_dir: A built curation directory.
    :param corpora: Which corpus subdirectories to read.
    :returns: Rows keyed by the headers of :data:`_REVIEW_COLUMNS`, ordered by corpus, then
        document id, then the order the extraction recorded, so the sheet is reproducible.
    """
    rows: List[Dict[str, Any]] = []
    for corpus in corpora:
        corpus_dir = os.path.join(output_dir, corpus)
        if not os.path.isdir(corpus_dir):
            continue
        singular = "patent" if corpus == "patents" else "paper"
        for document_id in sorted(os.listdir(corpus_dir)):
            target = os.path.join(corpus_dir, document_id)
            metadata_path = os.path.join(target, "metadata.json")
            if not os.path.isfile(metadata_path):
                continue
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            bibliographic = metadata.get("bibliographic") or {}

            extraction_path = os.path.join(target, "extraction.json")
            peptides: List[Dict[str, Any]] = []
            if os.path.isfile(extraction_path):
                with open(extraction_path, encoding="utf-8") as handle:
                    peptides = (json.load(handle).get("peptides") or [])

            common = {
                "id": document_id,
                "type": singular,
                "source": metadata.get("source", ""),
                "pipeline_found": "nothing" if metadata.get("is_negative") else "peptides",
                "pdf": (f"{corpus}/{document_id}/paper.pdf"
                        if (metadata.get("files") or {}).get("paper_pdf") else ""),
                "title": _sheet_text(bibliographic.get("title"), 300),
                "link": _sheet_text(bibliographic.get("link")
                                    or bibliographic.get("result link"), 300),
            }

            if not peptides:
                rows.append({**common, "row_key": f"{document_id}#-#-"})
                continue

            for peptide_index, peptide in enumerate(peptides):
                fields = {
                    "peptide_name": _sheet_text(peptide.get("peptide_name"), 120),
                    "peptide_sequence": _sheet_text(peptide.get("peptide_sequence"), 200),
                }
                activities = peptide.get("biological_activities") or []
                if not activities:
                    rows.append({**common, **fields,
                                 "row_key": f"{document_id}#{peptide_index}#-"})
                    continue
                for activity_index, activity in enumerate(activities):
                    rows.append({
                        **common, **fields,
                        "row_key": f"{document_id}#{peptide_index}#{activity_index}",
                        "target_organism": _sheet_text(activity.get("target_organism"), 120),
                        "target_cell_line": _sheet_text(activity.get("target_cell_line"), 120),
                        "activity_group": _sheet_text(activity.get("target_group"), 40),
                        "activity_type": _sheet_text(activity.get("activity_type"), 40),
                        "activity_value": _activity_value(activity),
                        "activity_unit": _sheet_text(activity.get("activity_unit")
                                                     or activity.get("value_raw_unit"), 24),
                        "evidence_sentence": _sheet_text(activity.get("context_sentence")),
                    })
    return rows


def write_review_workbook(output_dir: Optional[str] = None, data_dir: str = DATA_DIR) -> str:
    """
    Writes ``curation_review.xlsx`` for an already-built curation set.

    Three sheets. ``Review`` is the work: one row per extracted activity, the pipeline's values on
    the left of each verdict column. ``Legend`` says which columns to fill and what the allowed
    values mean, with one example row. ``Summary`` counts the verdicts with ``COUNTIFS`` formulas,
    so per-field accuracy appears as the sheet is filled rather than having to be recomputed
    afterwards -- these are the numbers Section 3.7 of the manuscript reports.

    Non-destructive and re-runnable: it only reads the curation directories, and overwrites its own
    output. It does NOT re-select documents, so running it never disturbs review already done in
    the directory tree.

    :param output_dir: A built curation directory; :data:`CURATION_DIR` when omitted.
    :param data_dir: The project data directory, used only to resolve the default output.
    :returns: Path to the workbook written.
    :raises FileNotFoundError: When no curation set exists at `output_dir`.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    output_dir = output_dir or (CURATION_DIR if data_dir == DATA_DIR
                                else os.path.join(data_dir, "manual_curation"))
    if not any(os.path.isdir(os.path.join(output_dir, c)) for c in CORPORA):
        raise FileNotFoundError(
            f"No curation set under {output_dir}. Run `sample-cli select-curation-set` first.")

    rows = _review_rows(output_dir, tuple(CORPORA))
    headers = [h for h, _ in _REVIEW_COLUMNS]
    index = {h: i + 1 for i, h in enumerate(headers)}

    body = Font(name="Arial", size=10)
    head = Font(name="Arial", size=10, bold=True, color="FFFFFFFF")
    head_fill = PatternFill("solid", fgColor="FF305B7F")
    fill_me = PatternFill("solid", fgColor="FFFFFF99")
    top = Alignment(vertical="top")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Review"
    sheet.append(headers)
    for column in range(1, len(headers) + 1):
        cell = sheet.cell(row=1, column=column)
        cell.font, cell.fill = head, head_fill
        cell.alignment = Alignment(vertical="top", horizontal="center", wrap_text=True)
        sheet.column_dimensions[get_column_letter(column)].width = _REVIEW_COLUMNS[column - 1][1]

    for offset, row in enumerate(rows, start=2):
        for header, column in index.items():
            cell = sheet.cell(row=offset, column=column)
            cell.value = row.get(header, "")
            # openpyxl reads a leading "=" as a formula, and an extracted name or sentence is text
            # whatever it starts with. Forcing the type after assignment keeps the character.
            if isinstance(cell.value, str) and cell.value.startswith("="):
                cell.data_type = "s"
            cell.font, cell.alignment = body, top
            if header in _EDITABLE:
                cell.fill = fill_me

    last = len(rows) + 1
    sheet.freeze_panes = "I2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last}"

    # Dropdowns, so a verdict is one of the agreed values and the Summary COUNTIFS can rely on it.
    # allow_blank, because an unreviewed row is the normal state of most of this sheet.
    for column_name, options in (("verdict_name", FIELD_VERDICTS),
                                 ("verdict_sequence", FIELD_VERDICTS),
                                 ("verdict_organism", FIELD_VERDICTS),
                                 ("verdict_activity", FIELD_VERDICTS),
                                 ("verdict_document", DOCUMENT_VERDICTS)):
        letter = get_column_letter(index[column_name])
        validation = DataValidation(type="list", formula1=f'"{",".join(options)}"',
                                    allow_blank=True, showDropDown=False)
        validation.error = f"Use one of: {', '.join(options)}"
        validation.errorTitle = "Not an allowed verdict"
        sheet.add_data_validation(validation)
        validation.add(f"{letter}2:{letter}{last}")

    _write_legend_sheet(workbook, rows, body, head, head_fill, fill_me)
    _write_summary_sheet(workbook, index, last, body, head, head_fill)

    path = os.path.join(output_dir, REVIEW_WORKBOOK)
    workbook.save(path)
    logger.info("Review workbook: %d rows across %d documents -> %s",
                len(rows), len({r["id"] for r in rows}), path)
    return path


def _write_legend_sheet(workbook: Any, rows: List[Dict[str, Any]], body: Any, head: Any,
                        head_fill: Any, fill_me: Any) -> None:
    """
    Writes the ``Legend`` sheet: what to fill in, what each verdict means, and one example row.

    :param workbook: The open workbook.
    :param rows: The review rows, used only to quote the set's actual size.
    :param body: Body font.
    :param head: Header font.
    :param head_fill: Header fill.
    :param fill_me: The same highlight the editable columns use, shown here so the colour is
        self-explanatory rather than a convention the reviewer has to infer.
    """
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet("Legend")
    documents = len({r["id"] for r in rows})
    lines: List[Tuple[str, str]] = [
        ("How to use this workbook", ""),
        ("", ""),
        ("Scope", f"{len(rows)} rows over {documents} documents. One row per extracted activity: "
                  f"a peptide measured against several organisms appears on several rows, with its "
                  f"name and sequence repeated on each."),
        ("Fill in", "Only the highlighted columns: verdict_name, verdict_sequence, "
                    "verdict_organism, verdict_activity, verdict_document, notes. Every other "
                    "column is pipeline output -- leave it as it is, so the sheet still records "
                    "what was actually extracted."),
        ("What to check against", "Open the file named in the `pdf` column, relative to this "
                                  "workbook. `paper.txt` in the same directory is the text the "
                                  "extractor actually read; if the PDF and the text disagree, say "
                                  "so in notes -- that localises the fault to text extraction "
                                  "rather than to the extractor."),
        ("Rows where pipeline_found = nothing",
         "The pipeline reported no peptide for this document. The field verdicts do not apply "
         "(leave them blank or n/a); set verdict_document to `ok` if the document really reports "
         "no peptide bioactivity, or `missed_peptides` if it does and the pipeline missed it. "
         "These rows are the only measure of what the pipeline fails to find at all."),
        ("", ""),
        ("Field verdicts", "verdict_name / verdict_sequence / verdict_organism / verdict_activity"),
        ("correct", "The document states this, and the sheet records it correctly."),
        ("incorrect", "The document states something different from what is recorded."),
        ("partial", "Recorded value is right but incomplete -- a truncated sequence, a genus with "
                    "no species, a strain dropped."),
        ("missing", "The document states this fact and the field is empty. A recall failure."),
        ("spurious", "The document does not state this at all. A precision failure, and the one "
                     "that matters most for a published resource -- flag these."),
        ("n/a", "The field cannot apply to this row (e.g. cell line on an antibacterial assay)."),
        ("", ""),
        ("Document verdicts", "verdict_document"),
        ("ok", "Nothing wrong with the document as a whole."),
        ("missed_peptides", "The document reports peptide bioactivity the pipeline did not record."),
        ("not_relevant", "The document is not about peptide bioactivity; it should not have been "
                         "in the corpus."),
        ("unreadable", "The PDF or text is unusable -- scanned beyond OCR, truncated, wrong "
                       "document. Not an extraction error."),
    ]
    sheet.append(["", ""])
    for label, text in lines:
        sheet.append([label, text])
    for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, max_col=2):
        for cell in row:
            cell.font = body
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for row_index in (2, 9, 17):
        for column in (1, 2):
            cell = sheet.cell(row=row_index, column=column)
            cell.font, cell.fill = head, head_fill
    sheet.column_dimensions["A"].width = 26
    sheet.column_dimensions["B"].width = 110

    # The example row, as the skill for this file type requires: realistic values in the format
    # expected, kept on the Legend sheet so it is never mistaken for data on the Review sheet.
    start = sheet.max_row + 2
    sheet.cell(row=start, column=1, value="Example of a filled row").font = head
    sheet.cell(row=start, column=1).fill = head_fill
    sheet.cell(row=start, column=2).fill = head_fill
    example = [
        ("peptide_name (given)", "BmKn2"),
        ("peptide_sequence (given)", "FIGAIARLLSKIF"),
        ("target_organism (given)", "Staphylococcus aureus"),
        ("activity (given)", "MIC 5.52 uM"),
        ("verdict_name", "correct"),
        ("verdict_sequence", "correct"),
        ("verdict_organism", "partial"),
        ("verdict_activity", "incorrect"),
        ("verdict_document", "ok"),
        ("notes", "Paper reports 4.83-5.52 uM as a range over four species; the sheet kept the "
                  "upper bound and dropped the strain (MRSA)."),
    ]
    for offset, (label, value) in enumerate(example, start=start + 1):
        sheet.cell(row=offset, column=1, value=label).font = body
        cell = sheet.cell(row=offset, column=2, value=value)
        cell.font = body
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        if label in _EDITABLE:
            cell.fill = fill_me


def _write_summary_sheet(workbook: Any, index: Dict[str, int], last: int, body: Any, head: Any,
                         head_fill: Any) -> None:
    """
    Writes the ``Summary`` sheet: verdict counts and per-field accuracy, as live formulas.

    Formulas rather than numbers computed here, because the counts are zero at the moment this file
    is written and are meant to fill in as the review proceeds. A reviewer opening the workbook
    half-way through should see where they are without anyone re-running the pipeline.

    Accuracy divides by the reviewed count *excluding* ``n/a``, and is blank until something is
    reviewed -- a zero would read as "nothing is correct" rather than "nothing is graded yet".

    Papers and patents are reported separately and deliberately not totalled: they are different
    corpora with different error modes and different sampling, and the manifest's own caveat says
    not to pool them.

    :param workbook: The open workbook.
    :param index: Header -> 1-based column number on the ``Review`` sheet.
    :param last: Last row number on the ``Review`` sheet.
    :param body: Body font.
    :param head: Header font.
    :param head_fill: Header fill.
    """
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet("Summary")
    type_range = f"Review!${get_column_letter(index['type'])}$2:${get_column_letter(index['type'])}${last}"
    found_range = (f"Review!${get_column_letter(index['pipeline_found'])}$2:"
                   f"${get_column_letter(index['pipeline_found'])}${last}")

    columns = ["Corpus", "Field", "rows", "reviewed", *FIELD_VERDICTS, "accuracy"]
    sheet.append(columns)

    fields = (("name", "verdict_name"), ("sequence", "verdict_sequence"),
              ("organism", "verdict_organism"), ("activity", "verdict_activity"))
    row_number = 2
    for corpus_label in ("paper", "patent"):
        for field_label, verdict_column in fields:
            letter = get_column_letter(index[verdict_column])
            verdict_range = f"Review!${letter}$2:${letter}${last}"
            sheet.cell(row=row_number, column=1, value=corpus_label)
            sheet.cell(row=row_number, column=2, value=field_label)
            # Rows that CAN carry a field verdict: the positive half only. A negative row has no
            # extracted field to grade, so including it would make every denominator wrong.
            sheet.cell(row=row_number, column=3,
                       value=f'=COUNTIFS({type_range},$A{row_number},{found_range},"peptides")')
            sheet.cell(row=row_number, column=4,
                       value=f'=COUNTIFS({type_range},$A{row_number},{verdict_range},"<>")')
            for offset, verdict in enumerate(FIELD_VERDICTS):
                sheet.cell(row=row_number, column=5 + offset,
                           value=f'=COUNTIFS({type_range},$A{row_number},'
                                 f'{verdict_range},"{verdict}")')
            graded = get_column_letter(5 + FIELD_VERDICTS.index("n/a"))
            correct = get_column_letter(5 + FIELD_VERDICTS.index("correct"))
            accuracy = get_column_letter(len(columns))
            sheet.cell(row=row_number, column=len(columns),
                       value=f'=IF(D{row_number}-{graded}{row_number}<=0,"",'
                             f'{correct}{row_number}/(D{row_number}-{graded}{row_number}))')
            sheet.cell(row=row_number, column=len(columns)).number_format = "0.0%"
            row_number += 1

    # The negative half, whose only question is whether the pipeline was right to find nothing.
    row_number += 1
    sheet.cell(row=row_number, column=1, value="Negatives (pipeline_found = nothing)").font = head
    sheet.cell(row=row_number, column=1).fill = head_fill
    row_number += 1
    sheet.append([])
    document_letter = get_column_letter(index["verdict_document"])
    document_range = f"Review!${document_letter}$2:${document_letter}${last}"
    sheet.cell(row=row_number, column=1, value="Corpus")
    sheet.cell(row=row_number, column=2, value="rows")
    sheet.cell(row=row_number, column=3, value="reviewed")
    for offset, verdict in enumerate(DOCUMENT_VERDICTS):
        sheet.cell(row=row_number, column=4 + offset, value=verdict)
    for column in range(1, 4 + len(DOCUMENT_VERDICTS)):
        cell = sheet.cell(row=row_number, column=column)
        cell.font, cell.fill = head, head_fill
    header_row = row_number
    row_number += 1
    for corpus_label in ("paper", "patent"):
        sheet.cell(row=row_number, column=1, value=corpus_label)
        sheet.cell(row=row_number, column=2,
                   value=f'=COUNTIFS({type_range},$A{row_number},{found_range},"nothing")')
        sheet.cell(row=row_number, column=3,
                   value=f'=COUNTIFS({type_range},$A{row_number},{found_range},"nothing",'
                         f'{document_range},"<>")')
        for offset, verdict in enumerate(DOCUMENT_VERDICTS):
            sheet.cell(row=row_number, column=4 + offset,
                       value=f'=COUNTIFS({type_range},$A{row_number},{found_range},"nothing",'
                             f'{document_range},"{verdict}")')
        row_number += 1

    for column in range(1, len(columns) + 1):
        cell = sheet.cell(row=1, column=column)
        cell.font, cell.fill = head, head_fill
        cell.alignment = Alignment(vertical="top", horizontal="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=len(columns)):
        for cell in row:
            if cell.row not in (header_row,) and not cell.font.bold:
                cell.font = body
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 12
    for column in range(3, len(columns) + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 11

    note = sheet.max_row + 2
    sheet.cell(row=note, column=1,
               value="Counts are live COUNTIFS over the Review sheet and fill in as it is "
                     "reviewed. Accuracy = correct / (reviewed - n/a), blank until a field is "
                     "graded. Papers and patents are not totalled: different corpora, different "
                     "sampling -- see manifest.json's sampling_caveat.").font = body
    sheet.cell(row=note, column=1).alignment = Alignment(vertical="top", wrap_text=True)
