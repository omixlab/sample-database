"""
Genomic context: where, across sequenced genomes, does a peptide's gene actually occur?

A peptide record says what a sequence *does*; it rarely says which organisms carry the gene that
encodes it. This module answers that by searching each genome for the peptide with tBLASTn (a
protein query against the six-frame translation of a nucleotide genome), keeping only near-exact
hits, and organising the surviving hits by the taxonomy of the genomes they landed in -- a
Krona-style radial breakdown of "this peptide's gene is present in these clades".

The design is dominated by two constraints the project owner set explicitly:

* **Near-exact only.** A hit is kept only when it covers at least
  :data:`MIN_COVERAGE` of the peptide *and* is at least :data:`MIN_IDENTITY` identical. Both, not
  either: a short perfect match to a conserved motif is not evidence the whole gene is present, and
  a full-length 60%-identity match is a different peptide. This is what makes a "present in
  *Escherichia*" claim mean the gene, not a lookalike fragment.
* **Delete as you go.** A genome FASTA is tens to hundreds of MB and there are thousands of them, so
  each is downloaded, searched, and *deleted* along with its BLAST database and output before the
  next -- only the handful of surviving hit rows are retained. The actual download/search loop
  (:func:`build_context`) is therefore disk- and time-bound and is never started implicitly; the
  parsing, filtering and aggregation it depends on are pure and are what this module tests.

The taxonomy of a hit comes from the *genome* it was found in (its NCBI tax id / organism name from
the RefSeq assembly summary), not from the BLAST alignment. The broad clade is resolved through the
same ``data/taxonomy.db`` ``tax_domain`` table the pipeline already builds; genus and species come
from the organism name, giving a domain → genus → species → strain hierarchy without needing the
full NCBI lineage shipped alongside.
"""
import logging
import math
import os
import re
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger("sample.genomic_context")

#: Minimum fraction of the peptide the alignment must cover to be kept.
MIN_COVERAGE: float = 0.95

#: Minimum fraction of identical residues in the alignment to be kept.
MIN_IDENTITY: float = 0.95

#: The tBLASTn tabular columns requested, in order. ``qcovhsp`` is BLAST's own per-HSP query
#: coverage, so coverage is read rather than recomputed (and stays correct for gapped alignments).
TBLASTN_COLUMNS: Tuple[str, ...] = (
    "qseqid", "sseqid", "pident", "length", "qcovhsp", "evalue", "bitscore", "sstart", "send")


class Hit(NamedTuple):
    """
    One surviving tBLASTn hit of a peptide in a genome.

    :param peptide_id: The query peptide's id.
    :param contig: The genome sequence (contig/chromosome) the hit landed on.
    :param identity: Fraction identical over the aligned region, ``0``–``1``.
    :param coverage: Fraction of the peptide covered by the alignment, ``0``–``1``.
    :param evalue: BLAST E-value.
    :param bitscore: BLAST bit score.
    :param sstart: Hit start on the subject (1-based, as BLAST reports).
    :param send: Hit end on the subject.
    :param genome_accession: The genome's assembly accession, filled by :func:`search_genome`.
    :param taxid: The genome's NCBI tax id.
    :param organism: The genome's organism name.
    """
    peptide_id: str
    contig: str
    identity: float
    coverage: float
    evalue: float
    bitscore: float
    sstart: int
    send: int
    genome_accession: str = ""
    taxid: str = ""
    organism: str = ""


def parse_tblastn(tabular: str) -> List[Hit]:
    """
    Parses tBLASTn ``-outfmt 6`` output produced with :data:`TBLASTN_COLUMNS`.

    :param tabular: The raw tabular output (one hit per line, tab-separated).
    :returns: One :class:`Hit` per parsable line. Identity and coverage are converted from BLAST's
        percentages to ``0``–``1`` fractions. Malformed lines are skipped rather than aborting the
        parse of a multi-thousand-hit file.
    """
    hits: List[Hit] = []
    for line in tabular.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < len(TBLASTN_COLUMNS):
            continue
        try:
            hits.append(Hit(
                peptide_id=fields[0],
                contig=fields[1],
                identity=float(fields[2]) / 100.0,
                coverage=float(fields[4]) / 100.0,
                evalue=float(fields[5]),
                bitscore=float(fields[6]),
                sstart=int(fields[7]),
                send=int(fields[8]),
            ))
        except (ValueError, IndexError):
            continue
    return hits


def passes_thresholds(hit: Hit, min_coverage: float = MIN_COVERAGE,
                      min_identity: float = MIN_IDENTITY) -> bool:
    """
    :param hit: A parsed hit.
    :param min_coverage: Minimum query coverage fraction.
    :param min_identity: Minimum identity fraction.
    :returns: Whether the hit clears BOTH thresholds. The conjunction is the point -- see the module
        docstring on why coverage-or-identity would admit fragments and lookalikes.
    """
    return hit.coverage >= min_coverage and hit.identity >= min_identity


def filter_hits(hits: Sequence[Hit], min_coverage: float = MIN_COVERAGE,
                min_identity: float = MIN_IDENTITY) -> List[Hit]:
    """
    :param hits: Parsed hits.
    :param min_coverage: Minimum query coverage fraction.
    :param min_identity: Minimum identity fraction.
    :returns: Only the hits clearing both thresholds.
    """
    return [h for h in hits if passes_thresholds(h, min_coverage, min_identity)]


#: Tokens in an organism name that are not part of the binomial and must not be read as a genus or
#: species epithet (``Escherichia coli str. K-12`` -> genus ``Escherichia``, species ``coli``).
_ORGANISM_NOISE = re.compile(
    r"\b(str\.?|strain|subsp\.?|serovar|isolate|substr\.?|sp\.?|var\.?|pv\.?)\b", re.IGNORECASE)


def _binomial(organism: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts genus and species epithet from an organism name.

    :param organism: An organism name, e.g. ``Escherichia coli str. K-12 substr. MG1655``.
    :returns: ``(genus, species)``; either may be ``None`` when the name does not carry it.
    """
    cleaned = _ORGANISM_NOISE.split(str(organism or "").strip())[0].strip()
    tokens = [t for t in cleaned.split() if t]
    genus = tokens[0] if tokens else None
    species = tokens[1] if len(tokens) > 1 else None
    return genus, species


def genome_lineage(taxid: str, organism: str,
                   domain_lookup: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Builds the taxonomic path a hit is filed under in the Krona breakdown.

    :param taxid: The genome's NCBI tax id.
    :param organism: The genome's organism name.
    :param domain_lookup: ``tax_id -> broad domain`` (bacteria/archaea/...), as built by
        ``scripts/build_taxonomy_domains.py``. When absent or missing the id, the domain level is
        ``"unknown"`` rather than dropped, so every hit still lands somewhere.
    :returns: A list of labels from broadest to most specific: domain, genus, species (strain-level
        detail is intentionally not a ring -- it would explode the plot with singletons).
    """
    domain = (domain_lookup or {}).get(str(taxid), "unknown") or "unknown"
    genus, species = _binomial(organism)
    lineage = [domain]
    if genus:
        lineage.append(genus)
        if species:
            lineage.append(f"{genus} {species}")
    return lineage


class _KronaNode(dict):
    """A hierarchy node: ``{"count": int, "children": {label: _KronaNode}}``."""


def krona_hierarchy(hits: Sequence[Hit],
                    domain_lookup: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """
    Aggregates hits into a nested taxonomic hierarchy for a Krona-style plot.

    Each hit increments the count of every node along its lineage, so a parent's count is the total
    of its descendants -- the invariant a sunburst needs for arc sizes to nest correctly.

    :param hits: The surviving hits (already threshold-filtered).
    :param domain_lookup: ``tax_id -> domain`` map; see :func:`genome_lineage`.
    :returns: ``{"count": total, "children": {label: {"count", "children"}}}``.
    """
    root: Dict[str, object] = {"count": 0, "children": {}}
    for hit in hits:
        root["count"] = int(root["count"]) + 1  # type: ignore[arg-type]
        node = root
        for label in genome_lineage(hit.taxid, hit.organism, domain_lookup):
            children: Dict[str, object] = node["children"]  # type: ignore[assignment]
            child = children.get(label)
            if child is None:
                child = {"count": 0, "children": {}}
                children[label] = child
            child["count"] = int(child["count"]) + 1  # type: ignore[arg-type]
            node = child
    return root


#: Palette keyed by broad domain, so a clade keeps one hue across the plot.
_DOMAIN_COLORS = {
    "bacteria": "#4e79a7", "archaea": "#59a14f", "virus": "#e15759", "fungi": "#b07aa1",
    "plant": "#76b7b2", "animal": "#f28e2b", "protozoa": "#edc948", "unknown": "#bab0ac",
}


def _arc_path(cx: float, cy: float, r0: float, r1: float,
              a0: float, a1: float) -> str:
    """
    Builds the SVG path for one annular arc (a sunburst wedge).

    :param cx: Centre x. :param cy: Centre y.
    :param r0: Inner radius. :param r1: Outer radius.
    :param a0: Start angle (radians). :param a1: End angle (radians).
    :returns: An SVG ``d`` attribute.
    """
    large = 1 if (a1 - a0) % (2 * math.pi) > math.pi else 0
    x0o, y0o = cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)
    x1o, y1o = cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)
    x0i, y0i = cx + r0 * math.cos(a1), cy + r0 * math.sin(a1)
    x1i, y1i = cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)
    return (f"M {x0o:.2f} {y0o:.2f} A {r1:.2f} {r1:.2f} 0 {large} 1 {x1o:.2f} {y1o:.2f} "
            f"L {x0i:.2f} {y0i:.2f} A {r0:.2f} {r0:.2f} 0 {large} 0 {x1i:.2f} {y1i:.2f} Z")


def krona_svg(hierarchy: Dict[str, object], size: int = 420,
              domain_lookup: Optional[Dict[str, str]] = None) -> str:
    """
    Renders a Krona-style radial sunburst of the taxonomic breakdown.

    Rings are taxonomic depth (domain, genus, species from the centre out); each wedge's angular
    span is proportional to its hit count, so a clade's share of the plot is its share of the hits.
    A wedge is coloured by its broad domain (its top-level ancestor), so a whole clade reads as one
    hue.

    :param hierarchy: The output of :func:`krona_hierarchy`.
    :param size: SVG width/height in pixels.
    :param domain_lookup: Unused directly; the colour comes from the top-level label, which is
        already the domain.
    :returns: A standalone ``<svg>`` string. Renders an empty ring with a "no hits" note when the
        hierarchy is empty, rather than a zero-division.
    """
    cx = cy = size / 2.0
    max_r = size / 2.0 - 8
    total = int(hierarchy.get("count", 0))  # type: ignore[arg-type]
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}" role="img" aria-label="Genomic distribution sunburst">']

    if total <= 0:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r}" fill="none" stroke="#ccc"/>')
        parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" '
                     f'dominant-baseline="middle" font-size="13" fill="#888">no genomic hits</text>')
        parts.append("</svg>")
        return "".join(parts)

    # Determine the depth so rings divide the radius evenly.
    def _depth(node: Dict[str, object]) -> int:
        children = node.get("children") or {}
        return 1 + max((_depth(c) for c in children.values()), default=0) if children else 0

    depth = max(1, _depth(hierarchy))
    ring = max_r / depth

    def _draw(node: Dict[str, object], level: int, a0: float, a1: float, domain: str) -> None:
        children: Dict[str, object] = node.get("children") or {}  # type: ignore[assignment]
        span = a1 - a0
        node_count = int(node.get("count", 0))  # type: ignore[arg-type]
        if node_count <= 0 or span <= 0:
            return
        angle = a0
        for label, child in children.items():
            child_count = int(child.get("count", 0))  # type: ignore[arg-type]
            frac = child_count / node_count
            child_a0, child_a1 = angle, angle + span * frac
            top_domain = label if level == 0 else domain
            color = _DOMAIN_COLORS.get(str(top_domain).lower(), "#9c9c9c")
            r0, r1 = ring * level, ring * (level + 1)
            parts.append(
                f'<path d="{_arc_path(cx, cy, r0, r1, child_a0, child_a1)}" '
                f'fill="{color}" fill-opacity="{0.9 - 0.12 * level:.2f}" '
                f'stroke="#fff" stroke-width="1"><title>{_escape(label)}: {child_count}</title></path>')
            _draw(child, level + 1, child_a0, child_a1, top_domain)
            angle = child_a1

    _draw(hierarchy, 0, -math.pi / 2, -math.pi / 2 + 2 * math.pi, "unknown")
    parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
                 f'font-size="12" fill="#444">{total} hits</text>')
    parts.append("</svg>")
    return "".join(parts)


def _escape(text: str) -> str:
    """:param text: Raw text. :returns: XML-escaped text safe inside an SVG ``<title>``."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------------------------------------
# Orchestration below. These shell out to BLAST and touch the network/disk; they are deliberately
# thin over the pure functions above, and are NOT exercised by the test suite (no genome is
# downloaded during tests). The heavy build loop is never started implicitly.
# --------------------------------------------------------------------------------------------------

#: RefSeq ``assembly_summary.txt`` refseq_category values worth searching. The reference/
#: representative subset is ~13,000 bacterial genomes rather than the full ~250,000-assembly set --
#: the scope the project owner narrowed this to, so a presence/absence sweep stays tractable.
REFERENCE_CATEGORIES: Tuple[str, ...] = ("reference genome", "representative genome")


def parse_assembly_summary(text: str,
                           categories: Sequence[str] = REFERENCE_CATEGORIES) -> List[Dict[str, str]]:
    """
    Parses NCBI's ``assembly_summary.txt`` into the genome records :func:`build_context` consumes.

    The file is tab-separated with a two-line ``#`` header; the columns used are the documented
    fixed positions: 0 ``assembly_accession``, 4 ``refseq_category``, 5 ``taxid``, 7
    ``organism_name``, 19 ``ftp_path``. The per-genome FASTA URL is derived from ``ftp_path`` by
    NCBI's own naming convention (``<basename>_genomic.fna.gz``), and ``https`` is forced in place of
    the file's ``ftp`` scheme, which is no longer served.

    :param text: The contents of ``assembly_summary.txt``.
    :param categories: Keep only rows whose ``refseq_category`` is one of these (case-insensitive);
        pass an empty sequence to keep every row.
    :returns: One dict per kept genome with ``accession``, ``taxid``, ``organism`` and ``url``.
    """
    wanted = {c.lower() for c in categories}
    records: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) <= 19:
            continue
        category = fields[4].strip().lower()
        if wanted and category not in wanted:
            continue
        ftp_path = fields[19].strip()
        if not ftp_path or ftp_path == "na":
            continue
        basename = ftp_path.rstrip("/").split("/")[-1]
        url = ftp_path.replace("ftp://", "https://").rstrip("/") + f"/{basename}_genomic.fna.gz"
        records.append({
            "accession": fields[0].strip(),
            "taxid": fields[5].strip(),
            "organism": fields[7].strip(),
            "url": url,
        })
    return records


def download_refseq_genome(record: Dict[str, str], dest_dir: str) -> Optional[str]:
    """
    Fetches and decompresses one genome FASTA. Network I/O; not exercised by the test suite.

    :param record: A genome record from :func:`parse_assembly_summary` (needs ``url`` and
        ``accession``).
    :param dest_dir: Directory to write the decompressed FASTA into.
    :returns: The path to the ``.fna`` written, or ``None`` on any failure (the caller skips it).
    """
    import gzip
    import shutil
    import urllib.request

    url = record.get("url")
    if not url:
        return None
    os.makedirs(dest_dir, exist_ok=True)
    accession = record.get("accession", "genome")
    out_path = os.path.join(dest_dir, f"{accession}.fna")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, \
                gzip.GzipFile(fileobj=response) as gz, \
                open(out_path, "wb") as fh:
            shutil.copyfileobj(gz, fh)
    except Exception as exc:  # noqa: BLE001 -- one genome failing must not abort the whole sweep
        logger.warning("Failed to download %s: %s", accession, exc)
        if os.path.exists(out_path):
            os.remove(out_path)
        return None
    return out_path


def search_genome(peptide_fasta: str, genome_fasta: str, work_dir: str,
                  min_coverage: float = MIN_COVERAGE, min_identity: float = MIN_IDENTITY,
                  genome_accession: str = "", taxid: str = "", organism: str = "",
                  cleanup: bool = True) -> List[Hit]:
    """
    Runs tBLASTn of the peptides against one genome and returns the surviving hits.

    Makes a BLAST nucleotide database from the genome, runs tBLASTn, parses and threshold-filters
    the output, tags each hit with the genome's taxonomy, then (by default) deletes the database and
    output -- the genome FASTA itself is the caller's to remove, since the caller downloaded it.

    :param peptide_fasta: FASTA of the peptide(s) to search for.
    :param genome_fasta: The genome nucleotide FASTA.
    :param work_dir: Scratch directory for the BLAST database and output.
    :param min_coverage: Minimum query coverage fraction.
    :param min_identity: Minimum identity fraction.
    :param genome_accession: Assembly accession to tag hits with.
    :param taxid: The genome's tax id, to tag hits with.
    :param organism: The genome's organism name, to tag hits with.
    :param cleanup: Delete the BLAST database and output afterward.
    :returns: The surviving hits, each tagged with the genome's taxonomy.
    """
    import glob
    import subprocess

    os.makedirs(work_dir, exist_ok=True)
    db_prefix = os.path.join(work_dir, "genome_db")
    out_path = os.path.join(work_dir, "tblastn.tsv")
    outfmt = "6 " + " ".join(TBLASTN_COLUMNS)

    hits: List[Hit] = []
    try:
        subprocess.run(["makeblastdb", "-in", genome_fasta, "-dbtype", "nucl",
                        "-out", db_prefix], check=True, capture_output=True)
        subprocess.run(["tblastn", "-query", peptide_fasta, "-db", db_prefix,
                        "-outfmt", outfmt, "-out", out_path,
                        # A gene-presence question, not a remote-homology one: keep the search tight
                        # and fast, and rely on the coverage/identity gate for the real filtering.
                        "-evalue", "1e-5", "-max_target_seqs", "50"],
                       check=True, capture_output=True)
        with open(out_path, encoding="utf-8") as fh:
            parsed = parse_tblastn(fh.read())
        survivors = filter_hits(parsed, min_coverage, min_identity)
        hits = [h._replace(genome_accession=genome_accession, taxid=taxid, organism=organism)
                for h in survivors]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.warning("tBLASTn against %s failed: %s", genome_accession or genome_fasta, exc)
    finally:
        if cleanup:
            for path in glob.glob(db_prefix + "*") + [out_path]:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return hits


def build_context(peptide_fasta: str, genomes: Sequence[Dict[str, str]],
                  download: Callable[[Dict[str, str], str], Optional[str]],
                  work_dir: str, domain_lookup: Optional[Dict[str, str]] = None,
                  min_coverage: float = MIN_COVERAGE,
                  min_identity: float = MIN_IDENTITY) -> Dict[str, object]:
    """
    Searches many genomes for the peptides, deleting each genome as it goes.

    This is the disk-/time-bound loop the module docstring warns about; it is never started
    implicitly. Downloading is injected (``download``) rather than hard-wired, so the loop's
    accounting and cleanup are testable against local files and the real driver supplies an
    FTP/HTTPS fetch. Each genome is downloaded, searched, and its FASTA deleted before the next, so
    peak disk use is one genome plus the accumulated hit rows.

    :param peptide_fasta: FASTA of the peptide(s) to search for.
    :param genomes: One dict per genome, each with at least ``accession``; optionally ``taxid``,
        ``organism`` and whatever ``download`` needs (e.g. ``url``).
    :param download: ``(genome_record, dest_dir) -> path`` fetching the genome FASTA, or ``None`` to
        skip it. The returned file is deleted after the search.
    :param work_dir: Scratch directory.
    :param domain_lookup: ``tax_id -> domain`` for the Krona aggregation.
    :param min_coverage: Minimum query coverage fraction.
    :param min_identity: Minimum identity fraction.
    :returns: ``{"hits": [...], "krona": {...}, "genomes_searched": n, "genomes_with_hits": m}``.
    """
    os.makedirs(work_dir, exist_ok=True)
    all_hits: List[Hit] = []
    searched = with_hits = 0

    for record in genomes:
        accession = record.get("accession", "")
        genome_path = None
        try:
            genome_path = download(record, work_dir)
            if not genome_path or not os.path.exists(genome_path):
                continue
            searched += 1
            hits = search_genome(
                peptide_fasta, genome_path, work_dir,
                min_coverage=min_coverage, min_identity=min_identity,
                genome_accession=accession, taxid=record.get("taxid", ""),
                organism=record.get("organism", ""))
            if hits:
                with_hits += 1
                all_hits.extend(hits)
        finally:
            # Delete the genome FASTA regardless of outcome -- the whole point is not to accumulate
            # thousands of multi-MB genomes on disk.
            if genome_path and os.path.exists(genome_path):
                try:
                    os.remove(genome_path)
                except OSError:
                    pass

    return {
        "hits": [h._asdict() for h in all_hits],
        "krona": krona_hierarchy(all_hits, domain_lookup),
        "genomes_searched": searched,
        "genomes_with_hits": with_hits,
    }


def sweep_to_store(peptide_id: str, sequence: str, genomes: Sequence[Dict[str, str]],
                   download: Callable[[Dict[str, str], str], Optional[str]], work_dir: str,
                   store_path: Optional[str] = None, min_coverage: float = MIN_COVERAGE,
                   min_identity: float = MIN_IDENTITY, resume: bool = True,
                   limit: Optional[int] = None) -> Dict[str, object]:
    """
    Sweeps genomes for one peptide, persisting each genome's hits to the database as it goes.

    This is the resumable, memory- and disk-bounded form of :func:`build_context`: one genome is
    downloaded, searched, its hits written to :mod:`sample.genomic_context_store`, and the genome
    FASTA deleted before the next. Because each genome is recorded as processed in the same
    transaction as its hits, an interrupted sweep (or one run in nightly slices) resumes by skipping
    the genomes already done, without re-downloading them. Nothing accumulates in memory across
    genomes and never more than one genome sits on disk.

    :param peptide_id: The query peptide's id (the key the hits are stored and read back under).
    :param sequence: The peptide's one-letter sequence.
    :param genomes: One dict per genome (from :func:`parse_assembly_summary`), each with at least
        ``accession`` and whatever ``download`` needs.
    :param download: ``(genome_record, dest_dir) -> path`` fetching the genome FASTA, or ``None`` to
        skip it. The returned file is deleted after the search.
    :param work_dir: Scratch directory (one genome plus the BLAST database live here transiently).
    :param store_path: The results database; the store default when omitted.
    :param min_coverage: Minimum query coverage fraction.
    :param min_identity: Minimum identity fraction.
    :param resume: Skip genomes already recorded for this peptide (the default). ``False`` re-searches
        every genome, overwriting its processed marker.
    :param limit: Stop after searching this many *new* genomes this run (for a bounded slice).
    :returns: The store's running :func:`sample.genomic_context_store.summary` for this peptide.
    """
    from sample import genomic_context_store as store

    store.init_db(store_path)
    os.makedirs(work_dir, exist_ok=True)
    peptide_fasta = os.path.join(work_dir, "query.faa")
    with open(peptide_fasta, "w", encoding="utf-8") as fh:
        fh.write(f">{peptide_id}\n{str(sequence).strip()}\n")

    already = store.processed_accessions(peptide_id, store_path) if resume else set()
    searched_this_run = 0

    for record in genomes:
        accession = record.get("accession", "")
        if resume and accession in already:
            continue
        genome_path = None
        try:
            genome_path = download(record, work_dir)
            if not genome_path or not os.path.exists(genome_path):
                continue
            hits = search_genome(
                peptide_fasta, genome_path, work_dir,
                min_coverage=min_coverage, min_identity=min_identity,
                genome_accession=accession, taxid=record.get("taxid", ""),
                organism=record.get("organism", ""))
            store.record_genome(peptide_id, accession, hits, store_path)
            searched_this_run += 1
        finally:
            # Delete the genome FASTA regardless of outcome; the whole point is one genome on disk.
            if genome_path and os.path.exists(genome_path):
                try:
                    os.remove(genome_path)
                except OSError:
                    pass
        if limit and searched_this_run >= limit:
            break

    result = store.summary(peptide_id, store_path)
    logger.info("Sweep for %s: %d genome(s) searched this run; store now holds %d hit(s) across "
                "%d/%d genomes.", peptide_id, searched_this_run, result["total_hits"],
                result["genomes_with_hits"], result["genomes_searched"])
    return result


def load_domain_lookup(taxonomy_db: Optional[str] = None) -> Dict[str, str]:
    """
    Reads the ``tax_id -> domain`` map from ``data/taxonomy.db``'s ``tax_domain`` table.

    :param taxonomy_db: Path to ``taxonomy.db``; the project default when omitted.
    :returns: The mapping, or an empty dict when the table or file is absent (the Krona plot then
        files every hit under ``"unknown"`` rather than failing).
    """
    import sqlite3

    from sample.config import DATA_DIR

    taxonomy_db = taxonomy_db or os.path.join(str(DATA_DIR), "taxonomy.db")
    if not os.path.exists(taxonomy_db):
        logger.warning("No taxonomy.db at %s; genomic-context domains will be 'unknown'.",
                       taxonomy_db)
        return {}
    lookup: Dict[str, str] = {}
    try:
        conn = sqlite3.connect(taxonomy_db)
        try:
            for tax_id, domain in conn.execute("SELECT tax_id, domain FROM tax_domain"):
                lookup[str(tax_id)] = str(domain)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("Could not read tax_domain from %s: %s", taxonomy_db, exc)
    return lookup
