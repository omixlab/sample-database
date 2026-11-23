"""
Gene Ontology terms, harvested from CAMP's own scraped GO field.

CAMP records a ``Gene Ontology`` cell for entries where UniProt's own annotation was pulled in at
scrape time -- one row per GO term, each carrying the term's category (biological process,
molecular function or cellular component), its definition and its evidence code. Nothing in this
pipeline reads it: it sits beside `Pfam` and `InterPro`, similarly unread, in files that were
harvested for their sequence and activity data only.

This is deliberately the *only* source implemented here, not a first step toward assigning GO
terms from Pfam or InterPro accessions. CAMP's own Pfam/InterPro fields are free text scraped from
a rendered page (``PF16047: Antimicrobial22 PF03032 : FSAP_sig_propep (...)`` -- no reliable
delimiter between consecutive entries) and would need their own fragile parser to turn into
accessions; Pfam family membership is already served far more reliably by the curated HMMER
assignment in ``peptide_families`` (608 families, gathering-threshold hits). Re-deriving the same
fact from CAMP's noisier text would create two disagreeing sources for one thing. GO terms have no
such existing source, so they are worth the parser; Pfam-to-GO mapping through QuickGO is future
work (**C5**), not blocked by this.

**Why regex, not an HTML/XML parser.** The field is already flattened to plain text by CAMP's own
scrape -- there is no markup left to parse, only run-together words. The category name (one of
three fixed phrases) and the evidence code (a GO evidence code, e.g. ``IEA``, ``IDA``) both come
from small closed vocabularies, which is what makes the boundary between one term's definition and
the next term's id findable at all in an unstructured blob.
"""
import datetime
import glob
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from sample.config import DATA_DIR

logger = logging.getLogger("sample.gene_ontology")

#: The three top-level GO namespaces, as CAMP spells them. A record's category is sometimes
#: omitted by the source scrape entirely (~29 % of terms observed), so it is optional in the
#: pattern and left blank rather than guessed.
_CATEGORIES = "Molecular function|Biological process|Cellular component"

#: One GO term entry: its id, an optional category, a free-text definition, and an evidence code.
#: The definition is matched non-greedily up to whichever comes first -- the next GO id, or the
#: end of the field -- which is what makes an unstructured, un-delimited blob decidable: neither
#: the category set nor the evidence-code shape can appear inside a real GO term definition (GO
#: definitions do not start with "Molecular function" etc., and do not end in a bare 2-4 letter
#: capitalised code).
_GO_ENTRY = re.compile(
    rf"GO:(\d{{7}})\s+(?:({_CATEGORIES})\s+)?(.*?)\s+([A-Z]{{2,4}})(?=\s+GO:\d{{7}}|\s*$)"
)

#: Category label -> short code, for compact display.
CATEGORY_LABELS = {
    "Molecular function": "MF",
    "Biological process": "BP",
    "Cellular component": "CC",
}


def parse_go_field(blob: Any) -> List[Dict[str, Optional[str]]]:
    """
    Parses one CAMP ``Gene Ontology`` cell into individual terms.

    :param blob: The raw cell value, e.g. ``"GO ID  Ontology  Definition  Evidence  GO:0005576
        Cellular component  Extracellular region  IDA  GO:0050829  ..."``.
    :returns: ``{"go_id", "category", "term", "evidence"}`` per entry found. A blob with no
        recognisable entries returns an empty list rather than raising -- the header row alone
        (``GO ID  Ontology  Definition  Evidence`` with nothing after it) is valid input meaning
        "no terms were actually scraped for this peptide".
    """
    text = str(blob or "")
    terms = []
    for go_num, category, term, evidence in _GO_ENTRY.findall(text):
        term = term.strip()
        if not term:
            continue
        terms.append({
            "go_id": f"GO:{go_num}",
            "category": category or None,
            "term": term,
            "evidence": evidence,
        })
    return terms


def harvest_gene_ontology(camp_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads every peptide's GO terms out of CAMP's scraped records.

    :param camp_dir: CAMP CSV directory; the packaged location when omitted.
    :returns: Sequence -> list of GO term dicts, for sequences carrying at least one parsed term.
    """
    camp_dir = camp_dir or os.path.join(str(DATA_DIR), "antimicrobial_databases", "camp")
    if not os.path.isdir(camp_dir):
        logger.warning(f"CAMP directory not found at {camp_dir}; no GO terms harvested.")
        return {}

    from sample.sequence import sequence_to_1_letter

    out: Dict[str, List[Dict[str, Any]]] = {}
    files = sorted(glob.glob(os.path.join(camp_dir, "*.csv")))
    unreadable = 0
    with_go = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            unreadable += 1
            continue

        go_match = re.search(r"^Gene Ontology,(.*)$", text, re.MULTILINE)
        seq_match = re.search(r"^Sequence,(.*)$", text, re.MULTILINE)
        if not go_match or not seq_match:
            continue

        terms = parse_go_field(go_match.group(1))
        if not terms:
            continue

        sequence = sequence_to_1_letter(seq_match.group(1).strip()).upper().strip()
        if not sequence:
            continue

        # A sequence can appear in more than one CAMP record (a peptide re-published under a
        # second accession); its GO terms accumulate rather than one record's overwriting another's.
        bucket = out.setdefault(sequence, [])
        seen = {(t["go_id"], t["evidence"]) for t in bucket}
        for term in terms:
            key = (term["go_id"], term["evidence"])
            if key not in seen:
                bucket.append(term)
                seen.add(key)
        with_go += 1

    if unreadable:
        logger.warning(f"{unreadable} of {len(files)} CAMP files could not be read.")
    logger.info(f"Gene Ontology: {sum(len(v) for v in out.values())} terms across "
               f"{len(out)} sequences ({with_go} CAMP records contributed).")
    return out


def build_gene_ontology(output_path: Optional[str] = None,
                        camp_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Harvests GO terms and writes them for the loader.

    :param output_path: Where to write; ``data/gene_ontology.json`` when omitted.
    :param camp_dir: CAMP CSV directory.
    :returns: A summary of what was written.
    """
    import json

    terms_by_sequence = harvest_gene_ontology(camp_dir)
    output_path = output_path or os.path.join(str(DATA_DIR), "gene_ontology.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"gene_ontology": terms_by_sequence}, handle, ensure_ascii=False, indent=1)

    total_terms = sum(len(v) for v in terms_by_sequence.values())
    summary = {"path": output_path, "sequences": len(terms_by_sequence), "terms": total_terms}
    logger.info(f"Wrote {total_terms} GO terms for {len(terms_by_sequence)} sequences to "
               f"{output_path}.")
    return summary


#: The official Gene Ontology, "basic" edition. Deliberately not the full `go.obo`: go-basic
#: strips every relation type except `is_a` and `part_of` specifically to guarantee an acyclic
#: hierarchy, which is what a term slice needs to stay well-formed -- the full edition adds
#: `regulates`-family relations that can introduce cycles a naive ancestor walk would loop on.
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"


def download_go_obo(output_path: Optional[str] = None, force: bool = False) -> str:
    """
    Downloads the official `go-basic.obo`.

    :param output_path: Local path; ``data/gene_ontology/go-basic.obo`` when omitted.
    :param force: Re-download even if the file already exists.
    :returns: The local path, or ``""`` if the download failed.
    """
    import requests

    output_path = output_path or os.path.join(str(DATA_DIR), "gene_ontology", "go-basic.obo")
    if os.path.exists(output_path) and not force:
        logger.info(f"{output_path} already present; skipping download (force=True to refetch).")
        return output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Downloading {GO_OBO_URL} ...")
    tmp_path = output_path + ".part"
    try:
        with requests.get(GO_OBO_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
        os.replace(tmp_path, output_path)
    except Exception as e:
        logger.error(f"Failed to download go-basic.obo: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return ""
    logger.info(f"Saved {os.path.getsize(output_path):,} bytes to {output_path}.")
    return output_path


def parse_obo(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses an OBO file's ``[Term]`` stanzas.

    Only what `go-basic.obo` actually needs: `id`, `name`, `namespace`, `is_a` and
    ``relationship: part_of`` parent edges, `alt_id` and `is_obsolete`. Not a general OBO parser --
    typedefs, other relationship types and instance stanzas are skipped, since go-basic does not
    use them.

    :param path: Local ``.obo`` file, e.g. from :func:`download_go_obo`.
    :returns: GO id -> ``{"name", "namespace", "is_a", "part_of", "alt_id", "is_obsolete"}``.
        `is_a` and `part_of` are lists of parent GO ids.
    """
    terms: Dict[str, Dict[str, Any]] = {}
    current: Optional[Dict[str, Any]] = None
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                current = {"name": None, "namespace": None, "is_a": [], "part_of": [],
                          "alt_id": [], "is_obsolete": False}
                continue
            if line.startswith("[") and line.endswith("]"):
                # [Typedef] or [Instance]: not a Term, and OBO stanzas do not nest, so whatever
                # comes next belongs to this new stanza, not the Term being built.
                current = None
                continue
            if current is None or ":" not in line:
                continue
            tag, _, value = line.partition(":")
            value = value.strip()
            if tag == "id":
                terms[value] = current
            elif tag == "name":
                current["name"] = value
            elif tag == "namespace":
                current["namespace"] = value
            elif tag == "is_a":
                current["is_a"].append(value.split("!")[0].strip())
            elif tag == "relationship" and value.startswith("part_of "):
                current["part_of"].append(value[len("part_of "):].split("!")[0].strip())
            elif tag == "alt_id":
                current["alt_id"].append(value)
            elif tag == "is_obsolete":
                current["is_obsolete"] = value.lower() == "true"
    return terms


def _ancestors(go_id: str, terms: Dict[str, Dict[str, Any]]) -> Set[str]:
    """:returns: Every `is_a`/`part_of` ancestor of `go_id` in `terms`, not including itself."""
    seen: Set[str] = set()
    root = terms.get(go_id, {})
    stack = list(root.get("is_a", []) + root.get("part_of", []))
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        term = terms.get(parent)
        if term:
            stack.extend(term["is_a"])
            stack.extend(term["part_of"])
    return seen


def build_go_slice(gene_ontology_json: Optional[str] = None, go_obo_path: Optional[str] = None,
                   output_path: Optional[str] = None, download_if_missing: bool = True) -> Dict[str, Any]:
    """
    Slices the real Gene Ontology down to the terms this project's peptides actually carry.

    Every GO id :func:`harvest_gene_ontology` pulled from CAMP is looked up in the official
    `go-basic.obo` -- real Gene Ontology accessions and names, not an invented vocabulary. Each
    resolves or it does not; the resolution rate is reported so a scraping regression shows up as a
    number, not a silent gap (see `sample.gene_ontology`'s own module docstring on why CAMP's
    scrape is trusted at all -- this is the check that trust was previously missing). Every
    resolved term's full `is_a`/`part_of` ancestor chain is included too, up to the ontology roots,
    so the slice is a connected sub-hierarchy of the real GO rather than a flat list with no
    structure -- a term with no path to a root is not usably placed in an ontology browser.

    :param gene_ontology_json: Harvested GO terms; ``data/gene_ontology.json`` when omitted.
    :param go_obo_path: Local `go-basic.obo`; ``data/gene_ontology/go-basic.obo`` when omitted.
    :param output_path: Where to write the slice; ``data/gene_ontology/sample_go_slice.obo`` when
        omitted.
    :param download_if_missing: Fetch `go-basic.obo` via :func:`download_go_obo` if it is not
        already on disk.
    :returns: A summary covering resolution (direct, via `alt_id`, unresolved) and term counts
        (directly used, ancestor-only, total). Zeroed out, rather than raising, when the harvested
        terms or the ontology file are unavailable -- this is a derived, best-effort artifact, not
        one anything else in the pipeline depends on to run.
    """
    import json

    gene_ontology_json = gene_ontology_json or os.path.join(str(DATA_DIR), "gene_ontology.json")
    output_path = output_path or os.path.join(
        str(DATA_DIR), "gene_ontology", "sample_go_slice.obo")
    empty_summary = {"path": output_path, "harvested_unique_ids": 0, "resolved_direct": 0,
                     "resolved_via_alt_id": 0, "unresolved": 0, "used_terms": 0,
                     "ancestor_terms": 0, "total_terms_in_slice": 0}

    if not os.path.exists(gene_ontology_json):
        logger.warning(f"No harvested GO terms at {gene_ontology_json}; nothing to slice.")
        return empty_summary

    go_obo_path = go_obo_path or os.path.join(str(DATA_DIR), "gene_ontology", "go-basic.obo")
    if not os.path.exists(go_obo_path):
        if not download_if_missing:
            logger.warning(f"No go-basic.obo at {go_obo_path} and download_if_missing is False.")
            return empty_summary
        go_obo_path = download_go_obo(go_obo_path)
        if not go_obo_path:
            return empty_summary

    with open(gene_ontology_json, encoding="utf-8") as handle:
        harvested = (json.load(handle) or {}).get("gene_ontology", {})

    used: Dict[str, int] = {}
    for entries in harvested.values():
        for entry in entries:
            go_id = entry.get("go_id")
            if go_id:
                used[go_id] = used.get(go_id, 0) + 1

    terms = parse_obo(go_obo_path)
    alt_to_primary = {alt: primary for primary, term in terms.items() for alt in term["alt_id"]}

    resolved_direct = 0
    resolved_via_alt = 0
    unresolved: List[str] = []
    used_primary_ids: Set[str] = set()
    # A harvested id maps to peptide-carried usage counts by its own id; once resolved to a
    # (possibly different) primary id, the counts for that primary id accumulate across every
    # harvested id -- direct or alt -- that resolves to it.
    usage_by_primary: Dict[str, int] = {}
    for go_id, count in used.items():
        primary = go_id if go_id in terms else alt_to_primary.get(go_id)
        if primary is None:
            unresolved.append(go_id)
            continue
        if go_id in terms:
            resolved_direct += 1
        else:
            resolved_via_alt += 1
        used_primary_ids.add(primary)
        usage_by_primary[primary] = usage_by_primary.get(primary, 0) + count

    if unresolved:
        preview = sorted(unresolved)[:10]
        logger.warning(f"{len(unresolved)} of {len(used)} harvested GO ids do not exist in "
                       f"go-basic.obo (obsolete or a scraping error): {preview}"
                       f"{'...' if len(unresolved) > 10 else ''}")

    all_ids: Set[str] = set(used_primary_ids)
    for go_id in used_primary_ids:
        all_ids |= _ancestors(go_id, terms)

    lines = [
        "format-version: 1.2",
        "ontology: sample-go-slice",
        f"date: {datetime.date.today().isoformat()}",
        f"remark: Subset of the official Gene Ontology ({GO_OBO_URL}) limited to terms this "
        f"project's peptides carry (harvested from CAMP) plus their is_a/part_of ancestors -- a "
        f"connected sub-hierarchy of the real GO, not an invented vocabulary.",
        "",
    ]
    for go_id in sorted(all_ids):
        term = terms[go_id]
        lines.append("[Term]")
        lines.append(f"id: {go_id}")
        if term["name"]:
            lines.append(f"name: {term['name']}")
        if term["namespace"]:
            lines.append(f"namespace: {term['namespace']}")
        for parent in term["is_a"]:
            if parent in all_ids:
                parent_name = terms.get(parent, {}).get("name")
                lines.append(f"is_a: {parent}" + (f" ! {parent_name}" if parent_name else ""))
        for parent in term["part_of"]:
            if parent in all_ids:
                parent_name = terms.get(parent, {}).get("name")
                lines.append(f"relationship: part_of {parent}"
                             + (f" ! {parent_name}" if parent_name else ""))
        if term["is_obsolete"]:
            # GO evolves; CAMP's scrape is old enough that some of what it annotated has since
            # been retired. An obsolete term is still a real fact about what CAMP recorded, so it
            # stays in the slice rather than being dropped -- but silently presenting it as a
            # live, current GO term would be wrong, and go-basic.obo's own convention for exactly
            # this is the `is_obsolete: true` tag, so that is what gets written. Obsolete terms
            # also carry no is_a/part_of in go-basic (their hierarchy is stripped on retirement),
            # so this term has no parent lines above regardless -- it is a deliberate, marked
            # exception to "every included term reaches a root", not a bug in the ancestor walk.
            lines.append("is_obsolete: true")
        if go_id in usage_by_primary:
            lines.append(f'comment: "{usage_by_primary[go_id]} SAMPLE peptide annotation(s) '
                         f'carry this term directly."')
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    obsolete_used = sum(1 for go_id in used_primary_ids if terms[go_id]["is_obsolete"])
    summary = {
        "path": output_path,
        "harvested_unique_ids": len(used),
        "resolved_direct": resolved_direct,
        "resolved_via_alt_id": resolved_via_alt,
        "unresolved": len(unresolved),
        "used_terms": len(used_primary_ids),
        "obsolete_used_terms": obsolete_used,
        "ancestor_terms": len(all_ids) - len(used_primary_ids),
        "total_terms_in_slice": len(all_ids),
    }
    logger.info(f"GO slice: {summary['used_terms']} directly-used terms "
               f"({resolved_direct} resolved directly, {resolved_via_alt} via alt_id, "
               f"{len(unresolved)} unresolved, {obsolete_used} of the resolved ones now obsolete "
               f"in GO but marked as such rather than dropped) + {summary['ancestor_terms']} "
               f"ancestor terms = {summary['total_terms_in_slice']} total -> {output_path}")
    return summary
