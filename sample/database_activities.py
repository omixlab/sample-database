"""
Structured activity, publication and provenance data from the raw antimicrobial-peptide databases.

`merge_databases` (step 7) reduces every source database to one row per sequence with a free-text
`Biological_Activity` column -- for DBAASP that column holds `"[{'name': 'Gram+'}]"`, the target
group labels and nothing else. Meanwhile each DBAASP record carries a `targetActivities` array with
the measured concentration, its unit, the target species, and the growth medium; across the corpus
that is roughly 194,500 measurements, against 29,900 the pipeline extracts from literature. All of
it was being discarded at parse time.

This module reads the raw files and returns measurements in the same shape the extraction pipeline
produces, so both go through one contract (sample/strict_schema.py) and land in one table.

Every value carries where it came from. A record states which database supplied it, that database's
own identifier, and the publication behind it, so a consumer can tell a DBAASP MIC from one this
project read out of a paper -- and so the claim that SAMPLE adds new data can be checked rather
than asserted.
"""
import csv
import glob
import json
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger("sample.database_activities")

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# Source assay names -> the pipeline's controlled vocabulary. The left side is what the databases
# actually write, including their spacing and suffix quirks ("IC50 I", "LC"), collected from the
# real files rather than from their documentation.
_MEASURE_MAP: Dict[str, str] = {
    "mic": "MIC", "mic50": "MIC50", "mic90": "MIC90", "mic80": "MIC80", "mic100": "MIC100",
    "mbc": "MBC", "mbc50": "MBC50", "mbc90": "MBC90", "mfc": "MFC",
    "ic50": "IC50", "ic50 i": "IC50", "ic90": "IC90",
    "ec50": "EC50", "ec90": "EC90",
    "lc50": "LC50", "lc90": "LD90", "ld50": "LD50", "ld90": "LD90",
    "cc50": "CC50", "gi50": "GI50", "ed50": "ED50", "hc50": "HC50",
    "mbic": "MBIC", "mbec": "MBEC", "zoi": "ZOI", "fic": "FIC",
    "mlc": "MLC", "mhc": "MHC",
}

# Target-group labels used by the databases, mapped to the pipeline's activity classes. These are
# clade labels rather than organisms ("Gram+" is not a species), so they set the class and never
# the target_organism field.
_GROUP_LABELS: Dict[str, str] = {
    "gram+": "antibacterial", "gram-": "antibacterial", "gram +": "antibacterial",
    "gram -": "antibacterial", "gram-positive": "antibacterial",
    "gram-negative": "antibacterial", "bacteria": "antibacterial",
    "fungi": "antifungal", "fungus": "antifungal", "yeast": "antifungal",
    "virus": "antiviral", "viruses": "antiviral",
    "parasite": "antiparasitic", "protozoa": "antiparasitic",
    "cancer": "anticancer", "cancer cells": "anticancer", "tumor": "anticancer",
    "mammalian cells": "other", "insect": "other",
}


# Unit equivalences and magnitude conversions, as (value -> (value, canonical_unit)). A molar
# concentration per litre IS the corresponding molarity -- 1 umol/L == 1 uM -- so these collapse to
# one spelling with no arithmetic; mM and mg/L carry a factor.
_EQUIVALENT_UNITS: Dict[str, Callable[[float], tuple]] = {
    "umol/L": lambda v: (v, "uM"),
    "nmol/L": lambda v: (v, "nM"),
    "mmol/L": lambda v: (v * 1000.0, "uM"),
    "mol/L": lambda v: (v * 1e6, "uM"),
    "mM": lambda v: (v * 1000.0, "uM"),
    "mg/L": lambda v: (v, "ug/mL"),
    "g/L": lambda v: (v, "mg/mL"),
}

# How a value entered this database. Anything read out of an external source is Third Party; only
# what this pipeline extracted from primary literature and patents is SAMPLE-original. Recorded on
# every activity so the distinction survives into the published table.
#: Marks a value read out of an external database rather than derived by this project.
PROVENANCE_THIRD_PARTY: str = "Third Party"
#: Marks a value this pipeline extracted from primary literature or patents.
PROVENANCE_SAMPLE: str = "SAMPLE"


def _clean(value: Any) -> str:
    """
    Collapses whitespace and coerces to ``str``, treating ``None`` as empty.

    :param value: Any scalar from a source record; these files mix ``None``, numbers and strings.
    :returns: Whitespace-normalized string, never ``None``.
    """
    return " ".join(str(value or "").split()).strip()


def resolve_target_species(raw: Any) -> str:
    """
    Resolves a database target-species string to an NCBI scientific name.

    Sources write the strain along with the species -- "Escherichia coli MTCC 443",
    "Staphylococcus aureus ATCC 29213" -- and no strain designation resolves against the taxonomy,
    so a direct lookup discards essentially every DBAASP target. Falls back to the binomial and
    then to the genus, which is what the strain is a strain OF.

    Returns "" for clade labels like "Gram+", which are not organisms; those become a target_group
    instead (see _GROUP_LABELS).
    """
    from sample.nlp import resolve_scientific_name

    name = _clean(raw)
    if not name or name.lower() in _GROUP_LABELS:
        return ""
    resolved = resolve_scientific_name(name)
    if resolved:
        return resolved
    tokens = re.sub(r'[^A-Za-z .\-]', ' ', name).split()
    for width in (3, 2, 1):
        if len(tokens) >= width:
            resolved = resolve_scientific_name(" ".join(tokens[:width]))
            if resolved:
                return resolved
    return ""


def _target_group_for(raw_label: Any, organism: str, activity_type: str) -> str:
    """Activity class from the database's own label, falling back to the pipeline's inference."""
    from sample.nlp import infer_activity_group

    label = _clean(raw_label).lower()
    if label in _GROUP_LABELS:
        return _GROUP_LABELS[label]
    return infer_activity_group(organism, activity_type) or "other"


def _build_activity(*, activity_type: Any, value: Any, unit: Any, organism: str = "",
                    cell_line: str = "", group_label: str = "", raw_value: Any = None,
                    extras: Optional[Dict[str, Any]] = None, database: str = "",
                    record_id: str = "", reference: str = "") -> Optional[Dict[str, Any]]:
    """
    Assembles one activity in the pipeline's record shape and puts it through the same contract as
    an extracted one. Returns None if it does not survive -- a database is not exempt from the
    contract, and these sources contain their share of unparseable values and unit-less rows.
    """
    from sample.preprocessing import canonicalize_unit
    from sample.nlp import parse_activity_value_fast
    from sample.strict_schema import enforce_activity

    measure = _MEASURE_MAP.get(_clean(activity_type).lower())
    if not measure:
        return None

    canonical_unit = canonicalize_unit(_clean(unit))
    operator, parsed, _ = ("", None, None)
    if value is not None and value != "":
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            operator, parsed, _ = parse_activity_value_fast(str(value))
    if parsed is None and raw_value:
        operator, parsed, _ = parse_activity_value_fast(str(raw_value))
    if parsed is None:
        return None

    # Magnitude conversions the contract expects to have happened already, plus the equivalences
    # that only show up in these sources: DRAMP reports molar concentrations as "umol/L" and DBAASP
    # as "uM", which are the same quantity written two ways. Left alone they split one column into
    # two units that no consumer can group -- exactly the defect the unit work fixed for the
    # extraction path, arriving here through a different door.
    parsed, canonical_unit = _EQUIVALENT_UNITS.get(canonical_unit, lambda v: (v, canonical_unit))(parsed)

    organism = resolve_target_species(organism)
    act = {
        "activity_type": measure,
        "activity_unit": canonical_unit,
        "value_raw_unit": canonical_unit,
        "value_raw": _clean(raw_value if raw_value is not None else value),
        "value_lower": parsed,
        "value_operator": operator or "",
        "target_organism": organism,
        "target_cell_line": _clean(cell_line),
        "target_group": _target_group_for(group_label, organism, measure),
        "context_sentence": "",
    }
    if extras:
        act.update({k: v for k, v in extras.items() if v})
    clean, _violations = enforce_activity(act)
    if clean is None:
        return None
    # Attached after the contract runs, so provenance describes a measurement that was actually
    # accepted, and so these fields can never be what makes a record pass.
    clean["provenance"] = PROVENANCE_THIRD_PARTY
    clean["source_database"] = database
    clean["source_record_id"] = record_id
    if reference:
        clean["source_reference"] = reference
    return clean


# --- DBAASP ------------------------------------------------------------------------------------

def _dbaasp_publications(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Publication metadata from a DBAASP record's `articles` array."""
    pubs = []
    for art in record.get("articles") or []:
        if not isinstance(art, dict):
            continue
        pubmed = (art.get("pubmed") or {}).get("pubmedId") if isinstance(art.get("pubmed"), dict) else None
        pubs.append({
            "pubmed_id": _clean(pubmed),
            "doi": _clean(art.get("doi")),
            "title": _clean(art.get("title")),
            "journal": _clean((art.get("journal") or {}).get("name")
                              if isinstance(art.get("journal"), dict) else art.get("journal")),
            "year": art.get("year"),
            "volume": _clean(art.get("volume")),
            "pages": _clean(art.get("pages")),
            "authors": [_clean(a.get("name")) for a in (art.get("authors") or [])
                        if isinstance(a, dict) and a.get("name")],
        })
    return [p for p in pubs if any(p.values())]


# DBAASP's targetActivities array is built for pathogen MIC/IC50 data, so targetSpecies.name goes
# through resolve_target_species()'s bacterial-strain-stripping logic everywhere in this module.
# Applied to a mammalian cell-line string -- "Human cervical carcinoma HeLa", "Murine macrophage
# cells RAW 264.7" -- that logic finds only "Human"/"Mouse" as an NCBI-resolvable token and
# discards the rest, collapsing a real, specific assay target down to a bare species name. Every
# such string in this corpus starts with one of these three words; a virus name sometimes does too
# ("Human Influenza A Virus H1N1") and is deliberately excluded below, since the host-species
# prefix there names who the virus infects, not a cell line to look up.
_MAMMALIAN_ORGANISM_PREFIXES: Dict[str, str] = {
    "human": "Homo sapiens", "mouse": "Mus musculus", "murine": "Mus musculus",
}


def _mammalian_cell_candidates(raw_name: str) -> List[str]:
    """
    Trailing-substring candidates, longest first, for a Cellosaurus cell-line lookup against a
    DBAASP targetSpecies name that names a mammalian cell line rather than a pathogen.

    Cell-line codes in this corpus are consistently the last 1-2 tokens of the name ("...HeLa",
    "...RAW 264.7", "...HCT 116", "...K562/ADM") after a "Human"/"Mouse"/"Murine" prefix and a
    tissue/cancer-type descriptor; trying widths 3, 2, 1 (the same shrinking-window idea
    :func:`resolve_target_species` uses for species/strain) covers every pattern seen without
    needing to know in advance how many words the code itself takes.

    :param raw_name: A ``targetSpecies.name`` string, uncleaned.
    :returns: Candidate substrings, longest (most specific) first; empty when `raw_name` does not
        start with a recognized organism prefix, or names a virus rather than a cell line.
    """
    name = _clean(raw_name)
    tokens = name.split()
    if not tokens or tokens[0].lower() not in _MAMMALIAN_ORGANISM_PREFIXES:
        return []
    if "virus" in name.lower():
        return []
    remainder = tokens[1:]
    return [" ".join(remainder[-width:]) for width in (3, 2, 1) if len(remainder) >= width]


def _resolve_mammalian_target(raw_name: str, cell_categories: Dict[str, Tuple[str, bool]]
                              ) -> Tuple[str, str, str]:
    """
    Splits a DBAASP ``targetActivities`` species string into a mammalian cell-line target, using
    `cell_categories` (a :func:`sample.cell_lines.build_cell_line_categories` result, precomputed
    once for the whole ingestion run rather than looked up per record) to confirm which candidate
    from :func:`_mammalian_cell_candidates` is a real, known cell line.

    :param raw_name: A ``targetSpecies.name`` string, uncleaned.
    :param cell_categories: Cleaned cell-line name -> ``(category, is_cancer)``, already resolved.
    :returns: ``(organism, cell_line, group_label)``. `organism` is ``"Homo sapiens"``/``"Mus
        musculus"``, or ``""`` when `raw_name` is not a recognized mammalian cell-line string (the
        caller falls back to :func:`resolve_target_species`'s existing behavior). `cell_line` is
        the matched, cleaned Cellosaurus name, or ``""`` when no candidate resolved -- still a real
        mammalian target, just one this pass could not confirm a cell-line spelling for.
        `group_label` is ``"cancer"`` when Cellosaurus classifies the matched line as tumour-derived
        (see :data:`_GROUP_LABELS`), otherwise ``"mammalian cells"`` -- e.g. RAW 264.7 macrophages
        or red-blood-cell hemolysis targets, which are toxicity/safety data, not efficacy.
    """
    from sample.cell_lines import clean_cell_line_name

    name = _clean(raw_name)
    tokens = name.split()
    if not tokens or tokens[0].lower() not in _MAMMALIAN_ORGANISM_PREFIXES:
        return "", "", ""
    organism = _MAMMALIAN_ORGANISM_PREFIXES[tokens[0].lower()]
    for candidate in _mammalian_cell_candidates(raw_name):
        cleaned = clean_cell_line_name(candidate)
        if not cleaned or cleaned not in cell_categories:
            continue
        _category, is_cancer = cell_categories[cleaned]
        return organism, cleaned, ("cancer" if is_cancer else "mammalian cells")
    return organism, "", "mammalian cells"


def load_dbaasp(base_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yields structured records from the DBAASP JSON dump -- the only source with per-target values."""
    from sample.cell_lines import build_cell_line_categories

    files = sorted(glob.glob(os.path.join(base_dir, "dbaasp", "*.json")))
    if limit:
        files = files[:limit]

    # First pass: every candidate mammalian cell-line substring across the whole corpus, resolved
    # against Cellosaurus in ONE batch call (loads/saves its disk cache once, per-name lookups only
    # for what is not already cached) rather than once per record -- the same corpus has only a few
    # hundred distinct targetSpecies strings worth checking, seen tens of thousands of times.
    candidates: Set[str] = set()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:
            continue
        for act in rec.get("targetActivities") or []:
            if not isinstance(act, dict):
                continue
            name = (act.get("targetSpecies") or {}).get("name")
            if name:
                candidates.update(_mammalian_cell_candidates(name))
    cell_categories = build_cell_line_categories(candidates) if candidates else {}

    unreadable = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:
            # Counted rather than silently dropped -- a bare `except: continue` in exactly this
            # shape once hid a 60% loss of the activity table with no error in the log. See
            # DECISIONS.md.
            unreadable += 1
            continue
        sequence = _clean(rec.get("sequence"))
        if not sequence:
            continue

        record_id = _clean(rec.get("dbaaspId")) or os.path.splitext(os.path.basename(path))[0]
        activities = []

        for act in rec.get("targetActivities") or []:
            if not isinstance(act, dict):
                continue
            raw_name = (act.get("targetSpecies") or {}).get("name")
            # DBAASP packs both pathogen MIC data and mammalian-cell IC50/cytotoxicity data into
            # this same array. A "Human"/"Mouse"/"Murine"-prefixed, non-virus name is the latter --
            # resolved to (organism, cell_line, group_label) via Cellosaurus above rather than
            # collapsed to a bare species name by the pathogen-strain-stripping path below.
            mammalian_organism, cell_line, group_label = _resolve_mammalian_target(
                raw_name, cell_categories)
            if cell_line:
                # A confirmed cell line implies its species -- target_organism stays empty, the
                # same convention hemoliticCytotoxicActivities below already uses, so a consumer
                # reads the cell line rather than two columns saying the same thing two ways.
                organism, group_label_for_build = "", group_label
            elif mammalian_organism:
                # Recognized as human/mouse-derived, but no candidate resolved in Cellosaurus for
                # this pass -- still real, just not a spelling this pass could confirm as a named
                # cell line, so at least the species survives instead of being lost entirely.
                organism, group_label_for_build = mammalian_organism, group_label
            else:
                # Not a mammalian-cell string at all -- today's pathogen/species path, unchanged.
                organism, group_label_for_build = raw_name, ""
            built = _build_activity(
                activity_type=(act.get("activityMeasureGroup") or {}).get("name"),
                value=act.get("activity"),
                raw_value=act.get("concentration"),
                unit=(act.get("unit") or {}).get("name"),
                organism=organism,
                cell_line=cell_line,
                group_label=group_label_for_build,
                extras={"assay_medium": _clean((act.get("medium") or {}).get("name")),
                        "assay_ph": _clean(act.get("ph"))},
                database="DBAASP", record_id=record_id,
            )
            if built:
                activities.append(built)

        # Hemolysis and cytotoxicity are reported against cells, not species, so the target lands in
        # target_cell_line and the class is fixed rather than inferred from an organism.
        for act in rec.get("hemoliticCytotoxicActivities") or []:
            if not isinstance(act, dict):
                continue
            built = _build_activity(
                activity_type=(act.get("activityMeasureGroup") or {}).get("name"),
                value=act.get("activity"), raw_value=act.get("concentration"),
                unit=(act.get("unit") or {}).get("name"),
                cell_line=(act.get("targetCell") or {}).get("name") if isinstance(act.get("targetCell"), dict) else "",
                group_label="mammalian cells",
                database="DBAASP", record_id=record_id,
            )
            if built:
                activities.append(built)

        for act in rec.get("antibiofilmActivities") or []:
            if not isinstance(act, dict):
                continue
            built = _build_activity(
                activity_type=(act.get("activityMeasureGroup") or {}).get("name"),
                value=act.get("activity"), raw_value=act.get("concentration"),
                unit=(act.get("unit") or {}).get("name"),
                organism=(act.get("targetSpecies") or {}).get("name"),
                database="DBAASP", record_id=record_id,
            )
            if built:
                activities.append(built)

        genes = rec.get("sourceGenes") or []
        source_organism = resolve_target_species(genes[0].get("source")) if genes else ""

        yield {
            "database": "DBAASP",
            "record_id": record_id,
            "url": _clean(rec.get("url")),
            "sequence": sequence,
            "name": _clean(rec.get("name")) or _clean(rec.get("majorName")),
            "source_organism": source_organism,
            "synthesis_type": _clean((rec.get("synthesisType") or {}).get("name")),
            "n_terminus": _clean((rec.get("nTerminus") or {}).get("name")),
            "c_terminus": _clean((rec.get("cTerminus") or {}).get("name")),
            "smiles": _clean(rec.get("smiles")),
            "cross_references": {
                "uniprot": [_clean(u.get("name")) for u in (rec.get("uniprots") or [])
                            if isinstance(u, dict)],
                "pdb": [_clean(p.get("name")) for p in (rec.get("pdbs") or []) if isinstance(p, dict)],
                "pubchem": _clean(rec.get("pubChem")),
            },
            "publications": _dbaasp_publications(rec),
            "activities": activities,
        }


# --- DRAMP -------------------------------------------------------------------------------------

# DRAMP packs its measurements into one free-text column:
#   "Gram-positive bacteria: Staphylococcus aureus 25923 (MIC=16 umol/L), S. epidermidis (MIC=2)##..."
# Species then optional strain, then the assay and value in parentheses. "##" separates groups.
_DRAMP_MEASUREMENT = re.compile(
    r'\(\s*(?P<assay>MIC\d*|MBC|MFC|IC50|EC50|LD50|LC50|MLC|MBIC|MBEC)\s*'
    r'(?:值)?\s*[=:＝]\s*'
    r'(?P<value>[<>≤≥~]?\s*\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?)\s*'
    r'(?P<unit>[^\s)]*)\s*\)',
    re.IGNORECASE
)

# Case-sensitive on purpose. The measurement pattern above is case-insensitive so it can match
# "mic=" as well as "MIC=", but a binomial is defined by its capitalization -- and running the
# organism pattern under IGNORECASE made [A-Z] match lowercase, so in "Gram-positive bacteria:
# Staphylococcus aureus 25923 (MIC=16)" it captured "positive bacteria" as the species and every
# DRAMP target resolved to nothing.
_BINOMIAL = re.compile(r'\b([A-Z][a-z]{2,}\.?(?:\s+[a-z][a-z-]{2,}){1,2})')


def _parse_dramp_targets(text: Any, group_hint: str = "",
                         record_id: str = "") -> List[Dict[str, Any]]:
    """
    Extracts every (organism, assay, value, unit) measurement from a DRAMP target column.

    Anchors on the parenthesised measurement and then searches BACKWARDS for the nearest binomial,
    rather than trying to match organism and measurement in one pattern. A single chunk lists
    several species with their own values -- "S. aureus 25923 (MIC=16), S. epidermidis (MIC=2)" --
    and the group prefix ("Gram-positive bacteria:") sits in front of all of them, so scanning
    forwards attaches the first name found to every measurement after it.
    """
    out = []
    for chunk in re.split(r'##', _clean(text)):
        label = chunk.split(":", 1)[0] if ":" in chunk[:60] else ""
        for m in _DRAMP_MEASUREMENT.finditer(chunk):
            preceding = chunk[:m.start()]
            names = _BINOMIAL.findall(preceding)
            built = _build_activity(
                activity_type=m.group("assay"),
                value=None, raw_value=m.group("value"),
                unit=m.group("unit") or "ug/mL",
                organism=names[-1] if names else "",
                group_label=label or group_hint,
                database="DRAMP", record_id=record_id,
            )
            if built:
                out.append(built)
    return out


def _dramp_rows(base_dir: str) -> Iterator[Dict[str, str]]:
    """Every DRAMP row, from both the Datasets and Activity exports, de-duplicated by DRAMP_ID."""
    seen = set()
    patterns = [os.path.join(base_dir, "dramp", "Datasets", "*.txt"),
                os.path.join(base_dir, "dramp", "Activity", "*.txt"),
                os.path.join(base_dir, "dramp", "Subclass", "*.txt")]
    total = unreadable = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            total += 1
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for row in csv.DictReader(fh, delimiter="\t"):
                        rid = _clean(row.get("DRAMP_ID"))
                        if rid and rid not in seen:
                            seen.add(rid)
                            yield row
            except Exception as e:
                unreadable += 1
                logger.debug(f"DRAMP file {path} unreadable: {e}")
    # Counted and surfaced at WARNING, not swallowed at DEBUG: this project's rule (DECISIONS.md)
    # is that a skipped input is never silent, because a bare except in the DBAASP ingest loop once
    # hid a 60% activity-table loss with no error anywhere. `load_dramp` used to carry a broken
    # attempt at this -- it referenced names from the DBAASP loader it was copied from, so a full
    # run (one that never returned early on `limit`) died with NameError right here.
    if unreadable:
        logger.warning(f"DRAMP: {unreadable} of {total} files could not be read.")


#: Column names that mean the same thing in different DRAMP exports.
#:
#: DRAMP's five Datasets files do not share a schema, and the differences are not systematic --
#: `general_amps.txt` has `Hemolytic_activity` and `Swiss_Prot_Entry`, `specific_amps.txt` has
#: `Hemolytic_Activity` and `Uniprot_Entry`, `stability_amps.txt` quotes every header. A reader
#: keyed on one spelling silently returns nothing for the files that use the other, and "nothing"
#: is indistinguishable here from "this entry has no hemolytic data".
#:
#: Verified against the 2026-03 dump and the live 2026-10 (DRAMP 5.0) exports; both carry the
#: same variants, so these are long-standing upstream inconsistencies rather than a 5.0 change.
_DRAMP_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sequence": ("Original_Sequence", "Sequence"),
    "hemolytic": ("Hemolytic_activity", "Hemolytic_Activity"),
    "uniprot": ("Swiss_Prot_Entry", "Uniprot_Entry"),
    "topology": ("Linear/Cyclic/Branched", "Linear/Cyclic"),
    "n_terminal": ("N-terminal_Modification", "N_terminal_Modification"),
    "c_terminal": ("C-terminal_Modification", "C_terminal_Modification"),
    "other_modifications": ("Other_Modifications",
                            "Special_Amino_Acid_and_Stapling_Position"),
    "protein_existence": ("Protein_existence", "Protein_Existence"),
    "smiles": ("SMILES",),
}


def dramp_field(row: Dict[str, str], key: str) -> str:
    """
    One logical DRAMP field, whichever column name this export happens to use.

    :param row: A row from :func:`_dramp_rows`.
    :param key: A key of :data:`_DRAMP_FIELD_ALIASES`.
    :returns: The first alias present and non-empty, cleaned; ``""`` when none is.
    :raises KeyError: For an unknown key -- a typo must fail loudly rather than return ``""``,
        which would look exactly like a legitimately empty field.
    """
    for name in _DRAMP_FIELD_ALIASES[key]:
        value = _clean(row.get(name))
        if value:
            return value
    return ""


def dramp_sequence(row: Dict[str, str]) -> str:
    """
    The plain one-letter sequence for a DRAMP row.

    :param row: A row from :func:`_dramp_rows`.
    :returns: The sequence, uppercased, or ``""``.

    ``Original_Sequence`` FIRST, ``Sequence`` as the fallback, and the order is load-bearing. In
    ``specific_amps.txt`` both columns exist and, measured over the 2026-03 dump, they differ in
    366 of 366 rows that populate both: ``Sequence`` marks stapling positions with Unicode
    circled letters (``KFFⓀKLKⓀAVKKGFKKFAKV``) while ``Original_Sequence`` holds the clean form
    (``KFFKKLKKAVKKGFKKFAKV``). The other exports carry only ``Sequence``, which in those files
    IS the clean form -- hence a fallback rather than one column for all.

    This lives here, beside :func:`_dramp_rows`, because two readers of this dump drifting apart
    is a mistake this project has already made twice: ``sample.merge`` implemented the rule and
    the activity loader did not, so the same 366 peptides entered the consensus clean and the
    activity table decorated.
    """
    return dramp_field(row, "sequence").upper().strip()


#: A plain one-letter peptide sequence, with the ambiguity codes real dumps use.
#:
#: Shared by both readers of the DRAMP dump, because both have to make the same judgement and
#: only one of them used to. ``sample.merge`` rejected non-plain sequences; the activity loader
#: did not, so strings that could not enter the consensus entered the activity table unchallenged.
DRAMP_PLAIN_RESIDUES = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUO]+$")


def dramp_sequence_is_plain(sequence: str) -> bool:
    """
    :param sequence: A sequence from :func:`dramp_sequence`.
    :returns: Whether it is a usable one-letter sequence.

    What this rejects, and why there is no repair for it. 507 rows of the 2026-03 dump carry
    stapling annotation in ``Sequence`` -- Unicode circled letters marking modified positions.
    366 of those also have ``Original_Sequence`` and are handled by :func:`dramp_sequence`; the
    remaining 141 (mostly ``Subclass/stapled_amps.txt``, which has no such column) have no clean
    form available in the file they come from.

    Stripping the circles looks like the obvious repair and is wrong. Tested against the 366 rows
    that carry both columns, un-circling ``Sequence`` reproduces ``Original_Sequence`` in only 34
    of them: ``Original_Sequence`` is *canonicalised*, not merely unmarked -- circled X marks a
    non-standard residue that DRAMP substitutes (``KKKKKKAAFⓍAWAⓍFAA`` is published as
    ``KKKKKKAAFAAWAAFAA``, not ``KKKKKKAAFXAWAXFAA``), and other positions are normalised too. So
    un-circling would have produced 332 sequences that disagree with what DRAMP publishes, and
    rejecting is the only honest option.
    """
    return bool(sequence) and bool(DRAMP_PLAIN_RESIDUES.match(sequence))


def dramp_is_modified(row: Dict[str, str]) -> bool:
    """
    Whether DRAMP records any chemical modification for this entry.

    :param row: A row from :func:`_dramp_rows`.
    :returns: ``True`` when a terminal modification, a non-linear topology or another
        modification is recorded.

    Used to judge SMILES provenance, not to annotate peptides. A sequence-derived structure
    describes a linear, unmodified molecule, so for a modified peptide it is wrong in a way that
    still parses -- which makes "is this entry modified?" the question that decides whether
    DRAMP's own curated SMILES should be preferred over one computed from the sequence.
    """
    topology = dramp_field(row, "topology").lower()
    if topology and not topology.startswith("linear"):
        return True
    for key in ("n_terminal", "c_terminal", "other_modifications"):
        value = dramp_field(row, key).lower()
        if value and value not in _DRAMP_NO_MODIFICATION:
            return True
    return False


#: Values DRAMP writes in a modification column to mean "none". Checked lowercased.
_DRAMP_NO_MODIFICATION = frozenset({
    "none", "no", "free", "nil", "not found", "not available", "n/a", "na", "-", "null",
})


def load_dramp_smiles(base_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    DRAMP's curated SMILES, keyed by DRAMP id.

    :param base_dir: Root of the raw database dumps (``data/antimicrobial_databases``).
    :returns: ``{DRAMP_ID: {"smiles", "modified", "sequence"}}``, empty when the dump predates
        DRAMP 5.0 -- which is a normal state, not an error, and is why the caller must treat an
        empty mapping as "no curated chemistry available" rather than as a failure.

    DRAMP 5.0 (Nucleic Acids Research, doi:10.1093/nar/gkag989) added a ``SMILES`` column to the
    Datasets exports, annotating 22,298 of its 32,296 entries. The paper is explicit about the
    scope: it encodes the first three modification categories -- unmodified, cyclized or
    disulfide-bonded, and terminally modified peptides -- and leaves more complex chemistry for
    a later release. So this is a *partial* but *curated* source, which is exactly the
    combination that makes it worth keeping separately from anything computed.

    Read from the ``SMILES`` column of the existing exports rather than from DRAMP's separate
    ``*_smiles.txt`` downloads, which carry the same strings; see
    :data:`sample.databases.DRAMP_FILES` for why fetching those would be actively harmful.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in _dramp_rows(base_dir):
        smiles = dramp_field(row, "smiles")
        if not smiles:
            continue
        rid = _clean(row.get("DRAMP_ID"))
        if not rid:
            continue
        out[rid] = {"smiles": smiles,
                    "modified": dramp_is_modified(row),
                    "sequence": dramp_sequence(row)}
    if out:
        modified = sum(1 for entry in out.values() if entry["modified"])
        logger.info(f"DRAMP: {len(out)} curated SMILES strings loaded "
                    f"({modified} for entries DRAMP records as modified).")
    else:
        logger.info("DRAMP: no SMILES column in this dump (pre-5.0); no curated chemistry "
                    "available.")
    return out


def load_dramp(base_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Yields structured records from the DRAMP tab-separated exports.

    DRAMP states its measurements as free text in a single ``Target_Organism`` column, so the
    numbers are recovered by :func:`_parse_dramp_targets` rather than read from typed fields.

    :param base_dir: Root of the raw database dumps (``data/antimicrobial_databases``).
    :param limit: Stop after this many records; ``None`` reads everything.
    :yields: One record per DRAMP entry, de-duplicated by ``DRAMP_ID``.
    """
    count = 0
    nonstandard = 0
    for row in _dramp_rows(base_dir):
        # dramp_sequence, not row["Sequence"]: in specific_amps.txt that column carries the
        # stapling-annotated form, and this loader used to yield those 366 sequences decorated
        # with Unicode circled letters while sample.merge yielded the same peptides clean.
        sequence = dramp_sequence(row)
        if not sequence:
            continue
        # The same judgement sample.merge makes, from the same shared predicate. Without it, the
        # 141 rows whose only sequence column carries stapling annotation entered the activity
        # table with a string that is not a peptide sequence -- while being rejected from the
        # consensus, so the two outputs disagreed about the same peptides.
        if not dramp_sequence_is_plain(sequence):
            nonstandard += 1
            continue
        rid = _clean(row.get("DRAMP_ID"))
        activities = _parse_dramp_targets(row.get("Target_Organism"), row.get("Activity") or "", rid)
        # dramp_field: specific_amps.txt spells this `Hemolytic_Activity`, so a lookup keyed on
        # general_amps.txt's `Hemolytic_activity` returned nothing for all 6,286 of its rows --
        # indistinguishable, in the output, from "this entry has no hemolytic data".
        activities += _parse_dramp_targets(dramp_field(row, "hemolytic"), record_id=rid)
        activities += _parse_dramp_targets(row.get("Cytotoxicity"), record_id=rid)
        yield {
            "database": "DRAMP",
            "record_id": _clean(row.get("DRAMP_ID")),
            "sequence": sequence,
            "name": _clean(row.get("Name")),
            "source_organism": resolve_target_species(row.get("Source")),
            "activity_labels": [s for s in re.split(r'[,;]', _clean(row.get("Activity"))) if s.strip()],
            # `Uniprot_Entry` in specific_amps.txt, `Swiss_Prot_Entry` everywhere else.
            "cross_references": {"uniprot": [dramp_field(row, "uniprot")] if dramp_field(row, "uniprot") else [],
                                 "pdb": [_clean(row.get("PDB_ID"))] if row.get("PDB_ID") else []},
            "publications": [{
                "pubmed_id": "", "doi": _clean(row.get("doi")), "title": _clean(row.get("Title")),
                "journal": _clean(row.get("Reference")), "year": None,
                "authors": [a for a in re.split(r'[,:;]', _clean(row.get("Author"))) if a.strip()],
            }] if (row.get("doi") or row.get("Title")) else [],
            "activities": activities,
        }
        count += 1
        if limit and count >= limit:
            return

    # Reported, never silent -- this project's rule after a bare except in the DBAASP ingest hid
    # a 60% activity-table loss. Logged only on a full pass: an early `limit` return above skips
    # this deliberately, because a partial count read as a corpus figure is worse than none.
    if nonstandard:
        logger.info(f"DRAMP: {nonstandard} rows had a sequence that is not plain one-letter "
                    f"residues (stapling annotation with no Original_Sequence to fall back on) "
                    f"and were skipped, matching what sample.merge already rejects.")


# --- CAMP / ADP / dbAMP ------------------------------------------------------------------------
#
# These three publish activity CLASSES ("Antibacterial") and a source organism, but no measured
# values -- there is no number to import. They still contribute identity, provenance and the
# publication behind each sequence, which is what makes a SAMPLE-extracted measurement checkable
# against a known peptide.

def load_camp(base_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Yields structured records from the CAMP per-entry CSV files.

    CAMP publishes activity CLASSES and a source organism but no measured values, so
    ``activities`` is always empty; the contribution is identity, provenance and the publication.

    :param base_dir: Root of the raw database dumps.
    :param limit: Stop after this many files; ``None`` reads everything.
    :yields: One record per CAMP entry that carries a sequence.
    """
    files = sorted(glob.glob(os.path.join(base_dir, "camp", "*.csv")))
    if limit:
        files = files[:limit]
    unreadable = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                data = {r[0].strip(): r[1].strip()
                        for r in csv.reader(fh) if len(r) >= 2}
        except Exception:
            unreadable += 1
            continue
        sequence = _clean(data.get("Sequence"))
        if not sequence:
            continue
        pmid = _clean(data.get("PubMed") or data.get("Pubmed Id"))
        yield {
            "database": "CAMP",
            "record_id": os.path.splitext(os.path.basename(path))[0],
            "sequence": sequence,
            "name": _clean(data.get("Title")),
            "source_organism": resolve_target_species(re.sub(r'\[.*?\]', '', data.get("Source") or "")),
            "activity_labels": [s for s in re.split(r'[,;]', _clean(data.get("Activity"))) if s.strip()],
            "evidence": _clean(data.get("Validated")),
            "publications": [{"pubmed_id": pmid, "doi": "", "title": "", "journal": "",
                              "year": None, "authors": []}] if pmid else [],
            "activities": [],
        }
    if unreadable:
        logger.warning(f"CAMP: {unreadable} of {len(files)} files could not be read.")


def load_adp(base_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Yields structured records from the ADP (APD) single-CSV export.

    Like CAMP, ADP carries activity labels rather than measured values.

    :param base_dir: Root of the raw database dumps.
    :param limit: Stop after this many rows; ``None`` reads everything.
    :yields: One record per APD entry that carries a sequence.
    """
    path = os.path.join(base_dir, "adp", "adb.csv")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            rows = list(csv.reader(fh))
    except Exception as e:
        logger.debug(f"ADP unreadable: {e}")
        return
    if len(rows) < 2:
        return
    header = [h.strip().rstrip(":") for h in rows[1]]
    for count, row in enumerate(rows[2:]):
        if limit and count >= limit:
            return
        data = dict(zip(header, row))
        sequence = _clean(data.get("Sequence"))
        if not sequence:
            continue
        swiss = re.sub(r'SwissProt ID:|Go to SwissProt', '', data.get("SwissProt ID") or "").strip()
        yield {
            "database": "ADP",
            "record_id": _clean(data.get("APD ID")),
            "sequence": sequence,
            "name": _clean((data.get("Name/Class") or "").split("(")[0]),
            "source_organism": resolve_target_species(data.get("Source")),
            "activity_labels": [s for s in re.split(r'[,;]', _clean(data.get("Activity"))) if s.strip()],
            "cross_references": {"uniprot": [swiss] if swiss else []},
            "publications": [{"pubmed_id": "", "doi": "", "title": _clean(data.get("Title")),
                              "journal": _clean(data.get("Reference")), "year": None,
                              "authors": [a for a in re.split(r'[,;]', _clean(data.get("Author"))) if a.strip()]}]
                            if (data.get("Title") or data.get("Reference")) else [],
            "activities": [],
        }


def load_dbamp(base_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Yields structured records from the dbAMP per-entry metadata CSVs.

    :param base_dir: Root of the raw database dumps.
    :param limit: Stop after this many entries; ``None`` reads everything.
    :yields: One record per dbAMP entry that carries a sequence.
    """
    # dbAMP's activity CLASSES live in its FASTA filenames (dbAMP_Antibacterial.fasta,
    # dbAMP_Antifungal.fasta, ...), covering the whole release; the per-entry metadata directory
    # holds only 394 scraped entries. Reading just the metadata -- which this did -- reported 393
    # records for a source shipping 35,599, so 99% of dbAMP's activity labels never reached the
    # task builder. The metadata records are still yielded first and unchanged (they carry the
    # PubMed id and source organism the FASTAs lack); the FASTA-derived ones fill in the rest.
    fasta_labels: Dict[str, Dict[str, Any]] = {}
    for fasta_path in sorted(glob.glob(os.path.join(base_dir, "dbamp", "*.fasta"))
                             + glob.glob(os.path.join(base_dir, "dbamp", "dbAMP2024", "*.fasta"))):
        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        label = stem[len("dbAMP_"):] if stem.startswith("dbAMP_") and stem != "dbAMP3" else ""
        try:
            with open(fasta_path, "r", encoding="utf-8", errors="ignore") as fh:
                record_id = None
                for line in fh:
                    line = line.strip()
                    if line.startswith(">"):
                        record_id = line[1:].strip()
                    elif line and record_id:
                        entry = fasta_labels.setdefault(
                            record_id, {"sequence": line.upper(), "labels": set()})
                        if label:
                            entry["labels"].add(label)
        except Exception as e:  # noqa: BLE001 -- one unreadable file must not lose the others
            logger.warning(f"dbAMP FASTA {fasta_path} could not be read: {e}")

    dirs = sorted(glob.glob(os.path.join(base_dir, "dbamp", "extracted_metadata", "*")))
    if limit:
        dirs = dirs[:limit]
    unreadable = 0
    yielded_ids = set()
    for d in dirs:
        data = {}
        for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for r in csv.reader(fh):
                        if len(r) >= 2 and r[0].strip() and r[0].strip() not in data:
                            data[r[0].strip()] = re.sub(r'&nbsp;?', ' ', r[1]).strip()
            except Exception:
                unreadable += 1
                continue
        sequence = _clean(data.get("Sequence"))
        if not sequence:
            continue
        pmid = _clean(data.get("Pubmed_Id") or data.get("PubMed ID") or data.get("PMID"))
        yielded_ids.add(os.path.basename(d))
        yield {
            "database": "DBAMP",
            "record_id": os.path.basename(d),
            "sequence": sequence,
            "name": _clean(data.get("Name") or data.get("dbAMP ID")),
            "source_organism": resolve_target_species(data.get("Source") or data.get("Species")),
            "activity_labels": [s for s in re.split(r'[,;]', _clean(data.get("Activity"))) if s.strip()],
            "publications": [{"pubmed_id": pmid, "doi": "", "title": "", "journal": "",
                              "year": None, "authors": []}] if pmid else [],
            "activities": [],
        }
    # The release itself, for every entry the metadata directory does not already cover.
    from_fasta = 0
    for record_id, entry in fasta_labels.items():
        if record_id in yielded_ids or not entry["sequence"]:
            continue
        if limit and from_fasta >= limit:
            break
        from_fasta += 1
        yield {
            "database": "DBAMP",
            "record_id": record_id,
            "sequence": entry["sequence"],
            "name": record_id,
            "source_organism": "",
            "activity_labels": sorted(entry["labels"]),
            "publications": [],
            "activities": [],
        }
    if from_fasta:
        logger.info(f"dbAMP: {from_fasta} records from the release FASTAs "
                    f"(activity classes taken from their filenames).")
    if unreadable:
        logger.warning(f"dbAMP: {unreadable} metadata files across {len(dirs)} entries could "
                       f"not be read.")


def load_anticancer(base_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Yields anticancer-peptide records imported by :mod:`sample.anticancer` (AntiCP2 today).

    Unlike the antimicrobial sources this reads from ``data/anticancer_databases`` rather than
    ``base_dir`` -- the anticancer imports live in their own tree -- and it degrades to yielding
    nothing when nothing has been imported, so ``load_all`` stays safe on a checkout that never ran
    ``import-anticancer``. Each record carries one qualitative ``anticancer`` activity so the
    sequence lands in the ``anticancer`` group task; see
    :func:`sample.anticancer.iter_activity_records`.

    :param base_dir: Unused (the anticancer tree is separate); accepted for loader-signature parity.
    :param limit: Stop after this many records; ``None`` reads everything.
    :yields: One record per anticancer-positive sequence.
    """
    from sample import anticancer

    for count, record in enumerate(anticancer.iter_activity_records()):
        if limit and count >= limit:
            return
        yield record


_LOADERS: Dict[str, Callable[..., Iterator[Dict[str, Any]]]] = {"DBAASP": load_dbaasp, "DRAMP": load_dramp, "CAMP": load_camp,
            "ADP": load_adp, "DBAMP": load_dbamp, "ANTICANCER": load_anticancer}


def load_all(base_dir: str = "data/antimicrobial_databases",
             databases: Optional[Iterable[str]] = None,
             limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Yields structured records from every requested source database.

    Records are emitted rather than accumulated: DBAASP alone is ~25,000 JSON files carrying
    ~194,500 measurements, and the caller indexes them by sequence as they arrive.
    """
    for name in (databases or _LOADERS):
        loader = _LOADERS.get(name.upper())
        if not loader:
            logger.warning(f"Unknown source database {name!r}")
            continue
        n_records = n_activities = 0
        for record in loader(base_dir, limit=limit):
            n_records += 1
            n_activities += len(record.get("activities") or [])
            yield record
        logger.info(f"{name}: {n_records} records, {n_activities} measured activities")
