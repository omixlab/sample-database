"""
Peptide names and synonyms, harvested from every source database.

Peptides were unsearchable by name. The ``peptides`` table carries an entry id, a sequence and a
SMILES string and no name at all, so a search for "magainin" returned nothing while the sequence
that *is* magainin sat in the database. Every source ships names; none of them reached the release.

They also ship **synonyms**, which matter more than they look. The same peptide is called
``Dermaseptin-B2``, ``DRS-B2``, ``DS bII`` and ``ADENOREGULIN`` in different papers, and a reader
who knows one of those names cannot otherwise find the entry.

**Distinguishing a synonym from a class label is the whole problem.** Sources pack both into the
same parenthetical::

    Dermaseptin-B2 (XXA, DRS-B2, Dermaseptin B2, DS bII, ADENOREGULIN; natural AMPs; Ala-rich; frog)
    Abaecin (natural AMPs; Pro-rich; PrAMPs; insects, arthropods, invertebrates, animals)

The first carries real synonyms; the second carries none, only classification. Position does not
separate them -- and neither does any hand-written stoplist, which would need to anticipate every
vocabulary term the sources use.

What does separate them is how many *different* peptides a token is attached to. A synonym belongs
to one peptide almost by definition; a class label is shared by hundreds. Measured over ADP: 2,819
tokens appear for exactly one peptide, while ``natural amps`` appears for 3,202, ``animals`` for
2,481 and ``frog`` for 969. The threshold also catches terms no stoplist would have predicted --
``XXA`` looks like a synonym in the Dermaseptin line above but appears for 663 peptides, because
it is the source's code for C-terminal amidation.

So candidates are counted first and emitted second, and anything shared by
:data:`VOCABULARY_THRESHOLD` or more distinct peptides is dropped as vocabulary. This errs toward
discarding a real synonym shared by a large family, which costs a search hit; the opposite error
would file "natural AMPs" as a name for 3,202 peptides, which makes name search useless.
"""
import csv
import glob
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

logger = logging.getLogger("sample.names")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: A parenthetical token attached to this many distinct peptides is classification vocabulary
#: rather than a synonym. Five is comfortably above the noise: in ADP the counts jump from a long
#: tail of ones and twos straight to the hundreds.
VOCABULARY_THRESHOLD: int = 5

#: Longest accepted name. Anything longer is a description, not a name.
MAX_NAME_LENGTH: int = 120

#: A name attached to this many distinct sequences is not identifying anything. dbAMP's ``Name``
#: column mixes real names with source organisms and free text, so ``Mouse-ear cress`` arrived as a
#: name for 300 sequences. Set well above the handful of sequences a genuine analogue series shares
#: (``Magainin-2`` and its amidated variants, say), so those survive.
NAME_SEQUENCE_THRESHOLD: int = 20

#: Names that identify nothing.
_NULL_NAMES = frozenset({
    "", "-", "na", "n/a", "none", "null", "nan", "unknown", "not available", "unnamed",
    "name/class:", "peptide", "antimicrobial peptide", "synthetic construct",
})

#: A trailing parenthetical, e.g. ``Epicidin 280 (Bacteriocin)``.
_PARENTHETICAL = re.compile(r"^([^(]+)\((.*)\)\s*$", re.DOTALL)

#: dbAMP joins several names with ``&&``.
_DBAMP_SEPARATOR = re.compile(r"\s*&&\s*")


class NameRecord(NamedTuple):
    """
    One name for one peptide.

    :param sequence: The peptide's sequence, which is how names are joined to the release --
        source accessions differ per database but the sequence is common to all of them.
    :param name: The name as written.
    :param kind: ``primary`` or ``synonym``.
    :param source: Which database supplied it.
    """
    sequence: str
    name: str
    kind: str
    source: str


def is_usable_name(name: Any) -> bool:
    """
    :param name: A candidate name.
    :returns: Whether it identifies a peptide. Rejects placeholders, bare numbers and anything
        long enough to be a description.
    """
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    if not text or text.lower() in _NULL_NAMES:
        return False
    if len(text) > MAX_NAME_LENGTH:
        return False
    # A bare number is a row index or a length, never a name.
    if re.fullmatch(r"[\d.,\s]+", text):
        return False
    if looks_like_junk_name(text):
        return False
    return True


#: Phrases that only appear when a spreadsheet cell ran two fields together. dbAMP's name column
#: yielded `Adenoregulindata about MIC or HC50 is present` -- a peptide name with a coverage note
#: welded onto it, no separator between them.
_RUN_ON_PHRASES = (
    "data about", "is present", "no data", "not available", "information about",
    "see reference", "unknown",
)

#: A UniProt entry name (``DMS2_PHYBI``, ``CECA1_DROME``). A real identifier, but not a name any
#: reader would search for, and it is already covered by the cross-reference links.
_ACCESSION_LIKE = re.compile(r"^[A-Z0-9]{1,10}_[A-Z]{3,6}\d?$")


def looks_like_junk_name(name: Any) -> bool:
    """
    Whether a candidate is a parsing artifact rather than a name.

    Cheap and self-contained -- no taxonomy lookup -- so it can run at display time over a
    database built before the harvest was fixed, as well as during the harvest itself. All four
    shapes below were live on one peptide's page:

    * ``Adenoregulin (Dermasepti`` -- truncated mid-word by a fixed-width export, which an
      unbalanced bracket detects without guessing at word boundaries.
    * ``Adenoregulindata about MIC or HC50 is present`` -- two fields run together.
    * ``DMS2_PHYBI`` -- a UniProt entry name.
    * ``South American frog Phyllomedusa bicolor`` -- an organism with a descriptive prefix, which
      :func:`is_organism_name` misses because the whole string does not resolve as a taxon.

    :param name: A candidate name.
    :returns: Whether it should be discarded.
    """
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    if not text:
        return True
    lowered = text.lower()
    if any(phrase in lowered for phrase in _RUN_ON_PHRASES):
        return True
    # Truncation: an opening bracket the export cut before closing.
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        return True
    if _ACCESSION_LIKE.match(text):
        return True
    # A descriptive phrase ending in a binomial is the organism, not the peptide.
    words = text.split()
    if len(words) >= 3 and re.fullmatch(r"[A-Z][a-z]+ [a-z]+", " ".join(words[-2:])):
        return True
    return False


def is_organism_name(name: str) -> bool:
    """
    Whether a candidate is a species rather than a peptide.

    dbAMP files the source organism in the same column as the name, so ``Xenopus laevis``,
    ``Phyllomedusa bicolor`` and the common names ``Bovine`` and ``Mouse-ear cress`` all arrived as
    peptide names. Resolving against NCBI taxonomy separates them exactly: organisms resolve --
    including common names, ``Mouse-ear cress`` to *Arabidopsis thaliana* -- and peptide names
    (``Magainin 2``, ``DRS-B2``, ``ADENOREGULIN``) do not.

    :param name: A candidate name.
    :returns: Whether it names an organism.
    """
    try:
        from sample.nlp import resolve_taxon

        resolved, taxid = resolve_taxon(name)
        return bool(taxid)
    except Exception:
        # No taxonomy database available: keep the name rather than discard it on a failed lookup.
        return False


def split_name(raw: Any) -> Tuple[str, List[str]]:
    """
    Splits a source's name field into the primary name and its synonym candidates.

    :param raw: The field as written, e.g. ``Ruminococcin A (RumA; Bacteriocin)``.
    :returns: ``(primary, candidates)``. Candidates are only *candidates* -- they are filtered by
        shared-token frequency in :func:`harvest_names`, because at this point a synonym and a
        class label are indistinguishable.
    """
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return "", []

    match = _PARENTHETICAL.match(text)
    if not match:
        return text, []

    primary = match.group(1).strip()
    candidates: List[str] = []
    for group in match.group(2).split(";"):
        for token in group.split(","):
            token = token.strip()
            if token:
                candidates.append(token)
    return primary, candidates


def _read_adp(path: str) -> Iterator[Tuple[str, str]]:
    """
    :param path: ADP ``adb.csv``.
    :returns: ``(sequence, name_field)`` pairs. Columns are positional because the file carries a
        numeric header line and then a label line, neither of which is a usable header.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) > 3 and re.fullmatch(r"AP\d{3,6}", row[0].strip(), re.IGNORECASE):
                yield row[3].strip().upper(), row[1]


def _read_dbaasp(directory: str) -> Iterator[Tuple[str, str]]:
    """
    :param directory: DBAASP JSON directory.
    :returns: ``(sequence, name_field)`` pairs, including ``majorName`` where it differs.
    """
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                card = json.load(handle)
        except (OSError, ValueError):
            continue
        sequence = str(card.get("sequence") or "").strip().upper()
        if not sequence:
            continue
        for key in ("name", "majorName"):
            value = card.get(key)
            if value:
                yield sequence, str(value)


def _read_dramp(path: str) -> Iterator[Tuple[str, str]]:
    """
    :param path: DRAMP ``general_amps.txt``.
    :returns: ``(sequence, name_field)`` pairs.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequence = str(row.get("Sequence") or "").strip().upper()
            if sequence and row.get("Name"):
                yield sequence, str(row["Name"])


def _read_camp(directory: str) -> Iterator[Tuple[str, str]]:
    """
    :param directory: CAMP directory of per-peptide key/value CSVs.
    :returns: ``(sequence, name_field)`` pairs.
    """
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        fields: Dict[str, str] = {}
        try:
            with open(path, encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("key") or "").strip()
                    if key and key not in fields:
                        fields[key] = str(row.get("value") or "").strip()
        except OSError:
            continue
        sequence = fields.get("Sequence", "").strip().upper()
        if sequence and fields.get("Title"):
            yield sequence, fields["Title"]


def _read_dbamp(path: str) -> Iterator[Tuple[str, str]]:
    """
    :param path: dbAMP ``dbAMP3_pepinfo.xlsx``.
    :returns: ``(sequence, name_field)`` pairs, one per ``&&``-joined name.
    """
    if not os.path.exists(path):
        return
    try:
        import pandas as pd

        frame = pd.read_excel(path)
    except Exception as e:
        # The file is present, so this is a broken environment rather than absent data -- most
        # often a missing openpyxl, which pandas reports only when read_excel is called. Logged at
        # ERROR because the harvest otherwise completes "successfully" with dbAMP contributing
        # nothing: 47,732 names instead of 60,649, with no failure anywhere to explain it.
        logger.error(f"Could not read dbAMP names from {path}: {e}. "
                     f"dbAMP will contribute no names to this run.")
        return
    for sequence, name in zip(frame.get("Seq", []), frame.get("Name", [])):
        sequence = str(sequence or "").strip().upper()
        if not sequence or not name:
            continue
        for part in _DBAMP_SEPARATOR.split(str(name)):
            if part.strip():
                yield sequence, part


def _read_activities(path: str) -> Iterator[Tuple[str, str]]:
    """
    :param path: ``activities.csv``.
    :returns: ``(sequence, name_field)`` pairs from the merged activity table, which carries a
        ``peptide_name`` for 87 % of its rows.
    """
    if not os.path.exists(path):
        return
    try:
        import pandas as pd

        frame = pd.read_csv(path, low_memory=False, usecols=["sequence", "peptide_name"])
    except Exception as e:
        logger.warning(f"Could not read names from {path}: {e}")
        return
    frame = frame[frame["sequence"].notna() & frame["peptide_name"].notna()]
    for sequence, name in zip(frame["sequence"], frame["peptide_name"]):
        yield str(sequence).strip().upper(), str(name)


#: Source name -> a reader yielding ``(sequence, name_field)``.
def _sources(data_dir: str) -> Dict[str, Any]:
    """
    :param data_dir: Root data directory.
    :returns: Source name -> a zero-argument iterator factory.
    """
    databases = os.path.join(data_dir, "antimicrobial_databases")
    return {
        "ADP": lambda: _read_adp(os.path.join(databases, "adp", "adb.csv")),
        "DBAASP": lambda: _read_dbaasp(os.path.join(databases, "dbaasp")),
        "DRAMP": lambda: _read_dramp(
            os.path.join(databases, "dramp", "Datasets", "general_amps.txt")),
        "CAMP": lambda: _read_camp(os.path.join(databases, "camp")),
        "DBAMP": lambda: _read_dbamp(
            os.path.join(databases, "dbamp", "dbAMP2024", "dbAMP3_pepinfo.xlsx")),
        "SAMPLE": lambda: _read_activities(
            os.path.join(data_dir, "sample_raw_release", "activities.csv")),
    }


def harvest_names(data_dir: Optional[str] = None,
                  vocabulary_threshold: int = VOCABULARY_THRESHOLD) -> List[NameRecord]:
    """
    Collects primary names and synonyms from every source database.

    Two passes. The first records, for each synonym candidate, how many distinct primary names it
    is attached to; the second emits only those below the threshold. See the module docstring for
    why frequency is the discriminator and a stoplist is not.

    :param data_dir: Root data directory; the packaged one when omitted.
    :param vocabulary_threshold: Candidates attached to this many distinct primary names or more
        are treated as classification vocabulary and dropped.
    :returns: Deduplicated name records.
    """
    data_dir = data_dir or DATA_DIR
    sources = _sources(data_dir)

    # Pass 1 -- how widely is each candidate shared?
    shared_by: Dict[str, Set[str]] = defaultdict(set)
    raw_rows: List[Tuple[str, str, str]] = []
    for source, reader in sources.items():
        count = 0
        for sequence, field in reader():
            primary, candidates = split_name(field)
            if not is_usable_name(primary):
                continue
            raw_rows.append((sequence, field, source))
            for candidate in candidates:
                shared_by[candidate.strip().lower()].add(primary.lower())
            count += 1
        if count:
            logger.info(f"{source}: read {count} named records.")
        else:
            # A source that is configured but yields nothing is nearly always a broken read, not
            # an empty database. Saying so here means the anomaly appears next to the counts a
            # reader is already comparing, rather than only in an earlier traceback.
            logger.error(f"{source}: read 0 named records. Its data is either missing or "
                         f"unreadable -- check the messages above.")

    vocabulary = {token for token, primaries in shared_by.items()
                  if len(primaries) >= vocabulary_threshold}
    logger.info(f"Treating {len(vocabulary)} shared tokens as classification vocabulary "
                f"rather than synonyms.")

    # Pass 2 -- how many distinct sequences does each name claim? A name spanning hundreds is a
    # description or an organism, whatever column it arrived in, so this applies to primary names
    # as well as synonyms.
    sequences_per_name: Dict[str, Set[str]] = defaultdict(set)
    for sequence, field, _ in raw_rows:
        primary, candidates = split_name(field)
        for name in [primary] + candidates:
            if is_usable_name(name):
                sequences_per_name[name.strip().lower()].add(sequence)
    overused = {name for name, seqs in sequences_per_name.items()
                if len(seqs) >= NAME_SEQUENCE_THRESHOLD}

    # Pass 3 -- emit.
    seen: Set[Tuple[str, str]] = set()
    records: List[NameRecord] = []
    organism_rejects: Set[str] = set()
    for sequence, field, source in raw_rows:
        primary, candidates = split_name(field)
        for name, kind in [(primary, "primary")] + [(c, "synonym") for c in candidates]:
            if not is_usable_name(name):
                continue
            lowered = name.strip().lower()
            if kind == "synonym" and lowered in vocabulary:
                continue
            if lowered in overused:
                continue
            key = (sequence, lowered)
            if key in seen:
                continue
            if lowered in organism_rejects:
                continue
            if is_organism_name(name):
                organism_rejects.add(lowered)
                continue
            seen.add(key)
            records.append(NameRecord(sequence, name.strip(), kind, source))

    primaries = sum(1 for r in records if r.kind == "primary")
    logger.info(f"Dropped {len(overused)} names claimed by >= {NAME_SEQUENCE_THRESHOLD} sequences "
                f"and {len(organism_rejects)} that resolve as organisms.")
    logger.info(f"Harvested {len(records)} names for "
                f"{len({r.sequence for r in records})} sequences "
                f"({primaries} primary, {len(records) - primaries} synonyms).")
    return records


def build_names(output_path: Optional[str] = None, data_dir: Optional[str] = None,
                vocabulary_threshold: int = VOCABULARY_THRESHOLD) -> Dict[str, Any]:
    """
    Harvests names and writes them for the SQLite load to pick up.

    :param output_path: Where to write; ``data/peptide_names.json`` when omitted.
    :param data_dir: Root data directory.
    :param vocabulary_threshold: See :func:`harvest_names`.
    :returns: A summary of what was written.
    """
    records = harvest_names(data_dir=data_dir, vocabulary_threshold=vocabulary_threshold)

    by_sequence: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        by_sequence[record.sequence].append(
            {"name": record.name, "kind": record.kind, "source": record.source})

    output_path = output_path or os.path.join(DATA_DIR, "peptide_names.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(by_sequence, handle, ensure_ascii=False, indent=1)

    summary = {
        "path": output_path,
        "sequences": len(by_sequence),
        "names": len(records),
        "primary": sum(1 for r in records if r.kind == "primary"),
        "synonyms": sum(1 for r in records if r.kind == "synonym"),
    }
    logger.info(f"Wrote {summary['names']} names for {summary['sequences']} sequences "
                f"to {output_path}.")
    return summary
