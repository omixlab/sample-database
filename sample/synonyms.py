"""
Resolve a name-only peptide to a sequence, using the paper itself and the local catalogue.

WHY THIS EXISTS. 81% of the peptides the extraction step produces carry a name and an activity
measurement but no sequence (8,270 of 10,193 in the current pubmed run), and
``sample.merge`` keys on ``peptide_sequence`` with no name fallback -- ``if not seq: continue`` at
two call sites -- so every one of them is silently discarded before it reaches the database. The
activities attached to them go with them. This module tries to give those records a sequence, from
evidence that can be checked afterwards.

WHAT IT WILL NOT DO. It never joins on a name alone. Measured over the catalogue, 9.4% of names map
to MORE THAN ONE distinct sequence, so a bare name match is a coin toss on those; and measured over
the extracted records, 1,188 of 1,245 "sequence found near the name" hits were coincidental
proximity inside flattened table text -- a unit column landing next to an unrelated residue run --
not a statement by the paper. Joining on proximity would fabricate bindings that look like data.
Every tier below therefore requires either an explicit statement in the document or independent
corroboration, records the evidence it used, and refuses the join when the evidence is thin.

LOCAL ONLY. The catalogue is read from ``data/sample.db`` read-only and accessions are matched
against text already on disk. Nothing here reaches the network.

:data:`MIN_CONFIDENCE_TO_JOIN` is the gate. A resolution below it is still RECORDED on the record,
so a later pass can review what was rejected and why, but the sequence is not set.
"""
import functools
import json
import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("sample.synonyms")

#: The catalogue. Opened read-only; a missing file degrades every catalogue tier to "no candidate".
CATALOGUE_DB = os.getenv("SAMPLE_CATALOGUE_DB", os.path.join("data", "sample.db"))

AA = "ACDEFGHIKLMNPQRSTVWY"

#: Shortest run accepted as a sequence. Matches the table reader's floor; it matters because
#: every shorter run is far likelier to be a word that lost its spaces.
_MIN_SEQUENCE_LEN = 10

#: A candidate run, before validation.
_SEQUENCE = re.compile(rf"[{AA}]{{{_MIN_SEQUENCE_LEN},}}")

#: Names that identify nothing on their own. "Peptide 1" means a different molecule in every paper
#: that uses it, so a catalogue hit on it is meaningless; and the table reader sometimes captures a
#: measurement as a name ("0.62 l g ml - 1" appears in the current run), which must never be looked
#: up at all.
_GENERIC_LABEL = re.compile(
    r"^(?:peptide|compound|analog(?:ue)?|derivative|fraction|sample|pep|p|cpd|no|#)\s*[-_.]?\s*\d*$",
    re.I)
#: Unit tokens, concatenated. Tested against the string's LETTERS ONLY, after digits, spaces and
#: punctuation are removed, because the table reader captures measurements with their spacing
#: mangled: "0.62 l g ml - 1" and "25 ug/mL" both reduce to a run of unit letters and must be
#: rejected, while "PBD-1" ("pbd") and "BmKn2" ("bmkn") must not.
#: The lookahead requires at least one SUBSTANTIVE unit token (ml, mol, mg, ug ...). Without it,
#: a match could be made entirely of repeated single-letter tokens, and "LL-37" -- one of the most
#: studied human antimicrobial peptides -- was read as litre+litre and discarded as a measurement.
_UNIT_LETTERS = re.compile(
    r"^(?=.*(?:ml|dl|mol|mm|cm|nm|um|min|kda|da|[munpµμ]g))"
    r"(?:[munpµμ]?g|ml|dl|l|mol|mm|cm|nm|um|min|kda|da|h)+$", re.I)

#: Accession shapes, matched only in the document's own text.
_ACCESSION_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("uniprot", re.compile(r"\b(?:UniProt(?:KB)?[ :/]*)?([OPQ][0-9][A-Z0-9]{3}[0-9]|"
                           r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")),
    ("pdb", re.compile(r"\b(?:PDB[ :/]*(?:ID[ :]*)?)([0-9][A-Za-z0-9]{3})\b")),
    ("genbank", re.compile(r"\b(?:GenBank|GenPept|NCBI)[ :/]*([A-Z]{1,3}_?[0-9]{5,8}(?:\.[0-9]+)?)\b"),),
)

#: How far from the name a statement has to be to count as being about it. One clause, not one page:
#: a 200-character window over flattened table text was what produced the 1,188 coincidences.
_BINDING_WINDOW = 120

#: Confidence per resolution method. These are ORDINAL -- they rank evidence, they are not
#: probabilities -- and the only number that changes behaviour is the threshold below.
CONFIDENCE = {
    "internal_explicit": 0.95,     # the paper writes "NAME (SEQUENCE)"
    "accession": 0.90,             # the paper cites an accession beside the name
    # "NAME (ALIAS)" and ALIAS resolves explicitly. Deliberately BELOW the join threshold: the
    # alias hop doubles the chance of binding the wrong thing, and measured over 900 documents it
    # produced 8 resolutions of which at least one was a fabrication -- 'AcSDKP' -> "SERASPENIDE",
    # which is a DRUG NAME whose letters happen to all be residue codes and which no dictionary
    # carries. One in eight wrong is far too high for something that reaches the database, so this
    # tier is recorded as evidence for a curator and never applied on its own.
    "internal_alias": 0.70,
    "catalogue_corroborated": 0.80,  # one catalogue sequence, and organism or activity agrees
    "catalogue_disambiguated": 0.75,  # several candidates, evidence leaves exactly one
    "catalogue_unique_bare": 0.50,  # one candidate, nothing corroborates it -- recorded, not joined
}

#: Resolutions at or above this are written onto the record. Below it they are recorded as a
#: rejected attempt and the record keeps its empty sequence.
MIN_CONFIDENCE_TO_JOIN = float(os.getenv("SAMPLE_SYNONYM_MIN_CONFIDENCE", "0.75"))


def _normalise(name: Any) -> str:
    """
    :param name: A peptide name as written.
    :returns: Lower-cased alphanumerics only, so "PBD-1", "pbd 1" and "PBD1" collapse together.
        Punctuation carries no meaning in these names and varies between papers and databases.
    """
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


#: Table headers and placeholders the reader captures from a header or a blank cell.
_HEADER_WORDS = frozenset({
    "range", "mean", "sd", "sem", "control", "total", "value", "values", "min", "max", "median",
    "mic", "mbc", "mfc", "ic50", "ec50", "peptide", "peptides", "sequence", "name", "strain",
    "strains", "organism", "organisms", "test", "assay", "sample", "samples", "concentration",
    "activity", "result", "results", "reference", "ref", "nd", "na", "none", "blank",
})

#: An abbreviated binomial: "E. coli", "S. aureus". These are the TEST organisms of an assay
#: table, and the reader captures them as peptide names when it picks the wrong column.
_ABBREVIATED_BINOMIAL = re.compile(r"^[A-Z]\.\s*[a-z]{3,}$")

#: A bare table row label: a number with an optional trailing letter ("13a", "4a", "1 b"), or a
#: SINGLE letter followed by digits with no hyphen ("C2", "T1", "I1", "N17").
#:
#: The hyphenated two-letter form is deliberately EXCLUDED, because that is how real peptides are
#: named -- NK-2, LL-37, HD-6, PR-39. An earlier version of this pattern matched them and would
#: have discarded correct resolutions (NK-2 resolves to KILRGVCKKIMRTFLRRISKDILTGKK).
_ROW_LABEL = re.compile(r"^\s*(?:\d{1,3}\s*[a-z]?|[A-Za-z]\s*\d{1,3})\s*$")


@functools.lru_cache(maxsize=1)
def _organism_names() -> frozenset:
    """
    :returns: Normalised organism names from ``data/organism_names.txt``, or an empty set when it
        is absent -- in which case the abbreviated-binomial and header rules below still apply.
    """
    path = os.path.join("data", "organism_names.txt")
    names = set()
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                normalised = _normalise(line)
                if len(normalised) > 3:
                    names.add(normalised)
    except OSError:
        logger.debug(f"{path} absent; organism names will not be filtered by list")
    return frozenset(names)


def is_not_a_peptide_name(name: Any) -> bool:
    """
    Whether a captured "peptide name" is really an organism or a table header.

    Measured over 2,813 name-only records, 3.9% are neither: 2.7% abbreviated binomials
    ("E. coli", "S. aureus", "P. aeruginosa"), 0.4% full organism names and 0.8% headers and
    placeholders ("Range", "Control", "None"). They are the most FREQUENT unresolvable "names" in
    the corpus, and they are extraction defects rather than resolution failures -- the table reader
    picked the test-organism or header column. Resolving them is impossible and recording them as
    peptides is wrong, so they are refused outright.

    :param name: The extracted peptide name.
    :returns: True when the string names an organism or a table column rather than a peptide.
    """
    raw = str(name or "").strip()
    if not raw:
        return False
    if _ABBREVIATED_BINOMIAL.match(raw):
        return True
    normalised = _normalise(raw)
    if normalised in _HEADER_WORDS:
        return True
    return normalised in _organism_names()


def is_junk_name(name: Any) -> bool:
    """
    Whether a string is not a peptide name at all, so no tier should touch it.

    Blank, a bare number, a single character, or a measurement the table reader captured as a name
    -- "0.62 l g ml - 1" occurs in the current run. These cannot be resolved by ANY route, because
    there is nothing there to resolve.

    :param name: The extracted peptide name.
    :returns: True when the string should be ignored entirely.
    """
    raw = str(name or "").strip()
    if len(raw) < 2:
        return True
    normalised = _normalise(raw)
    if not normalised or normalised.isdigit() or not any(c.isalpha() for c in normalised):
        return True
    letters = re.sub(r"[^A-Za-z]", "", raw)
    if bool(letters) and bool(_UNIT_LETTERS.match(letters)):
        return True
    return is_not_a_peptide_name(raw)


def is_document_local_name(name: Any) -> bool:
    """
    Whether a name identifies a molecule only inside its own paper.

    "Peptide 1", "compound 3", "P40" mean a different molecule in every paper that uses them, so a
    CATALOGUE match on one is meaningless -- it says two papers chose the same label, nothing more.

    Such a name is still perfectly resolvable from the document ITSELF: if the paper writes
    "Peptide 1 (GIGKFLHSAK)" that is the strongest evidence available, and rejecting it because the
    label is generic would throw away the one tier that cannot be wrong. So this gates the
    catalogue tiers only.

    :param name: The extracted peptide name.
    :returns: True when catalogue lookup on this name is not meaningful.
    """
    raw = str(name or "").strip()
    if _GENERIC_LABEL.match(raw):
        return True
    # Bare table row labels: "1 b", "13a", "4a", "6e", "C2", "T1", "I1", "N17". Measured, 67 of
    # 1,171 applied resolutions were keyed on one of these, including 'C2' -> ACETYLATED. A label
    # like this is an index into one paper's table, not a name any catalogue can be asked about.
    return bool(_ROW_LABEL.match(raw))


def _splits_into_words(sequence: str) -> bool:
    """
    Whether a run is two English words whose separating space was lost.

    The shared test checks the run as ONE token, so it rejects ``THEPEPTIDES`` and
    ``GRAMNEGATIVE`` but accepts ``TREATEDSAMPLES`` -- "treated" and "samples" are each ordinary
    words, and nothing in a single-token dictionary lookup notices. The extracted text loses spaces
    constantly, so this is the common case rather than an edge one.

    :param sequence: An upper-cased candidate.
    :returns: True when it splits cleanly into two dictionary words of 4+ letters each.
    """
    try:
        from sample.docling_tables import _dictionary_words

        words = _dictionary_words()
    except Exception:  # noqa: BLE001
        return False
    if not words:
        return False
    for cut in range(4, len(sequence) - 3):
        if sequence[:cut] in words and sequence[cut:] in words:
            return True
    return False


def _clean_sequence(candidate: str) -> Optional[str]:
    """
    Validate a residue run taken from running text.

    Delegates to :func:`sample.docling_tables.sequence_from_cell` -- the project's existing and
    hard-won test: residue alphabet, a length floor, an English dictionary filter, a prose filter,
    and ``nlp.is_valid_sequence``'s log-likelihood check that rejects nucleotide runs spelling
    valid residue letters (a PCR primer clears every other rule).

    Writing a weaker test here reproduced a failure this project has recorded twice already. The
    first version of this module accepted any 8+ residue-alphabet run and "recovered" ENGLISH PROSE
    with its spaces missing: ``THEPEPTIDES`` from "...thepeptides beh", ``CASEICIN``,
    ``ASMELITTIN`` from "as melittin", ``ACATHELICIDIN`` from "a cathelicidin", ``TREATEDSAMPLES``
    from "Magainin-treatedsamples". Every one would have been written to a record as a sequence.

    :param candidate: A residue run taken from text.
    :returns: The validated one-letter sequence, or None.
    """
    raw = (candidate or "").strip()
    if not raw:
        return None
    try:
        from sample.docling_tables import sequence_from_cell

        sequence = sequence_from_cell(raw, min_length=_MIN_SEQUENCE_LEN)
    except Exception as e:  # noqa: BLE001 -- without the shared test, accept nothing
        logger.debug(f"shared sequence test unavailable ({type(e).__name__}: {e}); "
                     f"refusing {raw[:20]!r}")
        return None
    if not sequence or _splits_into_words(sequence):
        return None
    return sequence


# --------------------------------------------------------------------------------------------
# Tier 1 and 2: what the document itself states
# --------------------------------------------------------------------------------------------

def internal_explicit_sequence(name: str, text: str,
                               text_lower: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    A sequence the document explicitly binds to this name.

    Only the forms where the paper is making the statement: ``Name (SEQUENCE)``,
    ``Name: SEQUENCE`` (how a table row or a sequence list reads once flattened) and
    ``SEQUENCE (Name)``. Mere co-occurrence in the same window is NOT accepted -- measured, that is
    where 1,188 of 1,245 apparent hits came from, nearly all of them a unit column sitting beside an
    unrelated residue run.

    :param name: The peptide name as extracted.
    :param text: The document text.
    :returns: ``(sequence, evidence_snippet)``, or None.
    """
    if not name or not text:
        return None
    # A name the document never writes cannot be bound by it, and this check short-circuits the
    # common case: the table reader often records a name that appears only in a cell the text layer
    # lost. `text_lower` is supplied by the caller because lowering the document PER PEPTIDE costs
    # an O(len(text)) copy each time -- measured, doing that made resolution slower than the regex
    # scans it was meant to avoid (835 ms/document -> 1,101).
    haystack = text_lower if text_lower is not None else text.lower()
    if name.strip().lower() not in haystack:
        return None
    escaped = re.escape(name.strip())
    forms = (
        (rf"{escaped}\s*[\(\[]\s*([ACDEFGHIKLMNPQRSTVWY]{{10,}})\s*[\)\]]", "name (sequence)"),
        (rf"{escaped}\s*:\s*([ACDEFGHIKLMNPQRSTVWY]{{10,}})\b", "name: sequence"),
        (rf"\b([ACDEFGHIKLMNPQRSTVWY]{{10,}})\s*[\(\[]\s*{escaped}\s*[\)\]]", "sequence (name)"),
    )
    for pattern, label in forms:
        for match in re.finditer(pattern, text, re.I):
            sequence = _clean_sequence(match.group(1))
            if sequence:
                start = max(0, match.start() - 40)
                snippet = re.sub(r"\s+", " ", text[start:match.end() + 40]).strip()
                return sequence, f"{label}: ...{snippet}..."
    return None


def internal_aliases(name: str, text: str,
                     text_lower: Optional[str] = None) -> List[str]:
    """
    Other names the document uses for this peptide.

    Papers introduce a short form once and then use it throughout: "gomesin (Gm)", "the cyclic RGD
    peptide ... hereafter referred to as RGDpep". Resolving the alias is often what reaches the
    sequence, because the sequence is stated beside the short form rather than the long one.

    :param name: The peptide name as extracted.
    :param text: The document text.
    :returns: Candidate aliases, de-duplicated, excluding the name itself and anything unusable.
    """
    if not name or not text:
        return []
    haystack = text_lower if text_lower is not None else text.lower()
    if name.strip().lower() not in haystack:
        return []
    escaped = re.escape(name.strip())
    found: List[str] = []
    forms = (
        rf"{escaped}\s*[\(\[]\s*([A-Za-z][A-Za-z0-9\-]{{2,24}})\s*[\)\]]",
        rf"\b([A-Za-z][A-Za-z0-9\-]{{2,24}})\s*[\(\[]\s*{escaped}\s*[\)\]]",
        rf"{escaped}[^.\n]{{0,60}}?(?:hereafter|referred to as|abbreviated|designated)\s+"
        rf"(?:as\s+)?([A-Za-z][A-Za-z0-9\-]{{2,24}})",
    )
    for pattern in forms:
        for match in re.finditer(pattern, text, re.I):
            alias = match.group(1).strip()
            if _normalise(alias) and _normalise(alias) != _normalise(name) \
                    and not is_junk_name(alias) and alias not in found:
                found.append(alias)
    return found[:5]


def accessions_near(name: str, text: str, window: int = _BINDING_WINDOW,
                    text_lower: Optional[str] = None) -> Dict[str, str]:
    """
    External accessions the document cites beside this name.

    These are the paper's own cross-references, so they are evidence about THIS peptide rather than
    an inference. They are returned whether or not a sequence can be obtained from them: an
    accession recorded on the record is useful to a curator even when nothing local can resolve it.

    :param name: The peptide name.
    :param text: The document text.
    :param window: Characters either side of the name to search.
    :returns: ``{"uniprot": "P15450", ...}``, empty when none are near.
    """
    out: Dict[str, str] = {}
    if not name or not text:
        return out
    haystack = text_lower if text_lower is not None else text.lower()
    if name.strip().lower() not in haystack:
        return out
    for match in re.finditer(re.escape(name.strip()), text, re.I):
        region = text[max(0, match.start() - window): match.end() + window]
        for kind, pattern in _ACCESSION_PATTERNS:
            hit = pattern.search(region)
            if hit and kind not in out:
                out[kind] = hit.group(1)
        if out:
            break
    return out


# --------------------------------------------------------------------------------------------
# Tier 3: the local catalogue
# --------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _catalogue() -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalised name -> candidate catalogue entries.

    Built once per process from ``peptide_names`` joined to ``peptides``. Decoys are excluded: a
    fabricated training sequence must never be handed to a real record.

    :returns: The index, or ``{}`` when the catalogue is absent or unreadable.
    """
    if not os.path.exists(CATALOGUE_DB):
        logger.debug(f"catalogue {CATALOGUE_DB} absent; name resolution disabled")
        return {}
    index: Dict[str, List[Dict[str, Any]]] = {}
    try:
        connection = sqlite3.connect(f"file:{CATALOGUE_DB}?mode=ro", uri=True)
        rows = connection.execute("""
            SELECT n.name, n.name_kind, p.entry_id, p.sequence, p.source_organism,
                   p.Biological_Activity, p.cross_references
            FROM peptide_names n
            JOIN peptides p ON p.entry_id = n.entry_id
            WHERE p.sequence IS NOT NULL AND TRIM(p.sequence) <> ''
              AND COALESCE(p.is_decoy, 0) = 0
        """).fetchall()
        connection.close()
    except Exception as e:  # noqa: BLE001 -- a missing catalogue must not break extraction
        logger.warning(f"could not read the catalogue ({type(e).__name__}: {e}); "
                       f"name resolution disabled")
        return {}
    for name, kind, entry_id, sequence, organism, activity, xrefs in rows:
        key = _normalise(name)
        if not key:
            continue
        index.setdefault(key, []).append({
            "entry_id": entry_id, "sequence": sequence, "name": name, "name_kind": kind,
            "organism": organism or "", "activity": str(activity or ""),
            "cross_references": xrefs or "",
        })
    logger.info(f"catalogue index: {len(index):,} distinct names over {len(rows):,} rows")
    return index


#: Shortest sequence accepted from the CATALOGUE. Matches sample.nlp.MIN_CATALOGUED_SEQUENCE_LEN,
#: the project's existing floor for catalogued keys.
_MIN_CATALOGUE_SEQUENCE_LEN = 5

#: Strings found in catalogue sequence columns that are descriptions, not sequences. Residue-code
#: letters throughout and absent from any English word list, so only an explicit list catches them.
_NON_SEQUENCE_WORDS = frozenset({
    "ACETYLATED", "AMIDATED", "ACETYL", "AMIDE", "AMIDATION", "ACETYLATION", "CYCLIC",
    "LINEAR", "FREE", "NONE", "NA", "ND", "NT", "NAN", "NULL", "MISSING", "UNKNOWN",
    "PEGYLATED", "METHYLATED", "GLYCATED", "PALMITATED", "MYRISTOYLATED", "FORMYLATED",
})


def _usable_catalogue_sequence(sequence: Any) -> Optional[str]:
    """
    Validate a sequence taken from the catalogue before it is written to a record.

    The text tiers were validated from the start; these were not, and the catalogue is not clean.
    Measured over 1,171 applied resolutions: 30 were 1-4 residues, including ``'Asp' -> 'D'`` (one
    amino acid, not a peptide) and ``'G - I' -> 'GI'``, and ``'C2'`` resolved to the literal string
    ``ACETYLATED`` -- an English word sitting in a sequence column. A catalogue hit is evidence
    about WHICH entry matches, never evidence that the entry's sequence field is a sequence.

    :param sequence: The catalogue's sequence field.
    :returns: The upper-cased sequence, or None when it is too short or is not a sequence at all.
    """
    raw = re.sub(r"[^A-Za-z]", "", str(sequence or "")).upper()
    if len(raw) < _MIN_CATALOGUE_SEQUENCE_LEN:
        return None
    if any(c not in AA for c in raw):
        return None
    # Chemistry and placeholder terms that occupy sequence columns. These are NOT in the system
    # word list -- they are technical vocabulary -- so the dictionary check below misses them, and
    # every letter in them is a residue code. 'C2' resolved to the literal string ACETYLATED.
    if raw in _NON_SEQUENCE_WORDS:
        return None
    try:
        from sample.docling_tables import _dictionary_words, _is_ordinary_prose

        if raw in _dictionary_words() or _is_ordinary_prose(str(sequence or "")):
            return None
    except Exception:  # noqa: BLE001 -- the length and alphabet rules above still applied
        pass
    return raw


def catalogue_candidates(name: str) -> List[Dict[str, Any]]:
    """
    :param name: A peptide name.
    :returns: Catalogue entries whose name matches, collapsed so that entries sharing one sequence
        count once -- two database rows for the same molecule are not an ambiguity.
    """
    if is_junk_name(name) or is_document_local_name(name):
        return []
    by_sequence: Dict[str, Dict[str, Any]] = {}
    for candidate in _catalogue().get(_normalise(name), []):
        usable = _usable_catalogue_sequence(candidate.get("sequence"))
        if not usable:
            continue
        entry = dict(candidate)
        entry["sequence"] = usable
        by_sequence.setdefault(usable, entry)
    return list(by_sequence.values())


#: Words that introduce the organism a peptide CAME FROM, as opposed to one it was tested against.
_SOURCE_CUE = re.compile(
    r"(?:isolated|purified|derived|obtained|extracted|cloned|identified)\s+from|"
    r"\bfrom\s+the\s+(?:skin|venom|haemolymph|hemolymph|gland|leaves|seeds|serum)|"
    r"\bvenom\s+of|\bskin\s+of|\bsecretions?\s+of", re.I)


def _organism_terms(record: Dict[str, Any], context: str) -> Set[str]:
    """
    Organism words that describe where the peptide CAME FROM.

    The distinction is the whole point. The catalogue's ``source_organism`` is the species the
    peptide originates in; an activity's ``target_organism`` is the species it was tested
    AGAINST. Comparing the two is comparing a frog to *E. coli*, and the first version of this
    function did exactly that -- which is why organism corroboration fired 31 times against the
    activity signal's 416. A spurious agreement there would be worse than a miss, because it is the
    evidence that promotes a catalogue match to "usable".

    So target organisms and cell lines are excluded, and binomials are taken from the context only
    where a source cue introduces them ("isolated from Phyllomedusa bicolor", "venom of ...").

    :param record: The peptide record.
    :param context: Its context window.
    :returns: Lower-cased source-organism words, empty when the context states none.
    """
    terms: Set[str] = set()
    for activity in record.get("biological_activities") or []:
        # source_organism only -- never target_organism or target_cell_line.
        value = str(activity.get("source_organism") or "").strip().lower()
        if value:
            terms.update(w for w in re.split(r"[^a-z]+", value) if len(w) > 3)
    body = context or ""
    for cue in _SOURCE_CUE.finditer(body):
        window = body[cue.end(): cue.end() + 80]
        for match in re.finditer(r"\b([A-Z][a-z]{3,})\s+([a-z]{3,})\b", window):
            terms.add(match.group(1).lower())
            terms.add(match.group(2).lower())
    return terms


def _activity_terms(record: Dict[str, Any]) -> Set[str]:
    """
    :param record: The peptide record.
    :returns: Lower-cased activity descriptors available as evidence.
    """
    terms: Set[str] = set()
    for activity in record.get("biological_activities") or []:
        for field in ("activity_group", "activity_type", "target_group"):
            value = str(activity.get(field) or "").strip().lower()
            if value and value != "other":
                terms.add(value)
    return terms


def _corroboration(candidate: Dict[str, Any], organisms: Set[str],
                   activities: Set[str]) -> List[str]:
    """
    Independent agreement between a candidate and the record, beyond the shared name.

    This is what separates a usable catalogue match from a coincidence of naming. An organism match
    is the strongest: a peptide named in a paper studying *Staphylococcus aureus* that matches a
    catalogue entry recorded from *Staphylococcus aureus* is very likely the same molecule.

    :param candidate: A catalogue entry.
    :param organisms: Organism words from the record.
    :param activities: Activity descriptors from the record.
    :returns: Human-readable evidence strings, empty when nothing agrees.
    """
    evidence: List[str] = []
    catalogue_organism = (candidate.get("organism") or "").lower()
    if catalogue_organism and organisms:
        shared = {w for w in organisms if len(w) > 3 and w in catalogue_organism}
        if shared:
            evidence.append(f"organism agrees ({', '.join(sorted(shared)[:3])})")
    catalogue_activity = (candidate.get("activity") or "").lower()
    if catalogue_activity and activities:
        shared = {a for a in activities if a in catalogue_activity}
        if shared:
            evidence.append(f"activity agrees ({', '.join(sorted(shared)[:3])})")
    if (candidate.get("name_kind") or "") == "primary":
        evidence.append("matched the catalogue's primary name")
    return evidence


# --------------------------------------------------------------------------------------------
# The resolver
# --------------------------------------------------------------------------------------------

def resolve(record: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
    """
    Try to give a name-only record a sequence, from evidence that can be re-checked.

    Tiers run strongest-first and the first one that produces a sequence wins:

    1. The document explicitly binds name to sequence.
    2. The document cites an accession beside the name (recorded; resolves a sequence only when a
       catalogue entry carries that same accession).
    3. The document introduces an alias, and the alias is explicitly bound to a sequence.
    4. Exactly one catalogue entry matches the name, and the organism or activity corroborates it.
    5. Several catalogue entries match, and corroboration leaves exactly one standing.

    :param record: A peptide record with a name and no sequence.
    :param text: The document's text -- the OCR output when there is one, since that is what the
        rest of the pipeline read.
    :returns: A resolution dict with ``sequence``, ``method``, ``confidence``, ``evidence`` and any
        ``cross_references``, or None when nothing was found. A returned resolution may still be
        below :data:`MIN_CONFIDENCE_TO_JOIN`; the caller decides whether to apply it, and should
        record it either way.
    """
    name = str(record.get("peptide_name") or "").strip()
    if not name or (record.get("peptide_sequence") or "").strip():
        return None
    if is_junk_name(name):
        return {"sequence": None, "method": "rejected_not_a_name", "confidence": 0.0,
                "evidence": [f"{name!r} is not a peptide name (blank, numeric or a measurement)"],
                "cross_references": {}}

    # Lowered ONCE for every substring pre-check below.
    text_lower = (text or "").lower()

    # 1. the paper says so. Deliberately BEFORE the document-local check: "Peptide 1 (GIGKFLHSAK)"
    # is the strongest evidence there is, and the label being generic does not weaken it.
    explicit = internal_explicit_sequence(name, text, text_lower)
    if explicit:
        sequence, snippet = explicit
        return {"sequence": sequence, "method": "internal_explicit",
                "confidence": CONFIDENCE["internal_explicit"],
                "evidence": [snippet], "cross_references": {}}

    organisms = _organism_terms(record, record.get("context") or "")
    activities = _activity_terms(record)

    # Accessions are scanned ONCE and only when something will use them. Computing them eagerly
    # cost a full-text scan per peptide on every record, including the 60% that end at
    # "no_candidate" and never look at an accession -- measured, resolution was 835 ms/document
    # before this and the scans were most of it.
    candidates = catalogue_candidates(name)
    accessions = accessions_near(name, text, text_lower=text_lower)

    # 2. an accession beside the name, resolved only against local catalogue cross-references
    if accessions:
        wanted = {f"{k}:{v}".lower() for k, v in accessions.items()}
        for candidate in candidates or []:
            have = (candidate.get("cross_references") or "").lower()
            shared = [w for w in wanted if w in have]
            if shared:
                return {"sequence": candidate["sequence"], "method": "accession",
                        "confidence": CONFIDENCE["accession"],
                        "evidence": [f"document cites {', '.join(shared)} beside the name, and the "
                                     f"catalogue entry carries the same accession"],
                        "cross_references": accessions}

    # 3. an alias the paper introduces, explicitly bound
    for alias in internal_aliases(name, text, text_lower):
        bound = internal_explicit_sequence(alias, text, text_lower)
        if bound:
            sequence, snippet = bound
            return {"sequence": sequence, "method": "internal_alias",
                    "confidence": CONFIDENCE["internal_alias"],
                    "evidence": [f"document introduces alias {alias!r}", snippet],
                    "cross_references": accessions}

    # 4 and 5. the catalogue, never on the name alone
    if not candidates:
        return {"sequence": None, "method": "no_candidate", "confidence": 0.0,
                "evidence": ["no explicit binding in the document and no catalogue match"],
                "cross_references": accessions}

    scored = [(candidate, _corroboration(candidate, organisms, activities))
              for candidate in candidates]
    # "primary name" alone is weak -- it says the catalogue likes the spelling, not that this is the
    # same molecule. Require something about the biology too.
    strong = [(c, e) for c, e in scored
              if any(not x.startswith("matched the catalogue") for x in e)]

    if len(candidates) == 1:
        candidate, evidence = scored[0]
        if strong:
            return {"sequence": candidate["sequence"], "method": "catalogue_corroborated",
                    "confidence": CONFIDENCE["catalogue_corroborated"],
                    "evidence": evidence + [f"single catalogue match "
                                            f"({candidate['entry_id']})"],
                    "cross_references": accessions}
        return {"sequence": candidate["sequence"], "method": "catalogue_unique_bare",
                "confidence": CONFIDENCE["catalogue_unique_bare"],
                "evidence": [f"single catalogue match ({candidate['entry_id']}) but nothing in the "
                             f"record corroborates it"],
                "cross_references": accessions}

    if len(strong) == 1:
        candidate, evidence = strong[0]
        return {"sequence": candidate["sequence"], "method": "catalogue_disambiguated",
                "confidence": CONFIDENCE["catalogue_disambiguated"],
                "evidence": evidence + [f"{len(candidates)} catalogue candidates, evidence left "
                                        f"one ({candidate['entry_id']})"],
                "cross_references": accessions}

    return {"sequence": None, "method": "ambiguous", "confidence": 0.0,
            "evidence": [f"{len(candidates)} catalogue sequences share this name and "
                         f"{len(strong)} are equally corroborated"],
            "cross_references": accessions}


def apply_resolution(record: Dict[str, Any], resolution: Optional[Dict[str, Any]],
                     minimum: float = None) -> bool:
    """
    Write a resolution onto a record, if its evidence clears the bar.

    The attempt is recorded either way, under ``sequence_resolution``. A rejected attempt is as
    useful as an accepted one: it says this record was considered and why it was left alone, which
    is what makes a later pass able to revisit it instead of starting over.

    :param record: The peptide record, modified in place.
    :param resolution: A :func:`resolve` result, or None.
    :param minimum: Confidence required to set the sequence; defaults to
        :data:`MIN_CONFIDENCE_TO_JOIN`.
    :returns: True when a sequence was set.
    """
    if not resolution:
        return False
    bar = MIN_CONFIDENCE_TO_JOIN if minimum is None else minimum
    record["sequence_resolution"] = {
        "method": resolution.get("method"),
        "confidence": resolution.get("confidence"),
        "evidence": resolution.get("evidence") or [],
        "applied": False,
    }
    if resolution.get("cross_references"):
        record["sequence_resolution"]["cross_references"] = resolution["cross_references"]
        record.setdefault("cross_references", resolution["cross_references"])

    sequence = resolution.get("sequence")
    if not sequence or (resolution.get("confidence") or 0.0) < bar:
        return False
    record["peptide_sequence"] = sequence
    record["sequence_resolution"]["applied"] = True
    return True


# --------------------------------------------------------------------------------------------
# Revalidation of resolutions already written to disk
# --------------------------------------------------------------------------------------------

def revalidate_resolution(record: Dict[str, Any]) -> Optional[str]:
    """
    Whether a resolution already applied to a record still qualifies under the current rules.

    Needed because the rules tightened after records had been written. The first version applied a
    sequence from the catalogue without validating it, and over 1,172 applied resolutions in a real
    run that produced: ``'Asp' -> 'D'`` (one amino acid), ``'C2' -> 'ACETYLATED'`` (an English word
    in a sequence column), 30 sequences of 1-4 residues, and 67 keyed on a bare table row label
    that means nothing outside its own paper. Those are on disk and have to be withdrawn.

    Withdrawal is reversible by design: the sequence is cleared but the resolution record is kept
    and annotated, so what was removed and why is still legible.

    :param record: A peptide record, inspected not modified.
    :returns: A reason string when the resolution must be withdrawn, else None.
    """
    resolution = record.get("sequence_resolution") or {}
    if not resolution.get("applied"):
        return None
    name = record.get("peptide_name")
    sequence = (record.get("peptide_sequence") or "").strip()
    if not sequence:
        return "no sequence present despite applied=True"
    if is_junk_name(name):
        return f"{name!r} is not a peptide name (measurement, organism or blank)"
    method = str(resolution.get("method") or "")
    if method.startswith("catalogue") and is_document_local_name(name):
        return f"{name!r} is a document-local label; a catalogue match on it is meaningless"
    if (resolution.get("confidence") or 0.0) < MIN_CONFIDENCE_TO_JOIN:
        return (f"confidence {resolution.get('confidence')} is below the current threshold "
                f"{MIN_CONFIDENCE_TO_JOIN}")
    if method.startswith("catalogue"):
        if not _usable_catalogue_sequence(sequence):
            return f"catalogue sequence {sequence[:24]!r} fails validation (too short, or not a sequence)"
    elif _clean_sequence(sequence) is None:
        return f"sequence {sequence[:24]!r} fails the shared sequence test"
    return None


def revalidate_extraction_file(path: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Re-check every applied resolution in one extraction record, withdrawing those that no longer
    qualify.

    :param path: An extraction JSON written by the extraction step.
    :param dry_run: Report what would change without writing.
    :returns: ``{"withdrawn": n, "kept": n, "reasons": [...]}``; ``{"error": ...}`` when unreadable.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}", "withdrawn": 0, "kept": 0, "reasons": []}

    withdrawn = kept = 0
    reasons: List[str] = []
    for record in document.get("peptides") or []:
        if not isinstance(record, dict):
            continue
        reason = revalidate_resolution(record)
        if reason is None:
            if (record.get("sequence_resolution") or {}).get("applied"):
                kept += 1
            continue
        withdrawn += 1
        reasons.append(reason)
        if dry_run:
            continue
        # The sequence goes; the evidence trail stays, annotated with why it was withdrawn.
        record["peptide_sequence"] = ""
        resolution = record.setdefault("sequence_resolution", {})
        resolution["applied"] = False
        resolution["withdrawn_reason"] = reason

    if withdrawn and not dry_run:
        metadata = document.setdefault("metadata", {})
        metadata["sequences_resolved"] = max(
            0, int(metadata.get("sequences_resolved") or 0) - withdrawn)
        metadata["resolutions_withdrawn"] = int(
            metadata.get("resolutions_withdrawn") or 0) + withdrawn
        # Written through a temporary file in the same directory: a crash mid-write must not leave
        # a truncated extraction record, which later stages read as a document with no peptides.
        temporary = f"{path}.partial"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, ensure_ascii=False)
            os.replace(temporary, path)
        except OSError as e:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
            return {"error": f"write failed: {e}", "withdrawn": 0, "kept": kept, "reasons": []}
    return {"withdrawn": withdrawn, "kept": kept, "reasons": reasons}


def revalidate_extraction_dirs(directories: Sequence[str], dry_run: bool = False,
                               limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Revalidate every extraction record under the given directories.

    Idempotent: a record whose resolutions all still qualify is left untouched and not rewritten,
    so this can be re-run or interrupted freely.

    :param directories: Extraction output directories, read through
        :func:`sample.config.iter_extraction_files` so only recognised source subdirectories are
        visited -- never ``os.walk``, which would also sweep ``.bak``/``_old`` siblings.
    :param dry_run: Report without writing.
    :param limit: Stop after this many files.
    :returns: Aggregate counts plus the most common withdrawal reasons.
    """
    from sample.config import iter_extraction_files

    files: List[str] = []
    for directory in directories:
        files.extend(iter_extraction_files(directory))
    if limit:
        files = files[:limit]

    import collections

    totals = {"files": 0, "files_changed": 0, "withdrawn": 0, "kept": 0, "errors": 0}
    reasons: "collections.Counter[str]" = collections.Counter()
    for path in files:
        outcome = revalidate_extraction_file(path, dry_run=dry_run)
        totals["files"] += 1
        if outcome.get("error"):
            totals["errors"] += 1
            logger.warning(f"revalidation failed for {path}: {outcome['error']}")
            continue
        totals["withdrawn"] += outcome["withdrawn"]
        totals["kept"] += outcome["kept"]
        if outcome["withdrawn"]:
            totals["files_changed"] += 1
        for reason in outcome["reasons"]:
            # Collapse to the leading clause so the summary groups rather than lists every name.
            reasons[reason.split(";")[0][:70]] += 1
    totals["top_reasons"] = reasons.most_common(8)
    return totals
