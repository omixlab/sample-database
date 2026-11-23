"""
One-time offline build step: parses the raw CAMP/DBAASP/DRAMP/ADP database dumps under
data/antimicrobial_databases/ for each entry's name and canonical sequence, and writes a compact
name<TAB>sequence lookup to data/resources/name_to_sequence.tsv.

Why this exists: a paper commonly refers to a well-known peptide only by name ("magainin 2",
"FK-156", "LL-37") without restating its sequence anywhere nearby (or the sequence sits in an
OCR-mangled table far from the activity data). extract_peptides_from_text's in-document alias
mechanism (find_local_aliases) only binds a name to a sequence when both are written adjacently
in the SAME document -- it can't help when the document only ever gives the name. This global,
cross-document lookup (built once, offline, here) lets sample/extractor.py backfill
peptide_sequence for a name-only candidate whenever that name is itself a real, already-
catalogued peptide.

Re-run manually (`python scripts/build_name_sequence_lookup.py`) if the raw database dumps
change; this is not part of the runtime extraction pipeline.
"""
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample.config import (
    CAMP_DIR, DBAASP_DIR, DRAMP_GENERAL_TSV, ADP_CSV, NAME_SEQUENCE_MAP_TSV,
)
from sample.nlp import is_valid_peptide_name, is_valid_sequence

csv.field_size_limit(10_000_000)

# Normalized name -> Counter of sequences seen for it across all sources. CAMP/DBAASP in
# particular often reuse a parent peptide's bare name for engineered analogs and fragments
# without qualifying the Title/name field (e.g. "LL-37" alone labels the full 37-residue peptide
# in most entries, but also several single-residue-substituted analogs and N/C-terminal
# fragments) -- so a name resolving to more than one sequence is common and not automatically
# unusable. Resolved by strict plurality below: the single most frequent sequence wins only if no
# other sequence ties it: a genuine ambiguous case (e.g. the bare family name "Magainin" covering
# two, equally-represented, distinct peptides) still gets dropped, never guessed.
_candidates: dict[str, Counter] = defaultdict(Counter)
# normalized key -> one representative original-cased spelling, for output readability.
_display_name: dict[str, str] = {}

# Hyphen vs space is typographic variation in peptide nomenclature ("LL-37" / "LL 37",
# "Magainin-2" / "Magainin 2"), not a semantic difference -- collapsing them onto the same key
# merges what would otherwise be split, weaker pluralities for the exact same peptide.
def _normalize_key(name: str) -> str:
    return re.sub(r'[\s\-]+', ' ', name).strip().upper()


def _clean_sequence(raw) -> str | None:
    if not raw:
        return None
    seq = str(raw).upper().strip().replace('-', '')
    if len(seq) < 5 or not is_valid_sequence(seq):
        return None
    return seq


# Several sources embed a descriptive tag as a trailing parenthetical rather than an actual
# alternate name -- DRAMP: "Epicidin 280 (Bacteriocin)"; ADP-style: "Esculentin-2-OR4 (Frogs,
# amphibians, animals)". A paper almost always refers to the peptide by its bare name, never the
# database's internal descriptor, so the bare form is registered as its own candidate key too
# (square-bracketed analog suffixes like "Magainin-2 [F16W]" are a different, genuinely distinct
# peptide and are deliberately left untouched -- only trailing round parens are stripped).
_TRAILING_PAREN = re.compile(r'\s*\([^()]*\)\s*$')

# Names this short/generic collide with running English prose too often to trust blindly -- e.g.
# "Syn" (a real but obscure 3-letter DBAASP peptide name) matched "syn-ergistic" (a line-wrapped
# "synergistic") in a real document, fabricating a whole activity record for an unrelated
# peptide. Real peptide-specific codes referenced in papers (LL-37, FK-156, HNP-1, SP4080) are
# essentially always >=5 characters; the lab-notebook-style short codes below that length
# (~2,400 of the ~38,000 raw candidates -- "23a", "2D2", "ML1") are exactly the collision-prone
# tail, not the peptides this cross-reference is meant to recover.
_MIN_NAME_LEN = 5

# Bare AMP family/genus names cover many genuinely distinct peptides, so any single sequence
# "winning" the plurality vote for one of these is a curation artifact (whichever member happens
# to have the most same-named raw entries), not evidence that name refers to one specific
# sequence -- found via "Cecropin" (26 raw entries split across 10+ distinct sequences, top one
# winning 2-to-1 over each runner-up individually) resolving to an arbitrary single cecropin
# variant. Mirrors get_seq_names_kp()'s own macro_classes list (the codebase's existing set of
# names it already treats as class-level, not sequence-level), extended with other common AMP
# genus names that pose the same risk. A qualified form ("Cecropin A", "Cecropin-4") is a specific
# peptide and is NOT excluded -- only the exact bare family name is.
_GENERIC_FAMILY_NAMES = {
    "COLICIN", "BACTERIOCIN", "NISIN", "MUREIN", "CASIDIN", "CIDIN", "DEFENSIN", "DEFENSINS",
    "MAGAININ", "MAGAININS", "CECROPIN", "CECROPINS", "TEMPORIN", "TEMPORINS", "BREVININ",
    "BREVININS", "DERMASEPTIN", "DERMASEPTINS", "ANDROCTONIN", "THANATIN", "INDOLICIDIN",
    "PROTEGRIN", "PROTEGRINS", "TACHYPLESIN", "TACHYPLESINS", "LACTOFERRICIN", "HISTATIN",
    "HISTATINS", "BUFORIN", "BUFORINS",
}


def _add(name: str, seq: str) -> None:
    if len(name) < _MIN_NAME_LEN or not is_valid_peptide_name(name):
        return
    key = _normalize_key(name)
    if key in _GENERIC_FAMILY_NAMES:
        return
    _candidates[key][seq] += 1
    _display_name.setdefault(key, name)


def _register(name, seq) -> None:
    if not name:
        return
    name = str(name).strip()
    seq = _clean_sequence(seq)
    if seq is None:
        return
    _add(name, seq)

    bare = _TRAILING_PAREN.sub('', name).strip()
    if bare and bare != name:
        _add(bare, seq)


def parse_camp():
    n = 0
    for fp in CAMP_DIR.glob("*.csv"):
        n += 1
        title = None
        sequence = None
        try:
            with open(fp, encoding='utf-8', errors='replace') as f:
                for row in csv.reader(f):
                    if len(row) < 2:
                        continue
                    key, val = row[0].strip(), row[1].strip()
                    if key == "Title":
                        title = val
                    elif key == "Sequence":
                        sequence = val
        except Exception:
            continue
        if title and sequence:
            _register(title, sequence)
    print(f"CAMP: scanned {n} files")


def parse_dbaasp():
    n = 0
    for fp in DBAASP_DIR.glob("*.json"):
        n += 1
        try:
            data = json.loads(fp.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            continue
        sequence = data.get("sequence")
        if not sequence:
            continue
        for name in (data.get("name"), data.get("majorName")):
            if name:
                _register(name, sequence)
    print(f"DBAASP: scanned {n} files")


def parse_dramp():
    if not DRAMP_GENERAL_TSV.exists():
        print("DRAMP: general_amps.txt not found, skipping")
        return
    n = 0
    with open(DRAMP_GENERAL_TSV, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            n += 1
            name = row.get("Name")
            sequence = row.get("Sequence")
            if name and sequence:
                _register(name, sequence)
    print(f"DRAMP: scanned {n} rows")


# Only the primary name (text before the first parenthetical) is used from ADP -- its
# "Name/Class:" field packs alt-names and free-text classification tags ("natural AMPs",
# "Ala-rich", "frog, amphibians, animals"...) into one comma/semicolon-delimited parenthetical
# with no reliable way to tell an alias from a category label, and a wrong alias would silently
# fabricate a name->sequence binding. The primary name alone is unambiguous and still valuable.
_ADP_PRIMARY_NAME = re.compile(r'^([^(]+)')


def parse_adp():
    if not ADP_CSV.exists():
        print("ADP: adb.csv not found, skipping")
        return
    n = 0
    with open(ADP_CSV, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    # First line is a bare numeric column index (0,1,2,...), the real header is the second line.
    buf = io.StringIO(''.join(lines[1:]))
    reader = csv.DictReader(buf)
    for row in reader:
        n += 1
        name_field = row.get("Name/Class:")
        sequence = row.get("Sequence:")
        if not name_field or not sequence or name_field == "Name/Class:":
            continue  # skips the repeated-header artifact rows seen in this dump
        m = _ADP_PRIMARY_NAME.match(name_field)
        if m:
            _register(m.group(1).strip(), sequence)
    print(f"ADP: scanned {n} rows")


def main():
    parse_camp()
    parse_dbaasp()
    parse_dramp()
    parse_adp()

    resolved = 0
    dropped_ambiguous = 0
    NAME_SEQUENCE_MAP_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(NAME_SEQUENCE_MAP_TSV, 'w', encoding='utf-8') as out:
        for key, counts in _candidates.items():
            if len(counts) == 1:
                winner_seq = next(iter(counts))
            else:
                ranked = counts.most_common(2)
                (winner_seq, winner_n), (_, runner_up_n) = ranked[0], ranked[1]
                if winner_n <= runner_up_n:
                    dropped_ambiguous += 1
                    continue
            out.write(f"{_display_name[key]}\t{winner_seq}\n")
            resolved += 1

    print(f"Wrote {resolved} name->sequence entries to {NAME_SEQUENCE_MAP_TSV}")
    print(f"Dropped {dropped_ambiguous} ambiguous names (no single sequence is the clear plurality)")


if __name__ == "__main__":
    main()
