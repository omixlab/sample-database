"""
What it takes for an extracted sequence to count as new to the field.

Novelty used to be set membership: canonicalize the sequence, ask whether the string is in the
union of the source databases, and if not, call it ``SAMPLE (new)``. That test is right about
what it literally asks and wrong about what the column is read as meaning, because a string
absent from the catalogue is not the same thing as a peptide absent from the literature. Measured
over 32,408 refined documents, 975 extracted sequences carried a trustworthy measurement and were
absent from the 54,180 curated sequences. Classifying all 975 by hand:

    145  a fragment of a catalogued sequence       the extractor caught part of a known peptide
    121  a catalogued sequence plus flanking text
    103  three-letter notation, never decoded      ARGLEUTYRLEU... cannot match a one-letter set
     60  within two substitutions of a catalogue entry
     46  identifiable MHC epitopes                 SIINFEKL in 76 documents, NLVPMVATV in 46
     36  carrying an ambiguity code                IIIUKIKTDO, labelled melittin
      5  within two edits once indels are allowed  RRGWVLALYLRLYGRR vs RRGWVLALYLRYGRR
      4  a name or prose spelled in residue letters  ISGRIFFITHSIN -- "is griffithsin"

leaving roughly 230 for which the claim is defensible, so the old test overstated novelty about
fourfold. The decisive measurement is narrower and worse: for the 40 of those sequences whose
extracted *name* is a peptide this project already catalogues, the sequence disagreed with the
catalogued one **40 times out of 40**. Where novelty is checkable at all, it was always an
extraction error. That is why the verdict below is graded rather than binary, and why a name that
contradicts its sequence produces its own verdict instead of a novelty claim.

Three principles follow from the table.

**A sequence that cannot be compared is not novel.** An ambiguity code (B=Asx, Z=Glx, X=any,
U=Sec, O=Pyl) is legitimate in a sequence -- `sequence.validate_extracted_sequence` accepts all
five on purpose, because real dumps use them -- but it makes identity undecidable: ``AINXDAAYL``
might equal a catalogued sequence at the masked position. Novelty needs identity, so these get
:data:`NOT_COMPARABLE` rather than a guess in either direction. The same holds below
:data:`MIN_COMPARABLE_LENGTH` residues, where the catalogue contains so many short entries that
absence carries no information.

**Notation is not identity.** ``ARGLEUTYRLEU`` and ``RLYL`` are the same peptide, so novelty is
judged on a decoded form. :func:`sample.sequence.sequence_to_1_letter` handles separated
three-letter code; the unseparated run that OCR produces when it drops the hyphens is decoded
here, by :func:`decode_run_of_three_letter_codes`.

**The name is evidence about the sequence.** A name that identifies exactly one catalogued
sequence adjudicates: agreement confirms, disagreement is a defect report. A name attached to
many sequences adjudicates nothing, and :mod:`sample.names` already measured where that line
falls -- see :data:`sample.names.NAME_SEQUENCE_THRESHOLD`, reused here rather than restated.

This module decides; it does not rewrite records. Step 21 (:mod:`sample.merge`) applies it to the
released activity table, and step 12 (:mod:`sample.field_review`) uses the name adjudication to
turn `sequence_of_another_peptide` from a failure mode nobody could detect into a rule.
"""
import bisect
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

logger = logging.getLogger("sample.novelty")

#: Ambiguity and non-standard residue codes. Accepted by the sequence validator on purpose, but
#: each one masks a position, so a sequence carrying any of them cannot be compared for identity.
#: B=Asx, Z=Glx, X=any, U=Sec, O=Pyl.
AMBIGUITY_CODES: frozenset = frozenset("BZXUO")

#: Shortest sequence whose absence from the catalogue means anything. Measured: of the extracted
#: sequences absent from the catalogue and carrying a trustworthy measurement, every one below
#: eight residues was a table label, an abbreviation or a fragment -- 62 of them. The sequence
#: validator's floor of five is about whether a string is a sequence at all; this is the higher
#: bar for whether its absence is evidence.
MIN_COMPARABLE_LENGTH: int = 8

#: A sequence within this many edits of a catalogued one is reported as that one's analogue
#: rather than as new. Two, because the measured failures were single-residue insertions from OCR
#: (``RRGWVLALYLRLYGRR`` for ``RRGWVLALYLRYGRR``) and deliberate point substitutions in published
#: analogue series (the TL19 nisin variants, 2Asp and 34Ala). Both are things the field already
#: knows about; neither is a new peptide. Raising it would start swallowing genuine family members.
MAX_ANALOGUE_EDITS: int = 2

#: Shortest catalogued sequence that may be reported as the parent of a fragment or extension.
#: Below this, a short catalogued entry occurs inside unrelated longer sequences by chance.
MIN_PARENT_LENGTH: int = 8

#: Verdicts, worst-supported claim first. Only :data:`NOVEL` asserts that the field has not seen
#: this peptide; every other value says what was found instead, so a reader can audit the
#: refinement rather than take it on trust.
NAME_CONFLICT: str = "name_conflict"
NOT_COMPARABLE: str = "not_comparable"
KNOWN_EXACT: str = "known_exact"
KNOWN_ENCODING: str = "known_encoding"
KNOWN_FRAGMENT: str = "known_fragment"
KNOWN_EXTENSION: str = "known_extension"
ANALOGUE: str = "analogue"
NOVEL: str = "novel"

VERDICTS: Tuple[str, ...] = (NAME_CONFLICT, NOT_COMPARABLE, KNOWN_EXACT, KNOWN_ENCODING,
                             KNOWN_FRAGMENT, KNOWN_EXTENSION, ANALOGUE, NOVEL)

#: How a name relates to the catalogue. Only the first two adjudicate anything: a name shared by
#: :data:`sample.names.NAME_SEQUENCE_THRESHOLD` sequences or more is a class label or an organism
#: that a source filed in a name column, and a name the catalogue has never seen cannot confirm or
#: contradict a sequence. Treating either as evidence is how a name ends up overruling a sequence
#: it knows nothing about.
NAME_UNIQUE: str = "unique"           # exactly one catalogued sequence: it adjudicates
NAME_SERIES: str = "series"           # a handful, as an analogue series shares a name
NAME_AMBIGUOUS: str = "ambiguous"     # too many to identify anything
NAME_UNKNOWN: str = "unknown"         # not in the catalogue
NAME_ABSENT: str = "absent"           # the record carries no name

#: Names that may be used as evidence about a sequence.
ADJUDICATING: Tuple[str, ...] = (NAME_UNIQUE, NAME_SERIES)

#: What an adjudicating name says about the sequence it was extracted beside. Recorded separately
#: from the novelty verdict because the two are different questions and were conflated at first,
#: with a measurable cost: judging the name before the sequence reported ``DMS-H4`` as a conflict
#: for carrying ``GLWSTIKNVGKEAAIAAGKAALGAL`` when the catalogued sequence is
#: ``GLWSTIKNVGKEAAIAAGKAALGALG`` -- the same peptide, one residue short. Agreement is tested
#: against the same graded relations novelty uses, so a truncation reads as a truncation.
NAME_AGREES: str = "agrees"
NAME_DISAGREES: str = "disagrees"
NAME_SILENT: str = "silent"           # the name adjudicates nothing, so it is not evidence

#: Three-letter codes, including the non-standard ones that appear in unseparated runs. Kept
#: local and explicit because the decoder below walks a string three characters at a time and
#: needs a flat token set, not the mapping tables' bracket notation.
_RUN_CODES: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

#: Leading and trailing decoration an unseparated run carries: cyclization markers and terminal
#: groups. Stripped before decoding so ``CYCLOLEUARGLEU`` and ``ARGLEUNH2`` decode.
_RUN_PREFIXES: Tuple[str, ...] = ("CYCLO", "CYCLIC", "LINEAR")
_RUN_SUFFIXES: Tuple[str, ...] = ("NH2", "NH", "OH", "ACID", "AMIDE", "OME", "COOH")

#: Fraction of an unseparated run that must parse as three-letter codes before it is read as
#: three-letter notation. Four fifths, not all: the measured runs carry stereochemical prefixes
#: and linkers between codes (``NOCTANOYLAIBGLYLEUAIBGLY``), and demanding a clean parse rejected
#: them. Below this the string is far more likely a one-letter sequence that happens to contain
#: a few triplets that spell residue names.
_RUN_COVERAGE: float = 0.80

#: Shortest decoded run worth preferring over the original string. A three-residue decode from a
#: nine-character string is as likely coincidence as notation.
_RUN_MIN_DECODED: int = 4


def _block_spans(length: int) -> List[Tuple[int, int]]:
    """
    :param length: Length of a sequence to partition.
    :returns: ``(start, stop)`` for each of the ``MAX_ANALOGUE_EDITS + 1`` blocks, as evenly
        sized as the length allows. Deterministic, because the index and the query must agree on
        where the cuts fall.
    """
    count = MAX_ANALOGUE_EDITS + 1
    if length <= 0:
        return []
    if length < count:
        return [(index, index + 1) for index in range(length)]
    size, remainder = divmod(length, count)
    spans, start = [], 0
    for index in range(count):
        stop = start + size + (1 if index < remainder else 0)
        spans.append((start, stop))
        start = stop
    return spans


def _blocks_of(form: str) -> List[str]:
    """
    :param form: A canonical sequence.
    :returns: Its blocks, as strings. See :func:`_block_spans` for why the partition is fixed.
    """
    return [form[start:stop] for start, stop in _block_spans(len(form))]


def canonical(sequence: Any) -> str:
    """
    :param sequence: A sequence as written, in any notation.
    :returns: Upper-case residue letters only, with every separator and modifier stripped. Not
        itself a novelty judgment -- just the form two sequences are compared in.
    """
    return re.sub(r"[^A-Z]", "", str(sequence or "").upper())


def normalised_name(name: Any) -> str:
    """
    :param name: A peptide name as written.
    :returns: A lookup key for the name catalogue: lower-case alphanumerics only, so
        ``Dermaseptin-B2``, ``dermaseptin b2`` and ``DRS B2`` collapse the way a reader expects.
    """
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def decode_run_of_three_letter_codes(sequence: Any) -> str:
    """
    Decodes three-letter notation whose separators are gone.

    :param sequence: A sequence as written.
    :returns: The one-letter form, or ``""`` if the string does not read as a run of three-letter
        codes.

    :func:`sample.sequence.sequence_to_1_letter` handles ``Gly-Ile-Gly`` and ``Gly Ile Gly``,
    because it splits on the separator. PDF extraction routinely loses the hyphens, and the result
    reaches this project as ``GLYILEGLY`` -- which that function returns unchanged, since without
    a separator it has nothing to split on. 103 such strings were counted among the sequences the
    old novelty test called new, every one of them a peptide the catalogue already holds in
    one-letter form. They are the single largest source of false novelty.

    Deliberately greedy and forgiving: a character that starts no code is skipped, which is what
    lets a stereochemical ``D`` between codes through without derailing the parse. The
    :data:`_RUN_COVERAGE` floor is what keeps that forgiveness from decoding arbitrary strings.
    """
    body = canonical(sequence)
    if not body:
        return ""
    for prefix in _RUN_PREFIXES:
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    for suffix in _RUN_SUFFIXES:
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            break
    if len(body) < 3:
        return ""

    decoded: List[str] = []
    covered = index = 0
    while index + 3 <= len(body):
        token = body[index:index + 3]
        if token in _RUN_CODES:
            decoded.append(_RUN_CODES[token])
            covered += 3
            index += 3
        else:
            index += 1
    if len(decoded) < _RUN_MIN_DECODED or covered / len(body) < _RUN_COVERAGE:
        return ""
    return "".join(decoded)


def comparable_form(sequence: Any) -> Tuple[str, str]:
    """
    The form a sequence is compared in, and whether it can be compared at all.

    :param sequence: A sequence as written.
    :returns: ``(form, reason)``. ``reason`` is ``""`` when the sequence is comparable; otherwise
        ``form`` is ``""`` and ``reason`` is a stable slug saying why novelty cannot be decided.
    """
    from sample.sequence import sequence_to_1_letter

    raw = canonical(sequence)
    if not raw:
        return "", "empty"

    # Decode before judging. A three-letter run is a known peptide written differently, and
    # judging the undecoded string is how 103 of them were published as new.
    form = decode_run_of_three_letter_codes(sequence)
    if not form:
        separated = canonical(sequence_to_1_letter(str(sequence)) or "")
        form = separated or raw

    if set(form) & AMBIGUITY_CODES:
        return "", "ambiguity_code"
    if len(form) < MIN_COMPARABLE_LENGTH:
        return "", "too_short_to_adjudicate"
    return form, ""


class Verdict(NamedTuple):
    """
    One novelty judgment, with its grounds.

    :ivar verdict: One of :data:`VERDICTS`.
    :ivar parent: The catalogued sequence this was matched against, when there is one.
    :ivar distance: Edit distance to ``parent`` for :data:`ANALOGUE`, else ``0``.
    :ivar name_status: ``NAME_UNIQUE``/``SERIES``/``AMBIGUOUS``/``UNKNOWN``/``ABSENT`` -- how far
        the record's name could settle anything.
    :ivar name_agreement: :data:`NAME_AGREES`, :data:`NAME_DISAGREES` or :data:`NAME_SILENT`.
        Silent whenever ``name_status`` is not in :data:`ADJUDICATING`, so a reader can tell "the
        name corroborates the sequence" from "the name had no standing to say".
    :ivar named: The catalogued sequence the *name* identifies, when the name adjudicates. Not the
        same thing as ``parent``, which is what the *sequence* matched, and the two differ in
        exactly the case worth reporting: a sequence that is catalogued under some other peptide's
        name. Reporting ``parent`` as though the name pointed to it claimed that a sequence did
        not match itself.
    :ivar form: The comparable form actually judged, which is not always what was extracted.
    :ivar note: A short slug carrying the reason for :data:`NOT_COMPARABLE` or the relation found.
    """
    verdict: str
    parent: str = ""
    distance: int = 0
    name_status: str = NAME_UNKNOWN
    name_agreement: str = NAME_SILENT
    named: str = ""
    form: str = ""
    note: str = ""

    @property
    def is_novel(self) -> bool:
        """:returns: Whether this judgment supports claiming the peptide as new to the field."""
        return self.verdict == NOVEL

    @property
    def is_defect(self) -> bool:
        """
        :returns: Whether this judgment reports a problem with the record rather than a fact about
            the peptide -- so these are the rows worth sending back through review rather than
            counting either way. A disagreeing name qualifies even when the shape tests found a
            parent, because then the sequence belongs to a peptide other than the one it was
            filed under, which corrupts every activity attached to the record.
        """
        return (self.verdict in (NAME_CONFLICT, NOT_COMPARABLE)
                or self.name_agreement == NAME_DISAGREES)


def _within(a: str, b: str, limit: int) -> Optional[int]:
    """
    :param a: One sequence.
    :param b: The other.
    :param limit: Largest edit distance of interest.
    :returns: The Levenshtein distance when it is ``<= limit``, else ``None``.

    Banded, and it abandons a row whose cheapest cell already exceeds the limit, because this runs
    once per candidate against every catalogued sequence of a comparable length.
    """
    if abs(len(a) - len(b)) > limit:
        return None
    previous = list(range(len(b) + 1))
    for row, character in enumerate(a, 1):
        current = [row] + [limit + 1] * len(b)
        low, high = max(1, row - limit), min(len(b), row + limit)
        for column in range(low, high + 1):
            current[column] = min(previous[column] + 1,
                                  current[column - 1] + 1,
                                  previous[column - 1] + (character != b[column - 1]))
        if min(current[low:high + 1] or [limit + 1]) > limit:
            return None
        previous = current
    return previous[len(b)] if previous[len(b)] <= limit else None


class Catalogue:
    """
    Everything this project already knows, in the shapes the novelty tests need.

    Built once and reused, because the fragment test needs one big joined string and the analogue
    test needs sequences bucketed by length. Constructing those per call made the merge step
    quadratic in the catalogue.
    """

    def __init__(self, sequences: Iterable[str],
                 names: Optional[Dict[str, Set[str]]] = None,
                 ambiguous_at: Optional[int] = None):
        """
        :param sequences: Catalogued sequences, in any notation.
        :param names: Optional map from :func:`normalised_name` key to the set of canonical
            sequences bearing that name.
        :param ambiguous_at: Number of distinct sequences at which a name stops identifying
            anything. Defaults to :data:`sample.names.NAME_SEQUENCE_THRESHOLD`.
        """
        if ambiguous_at is None:
            from sample.names import NAME_SEQUENCE_THRESHOLD
            ambiguous_at = NAME_SEQUENCE_THRESHOLD
        self.ambiguous_at = ambiguous_at
        self.sequences: Set[str] = set()
        for raw in sequences:
            form = canonical(raw)
            if form:
                self.sequences.add(form)
        self.by_length: Dict[int, List[str]] = defaultdict(list)
        for form in self.sequences:
            self.by_length[len(form)].append(form)

        # The analogue test's index. Brute force is not available here: the length window around
        # a 20-mer holds 17,857 catalogued sequences, and judging 13,183 extracted ones against
        # them is 235 million edit-distance computations.
        #
        # Pigeonhole instead. If two sequences are within MAX_ANALOGUE_EDITS edits and one is cut
        # into MAX_ANALOGUE_EDITS + 1 contiguous blocks, at least one block survives untouched --
        # each edit damages at most one block, and there is always one block more than there are
        # edits -- so that block appears verbatim in the other sequence. Indexing every
        # catalogued sequence by its blocks therefore turns "find everything within two edits"
        # into "look up the query's own substrings", which is exact rather than heuristic: a
        # sequence the index does not return cannot be within two edits.
        self._blocks: Dict[str, Set[int]] = defaultdict(set)
        self._order: List[str] = sorted(self.sequences)
        for index, form in enumerate(self._order):
            for block in _blocks_of(form):
                self._blocks[block].add(index)

        # One joined string, so the fragment test is a single C-level substring search instead
        # of 59,060 Python-level ones, plus the offsets needed to say WHICH sequence matched. The
        # separator cannot occur in a sequence, so no match spans two entries.
        self._blob = "|".join(self._order)
        self._starts: List[int] = []
        offset = 0
        for form in self._order:
            self._starts.append(offset)
            offset += len(form) + 1

        self.names: Dict[str, Set[str]] = {key: set(value) for key, value in (names or {}).items()}

    def __len__(self) -> int:
        return len(self.sequences)

    @classmethod
    def load(cls, merged_path: str = "data/sample_raw_release/merged_from_database.csv",
             names_path: str = "data/peptide_names.json",
             extra_sequences: Optional[Iterable[str]] = None) -> "Catalogue":
        """
        Reads the catalogue off disk.

        :param merged_path: Step 7's merged source-database CSV, which holds the curated
            sequences. Pass ``""`` to skip it, for a caller that already holds those sequences
            and passes them as ``extra_sequences``.
        :param names_path: Step 35's name harvest, which maps sequence to names and so supplies
            the name adjudication *and* a second, wider set of known sequences -- it carries
            entries the merged CSV drops.
        :param extra_sequences: Further known sequences to fold in, for a caller that has already
            loaded them.
        :returns: The catalogue. A missing file is logged and skipped rather than raised: novelty
            degrades to a weaker claim without the catalogue, which is better than failing the
            merge step.
        """
        import csv

        sequences: List[str] = list(extra_sequences or ())
        if merged_path and os.path.exists(merged_path):
            limit = csv.field_size_limit()
            csv.field_size_limit(min(2 ** 31 - 1, 10 ** 9))
            try:
                with open(merged_path, newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        sequences.append(row.get("sequence") or "")
            finally:
                csv.field_size_limit(limit)
        elif merged_path:
            logger.warning("Novelty catalogue: %s is missing; curated sequences unavailable.",
                           merged_path)

        names: Dict[str, Set[str]] = defaultdict(set)
        if os.path.exists(names_path):
            with open(names_path, encoding="utf-8") as handle:
                harvest = json.load(handle)
            for sequence, entries in harvest.items():
                form = canonical(sequence)
                if not form:
                    continue
                sequences.append(form)
                for entry in entries or ():
                    key = normalised_name(entry.get("name") if isinstance(entry, dict) else entry)
                    if len(key) >= 3:
                        names[key].add(form)
        else:
            logger.warning("Novelty catalogue: %s is missing; names cannot adjudicate.", names_path)

        catalogue = cls(sequences, names)
        logger.info("Novelty catalogue: %d sequences, %d names.", len(catalogue), len(names))
        return catalogue

    # -- name adjudication ---------------------------------------------------------------------

    def name_status(self, name: Any) -> str:
        """
        :param name: A peptide name as written.
        :returns: One of the ``NAME_*`` constants.

        The threshold is :mod:`sample.names`' own, measured there: a name attached to 20 or more
        distinct sequences is not identifying a peptide (dbAMP filed ``Mouse-ear cress`` as a name
        for 300 of them), while a genuine analogue series shares a name across a handful.
        """
        key = normalised_name(name)
        if not key:
            return NAME_ABSENT
        known = self.names.get(key)
        if not known:
            return NAME_UNKNOWN
        if len(known) == 1:
            return NAME_UNIQUE
        return NAME_SERIES if len(known) < self.ambiguous_at else NAME_AMBIGUOUS

    def sequences_for_name(self, name: Any) -> Set[str]:
        """
        :param name: A peptide name as written.
        :returns: The catalogued sequences bearing that name, empty when the name is unknown.
        """
        return set(self.names.get(normalised_name(name), ()))

    # -- the tests, cheapest and most decisive first --------------------------------------------

    def _parent_containing(self, form: str) -> str:
        """
        :param form: The comparable form.
        :returns: A catalogued sequence that contains ``form``, or ``""``.

        The containing sequence, not ``form`` itself: :attr:`Verdict.parent` is read as "what
        this was matched against", and returning the fragment made the field say a sequence was
        a fragment of itself.
        """
        position = self._blob.find(form)
        if position < 0:
            return ""
        return self._order[bisect.bisect_right(self._starts, position) - 1]

    def _parent_inside(self, form: str) -> str:
        """
        :param form: The comparable form.
        :returns: The longest catalogued sequence occurring inside ``form``, or ``""``.

        Enumerates the query's own substrings and asks the sequence set about each, which costs
        ``len(form)**2 / 2`` dictionary lookups. The other direction -- testing every catalogued
        sequence for occurrence in ``form`` -- costs one substring search per catalogued sequence
        shorter than the query, which for a 40-mer is most of the catalogue.
        """
        for length in range(len(form) - 1, MIN_PARENT_LENGTH - 1, -1):
            for start in range(0, len(form) - length + 1):
                candidate = form[start:start + length]
                if candidate in self.sequences:
                    return candidate
        return ""

    def _nearest(self, form: str) -> Tuple[str, int]:
        """
        :param form: The comparable form.
        :returns: ``(parent, distance)`` for the closest catalogued sequence within
            :data:`MAX_ANALOGUE_EDITS`, or ``("", 0)``. Closest, not merely first: a point
            substitution of a catalogued peptide and a two-edit neighbour are different claims.
        """
        window = range(len(form) - MAX_ANALOGUE_EDITS, len(form) + MAX_ANALOGUE_EDITS + 1)
        lengths = {stop - start for length in window if length > 0
                   for start, stop in _block_spans(length)}
        candidates: Set[int] = set()
        for length in lengths:
            if length < 1 or length > len(form):
                continue
            for start in range(0, len(form) - length + 1):
                found = self._blocks.get(form[start:start + length])
                if found:
                    candidates |= found

        best, best_distance = "", MAX_ANALOGUE_EDITS + 1
        for index in candidates:
            candidate = self._order[index]
            if abs(len(candidate) - len(form)) > MAX_ANALOGUE_EDITS:
                continue
            distance = _within(form, candidate, min(MAX_ANALOGUE_EDITS, best_distance - 1))
            if distance is not None and distance < best_distance:
                best, best_distance = candidate, distance
                if distance == 1:
                    break          # cannot do better: distance 0 is the exact test, already run
        return (best, best_distance) if best else ("", 0)

    def _judge_shape(self, form: str, raw: str, status: str) -> Verdict:
        """
        The containment and distance tests, applied to an already-comparable form.

        :param form: The comparable form.
        :param raw: The canonical form of what was extracted, for the record.
        :param status: The name status, carried through.
        :returns: A :class:`Verdict`, never :data:`KNOWN_EXACT` -- identity is settled before
            this is reached.
        """
        parent = self._parent_containing(form)
        if parent:
            return Verdict(KNOWN_FRAGMENT, parent=parent, name_status=status, form=form,
                           note="fragment_of_a_catalogued_sequence")
        parent = self._parent_inside(form)
        if parent:
            return Verdict(KNOWN_EXTENSION, parent=parent, name_status=status, form=form,
                           note="catalogued_sequence_plus_flanking_text")
        parent, distance = self._nearest(form)
        if parent:
            return Verdict(ANALOGUE, parent=parent, distance=distance, name_status=status,
                           form=form, note=f"within_{distance}_edits_of_a_catalogued_sequence")
        return Verdict(NOVEL, name_status=status, form=form)

    def _relation(self, form: str, raw: str, targets: Set[str]) -> str:
        """
        :param form: The comparable form of the extracted sequence.
        :param raw: The canonical form of what was extracted.
        :param targets: Sequences to relate it to.
        :returns: ``""`` when ``form`` bears no relation to any of ``targets``, else a slug naming
            the relation found. Used to ask whether a name's sequence and an extracted sequence
            are the same peptide under any of the relations novelty already recognises, so that a
            truncation is not reported as a contradiction.
        """
        for target in targets:
            if form == target or raw == target:
                return "same"
            if form in target or target in form:
                return "contained"
            if _within(form, target, MAX_ANALOGUE_EDITS) is not None:
                return "analogue"
        return ""

    def judge(self, sequence: Any, name: Any = "") -> Verdict:
        """
        Decides whether an extracted sequence is new to the field.

        :param sequence: The sequence as extracted, in any notation.
        :param name: The peptide name as extracted. Adjudicating only when the catalogue says it
            identifies something -- see :func:`name_status`.
        :returns: A :class:`Verdict`.

        **The sequence is judged first, then the name is asked to corroborate it.** The reverse
        order was tried and is wrong: it reported ``DMS-H4`` as a name conflict for a sequence one
        residue shorter than the catalogued one, because a name test applied before the
        containment tests cannot tell a truncation from a different peptide.

        Within the sequence tests the order runs from the strongest claim about the catalogue to
        the weakest -- exact identity, then notation, then containment either way, then edit
        distance. Each still says this peptide is not new; they differ in what they found instead,
        and the strongest available relation is the one worth reporting.

        The name then does one of three things. It agrees, and the verdict stands with that
        corroboration recorded. It disagrees while the sequence matched something in the
        catalogue, which means the record pairs one peptide's name with another's sequence: the
        novelty verdict stands, but :attr:`Verdict.is_defect` is set, because every activity on
        that record is attributed to the wrong peptide. Or it disagrees while the sequence matched
        nothing, and then the novelty claim collapses into :data:`NAME_CONFLICT` -- the name says
        this peptide is catalogued, the sequence resembles nothing catalogued, and the measured
        base rate for that combination is 40 errors out of 40.

        A name that does not adjudicate stays silent and changes nothing, which is the whole
        point of separating :data:`NAME_AMBIGUOUS` from :data:`NAME_DISAGREES`.
        """
        status = self.name_status(name)
        raw = canonical(sequence)

        form, reason = comparable_form(sequence)
        if not form:
            return Verdict(NOT_COMPARABLE, name_status=status, form=raw, note=reason)

        if raw in self.sequences:
            base = Verdict(KNOWN_EXACT, parent=raw, name_status=status, form=raw)
        elif form in self.sequences:
            # Found only because the notation was decoded. Worth its own verdict: it says the
            # catalogue holds this peptide and the extractor wrote it in three-letter code, which
            # is a fixable extraction problem rather than a fact about the peptide.
            base = Verdict(KNOWN_ENCODING, parent=form, name_status=status, form=form,
                           note="three_letter_notation_of_a_catalogued_sequence")
        else:
            base = self._judge_shape(form, raw, status)

        if status not in ADJUDICATING:
            return base._replace(name_agreement=NAME_SILENT)

        named = self.sequences_for_name(name)
        shortest = sorted(named, key=lambda sequence: (len(sequence), sequence))[0] if named else ""
        if self._relation(form, raw, named):
            return base._replace(name_agreement=NAME_AGREES, named=shortest)

        if base.verdict == NOVEL:
            return Verdict(NAME_CONFLICT, parent=shortest, name_status=status,
                           name_agreement=NAME_DISAGREES, named=shortest, form=form,
                           note="name_identifies_a_different_sequence")
        return base._replace(name_agreement=NAME_DISAGREES, named=shortest)


def summarise(verdicts: Iterable[Verdict]) -> Dict[str, int]:
    """
    :param verdicts: Judgments to count.
    :returns: A count per verdict, every key in :data:`VERDICTS` present so a report does not have
        to guess whether a missing key means zero or means the test never ran.
    """
    counts = {verdict: 0 for verdict in VERDICTS}
    for verdict in verdicts:
        counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
    return counts
