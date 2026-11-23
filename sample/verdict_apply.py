"""
Acts on step 12's recorded verdicts, and applies the mechanical corrections that need no verdict.

Step 12 (:mod:`sample.field_review_run`) records a verdict per ambiguous field, but the Makefile
runs it without ``--apply``. The corpus therefore carries 34,071 verdicts and zero corrections:
37 hours of local inference produced a diagnosis nothing ever acted on. This module is the second
half of that stage.

Two layers, deliberately separated:

**The authorized layer** consumes the model's proposals. It never writes one as given. Every
proposal goes through the authority that owns the field -- NCBI taxonomy for an organism, the
29-unit vocabulary for a unit, the 38-term activity vocabulary for an assay, the amino-acid
alphabet for a sequence -- and is refused unless that authority confirms it. This is the same
"model proposes, deterministic rule authorizes" split :func:`sample.llm_validator
.fill_missing_fields_from_document` already uses, and it is load-bearing here because the
proposals are frequently unusable as text while still being right in substance: the review
answers ``target_organism`` with ``"Acinetobacter baumannii or Pseudomonas aeruginosa"``,
``"Listeria monocytogenes, Streptococcus suis, Campylobacter jejuni, and "`` and the literal
string ``"unsure"``; it answers ``peptide_sequence`` with three-letter notation; it answers
``value_raw`` with ``"3.8 +/- 0.4 ug/mL"``, unit included. Written verbatim, each of those makes
the field worse than leaving it alone.

**The mechanical layer** needs no model at all and is where most of the volume is. A metric name
sitting in the unit column, an activity type that differs from the vocabulary only in case, an
abbreviated genus the document itself spells out, a cell line spelled two ways, a sequence in
three-letter notation: each is decidable from the record plus a local authority, so each is
corrected for every activity that shows it, not only the 4,728 documents the review happened to
reach.

Both layers are idempotent and both record what they did. A correction leaves a provenance entry
naming the authority that allowed it, so a later pass can tell a repaired field from an extracted
one, and re-running this module changes nothing a previous run already settled.

What this module deliberately does **not** do:

- It does not rewrite a sequence over a ``sequence_of_another_peptide`` verdict. That verdict says
  the *name* and the *sequence* disagree, and which of the two is wrong is not determined by the
  disagreement. 1,059 verdicts say this; all 1,059 are recorded as conflicts and none is rewritten.
- It does not write a multi-organism answer. A measurement attributable to "A or B" is not
  attributable, and an empty target is honest where a conjunction is not.
- It does not invent an identifier. ``target_taxid`` is attached only where NCBI resolves the name;
  cell lines are canonicalized against the local Cellosaurus name list but get no accession,
  because the local copy carries names only.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from sample import config, nlp, novelty, scale
from sample.polish import _write_atomically
from sample.preprocessing import canonicalize_unit
from sample.validation_rules import VALID_ACTIVITY_TYPES, VALID_UNITS

logger = logging.getLogger(__name__)

#: Bumped when the correction logic changes in a way that should make a previous pass's decisions
#: eligible for revisiting. A document already at this version is skipped.
#:
#: 2: three guards added after a first full corpus pass wrote data that had to be reverted.
#: An OCR-fused conjunction ("S. epidermidis andE. faecium") resolved to its first organism and
#: filed a two-organism measurement against one of them; an organism proposal was accepted for a
#: microbial assay without checking its domain, which filed a crab AMP's MIC against the crab; and
#: a verdict whose peptide name matched several peptides in one document was applied to each of
#: them. Bumping alone undoes none of the three, because every written value is well-formed and a
#: later pass has nothing to object to -- the tree was restored from the pre-pass copy and redone.
#: That is the general rule for this module: a logic fix needs a restore, not a re-run.
APPLY_VERSION: int = 2

#: Where the provenance of this stage lands, under the document's ``metadata``.
APPLY_KEY: str = "verdict_apply"

#: Per-field provenance on the activity or peptide itself, so a consumer can tell a corrected
#: value from an extracted one without reading document metadata.
PROVENANCE_KEY: str = "corrections"


# --- Vocabularies, indexed for lookup -------------------------------------------------------------

#: Activity type by case-folded spelling. 3,810 activities in the corpus carry a type that differs
#: from the vocabulary only in case ("mic", "Mic", "ic50"), which no model call is needed to settle.
_TYPE_BY_FOLD: Dict[str, str] = {t.strip().lower(): t for t in VALID_ACTIVITY_TYPES if t}

#: Unit by case-folded spelling, for the same reason.
_UNIT_BY_FOLD: Dict[str, str] = {u.strip().lower(): u for u in VALID_UNITS if u}

#: Strings that occupy the unit column but name an assay, not a quantity. A metric in the unit
#: column means the extractor read a table whose header held the assay name and whose cells held
#: the numbers, and mapped the two columns the wrong way round. The metric is recoverable: it
#: belongs in ``activity_type``, and the unit is simply absent.
_METRIC_IN_UNIT: Dict[str, str] = dict(_TYPE_BY_FOLD)

#: Units that are not units: table-layout debris, column labels, and fragments of a caption that
#: the table backend emitted into the unit field. Each was counted in the corpus before being
#: listed, and none denotes a quantity in which any assay reports a result. The unit is cleared
#: rather than guessed, which leaves the number with no unit and so excludes it from any
#: quantitative view -- the honest outcome, since nothing in the record says what was measured.
_NOT_A_UNIT: Set[str] = {
    "a/co", "continued", "calc", "calcd", "i", "ii", "iii", "iv", "v", "m/z", "pmm", "fh",
    "n/a", "na", "none", "null", "nd", "n.d.", "-", "--", "?", "value", "values", "unit",
    "units", "conc", "conc.", "concentration", "ratio", "index", "fold", "x", "mean", "sd",
    "sem", "range", "no", "yes", "ref", "table", "fig", "figure", "data", "result", "results",
}

#: Off-vocabulary units that denote a quantity the vocabulary already has, under a different name.
#: Only exact physical identities are listed; nothing here changes a number.
#:
#: ppm is the one that matters by volume. For a dilute aqueous solution 1 ppm by mass is 1 mg/L,
#: which the vocabulary spells ug/mL -- the same quantity, the same number. Every ppm value in this
#: corpus is an aqueous peptide concentration, so the conversion is exact rather than approximate.
_UNIT_SYNONYM: Dict[str, str] = {
    "ppm": "ug/mL",
    "mg/litre": "mg/L",
    "mg/liter": "mg/L",
    "ug/ml.": "ug/mL",
    "micromolar": "uM",
    "millimolar": "mM",
    "nanomolar": "nM",
    "picomolar": "pM",
    "micrograms/ml": "ug/mL",
    "microgram/ml": "ug/mL",
    "micrograms per ml": "ug/mL",
    "mcg/ml": "ug/mL",
    "ug/cm3": "ug/mL",
    "mg/cm3": "mg/mL",
    "g/100ml": "mg/mL",
    "cfu/ml": "CFU/mL",
    "log cfu/ml": "log10 CFU/mL",
    "log10cfu/ml": "log10 CFU/mL",
    "iu/ml": "U/mL",
    "percent": "%",
    "percentage": "%",
}

#: Assay wordings the review uses that :data:`sample.nlp._METRIC_PHRASE_TYPE_MAP` does not cover.
#:
#: That table was derived from table captions, which is where the extractor meets an assay name,
#: and captions are written in a house style: "Minimum inhibitory concentrations of ...". The
#: review answers in ordinary prose and inverts the phrase -- "inhibition of growth", "the
#: peptide's hemolytic effect" -- so the caption patterns miss it. Listed here rather than added
#: to the nlp table because these are proposal wordings, and widening the caption table would
#: change what the extractor reads out of documents.
_PROPOSAL_PHRASE: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\binhibition of (?:bacterial |microbial |cell )?growth\b", re.IGNORECASE),
     "Growth Inhibition"),
    (re.compile(r"\bgrowth[- ]inhibitory (?:activity|concentration|effect)\b", re.IGNORECASE),
     "Growth Inhibition"),
    (re.compile(r"\binhibitory concentration\b", re.IGNORECASE), "MIC"),
    (re.compile(r"\bbactericidal concentration\b", re.IGNORECASE), "MBC"),
    (re.compile(r"\bfungicidal concentration\b", re.IGNORECASE), "MFC"),
    (re.compile(r"\blethal (?:dose|concentration)\b", re.IGNORECASE), "LD50"),
    (re.compile(r"\bh(?:a)?emolytic (?:effect|activity|concentration)\b", re.IGNORECASE),
     "Hemolysis"),
    (re.compile(r"\bcytotoxic (?:effect|concentration)\b", re.IGNORECASE), "Cytotoxicity"),
    (re.compile(r"\bcell viability\b|\bviability assay\b", re.IGNORECASE), "Cytotoxicity"),
    (re.compile(r"\binhibition (?:halo|zone)\b|\bhalo diameter\b", re.IGNORECASE), "ZOI"),
    (re.compile(r"\bbiofilm\b", re.IGNORECASE), "Biofilm Inhibition"),
    (re.compile(r"\bmembrane (?:permeabili|depolari)\w*\b", re.IGNORECASE),
     "Membrane Permeabilization"),
    (re.compile(r"\bselectivity index\b", re.IGNORECASE), "SI"),
    (re.compile(r"\btherapeutic index\b", re.IGNORECASE), "TI"),
]

#: Placeholders that mean "the extractor had nothing", in any field. Written through, they look
#: like data; cleared, they are absent, which is what they are.
_PLACEHOLDER: Set[str] = {
    "", "-", "--", "n/a", "na", "n.a.", "none", "null", "nil", "nd", "n.d.", "unknown",
    "unspecified", "not specified", "not determined", "not reported", "not available",
    "unsure", "?", "tbd", "todo", "various", "multiple", "several", "other", "others",
}

#: Three-letter residue codes. Written out rather than imported because the authorizer needs a
#: *strict* decode: :func:`sample.sequence.sequence_to_1_letter` drops a residue it cannot convert,
#: which turns "Ac-Aib-Ala-Aib-Gln-Vxx-Lxxol" into "AQ" -- a shorter peptide that was never in the
#: paper. Here an unconvertible token refuses the whole proposal instead.
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "ASX": "B", "GLX": "Z", "XAA": "X", "SEC": "U", "PYL": "O", "XLE": "J",
}

#: Terminal modifications that qualify a sequence without changing its residues. Stripped before a
#: decode so "Ac-Gly-Ile-Gly-NH2" is readable, and remembered so nothing is silently discarded.
_TERMINAL_PREFIX = re.compile(r"^(?:ac|acetyl|nh2|h|boc|fmoc|for|formyl|myr|pal|n-?term)\s*-\s*",
                              re.IGNORECASE)
_TERMINAL_SUFFIX = re.compile(
    r"\s*-\s*(?:nh2|nh₂|oh|cooh|co|amide|amid|conh2|ome|oet|c-?term)$", re.IGNORECASE)

#: Separators a three-letter sequence may use.
_RESIDUE_SPLIT = re.compile(r"[-\s‐-―·∙.]+")

#: Fewest decoded residues an unseparated run may yield. Four, matching
#: :data:`sample.novelty._RUN_MIN_DECODED`, which was measured for the same decision.
_MIN_UNSEPARATED_RESIDUES: int = 4


def _strict_unseparated_decode(body: str) -> str:
    """
    Decodes an unseparated run of three-letter codes, or returns "" -- with no forgiveness.

    :param body: Upper-case letters only, separators already removed.
    :returns: The one-letter form, or ``""`` when the string does not partition exactly.

    :func:`sample.novelty.decode_run_of_three_letter_codes` is deliberately forgiving: it skips a
    character that starts no code, which is what lets a stereochemical ``D`` through, and it
    accepts a run at 80% coverage. That is right for its job, which is producing a *comparison*
    form for the novelty test -- a slightly lossy key still compares correctly. It is wrong for
    this module's job, which is rewriting the stored sequence of record. Measured across the
    corpus, the forgiving decoder rewrites eight sequences by dropping a residue it could not
    read: ``VAL-PRO-GLY-XAA-GLY`` loses its ambiguous position, ``GLP-HIS-TRP-...`` loses its
    pyroglutamate, and ``CYCLO-(-LEU-ARG-LEE-LYS-...)`` loses the residue behind the typo ``LEE``.
    Each result is a well-formed sequence that is one residue shorter than the peptide in the
    paper, which is the worst kind of error: undetectable downstream.

    So this requires an exact partition -- length divisible by three, every triple a code -- and
    refuses otherwise, leaving those eight in three-letter notation rather than shortening them.

    The residual risk is a genuine one-letter sequence that happens to partition into valid
    three-letter codes: ``ALAALA`` is both ``A-L-A-A-L-A`` and ``Ala-Ala``. The four-residue floor
    is what bounds it, and across all 15,016 distinct sequences in this corpus the shortest
    sequence this claims is six residues long and every one of the sixteen it claims is a real
    three-letter run, ``GLYARGLYSLYSARGARGGLNARGARGARGCYSG`` -> ``GRKKRRQRRRCG`` among them.
    """
    if len(body) < _MIN_UNSEPARATED_RESIDUES * 3 or len(body) % 3:
        return ""
    decoded: List[str] = []
    for start in range(0, len(body), 3):
        code = body[start:start + 3]
        if code not in _THREE_TO_ONE:
            return ""
        decoded.append(_THREE_TO_ONE[code])
    return "".join(decoded)

#: Stereochemical qualifiers on a residue token, which do not change its one-letter code.
_STEREO_PREFIX = re.compile(r"^(?:d|l|dl|ld)-", re.IGNORECASE)

#: Stereodescriptors that stand as their own token once a separator has split "D-Ala" apart. Only
#: skipped inside a parse already established as three-letter notation -- in one-letter notation
#: "D" is aspartate and "L" is leucine, and skipping them there would delete real residues.
_STEREODESCRIPTORS: Set[str] = {"D", "L", "DL", "LD", "R", "S"}

#: The one-letter alphabet a stored sequence may use: the twenty standard residues plus the
#: ambiguity codes :data:`sample.novelty.AMBIGUITY_CODES` documents as legitimate IUPAC. Defined
#: here rather than imported because novelty.py exposes the ambiguity set but not the alphabet.
_ONE_LETTER: Set[str] = set("ACDEFGHIKLMNPQRSTVWY") | set(novelty.AMBIGUITY_CODES) | {"J"}

#: A sequence already written in that alphabet, so no decode is needed.
_PLAIN_RESIDUES = re.compile(r"^[%s]{2,}$" % "".join(sorted(_ONE_LETTER)))

#: A conjunction or list in an organism answer. "A or B" is not a target; it is two targets and no
#: measurement attributable to either.
#:
#: The lookahead alternative is not redundant. This corpus's OCR routinely drops the space between
#: words -- the same fault sample.nlp.collect_document_genera documents for "againstMicrococcus
#: luteus" -- so a conjunction arrives fused to the next genus: "S. epidermidis andE. faecium".
#: There is no word boundary between the "d" and the "E", so \band\b does not match, and the
#: string then resolves cleanly to Staphylococcus epidermidis alone, filing a measurement made on
#: two organisms against the first one. That slipped through a full corpus pass before this was
#: added.
_MULTI_ORGANISM = re.compile(
    r"\b(?:and|or|plus|versus|vs\.?)\b"
    r"|(?:and|or|plus)(?=[A-Z])"
    r"|[,;/&]|\+")

#: A second abbreviated binomial anywhere after the first, for a list whose separator OCR lost
#: entirely ("S. aureus E. coli").
_SECOND_BINOMIAL = re.compile(r"[A-Z]\.\s*[a-z]{3,}.*?[A-Z]\.\s*[a-z]{3,}")

#: Assays that measure a concentration against a microbe, by definition. An authorized organism
#: proposal whose domain is an animal or a plant is refused for these.
#:
#: Scoped to these assay names rather than derived from :data:`sample.validation_rules
#: .TARGET_GROUP_DOMAINS` because every rule there lists the "other" catch-all among its groups,
#: and "other" carries no domain restriction -- so the union is unrestricted for every assay and
#: that authority cannot reject anything. The catch-all is right for hemolysis and in vivo dosing,
#: which do have animal targets; it is simply not usable as a guard.
#:
#: This applies ONLY in the authorized layer, to a value being written where there was none. A
#: false refusal therefore costs an unfilled field, which is the state before this module ran. The
#: failure it catches is the review naming the peptide's SOURCE organism as its target: a crab
#: AMP's MIC filed against Scopimera globosa, which NCBI confirms is a real species and which no
#: name-based check would question.
_MICROBIAL_ASSAYS: Set[str] = {
    "MIC", "MIC50", "MIC80", "MIC90", "MIC100", "MBC", "MBC50", "MBC90",
    "MFC", "MBEC", "MBIC", "MLC", "ZOI",
}

#: Domains a microbial assay's target cannot belong to.
_NON_MICROBIAL_DOMAINS: Set[str] = {"animal", "plant"}

#: Strain designations and collection accessions that qualify a binomial. NCBI does not register
#: most of them, so they are stripped for the lookup and re-attached to the stored value -- the
#: strain is real information and discarding it would lose which isolate was tested.
_STRAIN_TAIL = re.compile(
    r"\s+(?:(?:ATCC|NCTC|DSM|DSMZ|CECT|KCTC|CCUG|NBRC|JCM|CIP|NRRL|MTCC|BCRC|IFO|LMG|"
    r"CGMCC|PCC|CBS|IAM|NCIMB|NCYC|ACCC|CMCC|KACC|GIM|MRSA|VRE|ESBL)\b.*"
    r"|(?:strain|isolate|serotype|serovar|biovar|subsp\.?|var\.?|pv\.?)\b.*"
    r"|[A-Z]{1,4}[-\s]?\d{1,5}[A-Za-z]?(?:\(.*\))?$"
    r"|K-?12$|BL21.*$|O157:?H7$|USA300.*$|PAO1$|PA14$|SC5314$|H37Rv$|UB1005$)",
    re.IGNORECASE)

#: Taxonomic ranks above species that the review offers as a "target". They resolve against NCBI,
#: which makes them look like a win, but a clade is not an organism a measurement was made on and
#: the pipeline already has ``target_group`` for that level of generality.
_ABOVE_SPECIES: Set[str] = {
    "bacteria", "archaea", "eukaryota", "viruses", "fungi", "viridiplantae", "metazoa",
    "proteobacteria", "firmicutes", "actinobacteria", "bacteroidetes", "ascomycota",
    "basidiomycota", "gammaproteobacteria", "bacilli", "enterobacteriaceae", "staphylococcus",
    "streptococcus", "escherichia", "pseudomonas", "candida", "aspergillus", "bacillus",
    "mycobacterium", "enterococcus", "klebsiella", "acinetobacter", "salmonella", "listeria",
    "cellular organisms", "root", "environmental samples", "unclassified",
    "gram-positive bacteria", "gram-negative bacteria", "gram positive bacteria",
    "gram negative bacteria", "bacterium", "fungus", "yeast", "yeasts", "mold", "moulds",
    "cancer cells", "cancer cell lines", "tumor cells", "human cells", "mammalian cells",
    "erythrocytes", "red blood cells", "microbes", "microorganisms", "pathogens",
}


class Change(NamedTuple):
    """
    One applied correction.

    :param field: The field written.
    :param before: Its previous value, as a string.
    :param after: Its new value, as a string.
    :param authority: What allowed the write -- the vocabulary, NCBI, the alphabet, or the
        document's own genera. Recorded so a consumer can weigh a corrected field.
    :param layer: ``"mechanical"`` for a correction derived from the record alone, ``"authorized"``
        for one that began as a model proposal and survived its authority.
    :param extra: Further fields set alongside, such as ``target_taxid``.
    """
    field: str
    before: str
    after: str
    authority: str
    layer: str
    extra: Dict[str, Any] = {}


class Refusal(NamedTuple):
    """A proposal that no authority confirmed, kept so the refusals can be counted and read."""
    field: str
    proposed: str
    why: str


def _text(value: Any) -> str:
    """Collapses whitespace and coerces to ``str``, the form every check below expects."""
    return " ".join(str(value or "").split())


def _is_placeholder(value: Any) -> bool:
    """Whether a field value carries no information."""
    folded = _text(value).lower().strip(" .")
    return folded in _PLACEHOLDER or bool(nlp._PLACEHOLDER_NAME.search(_text(value)))


# --- Authorizers ----------------------------------------------------------------------------------
#
# Each takes what the model proposed and returns the value an authority confirms, or "" with a
# reason. None of them trusts the proposal's spelling, and none of them accepts a proposal that
# merely looks plausible.


def authorize_activity_type(proposal: str, context: str = "") -> Tuple[str, str]:
    """
    Confirms a proposed assay name against the 38-term activity vocabulary.

    :param proposal: What the review answered.
    :param context: The activity's own ``context_sentence``, used as a second chance when the
        proposal is a description rather than a metric name.
    :returns: ``(canonical_type, authority)``, or ``("", reason)``.

    The review answers this field in prose more often than in vocabulary: "antimicrobial
    activity", "inhibition of growth", "minimum inhibitory concentration". Those are not types,
    but they *name* a type, and :func:`sample.nlp.detect_activity_type` already maps the phrase
    to the metric -- it is the same mapping the extractor uses on a table caption. So the proposal
    is first matched literally, then folded for case, then read as a phrase. A phrase that maps to
    nothing in the vocabulary is refused; "antimicrobial activity" is such a phrase, because an
    activity class is not a measurement.
    """
    text = _text(proposal)
    if not text or _is_placeholder(text):
        return "", "placeholder"
    if text in VALID_ACTIVITY_TYPES:
        return text, "vocabulary:exact"
    folded = _TYPE_BY_FOLD.get(text.lower())
    if folded:
        return folded, "vocabulary:case"
    detected = nlp.detect_activity_type(text, default="")
    if detected in VALID_ACTIVITY_TYPES:
        return detected, "phrase:proposal"
    if detected:
        folded = _TYPE_BY_FOLD.get(detected.lower())
        if folded:
            return folded, "phrase:proposal"
    for pattern, label in _PROPOSAL_PHRASE:
        if pattern.search(text) and label in VALID_ACTIVITY_TYPES:
            return label, "phrase:proposal_wording"
    if context:
        detected = nlp.detect_activity_type(_text(context), default="")
        if detected in VALID_ACTIVITY_TYPES:
            return detected, "phrase:context"
        if detected and _TYPE_BY_FOLD.get(detected.lower()):
            return _TYPE_BY_FOLD[detected.lower()], "phrase:context"
    return "", "not_in_activity_vocabulary"


def authorize_unit(proposal: str) -> Tuple[str, str]:
    """
    Confirms a proposed unit against the 29-unit vocabulary.

    :param proposal: What the review answered.
    :returns: ``(canonical_unit, authority)``, or ``("", reason)``.

    A proposal naming an assay rather than a quantity is refused here specifically, rather than
    allowed through to be caught later: ``MIC`` is a real token the review does offer for this
    field, it is not a unit, and moving it to ``activity_type`` is a different correction with a
    different precondition (see :func:`mechanical_unit_fixes`).
    """
    text = _text(proposal)
    if not text or _is_placeholder(text):
        return "", "placeholder"
    if text.lower() in _METRIC_IN_UNIT:
        return "", "names_an_assay_not_a_unit"
    if text.lower() in _NOT_A_UNIT:
        return "", "not_a_unit"
    canonical = canonicalize_unit(text)
    if canonical in VALID_UNITS and canonical:
        return canonical, "vocabulary:canonicalized"
    folded = _UNIT_BY_FOLD.get(canonical.lower()) or _UNIT_BY_FOLD.get(text.lower())
    if folded:
        return folded, "vocabulary:case"
    synonym = _UNIT_SYNONYM.get(text.lower()) or _UNIT_SYNONYM.get(canonical.lower())
    if synonym and synonym in VALID_UNITS:
        return synonym, "synonym:exact_identity"
    lookup = scale.canonical_unit(text)
    if lookup:
        spelled = canonicalize_unit(scale.display_unit(lookup))
        if spelled in VALID_UNITS and spelled:
            return spelled, "scale:display_form"
    return "", "not_in_unit_vocabulary"


def domain_permits_target(activity_type: str, organism: str) -> bool:
    """
    Whether an organism's NCBI domain is possible as the target of this assay.

    :param activity_type: The assay name, canonical spelling.
    :param organism: An organism name NCBI already resolved.
    :returns: ``True`` unless the assay measures a concentration against a microbe and the
        organism is an animal or a plant. An unknown domain returns ``True``.

    Only the assays in :data:`_MICROBIAL_ASSAYS` are constrained, and an empty answer from
    :func:`sample.nlp.classify_organism_domain` means unknown rather than "none of these" -- that
    function's own docstring says so -- so it is never read as a contradiction.
    """
    if activity_type not in _MICROBIAL_ASSAYS or not organism:
        return True
    # A virus is not a microbial-assay target either, and it has to be recognised before the
    # domain lookup: nlp.classify_organism_domain falls back to the first token when the full name
    # is absent from organism_names, and "Human immunodeficiency virus Type 1" begins with
    # "human", which resolves to Homo sapiens and reports the animal domain. Handled here rather
    # than in that function because it is shared with validation and a change there would move
    # other decisions. 19 organism proposals in this corpus are named this way.
    if re.search(r"\bvirus\b|\bviruses\b|\bphage\b", organism, re.IGNORECASE):
        return False
    domain = nlp.classify_organism_domain(organism)
    return domain not in _NON_MICROBIAL_DOMAINS


def authorize_organism(proposal: str, document_genera: Optional[frozenset] = None
                       ) -> Tuple[str, Optional[int], str]:
    """
    Confirms a proposed organism against NCBI taxonomy, and returns its tax id.

    :param proposal: What the review answered.
    :param document_genera: Capitalized words the document spells out, from
        :func:`sample.nlp.collect_document_genera`, used to settle an abbreviated binomial.
    :returns: ``(scientific_name, tax_id, authority)``, or ``("", None, reason)``.

    The order is the point. A conjunction is refused *before* any lookup, because
    ``"Escherichia coli and Staphylococcus aureus"`` has a prefix that resolves perfectly well and
    a lookup would therefore return a confident, wrong answer -- the measurement would be filed
    against E. coli alone on the strength of word order. An abbreviated binomial is resolved
    against the document's own genera before NCBI sees it, for the reason
    :func:`sample.nlp.resolve_abbreviated_binomial` documents: NCBI registers "m. luteus" as
    Methylobacter luteus, which is valid, well-formed and never what this literature means. A rank
    above species is refused last, because those *do* resolve and would otherwise look like wins.
    """
    text = _text(proposal)
    if not text or _is_placeholder(text):
        return "", None, "placeholder"
    if text.lower() in _ABOVE_SPECIES:
        return "", None, "rank_above_species"
    if _MULTI_ORGANISM.search(text) or _SECOND_BINOMIAL.search(text):
        return "", None, "names_more_than_one_organism"
    if len(text) < 4:
        return "", None, "too_short_to_be_a_binomial"

    strain = ""
    core = text
    tail = _STRAIN_TAIL.search(text)
    if tail and tail.start() > 3:
        core, strain = text[:tail.start()].strip(), text[tail.start():].strip()

    if document_genera:
        expanded = nlp.resolve_abbreviated_binomial(core, document_genera)
        if expanded:
            _, taxid = nlp.resolve_taxon(expanded)
            stored = f"{expanded} {strain}".strip() if strain else expanded
            return stored, taxid, "document_genera"

    resolved, taxid = nlp.resolve_taxon(core)
    if not resolved or taxid is None:
        return "", None, "unresolved_in_ncbi_taxonomy"
    if resolved.lower() in _ABOVE_SPECIES:
        return "", None, "resolves_to_a_rank_above_species"
    # A one-token name is a genus or higher: a binomial has two, and a virus name has several.
    # This is not pedantry about rank -- it is what stops a table artifact from being dignified
    # with a taxonomy id. "Zeta" (from "zeta potential") is NCBI 743421 and "Helix" (from
    # "alpha-helix") is NCBI 6534, a snail; both sat in target_organism against an MIC, and a
    # first corpus pass attached those ids, which makes the artifact look like a curated target.
    # The hand-written _ABOVE_SPECIES list cannot cover this -- every genus NCBI knows is a
    # candidate -- and the pipeline already carries clade-level generality in ``target_group``.
    if len(resolved.split()) < 2:
        return "", None, "resolves_to_a_single_token_name"
    if resolved.upper() in nlp._ORGANISM_BLOCKLIST:
        return "", None, "blocklisted_organism_name"
    stored = f"{resolved} {strain}".strip() if strain else resolved
    return stored, taxid, "ncbi_taxonomy"


def authorize_sequence(proposal: str) -> Tuple[str, str]:
    """
    Confirms a proposed sequence by decoding it strictly to one-letter code.

    :param proposal: What the review answered.
    :returns: ``(one_letter_sequence, authority)``, or ``("", reason)``.

    Strict is the whole contribution. The review's sequence proposals are overwhelmingly the
    document's own three-letter spelling, and most decode cleanly -- but a real minority name
    residues that have no one-letter code at all: ``Ac-Aib-Ala-Aib-Gln-Aib-Vxx-Aib-Lxxol`` is a
    trichogin analogue built from alpha-aminoisobutyric acid and a leucinol terminus.
    :func:`sample.sequence.sequence_to_1_letter` would drop those tokens and return a short,
    well-formed sequence that is not the peptide in the paper. Here any token outside
    :data:`_THREE_TO_ONE` refuses the proposal, so a non-canonical peptide keeps its own spelling
    instead of acquiring a plausible wrong one.
    """
    text = _text(proposal)
    # Checked before the placeholder test only so the refusal reads accurately: a bracketed
    # non-standard residue would otherwise be reported as a placeholder, because
    # nlp._PLACEHOLDER_NAME treats a bracket as template text. The outcome is the same refusal.
    if re.search(r"[\[\]{}]", text):
        return "", "bracketed_non_standard_residue"
    if not text or _is_placeholder(text):
        return "", "placeholder"

    # Terminal groups may stack ("Ac-GIGK-NH2", "NH2-...-COOH"), so strip until nothing more comes
    # off rather than once at each end.
    body = text
    for _ in range(4):
        stripped = _TERMINAL_SUFFIX.sub("", _TERMINAL_PREFIX.sub("", body)).strip()
        if stripped == body:
            break
        body = stripped
    if not body:
        return "", "nothing_left_after_terminal_groups"

    tokens = [t.strip("[]()") for t in _RESIDUE_SPLIT.split(body) if t.strip("[]()")]
    if len(tokens) > 1:
        # Decide the notation before parsing a single token. The separator that makes three-letter
        # code readable also splits "D-Ala" into "D" and "Ala", and "D" is a perfectly valid
        # one-letter code for aspartate -- so a parser that accepts both notations at once reads
        # "D-Ala-L-Lys-Gly" as Asp-Ala-Leu-Lys-Gly and invents two residues that were never in the
        # peptide. The notation is therefore settled by whether any token is a three-letter code,
        # and a mixed string is refused rather than guessed.
        three_letter = any(t.upper() in _THREE_TO_ONE for t in tokens)
        decoded: List[str] = []
        for token in tokens:
            residue = token.upper()
            if three_letter:
                if residue in _THREE_TO_ONE:
                    decoded.append(_THREE_TO_ONE[residue])
                    continue
                stripped = _STEREO_PREFIX.sub("", token).upper()
                if stripped != residue and stripped in _THREE_TO_ONE:
                    decoded.append(_THREE_TO_ONE[stripped])
                    continue
                if residue in _STEREODESCRIPTORS:
                    continue
                return "", f"unconvertible_residue:{residue[:12]}"
            if len(residue) == 1 and residue in _ONE_LETTER:
                decoded.append(residue)
                continue
            return "", f"unconvertible_residue:{residue[:12]}"
        if len(decoded) < 2:
            return "", "too_short_after_decoding"
        return "".join(decoded), ("three_letter_decode" if three_letter
                                  else "hyphenated_one_letter")

    # A lone token must be letters and nothing else. Stripping whatever is not a letter looks
    # harmless and is not: "KWKLFKKIŸPŸKFLHSAKKF" becomes KWKLFKKIPKFLHSAKKF, two
    # residues shorter than the peptide the paper reported, and "G[Orn]K" becomes GORNK, which
    # invents an R and an N -- the exact failure sample.sequence.sequence_to_1_letter documents
    # guarding against. A stray character marks a position whose residue is unknown, so the
    # sequence is refused rather than silently closed up. 122 sequences reach here with one.
    single = body.upper()
    if re.search(r"[^A-Z]", single):
        return "", "non_residue_character_in_sequence"
    # The run decode is tried before the plain-residue test, not after. "GLYILEGLY" matches the
    # one-letter alphabet perfectly -- every letter in it is a valid residue code -- so testing
    # for plain residues first accepts it as a nine-residue peptide and never decodes it. That is
    # the false identity this ordering exists to catch, and it is the shape the undecoded
    # sequences in this corpus have: 185 of the 783 whose stored spelling is not plain one-letter,
    # plus 16 whose stored spelling looks plain but is not.
    run = _strict_unseparated_decode(single)
    if run:
        return run, "unseparated_three_letter_decode"
    if _PLAIN_RESIDUES.match(single):
        return single, "already_one_letter"
    return "", "not_a_readable_sequence"


def authorize_value(proposal: str, activity: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Confirms a proposed measurement by parsing it and checking its magnitude.

    :param proposal: What the review answered.
    :param activity: The activity the value belongs to, read for its unit.
    :returns: ``(fields_to_write, authority)``, or ``({}, reason)``.

    Refused when the proposal carries a unit that disagrees with the activity's own. The review
    answers this field with strings like ``"3.8 +/- 0.4 ug/mL"``, and a proposal whose unit differs
    from the stored one is not a corrected number -- it is a different measurement, or the same one
    read off a different row, and nothing in the proposal says which. Accepting the digits and
    keeping the old unit would silently combine two sources.
    """
    text = _text(proposal)
    if not text or _is_placeholder(text):
        return {}, "placeholder"

    stored_unit = _text(activity.get("activity_unit"))
    embedded = scale.quantities_in(text)
    if embedded:
        proposed_unit = canonicalize_unit(scale.display_unit(embedded[0].unit))
        if stored_unit and proposed_unit and proposed_unit.lower() != stored_unit.lower():
            return {}, f"proposal_carries_a_different_unit:{proposed_unit}"

    operator, lower, upper = nlp.parse_activity_value_fast(text)
    if lower is None:
        return {}, "no_number_in_proposal"

    canonical_stored = scale.canonical_unit(stored_unit) if stored_unit else ""
    basis = scale.basis_of(canonical_stored)
    if basis in scale.PLAUSIBLE:
        floor, ceiling = scale.PLAUSIBLE[basis]
        # Argument order is (unit, value). Reversed, this returns None for every input and the
        # guard passes everything silently, which is how it was first written.
        canonical_value = scale.to_canonical_value(canonical_stored, lower)
        if canonical_value is not None and not (floor <= canonical_value <= ceiling):
            return {}, "proposed_magnitude_outside_the_plausible_band"

    number = re.search(r"[<>]=?\s*\d+(?:\.\d+)?(?:\s*[-–—~]\s*\d+(?:\.\d+)?)?"
                       r"|\d+(?:\.\d+)?(?:\s*[-–—~]\s*\d+(?:\.\d+)?)?", text)
    fields: Dict[str, Any] = {"value_raw": _text(number.group(0)) if number else str(lower),
                              "value_lower": lower}
    if upper is not None:
        fields["value_upper"] = upper
    if operator:
        fields["value_operator"] = operator
    return fields, "parsed_and_within_band"


# --- The mechanical layer -------------------------------------------------------------------------


def mechanical_type_fixes(activity: Dict[str, Any]) -> List[Change]:
    """
    Corrects an activity type that the vocabulary already holds under another spelling.

    :param activity: One activity, modified in place.
    :returns: The changes applied.

    Case folding alone accounts for 3,810 activities. The context-sentence fallback handles a
    placeholder type ("N/A", "unknown", "concentration" -- 3,136 between them) where the activity's
    own recorded sentence names the assay: that sentence was stored by the extractor precisely so a
    later stage would not have to reopen the PDF.
    """
    changes: List[Change] = []
    raw = _text(activity.get("activity_type"))
    if raw and raw not in VALID_ACTIVITY_TYPES:
        folded = _TYPE_BY_FOLD.get(raw.lower())
        if folded:
            activity["activity_type"] = folded
            return [Change("activity_type", raw, folded, "vocabulary:case", "mechanical")]

    if _is_placeholder(raw) or (raw and raw not in VALID_ACTIVITY_TYPES):
        context = _text(activity.get("context_sentence"))
        if context:
            detected = nlp.detect_activity_type(context, default="")
            canonical = detected if detected in VALID_ACTIVITY_TYPES else \
                _TYPE_BY_FOLD.get(detected.lower(), "")
            if canonical and canonical != raw:
                activity["activity_type"] = canonical
                if raw and not _is_placeholder(raw):
                    activity.setdefault("activity_type_original", raw)
                changes.append(Change("activity_type", raw, canonical,
                                      "phrase:context_sentence", "mechanical"))
    return changes


def mechanical_unit_fixes(activity: Dict[str, Any]) -> List[Change]:
    """
    Corrects the unit column: case, exact synonyms, an assay name filed as a unit, and debris.

    :param activity: One activity, modified in place.
    :returns: The changes applied.

    The assay-name case is the interesting one. ``activity_unit == "MIC"`` means the table's assay
    column and value column were read the wrong way round, so the metric is recoverable and the
    unit is genuinely absent. It is moved only when ``activity_type`` is a placeholder or already
    agrees: a record that says ``activity_type: IC50, activity_unit: MIC`` names two different
    assays and nothing in it settles which was measured, so the unit is cleared and the type left
    alone.
    """
    changes: List[Change] = []
    raw = _text(activity.get("activity_unit"))
    if not raw:
        return changes
    if raw in VALID_UNITS:
        return changes

    metric = _METRIC_IN_UNIT.get(raw.lower())
    if metric:
        current = _text(activity.get("activity_type"))
        if _is_placeholder(current) or current.lower() == metric.lower():
            activity["activity_type"] = metric
            activity["activity_unit"] = ""
            activity.setdefault("activity_unit_original", raw)
            changes.append(Change("activity_type", current, metric,
                                  "unit_column_held_the_assay_name", "mechanical"))
            changes.append(Change("activity_unit", raw, "",
                                  "unit_column_held_the_assay_name", "mechanical"))
        else:
            activity["activity_unit"] = ""
            activity.setdefault("activity_unit_original", raw)
            changes.append(Change("activity_unit", raw, "",
                                  "assay_name_in_unit_column_conflicts_with_type", "mechanical"))
        return changes

    folded = _UNIT_BY_FOLD.get(raw.lower())
    if folded:
        activity["activity_unit"] = folded
        changes.append(Change("activity_unit", raw, folded, "vocabulary:case", "mechanical"))
        return changes

    canonical = canonicalize_unit(raw)
    if canonical in VALID_UNITS and canonical and canonical != raw:
        activity["activity_unit"] = canonical
        activity.setdefault("activity_unit_original", raw)
        changes.append(Change("activity_unit", raw, canonical,
                              "canonicalize_unit", "mechanical"))
        return changes

    synonym = _UNIT_SYNONYM.get(raw.lower()) or _UNIT_SYNONYM.get(canonical.lower())
    if synonym and synonym in VALID_UNITS:
        activity["activity_unit"] = synonym
        activity.setdefault("activity_unit_original", raw)
        changes.append(Change("activity_unit", raw, synonym,
                              "synonym:exact_identity", "mechanical"))
        return changes

    if raw.lower() in _NOT_A_UNIT:
        activity["activity_unit"] = ""
        activity.setdefault("activity_unit_original", raw)
        changes.append(Change("activity_unit", raw, "", "not_a_unit", "mechanical"))
    return changes


def mechanical_organism_fixes(activity: Dict[str, Any],
                              document_genera: Optional[frozenset] = None) -> List[Change]:
    """
    Expands an abbreviated genus, canonicalizes against NCBI, and attaches the tax id.

    :param activity: One activity, modified in place.
    :param document_genera: The document's spelled-out genera.
    :returns: The changes applied.

    This closes the identifier gap the audit found: no activity in the corpus carried a taxonomy
    id under any name, so ``escherichia coli`` and ``e. coli`` were two unjoinable strings for one
    organism. ``target_taxid`` is written wherever NCBI resolves the name, including where the
    stored spelling needs no change -- the id is the join key, and 24,399 filled organism fields
    had none.
    """
    changes: List[Change] = []
    raw = _text(activity.get("target_organism"))
    if not raw:
        return changes
    if _is_placeholder(raw):
        activity["target_organism"] = ""
        changes.append(Change("target_organism", raw, "", "placeholder", "mechanical"))
        return changes

    stored, taxid, authority = authorize_organism(raw, document_genera)
    if not stored:
        return changes

    extra: Dict[str, Any] = {}
    if taxid is not None and not activity.get("target_taxid"):
        activity["target_taxid"] = taxid
        extra["target_taxid"] = taxid
    if stored != raw:
        activity["target_organism"] = stored
        activity.setdefault("target_organism_original", raw)
        changes.append(Change("target_organism", raw, stored, authority, "mechanical", extra))
    elif extra:
        changes.append(Change("target_taxid", "", str(taxid), authority, "mechanical", extra))
    return changes


def mechanical_cell_line_fixes(activity: Dict[str, Any],
                               cellosaurus: Optional[Dict[str, str]] = None) -> List[Change]:
    """
    Canonicalizes a cell-line name against the local Cellosaurus name list.

    :param activity: One activity, modified in place.
    :param cellosaurus: Index from a punctuation- and case-insensitive key to the Cellosaurus
        spelling, from :func:`cellosaurus_index`.
    :returns: The changes applied.

    ``raw264.7`` and ``raw 264.7`` are the same line filed twice; so are ``MCF7`` and ``MCF-7``.
    The index is keyed on the name with case, spaces and hyphens removed, which is exactly the
    variation this corpus shows, and resolves each to the one spelling Cellosaurus uses. No
    accession is attached: the local copy is a name list, and inventing a CVCL id from a name match
    would assert an identity the file cannot support.
    """
    changes: List[Change] = []
    raw = _text(activity.get("target_cell_line"))
    if not raw:
        return changes
    if _is_placeholder(raw):
        activity["target_cell_line"] = ""
        changes.append(Change("target_cell_line", raw, "", "placeholder", "mechanical"))
        return changes
    if not cellosaurus:
        return changes
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    canonical = cellosaurus.get(key)
    if canonical and canonical != raw:
        activity["target_cell_line"] = canonical
        activity.setdefault("target_cell_line_original", raw)
        changes.append(Change("target_cell_line", raw, canonical, "cellosaurus", "mechanical"))
    return changes


def mechanical_sequence_fixes(record: Dict[str, Any]) -> List[Change]:
    """
    Puts the one-letter spelling in ``peptide_sequence`` where the stored value is three-letter.

    :param record: One peptide, modified in place.
    :returns: The changes applied.

    The audit found three spellings living side by side -- ``peptide_sequence``,
    ``peptide_sequence_1letter`` and ``peptide_sequence_3letter`` -- with everything downstream
    reading the first. 756 peptides were held in three-letter notation there, invisible to every
    identity test including the novelty judgment. The original spelling is preserved in
    ``peptide_sequence_3letter`` rather than discarded.
    """
    changes: List[Change] = []
    raw = _text(record.get("peptide_sequence"))
    if not raw:
        return changes
    upper = raw.upper()

    # authorize_sequence runs before the plain-residue shortcut, not after it. 16 of this corpus's
    # stored sequences are three-letter runs spelled entirely in letters that are also valid
    # one-letter codes -- GLYARGLYSLYSARGARGGLNARGARGARGCYSG is the TAT peptide, GRKKRRQRRRCG --
    # so a shortcut that accepts a plain-looking string first would leave every one of them
    # undecoded, which is the state the audit found them in.
    decoded, authority = authorize_sequence(raw)
    if decoded and decoded != upper:
        record["peptide_sequence"] = decoded
        if authority != "already_one_letter":
            record.setdefault("peptide_sequence_3letter", raw)
        record["peptide_sequence_1letter"] = decoded
        changes.append(Change("peptide_sequence", raw, decoded, authority, "mechanical"))
        return changes

    if _PLAIN_RESIDUES.match(upper) and upper != raw:
        record["peptide_sequence"] = upper
        changes.append(Change("peptide_sequence", raw, upper, "case_fold", "mechanical"))
    return changes


def cellosaurus_index(path: Optional[str] = None) -> Dict[str, str]:
    """
    Indexes the local Cellosaurus name list by a punctuation-free, case-folded key.

    :param path: Override for ``data/cellosaurus_names.txt``.
    :returns: key -> Cellosaurus spelling. Empty when the file is absent, which degrades cell-line
        canonicalization to a no-op rather than failing the pass.

    Where two Cellosaurus names collapse to the same key the shorter is kept, which prefers the
    primary name over a longer synonym that differs only in punctuation.
    """
    location = path or os.path.join(os.path.dirname(__file__), "..", "data",
                                    "cellosaurus_names.txt")
    index: Dict[str, str] = {}
    if not os.path.exists(location):
        logger.warning("cellosaurus_names.txt absent; cell-line canonicalization is a no-op")
        return index
    with open(location, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            if not key:
                continue
            existing = index.get(key)
            if existing is None or len(name) < len(existing):
                index[key] = name
    return index


# --- Applying the recorded verdicts ---------------------------------------------------------------

#: Which authorizer owns which field, and whether a verdict for it may rewrite the field at all.
#: ``peptide_sequence`` is listed because its three-letter verdicts are applicable; the
#: name-conflict ones are filtered by reason, not by field.
_AUTHORIZED_FIELDS: Set[str] = {"activity_type", "activity_unit", "target_organism",
                                "peptide_sequence", "value_raw"}

#: Verdict reasons whose proposal must never be written, whatever it says.
#:
#: ``name_identifies_another_peptide`` is a disagreement between two fields, not an error in one of
#: them. The review's own wording is "the name 'Defensin 1' identifies X, which this sequence does
#: not match" -- a true statement that does not say the sequence is wrong. Rewriting the sequence
#: to the catalogued one would overwrite what the paper reported with what the catalogue holds, and
#: the 1,059 records carrying this reason are exactly the ones where the two sources disagree and
#: the disagreement is the finding.
_NEVER_APPLY: Set[str] = {"name_identifies_another_peptide"}


def apply_verdicts_to_record(record: Dict[str, Any], verdicts: Sequence[Dict[str, Any]],
                             document_genera: Optional[frozenset] = None
                             ) -> Tuple[List[Change], List[Refusal]]:
    """
    Applies the verdicts recorded against one peptide.

    :param record: One peptide, modified in place.
    :param verdicts: The document's verdict entries, filtered to this peptide by name.
    :param document_genera: The document's spelled-out genera.
    :returns: ``(changes, refusals)``.

    A verdict names its peptide but not which of that peptide's activities it concerned, because
    step 12 recorded ``peptide`` and not an activity index. Matching is therefore by field value:
    a verdict is applied to an activity whose current value for that field still equals what the
    review saw. That is strictly narrower than applying it to every activity of the peptide -- a
    peptide with eight MIC rows gets the correction on the row that still reads the reviewed value,
    and none at all if step 11 has since repaired it, which is the right outcome in both cases.
    """
    changes: List[Change] = []
    refusals: List[Refusal] = []
    activities = [a for a in (record.get("biological_activities") or []) if isinstance(a, dict)]

    for entry in verdicts:
        if not isinstance(entry, dict):
            continue
        field = entry.get("field")
        if field not in _AUTHORIZED_FIELDS:
            continue
        if entry.get("verdict") != "wrong":
            continue
        reason = entry.get("reason") or ""
        if reason in _NEVER_APPLY:
            refusals.append(Refusal(field, _text(entry.get("corrected")),
                                    "name_sequence_conflict_is_not_a_sequence_error"))
            continue
        proposal = _text(entry.get("corrected"))
        if not proposal:
            continue

        # Refusals are recorded once per verdict, not once per activity. A verdict names its
        # peptide, so the loop below tries it against each of that peptide's activities; a
        # peptide with twenty activities would otherwise contribute twenty refusal events for one
        # rejected proposal, and the summary reported 5,790 where 461 verdicts were refused.
        refused_here: Set[str] = set()

        def refuse(why: str) -> None:
            if why not in refused_here:
                refused_here.add(why)
                refusals.append(Refusal(field, proposal, why))

        if field == "peptide_sequence":
            decoded, authority = authorize_sequence(proposal)
            current = _text(record.get("peptide_sequence"))
            if not decoded:
                refusals.append(Refusal(field, proposal, authority))
            elif decoded != current.upper():
                record["peptide_sequence"] = decoded
                if not _PLAIN_RESIDUES.match(current.upper()):
                    record.setdefault("peptide_sequence_3letter", current)
                record["peptide_sequence_1letter"] = decoded
                changes.append(Change(field, current, decoded, authority, "authorized"))
            continue

        for activity in activities:
            if field == "activity_type":
                value, authority = authorize_activity_type(
                    proposal, _text(activity.get("context_sentence")))
                if not value:
                    refuse(authority)
                    break
                current = _text(activity.get("activity_type"))
                if current == value or (current and current in VALID_ACTIVITY_TYPES
                                        and not _is_placeholder(current)):
                    continue
                activity["activity_type"] = value
                if current and not _is_placeholder(current):
                    activity.setdefault("activity_type_original", current)
                changes.append(Change(field, current, value, authority, "authorized"))
            elif field == "activity_unit":
                value, authority = authorize_unit(proposal)
                if not value:
                    refuse(authority)
                    break
                current = _text(activity.get("activity_unit"))
                if current == value:
                    continue
                if current in VALID_UNITS and current:
                    # Step 11 already settled this one deterministically; a model proposal does not
                    # override a unit the vocabulary accepts.
                    refuse("current_unit_already_in_vocabulary")
                    break
                activity["activity_unit"] = value
                activity.setdefault("activity_unit_original", current)
                changes.append(Change(field, current, value, authority, "authorized"))
            elif field == "target_organism":
                stored, taxid, authority = authorize_organism(proposal, document_genera)
                if not stored:
                    refuse(authority)
                    break
                if not domain_permits_target(_text(activity.get("activity_type")), stored):
                    refuse("domain_impossible_for_a_microbial_assay")
                    continue
                current = _text(activity.get("target_organism"))
                if current and not _is_placeholder(current):
                    resolved_current, _ = nlp.resolve_taxon(current)
                    if resolved_current:
                        continue
                extra: Dict[str, Any] = {}
                if taxid is not None:
                    activity["target_taxid"] = taxid
                    extra["target_taxid"] = taxid
                activity["target_organism"] = stored
                if current:
                    activity.setdefault("target_organism_original", current)
                changes.append(Change(field, current, stored, authority, "authorized", extra))
            elif field == "value_raw":
                fields, authority = authorize_value(proposal, activity)
                if not fields:
                    refuse(authority)
                    break
                current = _text(activity.get("value_raw"))
                if current == _text(fields.get("value_raw")):
                    continue
                activity.setdefault("value_raw_original", current)
                activity.update(fields)
                changes.append(Change(field, current, _text(fields.get("value_raw")),
                                      authority, "authorized"))
    return changes, refusals


def correct_document(payload: Dict[str, Any], verdicts: Sequence[Dict[str, Any]],
                     text: str = "", cellosaurus: Optional[Dict[str, str]] = None
                     ) -> Tuple[List[Change], List[Refusal]]:
    """
    Applies both layers to one document's payload.

    :param payload: The parsed extraction document, modified in place.
    :param verdicts: Step 12's verdict entries for this document; may be empty.
    :param text: The document's own text, for the spelled-out genera that settle an abbreviation.
    :param cellosaurus: The cell-line index.
    :returns: ``(changes, refusals)``.

    The mechanical layer runs *first*, deliberately. It settles the case variants, the assay names
    in the unit column and the abbreviated genera, so an authorized correction is weighed against
    an already-repaired field rather than a raw one -- which is what lets the unit branch refuse a
    proposal on the grounds that the current value is in the vocabulary.
    """
    changes: List[Change] = []
    refusals: List[Refusal] = []
    genera = nlp.collect_document_genera(text) if text else frozenset()

    by_peptide: Dict[str, List[Dict[str, Any]]] = {}
    for entry in verdicts or ():
        if isinstance(entry, dict):
            by_peptide.setdefault(_text(entry.get("peptide")), []).append(entry)

    # A verdict identifies its peptide by name and nothing else -- step 12 records ``peptide`` but
    # neither a peptide index nor an activity index. So a name shared by two peptides in the same
    # document makes every verdict against it unattributable, and applying it to each in turn
    # spreads one proposal across peptides the review never looked at. 10.6% of verdicts carry an
    # EMPTY peptide name, 428 documents hold more than one unnamed peptide, and a further 511 hold
    # two peptides with the same name; between them that is where the fan-out happened.
    occurrences: Dict[str, int] = {}
    for record in payload.get("peptides") or []:
        if isinstance(record, dict):
            key = _text(record.get("peptide_name"))
            occurrences[key] = occurrences.get(key, 0) + 1
    ambiguous_reported: Set[str] = set()

    for record in payload.get("peptides") or []:
        if not isinstance(record, dict):
            continue
        changes.extend(mechanical_sequence_fixes(record))
        for activity in record.get("biological_activities") or []:
            if not isinstance(activity, dict):
                continue
            changes.extend(mechanical_type_fixes(activity))
            changes.extend(mechanical_unit_fixes(activity))
            changes.extend(mechanical_organism_fixes(activity, genera))
            changes.extend(mechanical_cell_line_fixes(activity, cellosaurus))

        name = _text(record.get("peptide_name"))
        mine = by_peptide.get(name, [])
        if mine and occurrences.get(name, 0) > 1:
            # Reported once per name, not once per peptide carrying it, or a document with three
            # unnamed peptides and five empty-name verdicts would contribute fifteen refusals.
            if name not in ambiguous_reported:
                ambiguous_reported.add(name)
                for entry in mine:
                    refusals.append(Refusal(str(entry.get("field") or ""),
                                            _text(entry.get("corrected")),
                                            "peptide_name_is_not_unique_in_this_document"))
            mine = []
        if mine:
            applied, refused = apply_verdicts_to_record(record, mine, genera)
            changes.extend(applied)
            refusals.extend(refused)

    return changes, refusals


def _document_text(path: str) -> str:
    """
    Reads the document's extracted text, for the genera that settle an abbreviated binomial.

    Returns "" when absent: the abbreviation then stays as NCBI reads it, which is the behaviour
    before this module existed.
    """
    for root in ("literature_data_text", "ocr_text"):
        candidate = os.path.join(str(config.DATA_DIR), root,
                                 os.path.basename(os.path.dirname(path)),
                                 os.path.splitext(os.path.basename(path))[0] + ".txt")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            except OSError:
                return ""
    return ""


def run(base_dir: Optional[str] = None, verdict_file: Optional[str] = None,
        dry_run: bool = True, limit: int = 0,
        progress_every: int = 2000) -> Dict[str, Any]:
    """
    Applies the corrections across the extraction corpus.

    :param base_dir: Extraction output; ``data/literature_data_json`` by default.
    :param verdict_file: JSONL snapshot of step 12's verdicts, one object per document with
        ``path`` and ``verdicts``. When omitted, each document's verdicts are read from its own
        ``metadata.field_review``, which is where step 12 writes them -- so the pipeline needs no
        snapshot and a snapshot is only for acting on verdicts a later step-12 pass has since
        overwritten (that stage has no resume check and rewrites the key unconditionally).
    :param dry_run: Count everything and write nothing.
    :param limit: Stop after this many documents; 0 means all.
    :param progress_every: Log a progress line every N documents.
    :returns: A summary of documents, changes by field and authority, and refusals by reason.

    Documents are located through :func:`sample.config.iter_extraction_files` rather than a raw
    walk, for the reason that function documents: the output tree accumulates manual backups
    beside the live sources, and a walk over it once shipped a database built from superseded
    extractions.
    """
    root = base_dir or str(config.DATA_DIR / "literature_data_json")
    paths = config.iter_extraction_files(root)
    if limit:
        paths = paths[:limit]

    verdicts_by_path: Dict[str, List[Dict[str, Any]]] = {}
    if verdict_file and os.path.exists(verdict_file):
        with open(verdict_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                stored = entry.get("path")
                if stored:
                    verdicts_by_path[os.path.abspath(stored)] = entry.get("verdicts") or []

    cellosaurus = cellosaurus_index()
    summary: Dict[str, Any] = {
        "documents": 0, "documents_changed": 0, "unreadable": 0,
        "skipped_already_applied": 0, "skipped_changed_underneath": 0,
        "changes": 0, "refusals": 0,
        "by_field": {}, "by_authority": {}, "by_layer": {}, "by_refusal": {},
        "verdict_documents_seen": 0, "cellosaurus_names": len(cellosaurus),
        "dry_run": dry_run,
    }

    for index, path in enumerate(paths, start=1):
        summary["documents"] += 1
        if progress_every and index % progress_every == 0:
            logger.info("verdict_apply: %d/%d documents, %d changes",
                        index, len(paths), summary["changes"])
        try:
            mtime = os.path.getmtime(path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            summary["unreadable"] += 1
            continue
        if not isinstance(payload, dict):
            summary["unreadable"] += 1
            continue

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            previous = metadata.get(APPLY_KEY)
            if isinstance(previous, dict) and previous.get("version") == APPLY_VERSION:
                summary["skipped_already_applied"] += 1
                continue

        if verdicts_by_path:
            mine = verdicts_by_path.get(os.path.abspath(path), [])
        else:
            recorded = metadata.get("field_review") if isinstance(metadata, dict) else None
            mine = recorded.get("verdicts") or [] if isinstance(recorded, dict) else []
            if isinstance(mine, dict):
                mine = [mine]
        if mine:
            summary["verdict_documents_seen"] += 1

        needs_text = any(
            re.match(r"^[A-Z]\.\s*[a-z]", _text(a.get("target_organism")))
            for record in (payload.get("peptides") or []) if isinstance(record, dict)
            for a in (record.get("biological_activities") or []) if isinstance(a, dict))
        text = _document_text(path) if needs_text else ""

        changes, refusals = correct_document(payload, mine, text, cellosaurus)

        for change in changes:
            summary["by_field"][change.field] = summary["by_field"].get(change.field, 0) + 1
            summary["by_authority"][change.authority] = \
                summary["by_authority"].get(change.authority, 0) + 1
            summary["by_layer"][change.layer] = summary["by_layer"].get(change.layer, 0) + 1
        for refusal in refusals:
            summary["by_refusal"][refusal.why] = summary["by_refusal"].get(refusal.why, 0) + 1
        summary["changes"] += len(changes)
        summary["refusals"] += len(refusals)

        if dry_run or not changes:
            continue

        metadata = payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = payload["metadata"] = {}
        record_entry: Dict[str, Any] = {"version": APPLY_VERSION, "changes": len(changes),
                                        "refusals": len(refusals), "by_field": {}}
        for change in changes:
            record_entry["by_field"][change.field] = \
                record_entry["by_field"].get(change.field, 0) + 1
        metadata[APPLY_KEY] = record_entry

        if not _write_atomically(path, payload, expect_mtime=mtime):
            summary["skipped_changed_underneath"] += 1
            continue
        summary["documents_changed"] += 1

    return summary
