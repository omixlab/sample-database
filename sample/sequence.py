import json
import logging
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sample.sequence")

AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Standard IUPAC ambiguity codes (a real single-letter equivalent exists, unlike the
    # non-proteinogenic residues below).
    'ASX': 'B', 'GLX': 'Z', 'XAA': 'X',
}
AA_1_TO_3 = {v: k.capitalize() for k, v in AA_3_TO_1.items()}

# Non-proteinogenic residues seen in this domain (nlp.py's _aa3_list recognizes them as candidate
# sequence residues too) that have no official single-letter IUPAC code. Represented as a
# bracketed monomer token in 1-letter/HELM output -- e.g. "Orn" -> "[Orn]" -- rather than silently
# dropped, which is what sequence_to_1_letter() did before (any 3-letter token not in AA_3_TO_1
# was skipped with no fallback, silently truncating the converted sequence).
_NONSTANDARD_3L = {"ORN": "Orn", "DAB": "Dab", "DAP": "Dap"}
_NONSTANDARD_1L_TOKEN = re.compile(r'\[([A-Za-z]+)\]')

#: Matches a string already in this module's canonical one-letter form: standard residues plus
#: bracketed non-standard monomers. Used to make :func:`sequence_to_1_letter` idempotent.
_CANONICAL_1L_FORM = re.compile(
    r'^(?:[A-Za-z]|\[(?:' + "|".join(_NONSTANDARD_3L.values()) + r')\])+$', re.IGNORECASE)

def sequence_to_1_letter(sequence: str) -> str:
    """
    Normalizes any supported sequence spelling to the canonical one-letter code.

    Accepts one-letter runs, hyphen- or space-separated three-letter codes ("Gly-Ile-Gly"), and
    mixed forms produced by OCR. Non-standard residues that have no one-letter equivalent are
    dropped rather than guessed.

    :param sequence: Sequence in one-letter, three-letter, or mixed notation.
    :returns: Uppercase one-letter sequence, or ``""`` if nothing convertible was found.
    """
    if not sequence:
        return ""
    seq = str(sequence).strip()

    # Idempotence. This function's own output spells residues with no one-letter code as
    # [Orn]/[Dab]/[Dap], and the bracket strip below would turn "G[Orn]K" into "GORNK" --
    # inventing an R and an N that were never in the peptide, silently. Normalizing an
    # already-normalized sequence has to be a no-op, because callers cannot always know whether
    # the string they hold has been through here already.
    # Gated on a bracket actually being present: a plain letter run like "GlyIleGlyLys" is
    # unseparated three-letter code and must still go through the conversion below.
    if "[" in seq and _CANONICAL_1L_FORM.match(seq):
        return re.sub(r'\[([A-Za-z]+)\]',
                      lambda m: f"[{m.group(1).capitalize()}]", seq.upper())

    # Strip brackets e.g. [D-Ala] -> D-Ala
    seq = re.sub(r'\[([^\]]+)\]', r'\1', seq)

    # Strip D/L stereochemical prefixes (e.g. D-Ala, l-Lys)
    seq = re.sub(r'(?i)\b[DL]-', '', seq)

    # Strip lowercase d/l prefixes (e.g. dKdW -> KW)
    seq = re.sub(r'\b[dl](?=[A-Z])|(?<=[A-Z])[dl](?=[A-Z])', '', seq)

    seq = seq.upper()

    # Terminal modifiers. The separator requirement is load-bearing, not cosmetic.
    #
    # AC, PEG, FITC, COOH, OH and AMIDE are also perfectly good residue runs -- Ala-Cys,
    # Pro-Glu-Gly, Phe-Ile-Thr-Cys, Ala-Met-Ile-Asp-Glu -- so matching them without a separator
    # eats real residues off the end of a real peptide. It did: "ACA" normalized to "A", and 258
    # sequences in the shipped release were silently truncated by two or three residues.
    # Requiring the separator costs nothing, because the notation for a modifier is "Ac-GIGK".
    #
    # NH2 and H2N carry a digit and so cannot be a residue run at all; those stay optional-
    # separator, which keeps the common "GIGKNH2" spelling working.
    seq = re.sub(r'^(?:(?:NH2|H2N)[-_\s]*|(?:AC|FITC|PEG)[-_\s]+)', '', seq)
    seq = re.sub(r'(?:[-_\s]*NH2|[-_\s]+(?:COOH|OH|AMIDE))$', '', seq)

    parts = re.split(r'[-_\s]+', seq)
    if len(parts) > 1 and any(len(p) >= 3 for p in parts):
        one_letter = ""
        for p in parts:
            if p in AA_3_TO_1:
                one_letter += AA_3_TO_1[p]
            elif p in _NONSTANDARD_3L:
                one_letter += f"[{_NONSTANDARD_3L[p]}]"
            elif len(p) == 1 and p in AA_3_TO_1.values():
                one_letter += p
        return one_letter
    else:
        # Check if it's unseparated 3-letter
        seq_clean = re.sub(r'[^A-Z]', '', seq)
        if len(seq_clean) >= 3 and len(seq_clean) % 3 == 0:
            chunks = [seq_clean[i:i+3] for i in range(0, len(seq_clean), 3)]
            if all(c in AA_3_TO_1 or c in _NONSTANDARD_3L for c in chunks):
                return "".join(
                    AA_3_TO_1[c] if c in AA_3_TO_1 else f"[{_NONSTANDARD_3L[c]}]"
                    for c in chunks
                )
        # Fallback to standard cleaning
        return seq_clean


def sequence_to_3_letter(sequence: str) -> str:
    """
    Converts any supported raw sequence (1-letter, hyphenated 3-letter code, or a mix, including
    the OCR bracket artifacts sequence_to_1_letter() already cleans) to a canonical, hyphenated
    3-letter-code string, e.g. "YGGFMR" -> "Tyr-Gly-Gly-Phe-Met-Arg". Returns "" if any residue
    can't be confidently identified, rather than silently emitting a partial/wrong conversion.
    """
    seq1 = sequence_to_1_letter(sequence)
    if not seq1:
        return ""
    tokens = []
    i = 0
    while i < len(seq1):
        m = _NONSTANDARD_1L_TOKEN.match(seq1, i)
        if m:
            tokens.append(m.group(1).capitalize())
            i = m.end()
            continue
        ch = seq1[i]
        if ch not in AA_1_TO_3:
            return ""
        tokens.append(AA_1_TO_3[ch])
        i += 1
    return "-".join(tokens)


def sequence_to_helm(sequence: str) -> str:
    """
    Converts a raw sequence to a simple linear-peptide HELM v2 string, e.g.
    "YGGFMR" -> "PEPTIDE1{Y.G.G.F.M.R}$$$$". Non-standard residues (Orn/Dab/Dap) use HELM's own
    bracketed-monomer-ID convention. Returns "" if any residue can't be confidently identified.
    """
    seq1 = sequence_to_1_letter(sequence)
    if not seq1:
        return ""
    monomers = []
    i = 0
    while i < len(seq1):
        m = _NONSTANDARD_1L_TOKEN.match(seq1, i)
        if m:
            monomers.append(f"[{m.group(1).capitalize()}]")
            i = m.end()
            continue
        ch = seq1[i]
        if ch not in AA_1_TO_3:
            return ""
        monomers.append(ch)
        i += 1
    return f"PEPTIDE1{{{'.'.join(monomers)}}}$$$$"

# EMBOSS pKa scale
PKA_EMBOSS = {
    'N_term': 8.6,
    'C_term': 3.6,
    'C': 8.5,
    'D': 3.9,
    'E': 4.1,
    'H': 6.5,
    'K': 10.8,
    'R': 12.5,
    'Y': 10.9
}

def is_valid_peptide(sequence: str) -> bool:
    """
    Checks if a peptide sequence consists of valid amino acid representations
    including standard IUPAC three-letter codes, chirality prefixes, and non-proteinogenic residues.
    """
    if not sequence:
        return False
    sequence = str(sequence).strip()
    if not sequence:
        return False
        
    # Enforce maximum peptide size of 50 amino acids
    seq_1 = sequence_to_1_letter(sequence)
    if len(seq_1) > 50:
        return False
        
    # Allow letters, hyphens, and spaces to support 3-letter codes and chirality (e.g. D-Phe, Dab)
    pattern = re.compile(r'^[A-Za-z\-\s,]+$')
    return bool(pattern.match(sequence))

def get_peptide_charge(sequence: str, ph: float = 7.4) -> float:
    """
    Computes net charge of peptide sequence at a specific pH using the Henderson-Hasselbalch equation.
    Default pH is 7.4 (physiological).
    """
    if not sequence:
        return 0.0

    # Read the amidation marker from the RAW string. sequence_to_1_letter strips the "-NH2"
    # suffix as a modifier, so testing the normalized sequence for it -- as this did -- can never
    # be true, and every amidated peptide was charged as a free acid: about 1.0 too low. That is
    # not a rare case. C-terminal amidation is the norm among natural AMPs (magainin and melittin
    # are both amidated), and this value feeds the `charge` descriptor and ML feature.
    amidated = "NH2" in str(sequence).upper()

    seq = sequence_to_1_letter(sequence)

    charge = 0.0

    # N-terminus (basic)
    charge += 1.0 / (1.0 + 10**(ph - PKA_EMBOSS['N_term']))

    # C-terminus: an amide has no ionizable carboxyl, so it contributes no negative charge.
    if not amidated:
        charge -= 1.0 / (1.0 + 10**(PKA_EMBOSS['C_term'] - ph))
        
    # Basic side chains (positive at low pH)
    charge += seq.count('K') / (1.0 + 10**(ph - PKA_EMBOSS['K']))
    charge += seq.count('R') / (1.0 + 10**(ph - PKA_EMBOSS['R']))
    charge += seq.count('H') / (1.0 + 10**(ph - PKA_EMBOSS['H']))
    
    # Acidic side chains (negative at high pH)
    charge -= seq.count('D') / (1.0 + 10**(PKA_EMBOSS['D'] - ph))
    charge -= seq.count('E') / (1.0 + 10**(PKA_EMBOSS['E'] - ph))
    charge -= seq.count('C') / (1.0 + 10**(PKA_EMBOSS['C'] - ph))
    charge -= seq.count('Y') / (1.0 + 10**(PKA_EMBOSS['Y'] - ph))
    
    return round(charge, 4)


#: Residue letters accepted in an extracted sequence: the 20 standard plus the ambiguity and
#: non-standard codes real dumps use (B=Asx, Z=Glx, X=any, U=Sec, O=Pyl).
_EXTRACTED_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBZXUO")

#: Shortest extracted sequence worth keeping. Below this, a "sequence" is overwhelmingly a table
#: label or an abbreviation: measured over one run's activity log, every 1-4 character value was
#: junk -- 'ALA', 'MYR', 'PGSA', 'CC', 'G', 'N', 'M', '21', 'LF'.
MIN_EXTRACTED_SEQUENCE = 5

#: Longest extracted sequence worth keeping. Not a biological limit -- it is there to reject a
#: whole paragraph captured into the sequence field, which is a failure mode seen in this corpus
#: ('SHARED ITS FIRST 17 OF 20 AMINO ACIDS WITH ...'). `sample.merge` applies the project's
#: 50-residue publication cap separately; this one only rejects things that are not sequences.
MAX_EXTRACTED_SEQUENCE = 200

#: Above this fraction of A/C/G/T letters, a long run is treated as a nucleotide read rather
#: than a peptide. Applied only from _DNA_RATIO_MIN_LENGTH residues up, because short peptides
#: genuinely made of these four letters exist and are not evidence of anything.
_DNA_RATIO_THRESHOLD = 0.85
_DNA_RATIO_MIN_LENGTH = 12


#: Terminal and protecting groups that legitimately appear as tokens in a three-letter sequence.
#: 'NH2' and 'OBU' carry digits or are not residues, so without this list the digit rule below
#: would reject octreotide's published form, 'D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2'.
_SEQUENCE_END_GROUPS = frozenset({
    "NH2", "NH", "OH", "OL", "AC", "COOH", "OME", "OET", "FMOC", "BOC", "Z", "TFA",
    "D", "L", "DL",                      # chirality prefixes, which split into their own tokens
})

#: Characters that can appear in a sequence as actually written: residues, three-letter codes,
#: hyphens, spaces and the comma that separates positions. Anything else -- '=', '(' , '・', '+',
#: '>' -- belongs to a chemical formula or a table cell, not to a sequence.
_SEQUENCE_PUNCTUATION = set("-, \t")


def _looks_like_three_letter_sequence(raw: str) -> bool:
    """
    :param raw: The sequence field as written.
    :returns: Whether it parses as hyphen-separated three-letter residue codes.

    Used to decide whether digits in the raw string are legitimate. They are in exactly one
    situation -- a terminal group such as ``NH2`` on ``D-PHE-CYS-...-TRP-NH2`` -- and are
    otherwise the signature of a chemical formula.
    """
    tokens = [token.strip().upper() for token in raw.split("-") if token.strip()]
    if len(tokens) < 2:
        return False
    recognised = sum(1 for token in tokens
                     if token in AA_3_TO_1 or token in _SEQUENCE_END_GROUPS
                     or token in _NONSTANDARD_3L)
    # A majority, not all: real published forms carry the occasional modified residue that is in
    # none of these tables, and demanding every token match would reject them.
    return recognised >= max(2, (len(tokens) + 1) // 2)


def _normalised_name(value: str) -> str:
    """
    :param value: A peptide name or sequence.
    :returns: Upper-case, stripped of spaces, hyphens and punctuation, for comparing the two.
    """
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def validate_extracted_sequence(sequence: Any, name: str = "") -> Tuple[bool, str]:
    """
    Whether an extracted string is usable as a peptide sequence, and if not, why.

    :param sequence: The value in the record's sequence field.
    :param name: The record's peptide name, used for the name-copied check below. Optional.
    :returns: ``(ok, reason)``. ``reason`` is ``""`` when ok, otherwise a short stable slug
        suitable for counting in metadata.

    Written against what a real run actually produced. Over 1,554 logged activities from 204
    documents, only 14.4% of sequence fields were plain residue strings; the rest were table
    labels (``N = 17``), chemical formulas (``C2H5``), catalogue codes (``RC-160``), OCR debris
    (``7.515.022.530.07,515.022.530.07,5``), DNA (``GCGTYFGCGGCGTGTGACC``), prose (``THE PEPTIDE
    WAS SYNTHESIZED IN OUR LABORATORY``) and compound names (``CHOLESTEROL``, ``PRIMAQUINE``).
    Every one of those reached the output because the identity gate asked only whether the field
    was non-empty.

    The layers, cheapest and most decisive first:

    1. **The name copied into the sequence field.** ``name=Cholesterol`` with
       ``seq=CHOLESTEROL``, ``Ricin``/``RICIN``, ``Technetium-99m``/``TECHNETIUM-99M``,
       ``myristoylglycine``/``MYRISTOYLGLYCINE``. A real peptide's sequence is essentially never
       its own name, so this is near-zero-risk and it catches the single largest class --
       compound names spelled entirely in residue letters, which no alphabet test can reject.
    2. **Structure**: three-letter forms are normalised first, then digits, whitespace and
       non-residue characters are rejected, then the length bounds.
    3. **The dictionary and DNA gates** in :func:`sample.nlp.is_valid_sequence`, which reject
       ordinary English words (``UNKNOWN``, ``INTERFERON``) and nucleotide runs that happen to
       spell valid residues.

    What this deliberately does NOT use: ``sample.rf_filter``. Measured on this corpus it
    rejects 6 of 11 junk strings but also rejects ``ACYCRIPACIA``, the core of human neutrophil
    peptide 1 -- a real published sequence. A classifier that deletes genuine records to catch
    half the junk is the wrong trade for a gate; it belongs in reporting, not in deletion.

    4. **A compound name spelled entirely in residue letters**, which no structural test can
       reject because every letter in it IS a residue: ``ANTIMYCINA``, ``SANDOSTATIN``,
       ``ENKEPHALIN``, ``LINOLEOYL``. Decided by :func:`name_likelihood`, a bigram log-odds test
       against this project's own catalogue -- see that function for the measured separation.
       Applied last, because it is the only statistical test here and so the only one that can
       be wrong about a real sequence; everything above it is a structural certainty.
    """
    from sample.nlp import is_valid_sequence

    raw = str(sequence or "").strip()
    if not raw:
        return False, "empty"

    if name and _normalised_name(raw) and _normalised_name(raw) == _normalised_name(name):
        return False, "name_copied_into_sequence"

    # STRUCTURE OF THE RAW STRING, before any conversion -- and this order is the fix for a hole
    # that shipped: sequence_to_1_letter STRIPS digits and chemical punctuation, so
    # 'C10H22N6O4・H2O' converted to 'CHNOHO' and then passed every check that followed, as did
    # 'C31H47N6O10CLC34H34N7O8CL' -> 'CHNOCLCHNOCL' and '2ABCDEFGH' -> 'ABCDEFGH'. Judging only
    # the converted form means judging a string the extractor never saw.
    #
    # Digits are allowed only in a genuine three-letter sequence, where they belong to a terminal
    # group ('...-TRP-NH2'). Everywhere else they mark a formula, a table label or a residue
    # range.
    three_letter = _looks_like_three_letter_sequence(raw)
    if not three_letter:
        if any(character.isdigit() for character in raw):
            return False, "contains_digits"
        stray = {character for character in raw.upper()
                 if not character.isalpha() and character not in _SEQUENCE_PUNCTUATION}
        if stray:
            return False, "non_residue_characters"

    # Three-letter and chirality-prefixed forms are legitimate ('GLY-ARG-GLY-ASP-SER',
    # 'D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2') and are judged on what they convert to.
    # Conversion that yields nothing usable leaves the original to be judged as-is.
    candidate = raw.upper()
    try:
        converted = (sequence_to_1_letter(raw) or "").upper()
    except Exception:                               # noqa: BLE001 -- a converter failure is not
        converted = ""                              # a verdict; fall back to the raw string
    if converted and set(converted) <= _EXTRACTED_RESIDUES:
        candidate = converted

    if any(character.isdigit() for character in candidate):
        return False, "contains_digits"
    if any(character.isspace() for character in candidate):
        return False, "contains_whitespace"
    if not set(candidate) <= _EXTRACTED_RESIDUES:
        return False, "non_residue_characters"
    if len(candidate) < MIN_EXTRACTED_SEQUENCE:
        return False, "too_short"
    if len(candidate) > MAX_EXTRACTED_SEQUENCE:
        return False, "too_long"
    if not is_valid_sequence(candidate):
        # Covers the dictionary words and nucleotide runs; keeps one definition of those tests
        # rather than restating them here.
        return False, "dictionary_word_or_dna"

    # A nucleotide read that OCR has peppered with stray letters. sample.nlp's DNA test is
    # restricted to strings of pure A/C/G/T, so 'GCGTYFGCGGCGTGTGACC' -- 89% ACGT, the Y and F
    # introduced by OCR -- slipped through it and was published as a peptide. A ratio test
    # catches that, and over a sequence this long a run that is overwhelmingly A/C/G/T is far
    # more likely a nucleotide read than a peptide.
    if len(candidate) >= _DNA_RATIO_MIN_LENGTH:
        acgt = sum(1 for character in candidate if character in "ACGT")
        if acgt / len(candidate) >= _DNA_RATIO_THRESHOLD:
            return False, "looks_like_dna"

    # LAST, because it is the only statistical test here and the only one that can be wrong about
    # a real sequence. Everything above is a structural certainty; this is a likelihood.
    score = name_likelihood(candidate)
    if score is not None and score < NAME_LIKELIHOOD_THRESHOLD:
        return False, "looks_like_a_name"
    return True, ""


# --- Telling a peptide sequence from a compound name spelled in residue letters ---------------
#
# The gap this closes: a drug or chemical name made only of residue letters passes every
# structural test, because every letter in it IS a residue. Real examples from one run, all
# published as peptide sequences: ANTIMYCINA, METHENAMINEMANDELATE, TETRACYCLINES, SANDOSTATIN,
# ASCIDIACYCLAMIDE, ENKEPHALIN, LINOLEOYL, CHOLESTEROL, PRIMAQUINE, DAUNORUBICIN. The dictionary
# gate in sample.nlp catches the ones that are ordinary English words and misses the rest, which
# are Latin/Greek drug coinages absent from any wordlist.
#
# Letter-PAIR statistics separate them cleanly, where composition does not: English and chemical
# coinages are built from TH/ER/IN/ON/TI, peptides from LL/KK/GG/AA/KL/GI. Measured on held-out
# data (40,000 catalogued sequences against 17,300 English words spelled only in residue
# letters): peptide median +1.99, English median -1.27, and at the threshold below 0.1% of real
# sequences are rejected.
#
# Modelled on sample.nlp.dna_log_likelihood_ratio, which does the same thing for nucleotide runs
# with a unigram model. Bigrams are needed here because, unlike DNA, the confusable strings draw
# on the same alphabet with a similar composition.

#: Where the trained table lives. Small (one entry per observed letter pair) and committed, so
#: there is no build step; :func:`train_name_discriminator` regenerates it.
BIGRAM_MODEL_PATH = "data/peptide_name_bigrams.json"

#: Below this mean per-bigram log-odds, a string is treated as a name rather than a sequence.
#:
#: -0.5, not 0, and the asymmetry is deliberate: rejecting a real published sequence is a worse
#: error than keeping a name, because the name is caught by other gates downstream while the
#: sequence is simply lost. At -0.5, measured on held-out data, 0.1% of real sequences are
#: rejected and ~5% of names survive -- and all 15 offenders observed in the run score between
#: -0.62 and -2.05, while the hardest real case (ACYCRIPACIA, the human neutrophil peptide 1
#: core, which sample.rf_filter also rejects) scores -0.05 and survives with margin.
NAME_LIKELIHOOD_THRESHOLD = float(os.getenv("SAMPLE_NAME_LIKELIHOOD_THRESHOLD", "-0.5"))

_BIGRAM_MODEL: Optional[Dict[str, Any]] = None
_BIGRAM_MODEL_TRIED = False


def _bigrams(text: str) -> List[str]:
    """:returns: Overlapping letter pairs of ``text``."""
    return [text[index:index + 2] for index in range(len(text) - 1)]


def _load_bigram_model(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    :param path: Model JSON; :data:`BIGRAM_MODEL_PATH` when omitted.
    :returns: The model, or ``None`` when it is absent or unreadable -- in which case the test is
        skipped rather than failing, and a warning is logged ONCE so a missing model is visible
        instead of silently disabling a quality gate.
    """
    global _BIGRAM_MODEL, _BIGRAM_MODEL_TRIED
    if _BIGRAM_MODEL is not None or _BIGRAM_MODEL_TRIED:
        return _BIGRAM_MODEL
    _BIGRAM_MODEL_TRIED = True
    target = path or BIGRAM_MODEL_PATH
    try:
        with open(target, encoding="utf-8") as handle:
            model = json.load(handle)
        if not isinstance(model.get("log_odds"), dict):
            raise ValueError("no log_odds table")
        _BIGRAM_MODEL = model
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"peptide/name discriminator unavailable ({target}: {type(e).__name__}: "
                       f"{e}); compound names spelled in residue letters will not be rejected. "
                       f"Regenerate with `sample-cli train-sequence-discriminator`.")
    return _BIGRAM_MODEL


def name_likelihood(sequence: str, path: Optional[str] = None) -> Optional[float]:
    """
    Mean per-bigram log-odds of a string under a peptide model versus a name model.

    :param sequence: A one-letter residue string.
    :param path: Model JSON; the default when omitted.
    :returns: Positive favours peptide, negative favours name. ``None`` when the model is
        unavailable or the string is too short to score, so callers can tell "no opinion" from
        "looks like a name".
    """
    model = _load_bigram_model(path)
    if model is None:
        return None
    pairs = _bigrams(str(sequence or "").upper())
    if len(pairs) < model.get("min_bigrams", 4):
        return None
    table = model["log_odds"]
    default = model.get("default_log_odds", 0.0)
    return sum(table.get(pair, default) for pair in pairs) / len(pairs)


def train_name_discriminator(output_path: Optional[str] = None,
                             sequence_csv: str = "data/antimicrobial_databases/merged_source.csv",
                             wordlist: str = "/usr/share/dict/american-english",
                             limit: int = 40000) -> Optional[str]:
    """
    Fits the bigram log-odds table from this project's own catalogue and a system wordlist.

    :param output_path: Where to write the JSON; :data:`BIGRAM_MODEL_PATH` when omitted.
    :param sequence_csv: Consensus CSV supplying the positive class (its ``sequence`` column).
    :param wordlist: One word per line, supplying the negative class. Only words spelled
        entirely in residue letters are used -- the rest are not confusable and would dilute the
        model with pairs that can never appear in a sequence.
    :param limit: Cap on positive examples, for a bounded fit.
    :returns: The written path, or ``None`` when an input is missing.

    Committed output, deliberately: the table is small, and a gate that depends on a build step
    is a gate that is off on a fresh clone.
    """
    import collections
    import csv as _csv
    import math

    if not os.path.exists(sequence_csv) or not os.path.exists(wordlist):
        logger.error(f"cannot train the peptide/name discriminator: need {sequence_csv} and "
                     f"{wordlist}")
        return None

    residues = set("ACDEFGHIKLMNPQRSTVWY")
    names = []
    with open(wordlist, encoding="utf-8", errors="ignore") as handle:
        for word in handle:
            token = word.strip().upper()
            if len(token) >= 7 and set(token) <= residues:
                names.append(token)

    sequences = []
    with open(sequence_csv, encoding="utf-8", errors="ignore") as handle:
        for row in _csv.DictReader(handle):
            value = (row.get("sequence") or "").strip().upper()
            if 7 <= len(value) <= 60 and set(value) <= residues:
                sequences.append(value)
            if len(sequences) >= limit:
                break

    if not names or not sequences:
        logger.error("cannot train the peptide/name discriminator: one class came out empty")
        return None

    def _fit(corpus):
        counts = collections.Counter()
        for item in corpus:
            counts.update(_bigrams(item))
        total = sum(counts.values()) + 400          # add-one over the 20x20 pair space
        return {pair: (count + 1) / total for pair, count in counts.items()}, 1.0 / total

    peptide_probability, peptide_floor = _fit(sequences)
    name_probability, name_floor = _fit(names)

    pairs = set(peptide_probability) | set(name_probability)
    log_odds = {pair: math.log(peptide_probability.get(pair, peptide_floor)
                               / name_probability.get(pair, name_floor))
                for pair in pairs}

    model = {
        "log_odds": {pair: round(value, 4) for pair, value in sorted(log_odds.items())},
        "default_log_odds": round(math.log(peptide_floor / name_floor), 4),
        "min_bigrams": 4,
        "trained_on": {"sequences": len(sequences), "names": len(names),
                       "sequence_csv": sequence_csv, "wordlist": wordlist},
    }
    target = output_path or BIGRAM_MODEL_PATH
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=1, sort_keys=True)
    logger.info(f"peptide/name discriminator: {len(log_odds)} letter pairs from "
                f"{len(sequences)} sequences and {len(names)} names -> {target}")
    return target
