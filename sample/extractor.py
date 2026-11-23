import re
import logging
import json
from sample.nlp import (
    SEQ_EXACT, SEQ_3LETTER, SEQ_ALPHANUM,
    get_seq_names_kp, get_known_seq_1l_kp, get_known_seq_3l_kp,
    get_organism_kp, get_cellosaurus_kp, is_valid_sequence, is_valid_peptide_name,
    is_valid_organism_match, find_local_aliases, parse_activity_value_fast,
    get_name_sequence_map, get_known_peptide_names_kp, find_whitespace_broken_sequences,
    _normalize_peptide_key,
    resolve_scientific_name, organism_contradicts_group,
    collect_document_genera, resolve_abbreviated_binomial,
    extract_context_window, detect_activity_type, infer_activity_group, truncate_references,
    METRIC_PATTERN_EXACT, METRIC_PATTERN_IGNORE, _METRIC_PHRASE_TYPE_MAP
)
from sample.preprocessing import standardize_and_filter_activities
from sample.llm import (
    recover_peptide_llm, verify_activities_llm, triage_candidates_llm, normalize_organisms_llm,
)
from sample.sequence import sequence_to_1_letter, sequence_to_3_letter, sequence_to_helm

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALUE_PATTERN = re.compile(
    r'((?:[<>≤≥=~]|&[gl]t;)?\s*\d+(?:[.,]\d+)?(?:\s*(?:x|×)\s*10\^[-+]?\d+)?(?:\s*(?:-|±|\+/-)\s*\d+(?:[.,]\d+)?)?)\s*'
    # "lM"/"lg/mL"/"lmol/L" are a common PDF-extraction OCR corruption of the micro sign (µ/μ)
    # misread as a lowercase "l" -- confirmed against a real document where every concentration
    # in a dose-response list ("0.5, 1, 2, ... 32 lM") was otherwise invisible to this pattern.
    #
    # The molar-unit alternatives below are deliberately matched case-SENSITIVELY (no (?i:...)
    # wrapper), unlike the rest of the pattern. Under a blanket re.IGNORECASE (the previous
    # behavior), "mM" (millimolar) also matched literal "mm" (millimeter -- a real, valid ZOI/
    # zone-of-inhibition unit in this domain) and "nM" (nanomolar) also matched literal "nm"
    # (nanometer -- a UV-Vis absorbance wavelength, never a reported bioactivity value). Confirmed
    # against real output: a UV-absorption wavelength ("...at 280 nm...") was captured as a fake
    # "280 nM" activity, and a zone-of-inhibition diameter would have been silently multiplied
    # ×1000 and relabeled µM by standardize_and_filter_activities' mM->uM conversion. Case carries
    # real meaning here (papers consistently write capital M for molar, lowercase m/n for the
    # length-derived units) so case-insensitive matching was actively wrong, not just permissive.
    r'(µM|μM|uM|lM|mM|nM'
    r'|(?i:µg/mL|μg/mL|ug/mL|lg/mL|mg/L|mg/kg|mg/mL|nmol/L|umol/L|μmol/L|lmol/L|g/L|ng/mL|IU|U/mL|%|mm|(?:\s*log10\s*)?CFU/mL))\b'
)

# A SEQ_EXACT candidate sitting directly inside parentheses right after a virus/strain/isolate-
# indicating word is a strain designation, not a peptide sequence -- e.g. "Influenza virus
# (AIWSN)" (OCR-flattened from "A/WSN", a real lab influenza strain) matched as if "AIWSN" were a
# 5-residue amino-acid sequence, purely because it happens to spell valid single-letter codes.
_STRAIN_CODE_PRECEDING = re.compile(
    r'\b\w*(?:virus|viruses|strain|isolate|serotype|subtype|clone|genotype)\w*\s*\($',
    re.IGNORECASE
)

# A candidate *name* directly preceded by a strain/isolate/culture-collection designator is a
# microbial strain label, not a peptide -- observed in real output: "Alteromonas strain CH-112
# (NCMB 2156)" produced a peptide record named "CH-112" whose "activities" were read out of the
# table's symbol legend. Unlike _STRAIN_CODE_PRECEDING above (which guards bare sequences inside
# parentheses), this guards the alphanumeric name path and does not require parentheses.
_STRAIN_NAME_PRECEDING = re.compile(
    r'\b(?:strain|isolate|serovar|serotype|biovar|subsp\.?|sp\.|spp\.|var\.|'
    r'ATCC|NCMB|NCTC|NCIMB|DSM|CECT|KCTC|JCM|IFO|NRRL|CCUG|LMG|CIP)\s+$',
    re.IGNORECASE
)

# OCR of dense result tables prefixes footnote markers onto the following cell's text, producing
# spurious name variants of one real compound -- observed in real output: a single agent "FK-565"
# yielded records named "cFK-5650" and "dFK-5650" (footnote letters c/d) alongside "FK-5650" and
# "FK-5651" (the agent name run together with its dose column). A lone lowercase letter
# immediately followed by an uppercase run is never real peptide nomenclature.
_FOOTNOTE_MARKER_PREFIX = re.compile(r'^[a-z][A-Z]{2,}')

# Assay-condition reagents/buffers/media whose reported concentration is an experimental
# parameter, not the peptide's own measured activity -- e.g. "...tested in 10 mM phosphate
# buffer..." or "penicillin G (100 U/ml)" in a cell-culture media recipe. Confirmed against real
# output: these were being captured as biological_activities purely because a number+unit sat in
# the same sentence as a valid organism/cell-line match, with no real experimental metric (MIC,
# IC50, etc.) named anywhere in the sentence -- used below only to gate values in sentences that
# have no recognized metric type at all, so a real "MIC of 10 ug/mL in PBS" (which does name a
# metric) is never affected.
# Chromatography columns, gels and instrument specs, whose dimensions are quoted in mm -- the same
# unit a zone-of-inhibition diameter uses. Observed in real output: "EnantioLI, 4.6 i.d. x150mm"
# became an MIC of 150 mm, and "C-18 column (4.6 by 250 mm, Vydac...)" became a 250 mm activity
# repeated across every peptide in the document.
#
# Unlike _REAGENT_CONTEXT this is applied even when the sentence does name a real metric, because
# OCR routinely fuses a methods paragraph and a results paragraph into one "sentence": the
# lanthiopeptin case put a column spec and the phrase "The MICs of lanthiopeptin" in the same
# block, so gating on "no metric present" let it straight through. Adjacency is what keeps that
# safe -- the cue has to sit right next to the number, not merely somewhere in the block.
_INSTRUMENT_CONTEXT = re.compile(
    r'\b(?:column|columns|i\.?d\.?|HPLC|RP-?HPLC|FPLC|GC|TLC|Sephadex|Sepharose|TSK|Vydac|'
    r'C-?18|C-?8|silica|gel|cartridge|capillary|syringe|needle|tubing|cuvette|path\s*length|'
    r'particle\s*size|pore\s*size|diameter\s+of\s+the\s+(?:column|tube)|'
    r'spectrophotometer|chromatograph\w*|electrode|filter\s+paper|membrane\s+filter)\b',
    re.IGNORECASE
)
_INSTRUMENT_WINDOW = 45

# Methodological definitions of what a metric *means*, as opposed to a result. The numbers in them
# are part of the definition, not measurements -- observed in real output: "CPE reduction activity
# was expressed as 50% inhibitory dose (ID50), which is defined as the concentration required to
# reduce CPE by 50%" produced two Cytotoxicity records of "50 %". Also covers table legends that
# define a symbol key ("+++, 15 to 20 mm; ++, 10 to 15 mm").
# \s* rather than \s+ between words throughout: this corpus's OCR routinely drops inter-word
# spaces, and the documents that need this guard are exactly the badly-OCR'd ones. The real text
# reads "CPEreductionactivitywas expressedas 50%inhibitorydose (ID50) which is definedas..." and
# "One unitof antibacterial activity was defined as ... 10 mm", so a pattern requiring real spaces
# matches neither.
_DEFINITION_CONTEXT = re.compile(
    r'\b(?:is|was|are|were)\s*(?:defined|expressed|calculated|determined|taken|considered)\s*as\b'
    r'|\bdefined\s*as\b|\bwe\s*define\b|\brefers?\s*to\b|\bthat\s*is,'
    r'|\bone\s*unit\s*(?:of|was)\b|\bunit\s*of\s*\w+\s*activity\s*was\b',
    re.IGNORECASE
)

_REAGENT_CONTEXT = re.compile(
    r'\b(PBS|phosphate\s+buffer(?:ed)?|HEPES|Tris(?:-HCl)?|EDTA|EGTA|FCS|FBS|BSA|DMEM|RPMI(?:-?1640)?|'
    r'trypticase(?:\s+soy)?(?:\s+broth)?|penicillin|streptomycin|glucose|NaCl|saline|DMSO|MOPS|Tween(?:-?20)?|'
    r'culture\s+medi(?:um|a)|growth\s+medi(?:um|a)|interferon|IFN'
    # Signalling proteins, enzymes and molecular-biology reagents dosed into an assay. Their
    # concentrations were still being captured as the peptide's own activity -- observed in real
    # output: "Human recombinant cytokines were used at a concentration of 10 ng/ml (IL-3, SCF),
    # 20 ng/ml (TNF-a)..." produced four separate activity records.
    r'|recombinant|cytokines?|interleukin|IL-\d+|TNF|GM-CSF|SCF|EGF|VEGF|TGF|PDGF|NGF'
    r'|DNase|RNase|trypsin|proteinase|protease|lysozyme|collagenase|BrdU|MTT'
    r'|serum|albumin|antibod(?:y|ies)|antiserum|conjugate|substrate|probe|primer'
    r'|transfect\w*|incubat\w+\s+with|dissolved\s+in|diluted\s+in|suspended\s+in'
    # Salts, media components and induction agents. "physiological salt concentrations (150 mM)"
    # reached the output because the sentence around it also said "demonstrated", satisfying the
    # results-language gate -- the ionic strength an assay was run at is a condition of the
    # experiment, never the peptide's own activity, however the sentence is phrased.
    # Deliberately NOT listed here: ampicillin, kanamycin, chloramphenicol, gentamicin. They are
    # selection antibiotics in cloning, but they are also the comparator drugs whose MICs this
    # literature reports beside the peptide's own in the same table -- and a table row usually
    # inherits its assay from the caption rather than naming it, so this guard would fire on it.
    # Adding them cost 30 fully-specified (typed + targeted) activities per 150 documents, measured;
    # the noise they removed was not worth real MIC rows.
    # "salt" is likewise scoped to the two phrases that denote assay ionic strength, not the bare
    # word, which also appears in "salt bridge" and "salt tolerance" (a genuine peptide property).
    r'|physiological\s+salt|salt\s+concentrations?|ionic\s+strength'
    r'|NaCl|KCl|MgCl2|CaCl2|MgSO4|NaHCO3|mercaptoethanol|IPTG|X-gal|NEAA|L-glutamine)\b',
    re.IGNORECASE
)

# A value introduced by an administration/exposure construction is the dose given, not the result
# measured -- "cells were pre-treated with WRW4 (5 uM)", "treatment with 2 mg/ml of AS2-1",
# "mice were injected with 10 mg/kg". Matched against the text immediately BEFORE the value, so it
# describes how that particular number was introduced rather than anything else in the sentence.
#
# Applied only where no assay type could be resolved. A named assay ("IC50 after treatment with
# 5 uM") states what was measured and is never touched by this; without one, the sentence is
# describing an experimental input and there is nothing else it could be.
_TREATMENT_DOSE = re.compile(
    r'\b(?:treated|pre-?treated|treatment|stimulated|challenged|exposed|injected|administered'
    r'|supplemented|induced|transfected|co-?incubated|fed|dosed|perfused|loaded)\b[^.;]{0,70}$'
    r'|\b(?:in\s+the\s+presence\s+of|containing|supplemented\s+with|along\s+with)\b[^.;]{0,50}$',
    re.IGNORECASE
)


# Language that marks a number as a reported outcome rather than an experimental input. Required
# near any value whose assay type could not be resolved from the sentence, its table caption, or
# its section heading.
#
# Why a positive cue is needed rather than more negative ones: a corpus audit found 16,812 records
# (56% of all activities) in exactly that state -- a value and a target, but nothing saying what
# was measured. Sampling them showed the population is not one failure mode but two. Roughly a
# third are genuine results whose assay simply has no standard acronym; the rest are experimental
# parameters that happen to share a sentence with a peptide and an organism: IPTG induction
# concentrations, media recipes ("RPMI containing 11 mM glucose, 10% FBS, 10 mM HEPES"),
# pre-treatment doses ("cells were pre-treated with WRW4 (5 uM)"), injection volumes, and antibody
# dilutions in control panels. Enumerating the reagents was already tried (_REAGENT_CONTEXT below)
# and cannot close this: the set of things a paper doses into an assay is open-ended, while the
# set of ways it announces a measurement is small and conventional. Ten of ten randomly sampled
# records lacking any of these cues were experimental parameters, none was a measurement.
_RESULT_CONTEXT = re.compile(
    r'\b(?:inhibit\w*|kill\w*|lys(?:e|ed|es|is)\b|lytic|reduc\w+|suppress\w*|eradicat\w*'
    r'|abolish\w*|attenuat\w*|prevent\w+|protect\w*|active\s+against|activity\s+against'
    r'|effective\s+against|efficac\w+|potenc\w+|potent\b|susceptib\w+|sensitiv\w+|resistan\w+'
    r'|bactericidal|bacteriostatic|fungicidal|fungistatic|virucidal|cytotoxic\w*|h(?:a)?emolytic'
    r'|antiprolifer\w+|viabilit\w+|survival|mortalit\w+|apopto\w+|necro\w+'
    r'|showed|exhibit\w+|display\w+|demonstrat\w+|observed|obtained|yielded|resulted\s+in'
    r'|was\s+found\s+to\b|values?\s+of\b|ranged?\s+from\b|as\s+low\s+as\b|up\s+to\b)',
    re.IGNORECASE
)
# Generous, because the cue and the number are often separated by a parenthetical or a list of
# other results in the same sentence ("...inhibited growth of E. coli (MIC 4), S. aureus (8) and
# P. aeruginosa (16 ug/mL)"). Narrower windows dropped the later items of exactly such lists.
_RESULT_WINDOW = 220


def get_sentences(text: str) -> list:
    """Split text into sentences using simple heuristics."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return sentences


# Lines that scope the measurements printed under them: table/figure captions and the section
# headings used in this literature. A results table's rows almost never repeat the assay name --
# it is stated once, in the caption ("Table 2. Minimum inhibitory concentrations of ..."), and
# every number below inherits it.
_CAPTION_LINE = re.compile(
    r'^\s*(?:table|tabla|fig(?:ure)?|chart)\s*[\dIVX]+\s*[.:\-]?\s*(.{0,200})',
    re.IGNORECASE | re.MULTILINE
)
_SECTION_HEADING = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*\.?\s*)?((?:anti\w+|antimicrobial|antibacterial|antifungal|antiviral|'
    r'anticancer|antitumou?r|antiparasitic|cytotoxic\w*|h(?:a)?emoly\w+|minimum\s+\w+\s+concentration|'
    r'susceptibilit\w+|inhibition|potency|activity)[^\n]{0,120})$',
    re.IGNORECASE | re.MULTILINE
)


def _is_name_fragment(text: str, name: str) -> bool:
    """
    True if `name` never once appears in the document as a standalone token -- every occurrence is
    glued inside a longer one.

    This is the only way to catch a fragment of a cell-line or compound designation, because the
    fragment is indistinguishable from a real peptide code in isolation: "MB-231" has exactly the
    shape of "FK-565", and no rule reading the string alone can separate them. What separates them
    is that the document never says "MB-231" on its own -- it always says "MDA-MB-231". Observed
    125 times in the corpus, always from that one cell line.

    Conservative by construction: a single standalone occurrence anywhere in the document is enough
    to keep the candidate, so a real peptide that also happens to appear inside a longer compound
    name is unaffected.
    """
    for m in re.finditer(re.escape(name), text):
        before = text[m.start() - 1] if m.start() else ''
        after = text[m.end()] if m.end() < len(text) else ''
        # A hyphen only joins when it is itself attached to more text on the far side, so that
        # "anti-LL-37" reads as joined while a line-ending hyphen or a dash does not.
        left = before.isalnum() or (
            before == '-' and m.start() >= 2 and text[m.start() - 2].isalnum())
        right = after.isalnum() or (
            after == '-' and m.end() + 1 < len(text) and text[m.end() + 1].isalnum())
        if not (left or right):
            return False
    return True


#: Candidate ``type`` -> the provenance label recorded on the output record. The internal type is
#: about SPELLING ("exact" = one-letter, "3letter" = Gly-Ile-Gly); the label is about WHERE the
#: candidate came from, which is what a precision post-mortem actually needs.
_DISCOVERY_PATHS: Dict[str, str] = {
    "exact": "sequence_match",
    "3letter": "sequence_match_3letter",
    "name": "name_match",
    "alphanum": "alphanumeric_name_match",
}

#: Weights for :func:`candidate_confidence`. Deliberately small integers rather than tuned
#: coefficients: this score is RECORDED, not yet acted on, and inventing precise weights before the
#: curation audit can calibrate them would dress a guess up as a measurement.
_CONFIDENCE_SIGNALS: Dict[str, int] = {
    "catalogued": 3,       # the sequence is in a curated database -- the strongest signal there is
    "has_name": 2,         # a name binding means the document called it something
    "long_sequence": 2,    # >= 8 residues: past the length where prose accidents live
    "has_measurement": 1,  # an activity value was paired to it
    "standalone": 1,       # occurs outside any longer candidate
}


def candidate_confidence(pep_data: Dict[str, Any], text: str = "",
                         others: "Optional[List[str]]" = None) -> Dict[str, Any]:
    """
    Scores how well-supported one candidate is, and says which signals it has.

    Replaces nothing yet. :func:`sample.nlp.is_valid_sequence` remains the binary gate; this runs
    alongside it and records its verdict, so the acceptance threshold can be set from the curation
    audit's labelled findings rather than from intuition. The gate this is meant to become cannot
    honestly be calibrated on the 13 documents audited so far.

    The signals are the ones the audit showed actually separate real peptides from noise: "IP" and
    "DEVD" had none of them beyond bare validity, while every genuine peptide had at least a name
    or catalogue membership.

    :param pep_data: A candidate record from :func:`extract_peptides_from_text`.
    :param text: The document text, for the standalone-occurrence signal; skipped when empty.
    :param others: Other sequence candidates in the document, for the same signal.
    :returns: ``{"score": int, "max": int, "signals": [...]}``.
    """
    from sample.nlp import get_known_seq_1l_kp

    signals = []
    sequence = (pep_data.get("peptide_sequence") or "").strip()
    if sequence:
        try:
            if get_known_seq_1l_kp().extract_keywords(sequence):
                signals.append("catalogued")
        except Exception:  # noqa: BLE001 -- scoring must never break extraction
            pass
        if len(sequence) >= 8:
            signals.append("long_sequence")
    if (pep_data.get("peptide_name") or "").strip():
        signals.append("has_name")
    if pep_data.get("biological_activities"):
        signals.append("has_measurement")
    if sequence and text and others and not _is_sequence_fragment(text, sequence, others):
        signals.append("standalone")
    return {
        "score": sum(_CONFIDENCE_SIGNALS[s] for s in signals),
        "max": sum(_CONFIDENCE_SIGNALS.values()),
        "signals": signals,
    }


def _is_sequence_fragment(text: str, sequence: str, others: "List[str]") -> bool:
    """
    True if `sequence` is a fragment of a longer candidate that the document always writes whole.

    The sequence counterpart of :func:`_is_name_fragment`, and written for the same reason: a
    substring of a real peptide is indistinguishable from a real peptide when read in isolation.
    Measured in the first run under the new gates -- "WGFTLC" was extracted from a paper whose
    actual peptide is "CTTHWGFTLC", and "DEVD" from one whose peptide is "GDEVDGSGK". Both look
    perfectly well-formed, both pass :func:`sample.nlp.is_valid_sequence`, and both survive the
    context guard, because the fragment genuinely does appear in its own context window -- inside
    the longer sequence.

    Conservative by construction, and deliberately so: truncation analogues are real in this
    literature. A paper studying both "RWQWR" and "RWQWRWQWR" as separate peptides writes the short
    one on its own at least once, and a single standalone occurrence anywhere in the document is
    enough to keep it. Only a candidate that NEVER stands alone is dropped.

    :param text: The full document text.
    :param sequence: The candidate under test.
    :param others: Every other sequence candidate found in this document.
    :returns: Whether every occurrence of `sequence` sits inside a longer candidate.
    """
    if not sequence:
        return False
    longer = [o for o in others if o != sequence and len(o) > len(sequence) and sequence in o]
    if not longer:
        return False
    # Spans covered by any longer candidate. An occurrence outside all of them is standalone.
    covered = []
    for other in longer:
        for m in re.finditer(re.escape(other), text):
            covered.append((m.start(), m.end()))
    for m in re.finditer(re.escape(sequence), text):
        if not any(start <= m.start() and m.end() <= end for start, end in covered):
            return False
    # Every occurrence was inside a longer candidate -- but only call it a fragment if it occurred
    # at all. A candidate with no occurrences is some other stage's problem, not this one's.
    return bool(re.search(re.escape(sequence), text))


def _preceding_sentence_type(sentences: list, offsets: list, idx: int) -> str:
    """
    Assay type carried over from the sentence immediately before `idx`, or "" if it does not
    settle one.

    Two conditions keep this from bleeding a type across unrelated text. The previous sentence must
    name exactly ONE assay type -- a methods paragraph listing several ("MIC, MBC and hemolysis
    were determined") says nothing about which one the next sentence reports -- and it must be
    contiguous with this one, since a sentence separated by a paragraph break or an intervening
    table is a different context whose assay does not carry forward.
    """
    if idx <= 0:
        return ""
    prev = sentences[idx - 1]
    gap = offsets[idx] - (offsets[idx - 1] + len(prev))
    if gap > _PRECEDING_SENTENCE_GAP:
        return ""
    types = {m.group(1).rstrip('s') for m in METRIC_PATTERN_EXACT.finditer(prev)}
    for pattern, label in _METRIC_PHRASE_TYPE_MAP:
        if pattern.search(prev):
            types.add(label)
    return types.pop() if len(types) == 1 else ""


# A normal sentence break is 1-2 characters. Anything larger means a paragraph break, a table, or
# an OCR page boundary sat between the two sentences.
_PRECEDING_SENTENCE_GAP = 2


def build_scope_types(text: str) -> list:
    """
    Returns [(offset, activity_type), ...] sorted by offset: the assay type declared by each
    caption or section heading in the document, at the position where its scope begins.

    Why this exists: detect_activity_type() only ever saw one sentence, so a measurement whose
    assay type is named in the table caption rather than repeated in the row -- the norm for
    tabular results -- came out as activity_type="unknown". Measured on real output, only 10% of
    surviving activities carried a resolved type; the rest were genuine measurements missing their
    label. A caption/heading is the correct scope to inherit from, and it is the same scope a
    human reader uses.
    """
    scopes = []
    for pattern in (_CAPTION_LINE, _SECTION_HEADING):
        for m in pattern.finditer(text):
            label = detect_activity_type(m.group(1), default="")
            if label:
                scopes.append((m.start(), label))
    scopes.sort(key=lambda x: x[0])
    return scopes


# How far a caption's declared assay type carries. A caption labels the table printed under it,
# not the remainder of the paper -- without a bound, one "Cytotoxicity" heading stamped its type
# onto every later unlabeled number in the document (observed: mg/kg dosing rows from an unrelated
# in-vivo table inheriting "Cytotoxicity"), which then let them past the known-type-or-target gate.
_SCOPE_MAX_CHARS = 3000


def scope_type_at(scopes: list, pos: int) -> str:
    """
    The activity type declared by the nearest caption/heading at or before `pos`, provided `pos`
    is still within that caption's reach. Returns "" when nothing is in scope.
    """
    label, label_offset = "", None
    for offset, lbl in scopes:
        if offset > pos:
            break
        label, label_offset = lbl, offset
    if label_offset is None or pos - label_offset > _SCOPE_MAX_CHARS:
        return ""
    return label

def _nearest_span(spans: List[tuple], pos: float) -> Optional[tuple]:
    """Returns the span tuple in `spans` whose start/end is closest to character position `pos`."""
    if not spans:
        return None
    return min(spans, key=lambda sp: min(abs(sp[1] - pos), abs(sp[2] - pos)))

# A run of unit-less numbers in list form ("5, 10 and ") sitting immediately before a
# number+unit match -- the earlier items of a dose/result list that states its unit only once, at
# the end. Anchored to the end of the preceding text so only a strictly contiguous run qualifies.
_VALUE_LIST_PREFIX = re.compile(
    r'(?:\d+(?:[.,]\d+)?\s*(?:,|and|&)\s*)+$', re.IGNORECASE
)
_LIST_NUMBER = re.compile(r'\d+(?:\.\d+)?')

# Coordinating conjunctions / clause separators that, in this domain's reporting style, mark the
# boundary between two independently-reported measurements: "...5 ug/mL against E. coli AND 40
# ug/mL against S. aureus", "peptide A showed X WHILE peptide B showed Y".
_CLAUSE_SEPARATOR = re.compile(r'\b(?:and|while|whereas|but|whilst|however)\b|[;]', re.IGNORECASE)


def _clause_bounds(sent: str) -> List[Tuple[int, int]]:
    """Character ranges of `sent`'s coordinate clauses, split on _CLAUSE_SEPARATOR."""
    bounds = []
    prev = 0
    for m in _CLAUSE_SEPARATOR.finditer(sent):
        bounds.append((prev, m.start()))
        prev = m.end()
    bounds.append((prev, len(sent)))
    return bounds


def _same_clause(spans: List[tuple], pos: float, bounds: List[Tuple[int, int]]) -> List[tuple]:
    """The subset of `spans` sitting in the same coordinate clause as character position `pos`."""
    for start, end in bounds:
        if start <= pos <= end:
            return [sp for sp in spans if start <= sp[1] <= end]
    return []


def _pair_span(spans: List[tuple], pos: float, bounds: List[Tuple[int, int]]) -> Optional[tuple]:
    """
    Picks the span belonging to the value at `pos`, preferring one in the same coordinate clause
    before falling back to nearest-by-distance across the whole sentence.

    Raw nearest-distance alone is wrong in both directions here, because the two entity kinds sit
    on opposite sides of their value in normal scientific phrasing: a peptide usually PRECEDES its
    value ("peptide A showed an MIC of 5 ug/mL") while a target organism usually FOLLOWS it
    ("5 ug/mL against E. coli"). Measured on real phrasings, distance alone put both values of
    "A showed MIC of 5 ug/mL while B showed MIC of 40 ug/mL" on peptide B, and both values of
    "...5 ug/mL against E. coli and 40 ug/mL against S. aureus" on E. coli. Clause membership
    resolves both without needing a per-entity-kind directional rule, and degrades to exactly the
    previous behavior for sentences with no coordinating conjunction (one clause = whole sentence).
    """
    local = _same_clause(spans, pos, bounds)
    return _nearest_span(local or spans, pos)


_HYPHEN_TAIL = re.compile(r'^[A-Za-z]+')
_HYPHEN_HEAD = re.compile(r'[A-Za-z]+$')
_SHORT_ALPHA_NAME = 5


def _is_hyphen_split_fragment(sent: str, name: str, s: int, e: int) -> bool:
    """
    True when a match is really a piece of one word that a PDF line-break hyphenated in two.

    PDF text extraction preserves the typesetter's end-of-line hyphens, so "alkaline
    phosphatase" arrives as "alkaline phos-phatase". Aho-Corasick treats "-" as a word boundary,
    so the fragment "phos" matched the taxonomy database as a genus and was accepted as a
    target_organism -- the last systematic source of junk organisms in the audit.

    Two independent tests, because the dictionary alone does not cover this domain's vocabulary:
    rejoining across the hyphen yields a known word (catches "phos-phatase" -> PHOSPHATASE), or
    the match is a short all-alphabetic name sitting against an intra-word hyphen at all (catches
    "phos-pholipase", whose rejoined form is absent from the dictionary). The second test is
    deliberately limited to short pure-alpha names: a bare 4-letter genus immediately hyphenated
    to the next word is vanishingly rare next to hyphenation artifacts, whereas longer and
    punctuated names ("E. coli-infected mice") legitimately take a trailing hyphen and are left
    alone.
    """
    from sample.nlp import get_dictionary_words

    tail = _HYPHEN_TAIL.match(sent[e + 1:]) if e < len(sent) and sent[e] == '-' else None
    head = _HYPHEN_HEAD.search(sent[:s - 1]) if s >= 2 and sent[s - 1] == '-' else None

    if tail and (name + tail.group(0)).upper() in get_dictionary_words():
        return True
    if head and (head.group(0) + name).upper() in get_dictionary_words():
        return True

    if len(name) <= _SHORT_ALPHA_NAME and name.isalpha() and (tail or head):
        return True
    return False


# Species adjectives papers use when naming a cell line's origin. Mapped to a form NCBI resolves
# (its organism_names table knows "human"/"mouse"/"rat" but not the adjectival "murine").
_CELL_LINE_SPECIES = re.compile(
    r'\b(human|murine|mouse|mice|rat|hamster|canine|dog|porcine|pig|bovine|cow|'
    r'monkey|simian|rabbit|chicken|feline|cat)\b',
    re.IGNORECASE
)
_SPECIES_WORD_TO_NAME = {
    "murine": "mouse", "mice": "mouse", "simian": "monkey", "canine": "dog",
    "porcine": "pig", "bovine": "cow", "feline": "cat",
}


def _is_vendor_mention(sent: str, s: int, e: int) -> bool:
    """
    True when an organism match sits inside a supplier/manufacturer parenthetical.

    Several vendor names are genuine NCBI genera -- Phoenix (the date palm), Chiron -- so they
    validate perfectly and are indistinguishable from a real organism by name alone. What gives
    them away is the company metadata printed alongside: "(Chiron, Emeryville, CA)", "(2200
    Ci/mmol, Phoenix)". Only the enclosing parenthetical is examined, so an organism named
    elsewhere in the same sentence is unaffected.
    """
    from sample.nlp import _VENDOR_CONTEXT

    open_paren = sent.rfind("(", 0, s)
    if open_paren == -1:
        return False
    close_paren = sent.find(")", e)
    if close_paren == -1:
        close_paren = min(len(sent), e + 80)
    if sent.find(")", open_paren, s) != -1:
        return False  # the nearest preceding "(" already closed -- match is not inside it
    return bool(_VENDOR_CONTEXT.search(sent[open_paren:close_paren + 1]))


_CITATION_WINDOW = 40


def _is_citation_mention(sent: str, s: int, e: int) -> bool:
    """
    True when a match sits inside a bibliographic citation rather than a description of an
    experiment. Author surnames collide with short Cellosaurus codes, so "Smith et al." and
    "(Jones, 1990)" were landing in target_cell_line; the citation markers around them are what
    distinguish that from a genuine cell-line mention.
    """
    from sample.nlp import _CITATION_CONTEXT

    window = sent[max(0, s - _CITATION_WINDOW):e + _CITATION_WINDOW]
    return bool(_CITATION_CONTEXT.search(window))


def _is_common_noun_use(sent: str, name: str, s: int, e: int) -> bool:
    """
    True when a taxonomy hit is really an ordinary English word used as a common noun.

    Binomial nomenclature capitalizes the genus ("Rana", "Mus"), so a taxonomy name that appears
    in the text in lowercase is being used as vocabulary, not as an organism. Observed in real
    output: "cell transformation" produced target_organism="Transformation" (a real taxonomy
    entry) for a melittin record. Restricted to single-token names that are also dictionary words,
    so genuinely lowercase-written identifiers and multi-word binomials are unaffected.
    """
    from sample.nlp import get_dictionary_words

    if " " in name or not name.isalpha():
        return False
    surface = sent[s:e]
    if not surface.islower():
        return False
    return name.upper() in get_dictionary_words()


# A block that carries several newlines and several measurements is a flattened results table,
# not prose: one "sentence" holding many `RowLabel \n value` pairs.
_TABLE_BLOCK_MIN_NEWLINES = 3
_TABLE_BLOCK_MIN_VALUES = 3
_TABLE_ROW_MAX_NEWLINES = 1


def _is_table_block(sent: str, n_values: int) -> bool:
    """
    Whether a "sentence" is really a flattened results table.

    :param sent: Block of text produced by sentence splitting.
    :param n_values: How many numeric values were found in it; a dense run of numbers across
        several lines is the signature of a table, not prose.
    :returns: ``True`` if the block should be treated as tabular, so a value is attributed only
        to peptides on its own row.
    """
    return sent.count("\n") >= _TABLE_BLOCK_MIN_NEWLINES and n_values >= _TABLE_BLOCK_MIN_VALUES


def _same_table_row(sent: str, name_start: int, name_end: int, v_start: int, v_end: int) -> bool:
    """
    Whether a peptide mention and a value plausibly sit in the same flattened table row.

    In a flattened table each row arrives as "DrugName\\nvalue", so the row's own value is on the
    same line or the next one. Without this, a table listing many agents gave every value in the
    whole block to whichever single row label happened to be a recognized peptide -- observed in
    real output: melittin (whose true value is 250 ng/ml) was credited with 1.95 ng/ml from the
    Griseofulvin row and 0.48 ng/ml from the EGF row.
    """
    lo, hi = (name_end, v_start) if name_end <= v_start else (v_end, name_start)
    return sent.count("\n", lo, hi) <= _TABLE_ROW_MAX_NEWLINES


def _maximal_spans(spans: List[tuple]) -> List[tuple]:
    """
    Drops any (name, start, end) span fully contained inside another, keeping the longest at each
    location. Both taxonomy and Cellosaurus catalog genus and species (and cell-line families and
    their members) as separate entries, so a single mention yields several nested matches --
    "Escherichia coli" also matches bare "Escherichia"; "MCF-7" also matches "MCF". Left as-is,
    the shorter/wrong name wins whenever it happens to sit marginally closer to a value
    (target_organism came out as "Escherichia"/"MCF" instead of "Escherichia coli"/"MCF-7" in real
    output), and the duplicates inflate the distinct-entity count that decides whether proximity
    pairing engages at all -- so a genuinely single-organism sentence looked ambiguous.
    """
    ordered = sorted(spans, key=lambda sp: (sp[1], -(sp[2] - sp[1])))
    kept = []
    for sp in ordered:
        if any(sp[1] >= k[1] and sp[2] <= k[2] for k in kept):
            continue
        kept.append(sp)
    return kept

def extract_peptides_from_text(text: str, doc_id: str = "") -> list[dict]:
    """
    Extracts peptide records and their biological activities from one document.

    The classical-NLP-first engine: Aho-Corasick and regex candidate discovery, in-document
    name/sequence alias resolution, proximity-based pairing of values to organisms and assays,
    then LLM triage and verification of what survives.

    :param text: Full document text. The bibliography is truncated first, so author surnames and
        journal abbreviations are not mined as peptide names.
    :param doc_id: Document identifier, used only for log correlation.
    :returns: One dict per peptide, each with ``peptide_sequence``, ``peptide_name`` and a
        ``biological_activities`` list. Empty when the document yields nothing.
    """
    if not text or not text.strip():
        return []

    # Drop the bibliography before candidate discovery -- otherwise author surnames, years, and
    # journal abbreviations in citations get picked up as if they were peptide names/sequences.
    text = truncate_references(text)
    if not text.strip():
        return []

    # 1. Sequence Discovery (Aho-Corasick & Regex)
    sequences = []

    for seq_val in get_known_seq_1l_kp().extract_keywords(text):
        if is_valid_sequence(seq_val):
            sequences.append({"type": "exact", "value": seq_val})

    for seq_val in get_known_seq_3l_kp().extract_keywords(text):
        sequences.append({"type": "3letter", "value": seq_val})

    for name_val in get_seq_names_kp().extract_keywords(text):
        if len(name_val) > 3 and is_valid_peptide_name(name_val):
            sequences.append({"type": "name", "value": name_val})

    for match in SEQ_EXACT.finditer(text):
        # Left un-stripped of stray "[" / "]" noise here (unlike the final peptide_sequence
        # output, cleaned below) -- this value is used as the sequences_list/dict key that the
        # sentence loop later re-finds via plain substring search against the original document
        # text, which still has the brackets. Stripping them here would make that search silently
        # find zero occurrences (the cleaned value no longer matches the bracketed text on the
        # page), dropping the whole record for lack of activities.
        val = match.group(1).replace('-', '')
        is_modified = match.group(0).startswith(('NH2-', 'Ac-')) or match.group(0).endswith(('-NH2', '-COOH'))
        if is_modified and len(val) < 5: continue
        if len(val) >= 5 and is_valid_sequence(val):
            if _STRAIN_CODE_PRECEDING.search(text[max(0, match.start(1) - 50):match.start(1)]):
                continue
            sequences.append({"type": "exact", "value": val})

    for match in SEQ_3LETTER.finditer(text):
        sequences.append({"type": "3letter", "value": match.group(0)})

    # Drop candidates that are only ever a fragment of a longer candidate in this same document.
    # Done here, once, over the assembled candidate list rather than per discovery path, because a
    # fragment and its parent routinely arrive from DIFFERENT paths -- "DEVD" from the catalogued
    # automaton and "GDEVDGSGK" from SEQ_EXACT -- so no single path can see the containment.
    _seq_values = [s["value"] for s in sequences if s.get("type") in ("exact", "3letter")]
    if len(_seq_values) > 1:
        _before = len(sequences)
        sequences = [
            s for s in sequences
            if s.get("type") not in ("exact", "3letter")
            or not _is_sequence_fragment(text, s["value"], _seq_values)
        ]
        if len(sequences) < _before:
            logger.debug(f"[{doc_id}] Dropped {_before - len(sequences)} sequence candidate(s) "
                         f"that never appear outside a longer candidate")

    # SEQ_ALPHANUM (2-6 letters + optional digit/greek suffix, meant to catch real alphanumeric
    # peptide names like "LL37") used to be accepted with no validation at all -- a corpus-wide
    # audit found this was the single largest source of junk peptide_name values (IC50, CO2, pH7,
    # CD4, NP-40, mM3...). Route it through the same is_valid_peptide_name() gate the "name"-type
    # path above already uses.
    for match in SEQ_ALPHANUM.finditer(text):
        val = match.group(0)
        if not is_valid_peptide_name(val):
            continue
        if _FOOTNOTE_MARKER_PREFIX.match(val):
            continue
        if _STRAIN_NAME_PRECEDING.search(text[max(0, match.start() - 40):match.start()]):
            continue
        # Cell-line designations share SEQ_ALPHANUM's exact letters+digits shape, so without this
        # a document's own assay targets get extracted as if they were the peptides under study --
        # observed in real output: "MCF-7" became the peptide_name of a record whose activity was
        # measured *against* MCF-7, while the actual peptide sequence in the same sentence was
        # dropped. Only an exact full-string cellosaurus match is rejected, so a real peptide name
        # that merely contains a cell-line-like substring is unaffected.
        if any(kw.upper() == val.upper() for kw in get_cellosaurus_kp().extract_keywords(val)):
            continue
        sequences.append({"type": "alphanum", "value": val})

    # Global name->sequence cross-reference (offline-built from CAMP/DBAASP/DRAMP/ADP -- see
    # scripts/build_name_sequence_lookup.py). A paper often names a well-known peptide ("magainin
    # 2", "LL-37") without ever restating its sequence anywhere nearby, or the sequence sits in an
    # OCR-mangled table far from the activity data -- find_local_aliases below can't help here
    # since it only binds a name to a sequence written adjacently in the SAME document. Found via
    # a real corpus example: a 1988 magainin paper's Edman-degradation table only gave a
    # substring of the sequence, ~6000 characters from the activity data; "magainin 2" itself
    # appears in prose right next to the activity, so this recovers the record correctly instead
    # of it being name-only. Registered as regular alias entries (name -> canonical sequence) so
    # they reuse the same surface-forms/owner machinery as the in-document aliases below.
    # Uses span_info so the literal surface text as it actually appears in this document (e.g.
    # "magainin 2", lowercase and space-separated) becomes the alias key -- not the map's
    # normalized/uppercase form ("MAGAININ-2") -- since the sentence loop below searches sentences
    # for each candidate's own literal value.
    name_seq_map = get_name_sequence_map()
    global_aliases = {}
    for name_key, start, end in get_known_peptide_names_kp().extract_keywords(text, span_info=True):
        canonical_seq = name_seq_map.get(name_key)
        if not canonical_seq or not is_valid_sequence(canonical_seq):
            continue
        surface = text[start:end]
        if not is_valid_peptide_name(surface):
            continue
        sequences.append({"type": "exact", "value": canonical_seq})
        sequences.append({"type": "name", "value": surface})
        global_aliases[surface] = canonical_seq

    # get_known_seq_1l_kp()/get_known_seq_3l_kp() (used above) require an exact contiguous
    # substring match, so a known sequence broken across a PDF line-wrap or extra internal
    # whitespace ("GIGKFLHSAK\nKFGKAFVGEIMNS") is otherwise invisible to them. Registered the same
    # way as the global name aliases above: the literal, still-broken surface text becomes the
    # alias key, so the sentence loop's plain substring search below -- which scans the same,
    # unmodified document text -- finds it.
    whitespace_aliases = {}
    for surface, (name_key, seq_type) in find_whitespace_broken_sequences(text).items():
        sequences.append({"type": seq_type, "value": name_key})
        sequences.append({"type": "name", "value": surface})
        whitespace_aliases[surface] = name_key
        # Each whitespace-separated piece of the broken run (e.g. "GIGKFLHSAK" and
        # "KFGKAFVGEIMNS") is independently a valid-looking SEQ_EXACT match, so without this it
        # also becomes its own standalone candidate -- one real peptide fragmenting into three
        # separate records, each carrying a duplicate copy of the same activity. Aliasing the
        # pieces onto the reassembled sequence merges them into the one correct record.
        for piece in surface.split():
            if len(piece) >= 5 and piece != name_key:
                whitespace_aliases.setdefault(piece, name_key)

    # Drop candidates that are just the species half of an organism named in this document --
    # "vulgaris" from Proteus vulgaris, "laevis" from Xenopus laevis. A binomial's epithet is not
    # a peptide name, but on its own it looks exactly like one to the name matcher, and it then
    # collects the organism's activities as though it were the agent under study.
    _epithets = set()
    for _nm in get_organism_kp().extract_keywords(text):
        _parts = _nm.split()
        if len(_parts) >= 2 and _parts[-1].isalpha() and len(_parts[-1]) >= 5:
            _epithets.add(_parts[-1].lower())
    if _epithets:
        sequences = [
            s for s in sequences
            if s["type"] not in ("name", "alphanum") or s["value"].lower() not in _epithets
        ]

    # Deduplicate sequences. Name-type candidates are keyed case-insensitively: a paper writing
    # "SV-3936" in a table and "sv-3936" in prose is naming one compound, but keying on the exact
    # string made them two peptides carrying duplicate copies of the same activities. Sequences
    # keep their exact key, since case is meaningful there (residue letters are uppercase and the
    # value is used for substring matching against the document).
    unique_seqs = {}
    for s in sequences:
        key = s["value"].lower() if s["type"] in ("name", "alphanum") else s["value"]
        if key not in unique_seqs:
            unique_seqs[key] = s
    sequences_list = list(unique_seqs.values())

    if not sequences_list:
        return []

    # In-document shorthand-name <-> sequence binding (e.g. a paper stating
    # "NH2-...KSPFAFS (SP4080)" once and then discussing activity using only "SP4080"
    # afterward). Without this, the sequence-bearing candidate never gets an activity sentence
    # linked to it and is silently dropped below -- confirmed against real documents in a
    # corpus-wide audit (493 records had an empty peptide_sequence despite the real sequence
    # being present in their own captured context, exactly this pattern).
    alias_map = find_local_aliases(text, sequences_list)
    for name_val, canonical_seq in global_aliases.items():
        if canonical_seq in unique_seqs:
            alias_map.setdefault(name_val, canonical_seq)
    for surface, name_key in whitespace_aliases.items():
        if name_key in unique_seqs:
            alias_map.setdefault(surface, name_key)

    # 2 & 3. Entity and Activity Discovery within Sentences
    sentences = get_sentences(text)
    # Assay types declared by table captions / section headings, so a measurement printed in a
    # table row can inherit the type its caption states instead of coming out as "unknown".
    scope_types = build_scope_types(text)
    # Genera the document spells out in full, used to settle abbreviated binomials ("M. luteus")
    # against what this paper actually studies rather than against whichever species NCBI happens
    # to register the abbreviation under. See nlp.resolve_abbreviated_binomial.
    document_genera = collect_document_genera(text)
    # Offset of each sentence in `text`, to look up which caption/heading scope it falls under.
    sentence_offsets = []
    _cursor = 0
    for _s in sentences:
        _found = text.find(_s, _cursor)
        if _found < 0:
            _found = _cursor
        sentence_offsets.append(_found)
        _cursor = _found + len(_s)

    extracted_peptides = {}
    for seq in sequences_list:
        val = seq["value"]
        if val in alias_map:
            continue  # merged into its target sequence's record below, not a standalone candidate
        extracted_peptides[val] = {
            "peptide_sequence": val if seq["type"] in ("exact", "3letter") else "",
            "peptide_name": val if seq["type"] in ("name", "alphanum") else "",
            "biological_activities": [],
            "context": extract_context_window(text, target_term=val, token_window=1000),
            "surface_forms": {val},
            # WHICH discovery path produced this candidate, carried through to the output record.
            #
            # Without it, a record from a 2-residue catalogued-automaton hit is indistinguishable
            # from one read off a results table, and diagnosing a precision problem means
            # re-deriving the origin of every record by hand -- which is exactly what the audit of
            # 2026-09-16 had to do to discover that "IP", "PG" and "DEVD" all came from the
            # catalogued automaton rather than from the regex everyone assumed. With it, precision
            # is reportable PER PATH, which is what says where to spend effort.
            "discovery_path": _DISCOVERY_PATHS.get(seq["type"], seq["type"]),
        }

    # Register alias surface forms on their target's record, and give the record a
    # human-readable name if it didn't already have one (pure-sequence candidates otherwise
    # carry no name at all).
    for shortcode, target in alias_map.items():
        pep_data = extracted_peptides.get(target)
        if pep_data is None:
            continue
        pep_data["surface_forms"].add(shortcode)
        if not pep_data["peptide_name"]:
            pep_data["peptide_name"] = shortcode

    # Resolves any discovered candidate string (including alias shortcodes) to the
    # extracted_peptides key that should receive its activity.
    owner = {val: val for val in extracted_peptides}
    for shortcode, target in alias_map.items():
        if target in extracted_peptides:
            owner[shortcode] = target

    for sent_idx, sent in enumerate(sentences):
        # Every occurrence of every discovered candidate, with position -- both to resolve alias
        # shortcodes to their real sequence's record, and to let proximity pairing (below) decide
        # which peptide a value belongs to when a sentence names more than one.
        seq_occurrences = []  # (owner_seq_val, start, end)
        for seq in sequences_list:
            val = seq["value"]
            target = owner.get(val)
            if target is None:
                continue
            flags = re.IGNORECASE if seq["type"] == "name" else 0
            for m in re.finditer(re.escape(val), sent, flags):
                seq_occurrences.append((target, m.start(), m.end()))
        if not seq_occurrences:
            continue
        present_seqs = sorted(set(t for t, _, _ in seq_occurrences))

        # Every experimental-metric occurrence in the sentence (MIC/MIC50/90/100, IC50/90,
        # EC50/90, MBC, MFC, CC50, LC50, LD50, ED50, ZOI, plus descriptive-phrase fallbacks like
        # "growth inhibition"), not just the first. Computed before values_found below so the
        # reagent-context guard can tell whether this sentence names a real metric at all.
        type_spans = [(m.group(1).rstrip('s'), m.start(), m.end()) for m in METRIC_PATTERN_EXACT.finditer(sent)]
        for pattern, label in _METRIC_PHRASE_TYPE_MAP:
            for m in pattern.finditer(sent):
                type_spans.append((label, m.start(), m.end()))
        distinct_types = {lbl for lbl, _, _ in type_spans}

        # Extract values. A contiguous digit run over 6 characters is never a real activity value
        # in this domain -- it's an OCR table-concatenation artifact (observed in real output:
        # "102103104105", adjacent table cells pressed together with no separator). Likewise a
        # "%" value over 100 is physically impossible -- observed in real output: "42148 %",
        # OCR-concatenated compound-code digits ("FR42148") misread as a percentage.
        values_found = []
        for v_match in VALUE_PATTERN.finditer(sent):
            raw_val, unit = v_match.group(1).strip(), v_match.group(2).strip()
            digits = re.sub(r'\D', '', raw_val)
            if len(digits) > 6:
                continue
            _, parsed_val, _ = parse_activity_value_fast(raw_val)
            if unit.lower() == "%":
                if parsed_val is not None and parsed_val > 105:
                    continue
            elif parsed_val == 0:
                # A zero concentration/dose is not a measurable result -- it is an OCR artifact or
                # a control row. Observed in real output: an OCR'd figure axis rendered "100" as
                # "tO0", yielding value_raw "00" at 0.0 ug/ml. Percentages are exempted above,
                # where a genuine 0 ("0% hemolysis") is a real and meaningful finding.
                continue
            # A number sitting next to an assay-condition reagent/buffer/media name (e.g. "10 mM
            # phosphate buffer", "penicillin G (100 U/ml)") is an experimental parameter, not the
            # peptide's own measured activity. Only applied when the sentence names no recognized
            # metric type at all -- a real "MIC of 10 ug/mL in PBS" (which does name a metric)
            # is never affected by this guard.
            if not type_spans:
                window = sent[max(0, v_match.start() - 40):v_match.end() + 40]
                if _REAGENT_CONTEXT.search(window):
                    continue
            # Instrument/column dimensions are rejected even when the block does name a metric,
            # since OCR frequently merges a methods paragraph into the same "sentence" as a results
            # one. Adjacency keeps this from touching a genuine "ZOI of 15 mm" that merely shares a
            # long block with an equipment description.
            near = sent[max(0, v_match.start() - _INSTRUMENT_WINDOW):v_match.end() + _INSTRUMENT_WINDOW]
            if _INSTRUMENT_CONTEXT.search(near):
                continue
            # A number inside a definition of the metric itself is not a result.
            if _DEFINITION_CONTEXT.search(near):
                continue
            values_found.append((raw_val, unit, v_match.start(), v_match.end()))

            # A dose/result list states the unit only once, on the last item ("MIC values of 5,
            # 10 and 40 ug/mL"), so every earlier number in the list has no adjacent unit for
            # VALUE_PATTERN to anchor on and was silently dropped -- only the final value of such
            # a list was ever captured. Walk back over an immediately-preceding run of
            # number/comma/"and" tokens and re-attach this match's unit to each. Deliberately
            # requires the run to be strictly contiguous back from the matched value (nothing but
            # digits, separators and whitespace in between), so an unrelated nearby number
            # ("Table 3 shows ... 40 ug/mL") is not swept in.
            head = sent[:v_match.start()]
            list_prefix = _VALUE_LIST_PREFIX.search(head)
            if list_prefix:
                for num in _LIST_NUMBER.findall(list_prefix.group(0)):
                    if len(re.sub(r'\D', '', num)) > 6:
                        continue
                    values_found.append((num, unit, list_prefix.start(), v_match.end()))
        if not values_found:
            continue

        # Every organism/cell-line span in the sentence, filtered through is_valid_organism_match
        # (rejects pure-numeric matches outright; additionally requires nearby corroboration for
        # short/ambiguous Cellosaurus cell-line codes specifically -- not for organism/taxonomy
        # matches, which are trustworthy at any length once past the numeric/blocklist gates).
        org_spans = _maximal_spans([
            (nm, s, e) for nm, s, e in get_organism_kp().extract_keywords(sent, span_info=True)
            # Both the canonical name and the literal surface text are checked against the
            # blocklist. The automaton returns NCBI's canonical name, so a blocked surface form
            # arrives already rewritten -- "Figs." (a figure cross-reference, catalogued as a
            # common name of the fig genus) comes back as "Ficus" and sails past a canonical-only
            # check. The surface form is what the document actually said, so it is what decides
            # whether this is an organism mention.
            if is_valid_organism_match(nm, sent)
            and is_valid_organism_match(sent[s:e], sent)
            and not _is_hyphen_split_fragment(sent, nm, s, e)
            and not _is_common_noun_use(sent, nm, s, e)
            and not _is_vendor_mention(sent, s, e)
            and not _is_citation_mention(sent, s, e)
        ])
        # An abbreviated binomial gets the genus this document actually names, where NCBI's own
        # abbreviation entry may point at an unrelated species that merely shares the initial.
        org_spans = [
            (resolve_abbreviated_binomial(sent[s:e], document_genera) or nm, s, e)
            for nm, s, e in org_spans
        ]
        # For short/ambiguous Cellosaurus codes, is_valid_organism_match requires a corroborating
        # cue ("cells", "cell line", "ATCC"...). That cue is looked for in the text immediately
        # around the match rather than anywhere in the sentence: a sentence-wide search let
        # "MSH" (melanocyte-stimulating hormone) and "MOR" (mu-opioid receptor) pass as cell lines
        # purely because the same long table caption elsewhere mentioned "xenografts"/"tumor".
        # Adjacency is what actually distinguishes a cell-line mention ("MCF-7 cells") from an
        # unrelated acronym that happens to collide with a Cellosaurus entry.
        cell_spans = _maximal_spans([
            (nm, s, e) for nm, s, e in get_cellosaurus_kp().extract_keywords(sent, span_info=True)
            if is_valid_organism_match(nm, sent[max(0, s - 15):e + 25], is_cell_line_source=True)
            # Same lowercase-surface test the organism path uses: "rat cell lines" satisfies the
            # adjacency requirement above but is a species written as a common noun, not the
            # Cellosaurus line "RAT" -- it was landing in target_cell_line as "RAT".
            and not _is_common_noun_use(sent, nm, s, e)
            and not _is_vendor_mention(sent, s, e)
            and not _is_citation_mention(sent, s, e)
        ])
        entity_spans = [(nm, s, e, False) for nm, s, e in org_spans] + [(nm, s, e, True) for nm, s, e in cell_spans]
        distinct_entities = {nm for nm, _, _, _ in entity_spans}

        # A sentence with at most one organism, one metric type, and one peptide is unambiguous --
        # every value found belongs to that single entity, exactly as before. Proximity-based
        # pairing only changes behavior once a sentence names more than one of a given entity
        # kind, which is where blindly cross-producting (the pre-fix behavior) misattributes data,
        # e.g. "MIC of peptide A was 5 ug/mL against E. coli and 10 ug/mL against S. aureus" no
        # longer stamps both values with the first-listed organism.
        multi_seq = len(present_seqs) > 1 and len(values_found) > 1
        clause_bounds = _clause_bounds(sent)

        for value_raw, unit, v_start, v_end in values_found:
            v_mid = (v_start + v_end) / 2

            if len(distinct_entities) > 1:
                paired = _pair_span(entity_spans, v_mid, clause_bounds)
                target_organism, is_cell_line = (paired[0], paired[3]) if paired else ("", False)
            elif entity_spans:
                target_organism, is_cell_line = entity_spans[0][0], entity_spans[0][3]
            else:
                target_organism, is_cell_line = "", False

            if len(distinct_types) > 1:
                paired_type = _pair_span(type_spans, v_mid, clause_bounds)
                activity_type = paired_type[0] if paired_type else detect_activity_type(sent)
            elif type_spans:
                activity_type = type_spans[0][0]
            else:
                # Nothing in the sentence names the assay. Inherit from the enclosing table
                # caption or section heading before giving up -- a results table states its assay
                # once in the caption and never repeats it per row, which is why only 10% of
                # extracted activities previously carried a resolved activity_type. Failing that,
                # carry over the immediately preceding sentence's assay, which covers the very
                # common two-sentence form "MICs were determined by broth microdilution. Peptide X
                # was active against E. coli at 4 ug/mL."
                activity_type = detect_activity_type(sent, default="") or \
                    scope_type_at(scope_types, sentence_offsets[sent_idx]) or \
                    _preceding_sentence_type(sentences, sentence_offsets, sent_idx) or "unknown"

            target_group = infer_activity_group(target_organism, activity_type, is_cell_line=is_cell_line)

            # Drop an organism that cannot be the target of the activity class it was labelled
            # with -- an "antibacterial" MIC whose target is a plant or an animal. The group is
            # kept (a MIC really is an antibacterial assay); it is the organism that is wrong, and
            # a blank target is honest where a plant is not.
            if target_organism and organism_contradicts_group(target_organism, target_group):
                target_organism, is_cell_line = "", False

            if multi_seq:
                paired_seq = _pair_span(seq_occurrences, v_mid, clause_bounds)
                targets = [paired_seq[0]] if paired_seq else []
            else:
                targets = present_seqs

            # In a flattened results table, a value belongs to its own row. Restrict attribution to
            # peptide mentions on the same (or the immediately adjacent) line, so a peptide listed
            # as one row among many other agents no longer absorbs the whole table's numbers.
            if _is_table_block(sent, len(values_found)):
                targets = [
                    t for t in targets
                    if any(o == t and _same_table_row(sent, s_, e_, v_start, v_end)
                           for o, s_, e_ in seq_occurrences)
                ]
                if not targets:
                    continue

            # An activity with NEITHER an identified assay type NOR a target organism is not a
            # scientific claim -- it is "some number with a unit appeared in a sentence that also
            # mentioned this peptide". Measured on a 120-document sample, 53% of all extracted
            # activities were in exactly this state, and spot-checking them found no true
            # positives: they were table footnotes ("1.5 mg/ml of solution was used"), reagent
            # recipes, UV wavelengths, and OCR'd figure axis labels. Requiring at least one of the
            # two keeps every real measurement -- a stated metric ("MIC of 5 ug/mL") supplies the
            # type, and a stated target ("killed E. coli at 5 ug/mL") supplies the organism.
            if activity_type == "unknown":
                if not target_organism:
                    continue
                # Having a target is necessary but not sufficient. Naming an organism is what a
                # methods sentence does too ("incubated with C. albicans at 30 C"), so an
                # unresolved assay additionally has to read as a reported outcome rather than an
                # experimental input -- see _RESULT_CONTEXT.
                if not _RESULT_CONTEXT.search(
                        sent[max(0, v_start - _RESULT_WINDOW):v_end + _RESULT_WINDOW]):
                    continue
                # ...and results language elsewhere in the sentence does not redeem a number that
                # is itself the dose administered. This is what the results-language gate alone
                # could not separate: "the methylation status was downregulated after treatment
                # with 2 mg/ml of AS2-1" reads as an outcome and states a dose.
                if _TREATMENT_DOSE.search(sent[max(0, v_start - 90):v_start]):
                    continue

            for target in targets:
                act = {
                    # A cell line is not an organism, so it gets its own field instead of being
                    # written into target_organism (where it could never validate against NCBI).
                    # target_organism then carries the species the line came from, resolved below.
                    "target_organism": "" if is_cell_line else target_organism,
                    "target_cell_line": target_organism if is_cell_line else "",
                    "target_group": target_group,
                    "activity_type": activity_type,
                    "value_raw": value_raw,
                    "activity_unit": unit,
                    "context_sentence": sent,
                }
                extracted_peptides[target]["biological_activities"].append(act)

    # 3a. Collapse duplicate activities within each peptide.
    #
    # One measurement legitimately reaches this point several times: a peptide named twice in the
    # same sentence produces one activity per occurrence, and the shared-unit list expansion can
    # re-emit a value already matched directly. Observed in real output, a single "10 mg/kg"
    # statement appeared three times under one peptide. Identity is the scientific content --
    # target, assay, value, unit -- deliberately excluding context_sentence, so the same fact
    # restated in two different sentences still collapses to one record.
    for pep_data in extracted_peptides.values():
        seen, unique_acts = set(), []
        for act in pep_data["biological_activities"]:
            key = (
                act.get("target_organism", ""), act.get("target_cell_line", ""),
                act.get("activity_type", ""), str(act.get("value_raw", "")).strip(),
                (act.get("activity_unit") or "").lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_acts.append(act)
        pep_data["biological_activities"] = unique_acts

    # 3b. Organism validation against NCBI taxonomy.
    #
    # Every target_organism written to the output must be a name NCBI actually knows, so the field
    # is joinable against the taxonomy downstream. Resolution also canonicalizes variants for free
    # ("human" -> "Homo sapiens", "mouse" -> "Mus musculus", "E. coli" -> "Escherichia coli"),
    # which is what makes a cell line's species resolvable from the word used beside it.
    #
    # Order matters: NCBI first, the LLM only for what NCBI rejected, and then NCBI again over the
    # LLM's suggestion. The model is used to guess a spelling, never to authorize a name -- an
    # activity whose organism cannot be resolved by NCBI in the end is discarded.
    unresolved = {}
    for pep_data in extracted_peptides.values():
        for act in pep_data["biological_activities"]:
            raw_org = act.get("target_organism") or ""
            if not raw_org:
                continue
            resolved = resolve_scientific_name(raw_org)
            if resolved:
                act["target_organism"] = resolved
            else:
                unresolved.setdefault(raw_org, []).append(act)

    if unresolved:
        try:
            suggestions = normalize_organisms_llm(list(unresolved), doc_id=doc_id)
        except Exception as e:
            logger.warning(f"LLM organism normalization skipped: {e}")
            suggestions = {}
        for raw_org, acts_with_org in unresolved.items():
            resolved = resolve_scientific_name(suggestions.get(raw_org, ""))
            for act in acts_with_org:
                act["target_organism"] = resolved  # "" marks it for the discard below

    # Species of origin for a cell line, read from the words the paper uses beside it ("human
    # MCF-7 cells", "murine P815 mastocytoma"). Resolved through NCBI like any other organism, so
    # "human" lands as "Homo sapiens". Deliberately NOT defaulted to human when the text gives no
    # species: Cellosaurus is human-dominated, but guessing is how P815 -- a mouse mastocytoma
    # line -- gets stamped Homo sapiens. An unknown species is left empty; the cell-line field
    # still carries the target, which is the part the paper actually stated.
    for pep_data in extracted_peptides.values():
        kept_acts = []
        for act in pep_data["biological_activities"]:
            if act.get("target_cell_line") and not act.get("target_organism"):
                m = _CELL_LINE_SPECIES.search(act.get("context_sentence") or "")
                if m:
                    act["target_organism"] = resolve_scientific_name(
                        _SPECIES_WORD_TO_NAME.get(m.group(0).lower(), m.group(0))
                    )
            # Discard an activity that claimed an organism which no longer validates. An activity
            # that never had one (its target is a cell line, or the assay type alone carried it)
            # is unaffected -- the known-type-or-target gate above already covers that case.
            if act.get("target_organism") or act.get("target_cell_line") or act.get("activity_type") != "unknown":
                kept_acts.append(act)
        pep_data["biological_activities"] = kept_acts

    # 4. LLM Verification, Normalization & LLM Fallback

    # One triage call per document decides which name-only candidates are really peptides. This
    # covers the open-ended junk classes that no blocklist can close -- bench chemicals (BCNU),
    # strain codes (CH-112), instruments/columns (DU-8, RP-8), plasmids (pUC19), and OCR fragments
    # of a real name fused to a table column ("FK-5650"). Candidates that carry a recovered
    # sequence are exempt: they are already strongly grounded, and there is no reason to risk a
    # small model discarding real data. Runs before verification so rejected candidates cost no
    # further LLM calls.
    # Deterministic pass first: a name-only candidate that never stands alone in the document is a
    # fragment of a longer designation, decided from the text rather than from a model. Doing it
    # here rather than at candidate discovery keeps sequence-bearing candidates exempt (they are
    # grounded by their sequence) and shrinks the triage prompt.
    for val, pd in extracted_peptides.items():
        if pd.get("peptide_sequence") or not pd.get("biological_activities"):
            continue
        name = pd.get("peptide_name") or val
        if len(name) >= 3 and _is_name_fragment(text, name):
            logger.debug(f"[{doc_id}] Dropping {name!r}: never appears as a standalone token")
            pd["biological_activities"] = []

    # A name catalogued in CAMP/DBAASP/DRAMP/ADP is a known peptide on the authority of those
    # databases, so no model verdict may veto it -- the same reasoning that already exempts
    # sequence-bearing candidates. This is not hypothetical: on a 45-candidate ballot the model
    # rejected "LL-37", one of the most-studied peptides in this literature. Ballot accuracy
    # degrades as the list grows, and a false negative here silently deletes real data.
    # The exemption is keyed on CATALOGUE MEMBERSHIP, not on merely carrying a sequence.
    #
    # It used to read `not pd.get("peptide_sequence")`, exempting every sequence-bearing candidate
    # on the reasoning that a sequence is "strongly grounded". That reasoning holds for a peptide
    # in CAMP/DBAASP/DRAMP/ADP -- which is what protects LL-37 -- and fails for a bare regex hit:
    # a nameless 5-mer scraped from prose is exactly as sequence-bearing as magainin, and it was
    # therefore never eligible for the one check that could have rejected it. Measured: 248 such
    # records in 866 documents, including "PISEMA" (an NMR technique) and "VCREMS" (a cell line).
    #
    # A catalogued sequence, a named candidate whose name is catalogued, and anything without
    # activities all stay exempt exactly as before.
    def _is_catalogued(val: str, pd: Dict[str, Any]) -> bool:
        if _normalize_peptide_key(pd.get("peptide_name") or val) in get_name_sequence_map():
            return True
        sequence = (pd.get("peptide_sequence") or "").strip()
        if not sequence:
            return False
        try:
            return bool(get_known_seq_1l_kp().extract_keywords(sequence))
        except Exception:  # noqa: BLE001 -- a lookup failure must not silently widen triage
            return True

    triage_pool = [
        (val, (pd.get("context") or "")[:200])
        for val, pd in extracted_peptides.items()
        if pd.get("biological_activities") and not _is_catalogued(val, pd)
    ]
    if triage_pool:
        try:
            kept = triage_candidates_llm(triage_pool, doc_id=doc_id)
            for val, _ in triage_pool:
                if val not in kept:
                    extracted_peptides[val]["biological_activities"] = []
        except Exception as e:
            logger.warning(f"LLM candidate triage skipped: {e}")

    final_peptides = []
    for seq_val, pep_data in extracted_peptides.items():
        acts = pep_data["biological_activities"]

        if acts:
            # Verify against this peptide's local context window, not the full document -- the
            # claim should be checked near where the peptide is actually discussed, and scanning
            # the whole paper for every candidate scales badly for documents with many candidates.
            #
            # This used to call the BERT/NLI entailment gate, which was measured to reject 100% of
            # activities while carrying no usable signal (0 of 84 across 25 documents; TRUE claims
            # scored 0.015-0.053 entailment and FALSE claims 0.034-0.354, i.e. overlapping, with
            # the top score belonging to a false claim). That module has since been removed;
            # verification is now done by the local LLM, which is asked the question directly and
            # fails open rather than silently emptying the database.
            try:
                acts = verify_activities_llm(
                    acts, pep_data["context"],
                    peptide_id=pep_data.get("peptide_name") or seq_val, doc_id=doc_id,
                )
            except Exception as e:
                logger.warning(f"LLM activity verification skipped: {e}")
            acts = standardize_and_filter_activities(acts)

        pep_data["biological_activities"] = acts

        # 6. LLM Fallback
        # Only for candidates that carry a real sequence. Before this guard, a name-only candidate
        # -- which is what every junk candidate (lab reagents, strain codes, instrument models,
        # OCR fragments) is -- also reached this path, so the pipeline spent an LLM call per junk
        # candidate per document trying to invent activities for things like "CO2" and "MW13".
        # That was the single largest consumer of pipeline wall time, and the class of candidate
        # least likely to yield a real result.
        #
        # The context must additionally contain both a number+unit and some activity cue before
        # the call is worth making. Measured on 10 documents, this fallback was 90% of total
        # pipeline wall time (41 calls, 158s of 175s) while verification was 5% -- and the great
        # majority of those calls were structurally incapable of returning anything, because there
        # was no measurement anywhere in the candidate's context for the model to find. Checking
        # that cheaply, with the regexes already used for classical extraction, avoids paying a
        # multi-second LLM round trip (sometimes a 30s timeout) to be told "nothing here".
        ctx = pep_data["context"]
        fallback_worthwhile = bool(
            VALUE_PATTERN.search(ctx)
            and (METRIC_PATTERN_EXACT.search(ctx) or METRIC_PATTERN_IGNORE.search(ctx))
        )
        if not acts and pep_data.get("peptide_sequence") and fallback_worthwhile:
            llm_acts = recover_peptide_llm(ctx, seq_val, doc_id=doc_id)
            if llm_acts:
                # recover_peptide_llm already applies _verify_grounded_activities (quote must be
                # real text naming the peptide; value must parse and match a number in that quote;
                # organism must appear in it; target_group recomputed deterministically), so its
                # output is grounded by construction. The separate second-opinion pass that used
                # to run here was the NLI gate, now removed as measurably signal-free.
                pep_data["biological_activities"] = llm_acts

        if pep_data["biological_activities"]:
            pep_data.pop("surface_forms", None)
            pep_data["document_id"] = doc_id

            # All sentence-level linking (seq_occurrences, context window) is done by this point,
            # so it's now safe to clean stray OCR bracket noise (e.g. "T[GPGRAF]", from an
            # HIV V3-loop-style figure) out of the stored peptide_sequence itself -- doing this
            # earlier, before linking, would make the sentence loop's plain substring search find
            # zero occurrences of the cleaned value in the still-bracketed document text and drop
            # the whole record. peptide_sequence otherwise keeps whatever raw form (1-letter or
            # hyphenated 3-letter) candidate discovery captured, for backward compatibility with
            # existing consumers (merge.py, external_db.py, llm_validator.py...); the three derived
            # fields below give a clean, unambiguous representation in each format regardless of
            # which raw form was captured, leaving any field that can't be confidently derived (an
            # unrecognized residue) empty rather than guessed.
            raw_seq = pep_data.get("peptide_sequence") or ""
            if raw_seq:
                raw_seq = raw_seq.replace('[', '').replace(']', '')
                pep_data["peptide_sequence"] = raw_seq
            pep_data["peptide_sequence_1letter"] = sequence_to_1_letter(raw_seq) if raw_seq else ""
            pep_data["peptide_sequence_3letter"] = sequence_to_3_letter(raw_seq) if raw_seq else ""
            pep_data["peptide_sequence_helm"] = sequence_to_helm(raw_seq) if raw_seq else ""

            final_peptides.append(pep_data)

    return final_peptides
