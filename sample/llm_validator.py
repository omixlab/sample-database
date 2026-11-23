import json
import logging
import re
import concurrent.futures
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple
from tqdm import tqdm
from sample.llm import llm_call
from sample.preprocessing import raw_value_and_unit


class _FieldVerdicts(NamedTuple):
    """Per-field judgement for one extracted record. `source_organism` is "" when the peptide is
    synthetic, the origin is not stated, or the model's suggestion failed NCBI resolution."""
    sequence_ok: bool
    name_ok: bool
    target_ok: bool
    value_ok: bool
    source_organism: str

logger = logging.getLogger("sample.llm_validator")

#: Units whose printed form is one OCR error away from a unit a thousand or a million times larger.
#: "pg" is the case that matters: the micro sign is routinely lost or misread (see
#: preprocessing._OCR_UNIT_CANONICAL for the confusions that ARE mechanically repairable), and
#: unlike "lM" or "Mg/mL" -- which are not units in any system -- "pg/mL" is perfectly real.
_AMBIGUOUS_MASS_UNITS = ("pg/ml", "pg/l", "pg ml", "pg/mL")

_UNIT_ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_picogram": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_picogram", "reason"],
}

_UNIT_SYSTEM_PROMPT = (
    "You judge whether a concentration unit printed in a scientific paper is genuinely PICOGRAMS "
    "or is a misread MICROGRAM sign. Scanned and OCR'd papers frequently turn the micro sign into "
    "other letters, so a printed 'pg/mL' is sometimes really 'ug/mL'. Judge only from the evidence "
    "given. Picograms are genuine and common for cytokine, hormone and immunoassay measurements "
    "(ELISA readouts, plasma peptide levels, limits of quantification). Picograms are implausible "
    "for an antimicrobial potency result such as MIC, MBC or IC50 against an organism, where real "
    "values are micrograms or micromolar. A value sitting in a list beside other measurements "
    "written in ug/mL is evidence of a misread. Answer with the JSON object only."
)


_FIELD_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_peptide_sequence_supported": {"type": "boolean"},
        "is_peptide_name_supported": {"type": "boolean"},
        "is_target_supported": {"type": "boolean"},
        "is_value_supported": {"type": "boolean"},
        "is_synthetic": {"type": "boolean"},
        "source_organism": {"type": "string"},
    },
    "required": [
        "is_peptide_sequence_supported", "is_peptide_name_supported",
        "is_target_supported", "is_value_supported", "is_synthetic", "source_organism",
    ],
}

# Separating TARGET from SOURCE is the whole point of this prompt. Measured on a 12-case benchmark
# built from real failures, a generic "is this field supported by the text?" instruction scored
# 8/12 and *every* miss was the same confusion: the organism a peptide was isolated FROM being
# accepted as the organism it was tested AGAINST (Xenopus laevis, Apis mellifera, Glycine max,
# Armoracia rusticana -- all sources or reagent origins, all marked supported). Spelling the
# distinction out, with a worked example, takes it to 12/12.
#
# Capturing the source as its own field then stops it competing for the target slot, and gives the
# database a provenance column it never had. The synthetic rules matter because much of this
# literature is designed analogs: naming an organism for those would be pure fabrication, so the
# model is told to return "" unless the origin is plainly stated, and an expression host used to
# produce a recombinant peptide is explicitly not its natural source. Scores 10/10 on a 10-case
# source benchmark, including every synthetic/designed/recombinant case.
_VALIDATION_SYSTEM_PROMPT = (
    "You audit fields extracted from a scientific paper.\n"
    "Judge ONLY from the TEXT provided. Never use prior knowledge about the peptide. If the TEXT "
    "does not say it, it is not supported -- even if you know it to be true.\n"
    "TARGET = the organism or cell line the peptide was tested AGAINST -- the one it killed, "
    "inhibited, or acted on. It is NOT the organism the peptide came from, and not the source of "
    'a reagent, control or growth medium. In "magainin from Xenopus laevis inhibited E. coli", '
    "the target is E. coli and is_target_supported=true; if the target field instead said "
    "Xenopus laevis, is_target_supported=false because that is where the peptide came from.\n"
    "  - target absent from the text, or present only as a source/reagent/control "
    "-> is_target_supported=false\n"
    "SOURCE = the organism the peptide naturally comes from (isolated from, produced by, venom of, "
    "skin of, leukocytes of).\n"
    "  - Many peptides are synthetic or designed. If the text says synthetic, synthesized, "
    'designed, or analog, set is_synthetic=true and source_organism="".\n'
    "  - An expression host used to PRODUCE a recombinant peptide is not its natural source "
    '-> "".\n'
    '  - If the origin is not stated IN THE TEXT, set source_organism="". Do not fill it from what '
    "you know about the peptide -- naming an organism the TEXT never mentions is a fabrication. "
    "Only name an organism when the text states it plainly. Prefer the scientific binomial "
    "(porcine -> Sus scrofa, human -> Homo sapiens).\n"
    "A value is supported only if the text states it as a measured result for this peptide; "
    "reject reagent or buffer concentrations, instrument settings, molecular weights and page "
    "numbers.\n"
    "Sequence and name are supported only if the text actually shows them.\n"
    "Answer only with the JSON schema given."
)

# The sentence the value was actually extracted from is the evidence that matters; the 4000-char
# window around the peptide is the fallback when a record has no sentence recorded. Capped so the
# prompt stays short -- gemma4 returns an EMPTY completion for long, multi-clause prompts (see
# DEFAULT_TRIAGE_MODEL in sample/config.py), and a curation pass that silently returns nothing
# would pass every record through unchecked.
_EVIDENCE_CHARS = 1200

# Phrases that introduce an organism as the peptide's ORIGIN or its production host rather than as
# what the peptide was tested against. Applied to the ~60 characters immediately before the target
# mention, so "isolated from the skin of Xenopus laevis" and "expressed in Escherichia coli BL21"
# are recognized as source roles while "inhibited Escherichia coli" is not.
# The gap after the phrase is deliberately tiny and may not cross a comma or semicolon. A looser
# window let the source phrase bind to an organism further down the sentence: in "Magainin 2,
# isolated from the skin of Xenopus laevis, inhibited E. coli", a 30-character gap reached past
# "Xenopus laevis," and wrongly marked E. coli -- the real target -- as a source mention. A clause
# boundary means the source phrase is finished, so it cannot govern what follows.
_SOURCE_ROLE_PREFIX = re.compile(
    r'\b(?:isolated|purified|extracted|obtained|derived|cloned|amplified)\s+from\b[^.,;]{0,12}$'
    r'|\bfrom\s+(?:the\s+)?(?:skin|venom|leukocytes?|granules?|serum|milk|plasma|haemolymph|'
    r'hemolymph|seeds?|leaves|roots?)\s+of\b[^.,;]{0,12}$'
    r'|\b(?:produced|expressed|secreted|synthesized|made)\s+(?:by|in)\b[^.,;]{0,12}$'
    r'|\b(?:venom|skin|leukocytes?|granules?)\s+of\b[^.,;]{0,12}$',
    re.IGNORECASE
)


def validate_extraction_details(context: str, peptide_name: str, peptide_seq: str,
                                activity: Dict[str, Any],
                                model: Optional[str] = None) -> Optional["_FieldVerdicts"]:
    """
    Asks the LLM to judge each extracted field separately against the text it came from, returning
    (sequence_ok, name_ok, target_ok, value_ok).

    Deliberately per-field. The previous version asked one question about "the peptide and its
    activity" together, so a record with a correct sequence but a wrong organism had no way to be
    partially corrected -- and its prompt ended with "If you are unsure, default to true", which,
    combined with defaulting missing keys to True and returning all-True on any exception, made
    the stage accept essentially everything. That is the same fail-open-everywhere pattern that
    made the NLI gate and candidate triage no-ops earlier in this pipeline.

    This still fails open on transport errors -- a curation pass must not delete data because the
    backend was briefly down -- but it no longer *instructs* the model to prefer acceptance, and
    the caller distinguishes "judged supported" from "not judged" (see llm_validate_single_json).
    Returns None instead of a verdict tuple when the model could not be consulted.
    """
    from sample.config import DEFAULT_EXTRACT_MODEL
    from sample.llm import clean_json, _VALIDATION_BREAKER
    from sample.nlp import resolve_scientific_name

    if _VALIDATION_BREAKER.is_open():
        return None

    evidence = (activity or {}).get("context_sentence") or context or ""
    evidence = " ".join(str(evidence).split())[:_EVIDENCE_CHARS]
    if not evidence:
        return None

    target = (activity or {}).get("target_organism") or (activity or {}).get("target_cell_line") or ""
    # The model is asked whether this value appears in the TEXT, so it has to be shown the value
    # and unit the text printed -- see preprocessing.raw_value_and_unit.
    _v, _u = raw_value_and_unit(activity or {})
    value = f"{_v} {_u}".strip()
    atype = (activity or {}).get("activity_type") or "unspecified"

    user = (
        f"TEXT: {evidence}\n\n"
        f"peptide_sequence: {peptide_seq or '(none)'}\n"
        f"peptide_name: {peptide_name or '(none)'}\n"
        f"target: {target or '(none)'}\n"
        f"activity: {atype} = {value or '(none)'}\n\n"
        'Return {"is_peptide_sequence_supported":<bool>,"is_peptide_name_supported":<bool>,'
        '"is_target_supported":<bool>,"is_value_supported":<bool>,'
        '"is_synthetic":<bool>,"source_organism":"<binomial or empty>"}'
    )

    try:
        raw = llm_call(
            [{"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            temperature=0.0, response_schema=_FIELD_VALIDATION_SCHEMA,
            model=model or DEFAULT_EXTRACT_MODEL, max_tokens=256, timeout=60,
            layer="extraction_validate",
        )
        data = clean_json(raw)
    except Exception as e:
        logger.debug(f"LLM validation unavailable: {e}")
        _VALIDATION_BREAKER.record_failure(e)
        return None

    if not isinstance(data, dict):
        return None
    _VALIDATION_BREAKER.record_success()

    def verdict(key: str) -> bool:
        """
        :param key: Schema field name, e.g. ``"is_target_supported"``.
        :returns: The model's boolean for that field. A missing key reads as ``False`` -- an
            omitted verdict is not evidence of support.
        """
        val = data.get(key)
        if isinstance(val, str):
            return val.strip().lower() in ("true", "yes", "1")
        # A field the model did not answer is treated as unsupported only when it had something to
        # judge; an absent field in the record itself is vacuously fine.
        return bool(val) if val is not None else True

    # Two deterministic target checks, ANDed with the model's judgement. Measured on a 9-case
    # suite, the model alone scored 6/9 and every miss was mechanical rather than semantic: it
    # accepted a target that is simply absent from the text (inferring "E. coli" from what it knows
    # about the peptide), and accepted an organism introduced by "isolated from" or "expressed in".
    # Those two questions have exact answers, so they are answered exactly, and the model is left
    # with the part that genuinely needs reading comprehension.
    target_ok = verdict("is_target_supported")
    if target and target_ok:
        norm_evidence = " ".join(evidence.lower().split())
        norm_target = " ".join(str(target).lower().split())
        if norm_target not in norm_evidence:
            target_ok = False                      # named target never appears in its own evidence
        elif _SOURCE_ROLE_PREFIX.search(norm_evidence[:norm_evidence.index(norm_target)][-60:]):
            target_ok = False                      # introduced as an origin or expression host

    # The model proposes a source organism; NCBI decides whether it is one. Same ordering used for
    # target_organism in extraction -- the model supplies a spelling, the taxonomy authorizes it --
    # so a hallucinated or vernacular origin cannot reach the database. Anything unresolvable, and
    # anything for a peptide judged synthetic, is dropped to "".
    source = "" if verdict("is_synthetic") else (data.get("source_organism") or "").strip()
    source = resolve_scientific_name(source) if source else ""

    return _FieldVerdicts(
        sequence_ok=verdict("is_peptide_sequence_supported"),
        name_ok=verdict("is_peptide_name_supported"),
        target_ok=target_ok,
        value_ok=verdict("is_value_supported"),
        source_organism=source,
    )

_PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')
_PARA_MAX_CHARS = 900
_PARA_MAX_CHECKED = 6

_TARGET_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {"confirms": {"type": "boolean"}},
    "required": ["confirms"],
}

_TARGET_EVIDENCE_PROMPT = (
    "You are searching a paper for evidence that a target organism was tested.\n"
    "confirms=true if this paragraph shows the target was one of the organisms or cell lines the "
    "peptide was tested against, assayed on, killed, inhibited, or acted on. A paragraph listing "
    "the panel of test organisms confirms EVERY organism in that list.\n"
    "The paragraph does NOT need to contain the numeric value -- papers state the organism panel "
    "in one paragraph and the numbers in another. Judge the organism's role only.\n"
    "confirms=false if the target appears only as the organism the peptide came from, as the "
    "source of a reagent or control, as a growth medium, in a citation, or not at all.\n"
    "Judge only this paragraph. Answer only with the JSON schema given."
)


def _relevant_paragraphs(source_text: str, target: str, peptide: str) -> List[str]:
    """
    Paragraphs worth asking about, best first.

    A paragraph that never mentions the target cannot confirm it, so the search is restricted to
    those that do -- that check is exact and free, and it keeps a full-text search from turning
    into an LLM call per paragraph of every paper. Among those, paragraphs that also name the
    peptide are examined first, since a confirmation is far likelier there and the search stops at
    the first one found.
    """
    if not source_text or not target:
        return []
    t = " ".join(str(target).lower().split())
    p = " ".join(str(peptide or "").lower().split())
    with_peptide, without = [], []
    for para in _PARAGRAPH_SPLIT.split(source_text):
        flat = " ".join(para.lower().split())
        if t not in flat:
            continue
        (with_peptide if p and p in flat else without).append(" ".join(para.split())[:_PARA_MAX_CHARS])
    return (with_peptide + without)[:_PARA_MAX_CHECKED]


def confirm_target_in_document(source_text: str, peptide_name: str, target: str,
                               activity: Dict[str, Any],
                               model: Optional[str] = None) -> Optional[bool]:
    """
    Re-reads the original document paragraph by paragraph looking for one that confirms the target.

    The sentence-level check judges a record against the single sentence its value came from, which
    is the right evidence for the value but often the wrong evidence for the target: papers
    routinely state the organism panel in one paragraph ("activity was assayed against E. coli,
    S. aureus and C. albicans") and the numbers in another, so a correct target can look
    unsupported simply because its own sentence does not repeat it. This is the confirming pass --
    before a target is discarded, the document is searched for a paragraph that actually states it.

    Returns True (a paragraph confirms it), False (searched, nothing confirms it), or None (the
    model could not be consulted, so nothing is concluded).
    """
    from sample.config import DEFAULT_EXTRACT_MODEL
    from sample.llm import clean_json, _VALIDATION_BREAKER

    paragraphs = _relevant_paragraphs(source_text, target, peptide_name)
    if not paragraphs:
        return False          # the target appears nowhere in the document
    if _VALIDATION_BREAKER.is_open():
        return None

    # The measured value is deliberately NOT shown. Including it made the model demand that the
    # number appear in the same paragraph: on a panel sentence naming E. coli, S. aureus and
    # C. albicans it confirmed only the organism whose value happened to be nearby and rejected
    # the rest. The value has already been judged against its own sentence; this pass asks about
    # the organism's role and nothing else.
    for para in paragraphs:
        user = (
            f"PARAGRAPH: {para}\n\n"
            f"peptide: {peptide_name or 'the peptide'}\n"
            f"target: {target}\n\n"
            'Was the target one of the organisms this peptide was tested against? '
            'Return {"confirms":<bool>}'
        )
        try:
            data = clean_json(llm_call(
                [{"role": "system", "content": _TARGET_EVIDENCE_PROMPT},
                 {"role": "user", "content": user}],
                temperature=0.0, response_schema=_TARGET_EVIDENCE_SCHEMA,
                model=model or DEFAULT_EXTRACT_MODEL, max_tokens=128, timeout=60,
                layer="target_confirm",
            ))
        except Exception as e:
            logger.debug(f"Target evidence search unavailable: {e}")
            _VALIDATION_BREAKER.record_failure(e)
            return None
        _VALIDATION_BREAKER.record_success()
        if isinstance(data, dict) and data.get("confirms") is True:
            return True        # stop at the first paragraph that confirms
    return False


_FILL_SCHEMA = {
    "type": "object",
    "properties": {
        "activity_type": {"type": "string"},
        "target_organism": {"type": "string"},
        "peptide_sequence": {"type": "string"},
    },
    "required": ["activity_type", "target_organism", "peptide_sequence"],
}

_FILL_PROMPT = (
    "You are filling gaps in an extracted record by reading one paragraph of the source paper.\n"
    "Return a value ONLY if this paragraph states it for this peptide. Otherwise return \"\".\n"
    "activity_type: the assay named for this measurement -- MIC, MBC, MFC, IC50, EC50, LD50, "
    "ED50, CC50, LC50, ZOI, Hemolysis or Cytotoxicity. \"\" if the paragraph names no assay.\n"
    "target_organism: the organism or cell line the peptide was tested AGAINST. Never the organism "
    "it came from, and never a reagent, control or growth medium. Prefer the scientific binomial.\n"
    "peptide_sequence: the amino-acid sequence of THIS peptide, one-letter or three-letter, "
    "exactly as written. \"\" if this paragraph does not show it.\n"
    "Never guess and never use prior knowledge -- an empty string is the correct answer whenever "
    "the paragraph does not state the field.\n"
    "Answer only with the JSON schema given."
)

# Assay names the pipeline recognizes. A filled activity_type must be one of these, so a free-text
# answer ("antibacterial assay", "growth inhibition test") cannot invent a new vocabulary in a
# column the rest of the pipeline treats as an enum.
_KNOWN_ACTIVITY_TYPES = {
    "MIC", "MIC50", "MIC90", "MIC100", "MBC", "MBC50", "MBC90", "MFC", "IC50", "IC90",
    "EC50", "EC90", "LD50", "ED50", "CC50", "LC50", "ZOI", "HEMOLYSIS", "CYTOTOXICITY",
    "GROWTH INHIBITION",
}


def fill_missing_fields_from_document(source_text: str, peptide_name: str, peptide_seq: str,
                                      activity: Dict[str, Any],
                                      model: Optional[str] = None) -> Dict[str, str]:
    """
    Searches the manuscript for fields the extractor could not determine, and returns only those
    it can substantiate: {"activity_type": ..., "target_organism": ..., "peptide_sequence": ...}.

    Complements confirm_target_in_document: that pass decides whether an extracted value is real,
    this one recovers what was never captured. Both exist because the extractor works sentence by
    sentence while papers spread a single result across a caption, a methods paragraph and a
    results paragraph -- which is why so many records carry activity_type "unknown" and an empty
    target despite the paper stating both plainly elsewhere.

    Every suggestion is checked before it is accepted, on the same principle used for
    target/source: the model proposes, a deterministic rule authorizes. An assay name must be one
    the pipeline already recognizes, an organism must resolve against NCBI, and a sequence must
    both pass is_valid_sequence and actually occur in the document. Anything else is discarded.
    """
    from sample.config import DEFAULT_EXTRACT_MODEL
    from sample.llm import clean_json, _VALIDATION_BREAKER
    from sample.nlp import resolve_scientific_name, is_valid_sequence

    need_type = (activity or {}).get("activity_type", "unknown") in ("unknown", "", None)
    need_target = not ((activity or {}).get("target_organism") or (activity or {}).get("target_cell_line"))
    need_seq = not peptide_seq
    if not (need_type or need_target or need_seq) or not source_text:
        return {}
    if _VALIDATION_BREAKER.is_open():
        return {}

    # Paragraphs are ranked by what is actually being looked for: the one printing this value is
    # the likeliest place to name its assay and target, and paragraphs naming the peptide are the
    # likeliest to show its sequence.
    value = str((activity or {}).get("value_raw", "")).strip()
    name_l = " ".join(str(peptide_name or "").lower().split())
    ranked, seen = [], set()
    paragraphs = [" ".join(p.split())[:_PARA_MAX_CHARS] for p in _PARAGRAPH_SPLIT.split(source_text)]
    for want_value, want_name in ((True, True), (True, False), (False, True)):
        for para in paragraphs:
            flat = para.lower()
            if want_value and (not value or value not in flat):
                continue
            if want_name and (not name_l or name_l not in flat):
                continue
            if para not in seen:
                seen.add(para)
                ranked.append(para)
    if not ranked:
        return {}

    filled = {}
    for para in ranked[:_PARA_MAX_CHECKED]:
        missing = [k for k, need in (("activity_type", need_type), ("target_organism", need_target),
                                     ("peptide_sequence", need_seq)) if need and k not in filled]
        if not missing:
            break
        user = (
            f"PARAGRAPH: {para}\n\n"
            f"peptide: {peptide_name or 'the peptide'}\n"
            f"measured value: {value or '(unspecified)'} "
            f"{raw_value_and_unit(activity or {})[1]}\n"
            f"fields needed: {', '.join(missing)}\n\n"
            'Return {"activity_type":"<or empty>","target_organism":"<or empty>",'
            '"peptide_sequence":"<or empty>"}'
        )
        try:
            data = clean_json(llm_call(
                [{"role": "system", "content": _FILL_PROMPT}, {"role": "user", "content": user}],
                temperature=0.0, response_schema=_FILL_SCHEMA,
                model=model or DEFAULT_EXTRACT_MODEL, max_tokens=200, timeout=60,
                layer="fill_missing_fields",
            ))
        except Exception as e:
            logger.debug(f"Field fill unavailable: {e}")
            _VALIDATION_BREAKER.record_failure(e)
            return filled
        _VALIDATION_BREAKER.record_success()
        if not isinstance(data, dict):
            continue

        if "activity_type" in missing:
            proposed = (data.get("activity_type") or "").strip()
            if proposed.upper() in _KNOWN_ACTIVITY_TYPES:
                filled["activity_type"] = proposed.upper() if len(proposed) <= 5 else proposed.title()
        if "target_organism" in missing:
            resolved = resolve_scientific_name((data.get("target_organism") or "").strip())
            if resolved:
                filled["target_organism"] = resolved
        if "peptide_sequence" in missing:
            proposed = re.sub(r'[^A-Za-z\-]', '', (data.get("peptide_sequence") or "").strip())
            # Must be a plausible sequence AND physically present in the paper. The second test is
            # what stops a well-known peptide's sequence being recalled from memory rather than
            # read off the page; whitespace is collapsed first so a line-wrapped sequence matches.
            flat_doc = re.sub(r'\s+', '', source_text).upper()
            if len(proposed) >= 5 and is_valid_sequence(proposed) and proposed.replace('-', '').upper() in flat_doc:
                filled["peptide_sequence"] = proposed.replace('-', '').upper()

    return filled


_COHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "field": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["verdict"],
}

# The model is told what the contradiction IS and asked only which side of it the paper supports.
# Framing matters: asked open-endedly to "fix" a record, a model rewrites fields that were never in
# question. Naming the two fields in conflict and demanding one of them keeps the edit contained,
# and "drop" is offered explicitly so an incoherent record has an honest exit that is not a guess.
_COHERENCE_SYSTEM = (
    "You resolve a contradiction between two fields extracted from a scientific paper.\n"
    "You are given the TEXT, the extracted record, and which two fields disagree. Decide from the "
    "TEXT which one is wrong, and return the corrected value for THAT field only.\n"
    'verdict="correct" with the field name and its new value, when the text settles it.\n'
    'verdict="drop" when the text does not describe this measurement at all.\n'
    'verdict="unknown" when the text does not say enough to choose. Prefer "unknown" over a '
    "guess -- a wrong correction is worse than an unresolved flag.\n"
    "Never change a field that is not one of the two named. Answer only with the JSON schema given."
)


def resolve_coherence_conflict(source_text: str, peptide_name: str, activity: Dict[str, Any],
                               violation: Any,
                               model: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Asks the model which side of a cross-field contradiction the source text supports.

    The deterministic rules in :func:`strict_schema.check_activity_coherence` can tell that two
    fields disagree but not which is at fault -- for an "anticancer" record targeting
    *Escherichia coli*, either the class or the organism is wrong, and only the paper says which.
    This reads the paragraphs the record came from and decides.

    :param source_text: Full text of the document the record was extracted from.
    :param peptide_name: Peptide the activity belongs to, for locating relevant paragraphs.
    :param activity: The activity record carrying the contradiction.
    :param violation: The :class:`strict_schema.Violation` describing which pair disagrees.
    :param model: Override the model used; defaults to ``DEFAULT_EXTRACT_MODEL``.
    :returns: ``{"field": name, "value": corrected}`` to apply, ``{"verdict": "drop"}`` when the
        text does not support the measurement at all, or ``None`` when nothing could be decided --
        including when the model is unavailable, so an outage leaves records untouched.
    """
    from sample.config import DEFAULT_EXTRACT_MODEL
    from sample.llm import clean_json, _VALIDATION_BREAKER, _LLM_CALL_TIMEOUT

    if _VALIDATION_BREAKER.is_open():
        return None

    target = activity.get("target_organism") or activity.get("target_cell_line") or ""
    paragraphs = _relevant_paragraphs(source_text, target, peptide_name)
    if not paragraphs:
        return None
    evidence = " ".join(" ".join(paragraphs).split())[:_EVIDENCE_CHARS]

    record = (
        f"peptide={peptide_name or 'unnamed'} | activity_type={activity.get('activity_type')} "
        f"| activity_unit={activity.get('activity_unit')} | value={activity.get('value_lower')} "
        f"| target_group={activity.get('target_group')} "
        f"| target_organism={activity.get('target_organism')!r} "
        f"| target_cell_line={activity.get('target_cell_line')!r}"
    )
    user = (
        f"TEXT: {evidence}\n\n"
        f"RECORD: {record}\n\n"
        f"CONTRADICTION: {violation.field} -- {violation.reason}\n\n"
        'Return {"verdict":"correct|drop|unknown","field":"<one of the two>","value":"<new value>"}'
    )

    try:
        raw = llm_call(
            [{"role": "system", "content": _COHERENCE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, model=model or DEFAULT_EXTRACT_MODEL,
            max_tokens=192, timeout=_LLM_CALL_TIMEOUT, response_schema=_COHERENCE_SCHEMA,
            layer="coherence_conflict",
        )
        _VALIDATION_BREAKER.record_success()
    except Exception as e:
        logger.debug(f"Coherence resolution unavailable: {e}")
        try:
            _VALIDATION_BREAKER.record_failure(e)
        except Exception:
            pass
        return None

    try:
        parsed = clean_json(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict == "drop":
        return {"verdict": "drop"}
    if verdict != "correct":
        return None

    field = str(parsed.get("field") or "").strip()
    value = str(parsed.get("value") or "").strip()
    # Only the two fields named in the contradiction may be touched, so a model that decides to
    # rewrite something else is ignored rather than obeyed.
    if field not in violation.field.split("/") or not value:
        return None
    return {"field": field, "value": value}


def apply_coherence_review(source_text: str, peptide_name: str, activity: Dict[str, Any],
                           model: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Runs the coherence rules over one activity and applies whatever the LLM can settle.

    The deterministic rules decide WHAT is inconsistent; the model decides WHICH field to change;
    and the contract then re-judges the result, so a model correction reaches the database only if
    :func:`strict_schema.enforce_activity` independently accepts it -- the same ordering step 17
    uses.

    :param source_text: Document text the record came from.
    :param peptide_name: Peptide the activity belongs to.
    :param activity: The activity record to review. Not mutated; a corrected copy is returned.
    :param model: Override the model used for the review.
    :returns: ``(activity_or_None, n_resolved)``. ``None`` means the record should be dropped:
        either the model said the text does not describe it, or a contradiction survived the
        review and the record cannot be published as a measurement.
    """
    from sample.strict_schema import check_activity_coherence, enforce_activity

    problems = check_activity_coherence(activity)
    if not problems:
        return activity, 0

    candidate = dict(activity)
    resolved = 0
    for violation in problems:
        decision = resolve_coherence_conflict(source_text, peptide_name, candidate, violation, model)
        if decision is None:
            continue
        if decision.get("verdict") == "drop":
            return None, resolved
        proposed = dict(candidate)
        proposed[decision["field"]] = decision["value"]
        # target_group is DERIVED, not observed: once the model has confirmed which organism the
        # paper tested, the class follows from that organism's domain and the assay. Taking the
        # model's own label instead throws that away -- asked to fix an "anticancer" MIC against
        # E. coli, it answered "other", which removes the contradiction while losing the fact that
        # this is an antibacterial result. Recomputing gives "antibacterial".
        if decision["field"] == "target_group":
            try:
                from sample.nlp import infer_activity_group
                derived = infer_activity_group(
                    proposed.get("target_organism", ""), proposed.get("activity_type", ""),
                    is_cell_line=bool(proposed.get("target_cell_line")))
                if derived:
                    proposed["target_group"] = derived
            except Exception as e:
                # Falls back to the model's own target_group label rather than the derived one --
                # worse (see the note above on why derivation is preferred) but not fatal, so the
                # record still proceeds to the coherence re-check below instead of being dropped
                # over a taxonomy lookup failure.
                logger.debug(f"target_group re-derivation failed for {peptide_name!r}: {e}")
        clean, _ = enforce_activity(proposed)
        if clean is not None:
            candidate = clean
            resolved += 1

    # FINAL GATE. Whatever the model proposed, the record leaves this function only if the rules
    # accept it -- both the field-level contract and the cross-field rules, re-run from scratch on
    # the corrected record. A model answer is a candidate value and nothing more; it cannot enter
    # the database on its own authority, and it cannot introduce a NEW violation while fixing an
    # old one, because the whole record is re-judged rather than just the field that changed.
    final, violations = enforce_activity(candidate)
    if final is None or violations:
        return None, resolved
    if check_activity_coherence(final):
        # A contradiction the review could not settle leaves a record whose fields still disagree.
        # Publishing it would put a measurement in the database describing no real experiment.
        return None, resolved
    final["coherence_reviewed"] = True
    return final, resolved


def llm_validate_single_json(args: Tuple[str, str]) -> Dict[str, int]:
    """
    Validates one extraction file field-by-field. The unit of work for the validation pool.

    :param args: ``(input_path, output_path)``. The validated copy is written to `output_path`.
    :returns: Per-file counters -- how many sequences, names and activities were seen and how
        many were dropped.
    """
    if isinstance(args, tuple) and len(args) == 2:
        file_path, output_path = args
    else:
        file_path = args
        output_path = args
        
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        peptides = data.get("peptides", []) if isinstance(data, dict) else data
        if not isinstance(peptides, list):
            return file_path, {}

        # The document the records were extracted from, for the paragraph-by-paragraph target
        # confirmation below. Read once per file rather than per activity. Absence is not an error
        # -- confirm_target_in_document treats empty text as "nothing found", which leaves targets
        # to the sentence-level judgement alone.
        source_text = ""
        source_file = (data.get("metadata") or {}).get("source_file") if isinstance(data, dict) else None
        if source_file and os.path.exists(source_file):
            try:
                with open(source_file, "r", encoding="utf-8", errors="ignore") as sf:
                    source_text = sf.read()
            except OSError as e:
                logger.debug(f"Could not read source text {source_file}: {e}")

        updated = False
        retained_peptides = []
        
        total_activities = 0
        dropped_activities = 0
        dropped_incoherent = 0
        resolved_conflicts = 0
        total_sequences = 0
        dropped_sequences = 0
        total_names = 0
        dropped_names = 0
        peptides_validated = 0
        
        for item in peptides:
            context = item.get("context", "")
            if not context:
                context = item.get("reasoning", "")
                
            peptides_validated += 1
            name = item.get("peptide_name", "")
            seq = item.get("peptide_sequence", "")
            activities = item.get("biological_activities", [])
            
            total_sequences += 1
            if name:
                total_names += 1
            total_activities += len(activities)
            
            valid_activities = []
            seq_supported = True

            if not activities:
                verdicts = validate_extraction_details(context, name, seq, {})
                if verdicts:
                    if seq and not verdicts.sequence_ok:
                        seq_supported = False
                    if not verdicts.name_ok and item.get("peptide_name"):
                        item["peptide_name"] = None
                        dropped_names += 1
                        updated = True
                    if verdicts.source_organism:
                        item["source_organism"] = verdicts.source_organism
                        updated = True
            else:
                for act in activities:
                    verdicts = validate_extraction_details(context, name, seq, act)
                    if verdicts is None:
                        # Not judged (backend unavailable). Keep the record rather than delete data
                        # over an outage, but mark it so a consumer can tell curated rows from
                        # rows that merely escaped curation.
                        act["llm_validated"] = False
                        valid_activities.append(act)
                        continue

                    # Only a sequence that EXISTS can be unsupported. Without the `seq and`
                    # guard the model is asked to judge "(none)", answers that it is not
                    # supported, and the entire peptide is discarded -- name, activities and
                    # provenance with it. About 80% of extracted records are name-only, so this
                    # deleted most of the database: a protegrin-1 record carrying a valid name, a
                    # real MIC against C. albicans and a resolvable porcine origin was dropped
                    # outright in testing.
                    if seq and not verdicts.sequence_ok:
                        seq_supported = False
                    if not verdicts.name_ok and item.get("peptide_name"):
                        item["peptide_name"] = None
                        dropped_names += 1
                        updated = True

                    # Provenance is a property of the peptide, not of one measurement, so the first
                    # confidently-identified origin is kept on the peptide record. Set only when
                    # non-empty: an empty answer means synthetic or unstated, and must not
                    # overwrite an origin another sentence in the same paper did state.
                    if verdicts.source_organism and not item.get("source_organism"):
                        item["source_organism"] = verdicts.source_organism
                        updated = True

                    # An unsupported value makes the record meaningless, so it goes. An unsupported
                    # target is a mis-pairing -- the measurement is real but was attached to the
                    # wrong organism or cell line -- so the field is cleared and the record kept.
                    # That is the failure mode behind "Theobroma cacao" on a Proteus MIC, and
                    # deleting the whole row for it would throw away a genuine measurement.
                    if not verdicts.value_ok:
                        dropped_activities += 1
                        updated = True
                        continue
                    # A target the sentence does not support is not discarded on that basis alone.
                    # The panel of test organisms is commonly stated in one paragraph and the
                    # numbers printed in another, so the document is searched for a paragraph that
                    # confirms it before the field is cleared. Only a completed search that found
                    # nothing (False) clears the target; None means the search could not run and
                    # the existing value is left untouched.
                    if not verdicts.target_ok and (act.get("target_organism") or act.get("target_cell_line")):
                        confirmed = confirm_target_in_document(
                            source_text, name, act.get("target_organism") or act.get("target_cell_line"), act,
                        )
                        if confirmed is False:
                            act["target_organism"] = ""
                            act["target_cell_line"] = ""
                            updated = True
                        elif confirmed is True:
                            act["target_confirmed_in_document"] = True
                            updated = True
                    # Cross-field coherence, last: the fields are individually valid by now, so
                    # what remains is records whose fields contradict each other -- an anticancer
                    # assay against a bacterium, a hemolysis reported in millimetres. The rules say
                    # what disagrees, the model says which side the paper supports, and the
                    # contract re-judges the correction before it is kept.
                    reviewed, n_resolved = apply_coherence_review(source_text, name, act)
                    if reviewed is None:
                        dropped_incoherent += 1
                        updated = True
                        continue
                    if n_resolved:
                        resolved_conflicts += n_resolved
                        updated = True
                    act = reviewed
                    act["llm_validated"] = True
                    valid_activities.append(act)
                        
            if seq_supported:
                item["biological_activities"] = valid_activities
                retained_peptides.append(item)
            else:
                updated = True # Entire peptide dropped
                dropped_sequences += 1
                dropped_activities += len(valid_activities)
                if item.get("peptide_name"):
                    dropped_names += 1
            
        if updated:
            if isinstance(data, dict):
                data["peptides"] = retained_peptides
            else:
                data = retained_peptides
                
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            # If not updated, we still need to write it to the new output_path
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
        stats = {
            "updated": 1 if updated else 0,
            "total_activities": total_activities,
            "dropped_activities": dropped_activities,
            "dropped_incoherent": dropped_incoherent,
            "resolved_conflicts": resolved_conflicts,
            "total_sequences": total_sequences,
            "dropped_sequences": dropped_sequences,
            "total_names": total_names,
            "dropped_names": dropped_names,
            "peptides_validated": peptides_validated
        }
        return file_path, stats
        
    except Exception as e:
        logger.error(f"Error in LLM validation for {file_path}: {e}")
        return file_path, {}

def run_llm_validation(json_files: Sequence[Tuple[str, str]], workers: int = 4) -> None:
    """
    Step 13: audit every extracted field against the text it came from.

    :param json_files: ``(input_path, output_path)`` pairs to validate.
    :param workers: Concurrent validation threads; each issues its own LLM calls.
    :returns: ``None``; totals are logged and the validated copies are written.
    """
    from sample.config import DEFAULT_EXTRACT_MODEL
    logger.info(f"Starting LLM validation ({DEFAULT_EXTRACT_MODEL}) on {len(json_files)} files...")
    
    total_stats = {
        "files_updated": 0,
        "total_activities": 0,
        "dropped_activities": 0,
        "dropped_incoherent": 0,
        "resolved_conflicts": 0,
        "total_sequences": 0,
        "dropped_sequences": 0,
        "total_names": 0,
        "dropped_names": 0,
        "peptides_validated": 0
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(llm_validate_single_json, f): f for f in json_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM Validating"):
            f, stats = future.result()
            if stats:
                total_stats["files_updated"] += stats.get("updated", 0)
                total_stats["total_activities"] += stats.get("total_activities", 0)
                total_stats["dropped_activities"] += stats.get("dropped_activities", 0)
                total_stats["dropped_incoherent"] += stats.get("dropped_incoherent", 0)
                total_stats["resolved_conflicts"] += stats.get("resolved_conflicts", 0)
                total_stats["total_sequences"] += stats.get("total_sequences", 0)
                total_stats["dropped_sequences"] += stats.get("dropped_sequences", 0)
                total_stats["total_names"] += stats.get("total_names", 0)
                total_stats["dropped_names"] += stats.get("dropped_names", 0)
                total_stats["peptides_validated"] += stats.get("peptides_validated", 0)
            
    logger.info(f"LLM Validation complete. Evaluated {total_stats['peptides_validated']} unique peptide entries.")
    logger.info(f"Sequences: {total_stats['total_sequences']} total, {total_stats['dropped_sequences']} dropped.")
    logger.info(f"Names: {total_stats['total_names']} total, {total_stats['dropped_names']} dropped.")
    logger.info(f"Activities: {total_stats['total_activities']} total, {total_stats['dropped_activities']} dropped.")
    logger.info(
        f"Coherence: {total_stats['resolved_conflicts']} cross-field conflicts resolved by review, "
        f"{total_stats['dropped_incoherent']} activities dropped as internally contradictory."
    )
    logger.info(f"Updated {total_stats['files_updated']} files in total.")
    return total_stats["dropped_activities"]


def is_ambiguous_mass_unit(unit: Any) -> bool:
    """
    Whether a unit is one whose printed form could be an OCR'd micro sign.

    :param unit: The unit as extracted.
    :returns: True for picogram-per-volume spellings. Cheap enough to call on every activity, so
        the LLM below is only ever reached for the units that genuinely need judging.
    """
    folded = str(unit or "").strip().lower().replace("µ", "u").replace("μ", "u")
    folded = re.sub(r"\s+", "", folded)
    return any(re.sub(r"\s+", "", candidate) == folded for candidate in _AMBIGUOUS_MASS_UNITS)


def micrograms_counterpart(unit: str) -> str:
    """
    The microgram spelling of an ambiguous picogram unit, keeping its DENOMINATOR.

    A verdict of "this was a misread micro sign" says the mass prefix is wrong; it says nothing
    about the volume. Returning a hardcoded "ug/mL" therefore silently rewrote a ``pg/L`` reading
    -- which is in :data:`_AMBIGUOUS_MASS_UNITS` -- as per-millilitre, a 1000x denominator error
    introduced by the very function meant to repair a 1000x numerator error.

    :param unit: The ambiguous unit as printed.
    :returns: The microgram equivalent, e.g. ``"ug/mL"`` for ``pg/ml`` and ``"ug/L"`` for ``pg/l``.
    """
    folded = str(unit or "").strip().lower().replace("\u00b5", "u").replace("\u03bc", "u")
    folded = re.sub(r"\s+", "", folded)
    if "/" in folded:
        denominator = folded.split("/", 1)[1]
    else:
        # Slash-less spellings such as "pg ml", whitespace already removed.
        denominator = folded[2:] if folded.startswith("pg") else ""
    canonical = {"ml": "mL", "l": "L", "dl": "dL", "ul": "uL", "": "mL"}.get(denominator,
                                                                             denominator)
    return f"ug/{canonical}"


def adjudicate_picogram_unit(unit: str, activity_type: str, value_raw: str, context: str,
                             model: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Decides whether a printed picogram unit is genuine or a misread microgram sign.

    A rule cannot make this call, which is why it is asked of a model. ``pg/mL`` is a REAL unit --
    measured across 6,000 extraction texts it appears in 4.58% of them, overwhelmingly in genuine
    cytokine and plasma-peptide measurements ("both <8 pg/ml", "limit of quantification was 10
    pg/ml", "ANP, BNP and CNP are 1528 +/- 158 pg mL-1"). A blanket repair, or a value-range floor,
    would corrupt all of those to catch the minority that are artefacts. But artefacts do occur, and
    they are recognisable from context: one observed sentence reads "TBSV (0.007-2 ug/ml) and TMVP
    (0.06-16 pg/ml)" -- the same assay, in the same list, with one unit a million-fold out of line.
    OCR output from a scanned paper showed the same column rendered "0 pg. /ml." on one page and
    "0 ug./ml." on the next.

    Unlike the mechanical repairs in ``preprocessing._OCR_UNIT_CANONICAL`` -- "lM", "Mg/mL" and the
    rest, which are not units in any system and so can be rewritten unconditionally -- this one
    needs the surrounding evidence.

    :param unit: The unit as printed, e.g. ``"pg/mL"``.
    :param activity_type: The assay, e.g. ``"MIC"`` or ``"Cytotoxicity"``. Antimicrobial potency in
        picograms is implausible; an immunoassay readout in picograms is routine.
    :param value_raw: The measurement as printed, which carries the magnitude the judgement turns on.
    :param context: Surrounding text from the source document.
    :param model: Override the local model; defaults to the local Ollama model in sample/config.py.
    :returns: ``(unit, reason)``. The unit is unchanged unless the model judges it a misread, in
        which case ``"ug/mL"`` is returned with the model's stated reason. On ANY failure -- the
        breaker being open, an unreachable backend, an unparseable reply -- the printed unit is
        returned unchanged with a ``None`` reason: a judge that cannot be consulted must not be
        allowed to silently rewrite a measurement.
    """
    from sample.config import DEFAULT_EXTRACT_MODEL
    from sample.llm import clean_json, _UNIT_BREAKER

    if not is_ambiguous_mass_unit(unit):
        return unit, None
    if _UNIT_BREAKER.is_open():
        return unit, None

    evidence = " ".join(str(context or "").split())[:_EVIDENCE_CHARS]
    user = (
        f"TEXT: {evidence}\n\n"
        f"assay: {activity_type or '(unspecified)'}\n"
        f"value as printed: {value_raw or '(none)'} {unit}\n\n"
        'Return {"is_picogram":<bool>,"reason":"<short justification>"}'
    )
    try:
        raw = llm_call(
            [{"role": "system", "content": _UNIT_SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            temperature=0.0, response_schema=_UNIT_ADJUDICATION_SCHEMA,
            model=model or DEFAULT_EXTRACT_MODEL, max_tokens=192, timeout=60,
            layer="unit_adjudicate",
        )
        data = clean_json(raw)
    except Exception as e:                                      # noqa: BLE001 - degrade, don't fail
        logger.debug(f"Unit adjudication unavailable: {e}")
        _UNIT_BREAKER.record_failure(e)
        return unit, None

    if not isinstance(data, dict) or "is_picogram" not in data:
        # An answer that does not fit the schema is not a verdict. Treated as no verdict rather
        # than as agreement, so a confused model cannot rewrite a real measurement.
        return unit, None
    _UNIT_BREAKER.record_success()

    if data.get("is_picogram"):
        return unit, None
    reason = str(data.get("reason") or "").strip()[:300] or "judged a misread micro sign"
    return micrograms_counterpart(unit), reason
