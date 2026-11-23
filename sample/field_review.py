"""
Per-field LLM review of extracted data: one prompt, schema, context budget and model per field.

Every extracted record has fields that fail in field-specific ways, and a single
"check this record" prompt cannot be good at all of them. This module holds one
:class:`FieldSpec` per field, each carrying:

* the failure modes MEASURED on this corpus, with counts, so the prompt names real errors
  rather than imagined ones;
* a deterministic pre-check, so a defect a rule already settles never costs a model call;
* a JSON schema narrow enough that the model's only freedom is the judgement itself;
* a character budget for the evidence text, sized to land on the bottom context rung;
* a default model, with the reason it is that model.

Three findings from this codebase shape the design, and ignoring any of them makes the system
slower or wrong rather than better:

**1. Most field defects are deterministically decidable, so the LLM is the exception.** Measured
over the refined output on disk: 1,423 activities pair a concentration type with a mass/time/
energy unit, 1,163 units carry U+FFFD from PDF decoding, 882 ranges have no upper bound, 68
activity types are prose. Not one of those needs a model -- they are settled by
`concentration.activity_unit_is_coherent`, `preprocessing.canonicalize_unit` and a length test,
all of which step 11 already applies. The model is for what is left: a name that might be a cell
line, a sequence that might be a compound, a target that is named nowhere in the text.
:func:`deterministic_verdict` is consulted first for every field and short-circuits when it
decides.

**2. Context must snap to a rung, not be fitted per field.** `llm.llm_call` pins num_ctx to
`DEFAULT_NUM_CTX` and escalates once to `FALLBACK_NUM_CTX`, because a continuously varying window
makes Ollama rebuild its runner -- measured at 7 restarts in 10 minutes across three sizes,
discarding the cache the fitting was meant to protect. So a field spec does NOT choose a context
size. It chooses how much EVIDENCE to include, in characters, so that the finished prompt lands
on the bottom rung. :data:`EVIDENCE_BUDGET_CHARS` derives that budget from the rung rather than
hardcoding it.

**3. On this host the best model for a field is usually the one already loaded.** Ollama evicts
one model to load another, and this pipeline has already paid for that: 31 llama-server restarts
and 23 evictions in 20 minutes when two stages disagreed about the model, with documents failing
on a different symptom each run. A spec may therefore name a model other than the default, but
:func:`review_many` GROUPS work by model so each is loaded at most once per pass. Interleaving
two models per record would cost more in eviction than any accuracy difference could return.
"""
import json
import logging
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sample.config import DEFAULT_EXTRACT_MODEL, DEFAULT_NUM_CTX

logger = logging.getLogger("sample.field_review")

#: The model every field review uses, PINNED here rather than inherited from
#: `config.DEFAULT_EXTRACT_MODEL`.
#:
#: gemma4:latest, by operator decision on 2026-10-22 and on the only judgement score this project
#: has measured: 12/12 on the triage verdict, the same shape of task as a field review. Its
#: alternatives are measured too, and measured worse HERE even though some are better at
#: extraction -- `gemma3:4b` wins on extraction recall but returned reagent names
#: (`o-phthalaldehyde`, `fluorescamine`) as peptides and 65% of its records without a sequence,
#: which is a judgement failure; `qwen3:14b` does not fit beside the KV cache on this host
#: (13%/87% CPU/GPU split).
#:
#: Pinned rather than inherited because the two settings answer different questions. Extraction's
#: model is chosen for recall on generation; a reviewer is chosen for precision on a verdict, and
#: this project has already recorded that those rankings disagree. Inheriting would mean a future
#: extraction-model change silently re-chose the reviewer as a side effect -- which is how
#: `TestModelSplitGuard` came to exist.
REVIEW_MODEL: str = "gemma4:latest"


def warn_if_review_model_differs_from_extraction() -> Optional[str]:
    """
    :returns: A warning string when the review model is not the extraction model, else ``None``.

    Running a review whose model differs from the loaded one makes Ollama evict and reload for
    every alternating call. This pipeline has already paid that bill: 31 llama-server restarts
    and 23 evictions in 20 minutes when two stages disagreed, with documents failing on a
    different symptom each run. The models agree today; this exists so that if they ever stop
    agreeing, the cost is stated rather than discovered.
    """
    if REVIEW_MODEL == DEFAULT_EXTRACT_MODEL:
        return None
    message = (f"field review uses {REVIEW_MODEL} but extraction uses {DEFAULT_EXTRACT_MODEL}. "
               f"Ollama will evict and reload between them -- measured at 31 llama-server "
               f"restarts in 20 minutes. Run the review when no extraction is in flight, or "
               f"align the two.")
    logger.warning(message)
    return message

#: Characters of prompt that fit the bottom context rung.
#:
#: Derived from `llm_call`'s own arithmetic rather than guessed: it estimates tokens as
#: ``chars / 3.5 + max_tokens + 256``, and escalates to the next rung above `DEFAULT_NUM_CTX`.
#: Reserving 384 tokens for the answer and the template margin leaves this many characters for
#: system prompt, record and evidence combined. A spec's `evidence_chars` is what remains after
#: its own prompt, which is why the budget is stated once here and divided per field below.
EVIDENCE_BUDGET_CHARS: int = int((DEFAULT_NUM_CTX - 384) * 3.5)

#: Prompt/schema variants, so a change to either is measured instead of asserted.
#:
#: ``v1`` is the first draft. ``v2`` differs in three ways, each a hypothesis about the weakness
#: `v1` measured -- 67% `unsure` on activity_unit, with 0% false alarms:
#:
#: 1. **reason before verdict.** The schema's property order is the generation order under
#:    constrained decoding, so `v1` made the model commit to a verdict and then justify it.
#:    `v2` puts `reason` first, which is a chain-of-thought that costs nothing extra.
#: 2. **a positive criterion for `ok`.** `v1` said when to answer `wrong` and when to answer
#:    `unsure` and never what makes a value correct, which leaves `ok` as the option with no
#:    stated grounds. That is the most likely cause of the `unsure` rate.
#: 3. **naming the evidence that settles it.** Each field states what a reader would look for,
#:    so the model knows what counts as having checked.
#:
#: Whichever measures better is what :data:`PROMPT_VARIANT` points at; the loser stays so the
#: comparison can be re-run after a model change.
PROMPT_V1: str = "v1"
PROMPT_V2: str = "v2"
PROMPT_V3: str = "v3"
PROMPT_VARIANTS: Tuple[str, ...] = (PROMPT_V1, PROMPT_V2, PROMPT_V3)

#: The variant in force.
#:
#: **v2 was measured WORSE and is kept only so the comparison can be re-run.** At n=10 it looked
#: better and was briefly shipped on that basis; at n=40 on the same cases, with gemma4:latest:
#:
#:     field          variant  caught  false  unsure
#:     activity_unit  v1          75%     5%     60%
#:     activity_unit  v2          10%     0%     70%
#:     activity_type  v1          35%    20%     72%
#:     activity_type  v2          30%    20%     60%
#:
#: 75% against 10% is 15 of 20 known-bad cases against 2, which is not noise; and v2's apparent
#: "consistent zero false alarms" was an n=10 artifact -- at n=40 it matches v1's 20% on
#: activity_type. The likeliest mechanism is v2's own positive criterion: telling the model that
#: "the text does not have to restate the value verbatim" made it lenient, and a unit of `min` on
#: an MIC does appear in the document somewhere. An instruction meant to cut the `unsure` rate
#: bought leniency instead.
#:
#: ``v3`` isolates the change most likely to have helped from the one that hurt: it is ``v1``
#: verbatim PLUS the `settles_it` clause, with no positive-`ok` criterion and no reason-first
#: reordering. Pending its own measurement, the shipped variant is the measured best.
PROMPT_VARIANT: str = PROMPT_V1

#: Verdicts a reviewer may return. Deliberately three, not two: `unsure` is what keeps a model
#: from being forced into a coin flip on evidence that genuinely does not settle the question,
#: and the caller treats it as "leave the record alone" rather than as either verdict.
VERDICT_OK: str = "ok"
VERDICT_WRONG: str = "wrong"
VERDICT_UNSURE: str = "unsure"
VERDICTS: Tuple[str, ...] = (VERDICT_OK, VERDICT_WRONG, VERDICT_UNSURE)


@dataclass(frozen=True)
class FailureMode:
    """
    One way a field is known to be wrong, with the evidence that it happens.

    :param name: Short slug used in prompts and counters.
    :param description: What the error looks like, phrased for the model.
    :param examples: Real values observed on this corpus.
    :param count: How many instances were measured, or ``None`` when not counted.
    :param deterministic: Whether a rule already settles it, in which case no model call is made.
    :param measured: Whether this mode was OBSERVED on this corpus. ``False`` marks one included
        on reasoning alone -- a known class of error for which no instance has been counted here.
        The flag exists so the two cannot be confused: this project's rule is that a model choice
        or a gate justified by reasoning rather than measurement has twice cost real output, and
        a prompt is no different. A reasoned mode is still worth telling the model about; it just
        must not be mistaken for evidence.
    """
    name: str
    description: str
    examples: Tuple[str, ...] = ()
    count: Optional[int] = None
    deterministic: bool = False
    measured: bool = True


@dataclass(frozen=True)
class FieldSpec:
    """
    Everything needed to review one field: prompt, schema, context budget and model.

    :param field: The record key being judged.
    :param scope: ``"peptide"`` when the field lives on the peptide record, ``"activity"`` when it
        lives on one of its activities.
    :param question: The single question the reviewer answers, in one sentence.
    :param failure_modes: Measured ways this field goes wrong.
    :param model: Local Ollama tag to use. Defaults to :data:`REVIEW_MODEL`, which is pinned
        rather than inherited from the extraction setting; see that constant.
    :param model_rationale: Why that model, stated so a change has something to argue against.
    :param max_output_tokens: Cap on the answer, which also feeds the context arithmetic.
    :param evidence_chars: Characters of document text to include.
    :param can_correct: Whether the reviewer may propose a replacement value, or only judge.
    :param needle_fields: Which record/activity keys locate this field's evidence in the
        document. Declared per field because the field's own value is often the WORST needle:
        measured directly, reviewing `activity_unit="min"` on an MIC selected 1,200 characters of
        windows around "minute", "determined" and "amino", and the model correctly answered
        "unsure -- the document text provided does not contain any information about the activity
        unit". The unit is written beside its VALUE, so `value_raw` is the needle that finds it.
    :param settles_it: What a careful reader would look for to answer the question, in one
        clause. Used by the ``v2`` prompt: naming the evidence tells the model what counts as
        having checked, rather than leaving "unsure" as the safe default.
    """
    field: str
    scope: str
    question: str
    failure_modes: Tuple[FailureMode, ...]
    model: str = REVIEW_MODEL
    model_rationale: str = ""
    max_output_tokens: int = 192
    evidence_chars: int = 1200
    can_correct: bool = True
    needle_fields: Tuple[str, ...] = ("peptide_name", "peptide_sequence")
    settles_it: str = ""

    # ---------------------------------------------------------------- schema

    def schema(self, variant: Optional[str] = None) -> Dict[str, Any]:
        """
        :param variant: One of :data:`PROMPT_VARIANTS`; :data:`PROMPT_VARIANT` when omitted.
        :returns: JSON schema for the reviewer's answer.

        Flat, closed and small on purpose. Ollama converts the schema to a grammar, and that
        converter has a documented limitation this codebase has already been bitten by: array
        length bounds (`minItems`/`maxItems`) make it return an EMPTY completion rather than an
        error, which a fail-open caller reads as "everything is fine". Nothing here uses them,
        and `reason` is capped by the prompt rather than by `maxLength` for the same reason --
        unsupported keywords are a silent failure, not a loud one.
        """
        # Property ORDER is generation order under constrained decoding, which makes it a
        # prompt decision rather than a formatting one: v2 emits `reason` first so the model
        # states its evidence before committing to a verdict.
        if (variant or PROMPT_VARIANT) == PROMPT_V2:
            properties: Dict[str, Any] = {
                "reason": {"type": "string"},
                "verdict": {"type": "string", "enum": list(VERDICTS)},
            }
            required = ["reason", "verdict"]
        # v3 keeps v1's order: the reason-first reordering is one of the three changes v2 bundled,
        # and v3 exists to isolate the other one.
        else:
            properties = {
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "reason": {"type": "string"},
            }
            required = ["verdict", "reason"]
        if self.can_correct:
            properties["corrected"] = {"type": "string"}
            properties["failure_mode"] = {
                "type": "string",
                "enum": [m.name for m in self.failure_modes] + ["other"],
            }
            required.append("failure_mode")
        return {"type": "object", "properties": properties, "required": required}

    # ---------------------------------------------------------------- prompt

    def system_prompt(self, variant: Optional[str] = None) -> str:
        """
        :param variant: One of :data:`PROMPT_VARIANTS`; :data:`PROMPT_VARIANT` when omitted.
        :returns: The system message, built from this field's measured failure modes.

        The failure list is the point. A generic "check this field is correct" prompt leaves the
        model to invent its own notion of correct; naming the errors that actually occur, with
        real examples from this corpus, turns the task into recognition rather than judgement.
        """
        lines = [
            f"You review one field of a peptide record extracted from a scientific paper.",
            f"Field: `{self.field}`.",
            f"Question: {self.question}",
            "",
            "These are the errors that actually occur in this field, with real examples:",
        ]
        for mode in self.failure_modes:
            if mode.deterministic:
                continue
            examples = "; ".join(mode.examples[:3])
            lines.append(f"- {mode.name}: {mode.description}"
                         + (f" Examples: {examples}." if examples else ""))
        chosen = variant or PROMPT_VARIANT
        if chosen in (PROMPT_V2, PROMPT_V3) and self.settles_it:
            lines += ["", f"What settles the question: {self.settles_it}"]

        lines += ["", "Rules:",
                  "- Judge ONLY the named field. Other fields may be wrong; that is not your "
                  "task.",
                  "- Use ONLY the supplied document text as evidence. Do not use outside "
                  "knowledge to confirm a value that the text does not support."]
        if chosen == PROMPT_V2:
            # The positive criterion. v1 stated when to answer `wrong` and when to answer
            # `unsure` and never what makes a value correct, leaving `ok` the one option with no
            # stated grounds -- measured at 67% `unsure` with 0% false alarms, i.e. a reviewer
            # that declined rather than one that erred.
            lines += [
                f'- Answer "{VERDICT_OK}" when the evidence is consistent with the value. The '
                "text does not have to restate it verbatim: a value that matches what the "
                "document reports, in the document's own units and for this peptide, is correct.",
                f'- Answer "{VERDICT_WRONG}" only when the evidence contradicts the value or '
                "shows it belongs to something else.",
                f'- Answer "{VERDICT_UNSURE}" only when the supplied text contains nothing '
                "bearing on the question. Do not use it to avoid deciding on evidence that is "
                "present but requires reading.",
                "- First write `reason`: one short sentence quoting the words or numbers you "
                "used. Then give `verdict`, consistent with that reason.",
            ]
        else:
            lines += [
                f'- Answer "{VERDICT_UNSURE}" when the text does not settle the question. That '
                "is a useful answer, not a failure; a guess is worse than an admission.",
                "- `reason` must be one short sentence quoting the evidence you used.",
            ]
        if self.can_correct:
            lines.append('- When the verdict is "wrong" and the text gives the right value, put '
                         "it in `corrected`; otherwise leave `corrected` empty.")
            lines.append("- `failure_mode` must be one of the names listed above, or \"other\".")
        return "\n".join(lines)

    def needles(self, record: Dict[str, Any],
                activity: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        :param record: The peptide record.
        :param activity: The owning activity, for activity-scoped fields.
        :returns: Values to search the document for, in priority order.
        """
        source: Dict[str, Any] = dict(record)
        if activity:
            source.update(activity)
        return [source.get(key) for key in self.needle_fields]

    def build_messages(self, value: Any, record: Dict[str, Any],
                       activity: Optional[Dict[str, Any]] = None,
                       evidence: str = "", variant: Optional[str] = None,
                       json_context: str = "") -> List[Dict[str, str]]:
        """
        :param value: The field's current value.
        :param record: The peptide record it belongs to.
        :param activity: The activity it belongs to, for activity-scoped fields.
        :param evidence: Document text; trimmed to :attr:`evidence_chars`.
        :param json_context: Evidence drawn from the extraction record itself -- see
            :func:`record_context`. Presented separately from the document text because the two
            have different standing: the document is the authority, while the record is what the
            extractor believed, and a question about whether the extractor erred cannot be
            settled by quoting the extractor back at itself.
        :returns: Chat messages for :func:`sample.llm.llm_call`.
        """
        context: Dict[str, Any] = {
            "peptide_name": record.get("peptide_name", ""),
            "peptide_sequence": record.get("peptide_sequence", ""),
        }
        if activity is not None:
            context.update({
                "activity_type": activity.get("activity_type", ""),
                "value_raw": activity.get("value_raw", ""),
                "activity_unit": activity.get("activity_unit", ""),
                "target_organism": activity.get("target_organism", ""),
                "target_cell_line": activity.get("target_cell_line", ""),
            })
        # The field under review is stated separately from the surrounding record so the model
        # cannot mistake a neighbouring value for the one it is judging.
        user = [
            f"Field under review: {self.field}",
            f"Current value: {json.dumps(value, ensure_ascii=False)}",
            "",
            "Surrounding record (context only, do not judge these):",
            json.dumps(context, ensure_ascii=False, indent=1),
        ]
        if json_context:
            user += ["",
                     "What the extractor recorded alongside this value. This is evidence about "
                     "what the extractor saw, not an authority -- if it disagrees with the "
                     "document text below, the document wins:",
                     json_context]
        user += ["",
                 "Document text:",
                 (evidence or "")[:self.evidence_chars]]
        return [{"role": "system", "content": self.system_prompt(variant)},
                {"role": "user", "content": "\n".join(user)}]


# --------------------------------------------------------------------------- evidence selection

#: Characters either side of a match when pulling evidence out of a document.
_WINDOW = 420

#: Shortest token worth searching for. See :func:`select_evidence`.
_MIN_NEEDLE_CHARS = 4


def select_evidence(text: str, needles: Sequence[Any], budget: int) -> str:
    """
    The passages of `text` that mention any of `needles`, up to `budget` characters.

    :param text: Full document text.
    :param needles: Values whose neighbourhood is relevant -- the peptide name, its sequence,
        the value being checked.
    :param budget: Maximum characters to return.
    :returns: Concatenated windows, separated by ``" ... "``; the head of the document when
        nothing matches.

    Passages rather than the whole document, because the whole document does not fit the bottom
    context rung and truncating it keeps whichever half happens to come first. A field's evidence
    is wherever its value is written, which is rarely the first 12,000 characters.
    """
    if not text:
        return ""
    spans: List[Tuple[int, int]] = []
    for needle in needles:
        token = str(needle or "").strip()
        # Four characters, not three. "min", "s", "h" and "%" are real units and all of them
        # match thousands of times in ordinary prose; a needle that matches everywhere selects
        # nothing. Longer tokens are what actually locate a passage.
        if len(token) < _MIN_NEEDLE_CHARS:
            continue
        for match in re.finditer(re.escape(token), text, re.IGNORECASE):
            spans.append((max(0, match.start() - _WINDOW), min(len(text), match.end() + _WINDOW)))
            if len(spans) >= 12:
                break
    if not spans:
        return text[:budget]
    spans.sort()
    merged: List[List[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: List[str] = []
    used = 0
    for start, end in merged:
        piece = text[start:end]
        if used + len(piece) > budget:
            piece = piece[:max(0, budget - used)]
        if piece:
            out.append(piece)
            used += len(piece)
        if used >= budget:
            break
    return " ... ".join(out)


# --------------------------------------------------------------------------- the field registry
#
# Counts below were measured on this corpus during the 2026-10-20/22 reviews, over 13,733 refined
# documents and 17,486 activities. They are in the prompts because a model recognising a named,
# exemplified error does better than one asked to judge correctness in the abstract -- and they
# are here, in code, so a later measurement can contradict them.

_NAME_SPEC = FieldSpec(
    field="peptide_name",
    scope="peptide",
    needle_fields=("peptide_name", "peptide_sequence"),
    settles_it=("a sentence or table caption that introduces this string as a peptide, or a "
                "sequence given for it"),
    question="Is this the name of a peptide, as opposed to something else in the same table column?",
    failure_modes=(
        FailureMode("cell_line_as_name",
                    "A cell line or bacterial strain read out of a table's row or column header.",
                    ("MCF-7", "HeLa", "SW-620"), count=None),
        FailureMode("dose_as_name",
                    "A number from a serial-dilution series, taken from the dose column.",
                    ("250", "125", "62.5", "31.2"), count=None),
        FailureMode("unit_string_as_name",
                    "A concentration with its unit, often with the micro sign mangled.",
                    ("50 afii9839 g ml ¡ 1", "100 �g/mL"), count=None),
        FailureMode("mass_list_as_name",
                    "A list of mass-spectrometry m/z values.",
                    ("892.306; 979.422; 1302.687", "2548.2"), count=None),
        FailureMode("non_peptide_compound",
                    "A small-molecule drug or reagent used as a comparator or stain.",
                    ("Diclofenac", "Nimesulide", "o-phthalaldehyde"), count=None),
        FailureMode("category_not_an_entity",
                    "A class or phrase rather than one peptide.",
                    ("magainin analogues A, B, C and F", "magainin peptides",
                     "20 essential amino acids"), count=None),
        FailureMode("table_label",
                    "A row label that means nothing outside its own paper.",
                    ("1", "4a", "C2", "Peptide 3"), count=None),
    ),
    model_rationale=(
        "The pipeline model. This is the same judgement `llm.triage_candidates_llm` already makes "
        "with a measured 12/12 on gemma4:latest, so the field inherits a known score rather than "
        "an assumed one."),
    evidence_chars=1400,
)

_SEQUENCE_SPEC = FieldSpec(
    field="peptide_sequence",
    scope="peptide",
    needle_fields=("peptide_sequence", "peptide_name"),
    settles_it=("the sequence written out for THIS peptide, in one-letter or three-letter form, in "
                "the text or a table row bearing its name"),
    question=("Does the document state this residue sequence for this peptide, and is it a "
              "peptide sequence rather than a compound formula or a name spelled in residue "
              "letters?"),
    failure_modes=(
        FailureMode("name_copied_into_sequence",
                    "The peptide's own name placed in the sequence field.",
                    ("SANDOSTATIN", "ANTIMYCINA"), count=4182, deterministic=True),
        FailureMode("contains_digits", "A molecular formula rather than a sequence.",
                    ("C10H22N6O4", "C2H5"), count=1875, deterministic=True),
        FailureMode("too_short", "Fewer than five residues.", (), count=698, deterministic=True),
        FailureMode("non_residue_characters", "Letters that are not amino acids.",
                    ("XXX-YYY-ZZZ",), count=522, deterministic=True),
        FailureMode("dictionary_word_or_dna", "An English word, or a nucleotide string.",
                    ("ATGCATGCATGC",), count=190, deterministic=True),
        FailureMode("three_letter_ocr_corruption",
                    "Three-letter notation with OCR damage: a residue triplet misread, or stray "
                    "punctuation inside the sequence. These PASS the structural rules, so they "
                    "are the ones worth your attention.",
                    ("D-PHE-CY'S-TYR", "LYS-ARG-ASP-ALA-GLU-ALA-VAL!GLY-QR_C)",
                     "H- (D)-PHE-CVS-PHE"), count=20),
        FailureMode("sequence_of_another_peptide",
                    "A real sequence, correctly transcribed, but belonging to a different "
                    "peptide in the same paper. Settled by rule when the name is one this "
                    "project catalogues; the model never sees those.",
                    ("name=GRFT with ISGRIFFITHSIN", "name=nisin with NISAPLINVR",
                     "name=MP1 with GIGFVFTL, the influenza M1 epitope"),
                    count=865, deterministic=True),
    ),
    model_rationale=(
        "The pipeline model. Every structural failure mode here is already settled by "
        "`sequence.validate_extracted_sequence`, so the model only sees sequences that passed "
        "every rule -- a narrow residue where the question is attribution and OCR damage, not "
        "shape."),
    evidence_chars=1600,
)

_ACTIVITY_TYPE_SPEC = FieldSpec(
    field="activity_type",
    scope="activity",
    needle_fields=("peptide_name", "value_raw", "target_organism"),
    settles_it=("the methods section naming the assay, or a table header giving the metric for "
                "this column"),
    question="Which assay does this measurement come from, and does the stated type name it?",
    failure_modes=(
        FailureMode("prose_instead_of_type",
                    "A sentence describing the assay instead of naming it.",
                    ("hydrophobicity (RP-HPLC retention time tR )",
                     "Not specified (measured by percentage relative to control cells)"),
                    count=68, deterministic=True),
        FailureMode("placeholder",
                    "A placeholder where the paper does name the assay.",
                    ("N/A", "unknown", "Not specified"), count=1193),
        FailureMode("wrong_metric",
                    "A named metric that is not the one measured -- an MBC reported as an MIC, "
                    "or a cytotoxicity reported as an IC50.", (), count=None, measured=False),
        FailureMode("not_a_bioactivity",
                    "A physical property rather than an activity: a molecular weight, a "
                    "retention time, a docking energy.",
                    ("molecular weight", "retention time"), count=None, measured=False),
    ),
    model_rationale=(
        "The pipeline model. 1,193 placeholders are the recoverable majority here and recovering "
        "one means reading the methods section, which is the same document-grounded reading the "
        "pipeline model already does for activity verification."),
    evidence_chars=1800,
)

_ACTIVITY_UNIT_SPEC = FieldSpec(
    field="activity_unit",
    scope="activity",
    needle_fields=("value_raw", "peptide_name", "target_organism"),
    settles_it=("the unit printed immediately beside this value, in the sentence or the table "
                "header that reports it"),
    question="Is this the unit the document printed beside this value for this assay?",
    failure_modes=(
        FailureMode("mojibake",
                    "The micro sign lost by PDF decoding.",
                    ("�M", "�g/mL"), count=1163, deterministic=True),
        FailureMode("non_concentration_unit",
                    "A mass, time, energy or count unit on a concentration metric -- usually a "
                    "molecular weight read out of a mass-spec table into the MIC column.",
                    ("MIC in Da", "IC50 in min"), count=1423, deterministic=True),
        FailureMode("metric_as_unit", "The assay name placed in the unit field.",
                    ("MIC", "ZOI"), count=100, deterministic=True),
        FailureMode("micro_lost_as_milli",
                    "The paper printed micromolar and the unit reads millimolar, because OCR "
                    "renders the micro sign as a latin m. The two readings differ by 1000x and "
                    "are indistinguishable once the character is gone, so the DOCUMENT is the "
                    "only way to settle it. This is the one worth your attention.",
                    ("mM on an MIC",), count=None),
        FailureMode("wrong_denominator",
                    "A per-volume unit where the paper reports per mass, or vice versa.",
                    ("ug/mL vs ug/mg",), count=None, measured=False),
        FailureMode("prefix_lost_at_an_implausible_magnitude",
                    "A unit the field barely uses, carrying a value that would be ordinary one "
                    "prefix down. Across 182,766 curated potency measurements mg/mL appears 38 "
                    "times; this extraction has 1,733, and where a curated database reports the "
                    "same finding the median disagreement for mg/mL is exactly 1000x. Decide "
                    "from the recorded sentence and the sibling measurements whether the paper "
                    "really works in this unit or a micro sign was lost.",
                    ("MIC 49.38 mg/mL beside siblings in ug/mL",
                     "IC50 6.62 mg/mL where the sentence confirms mg/ml"),
                    count=1848),
        FailureMode("implausible_magnitude",
                    "An ordinary unit at a value outside the measured potency band (0.05 to "
                    "2000 uM or ug/mL). Either the unit or the number is wrong and a rule cannot "
                    "say which, so this is the one where the recorded sentence matters most.",
                    ("MIC 2860 uM with a context sentence reading 'MIC (mg/ml)'",
                     "MIC 5130.4 uM from 'MIC (lM) 5130.4'"),
                    count=1945),
    ),
    model_rationale=(
        "The pipeline model. `llm_validator.adjudicate_picogram_unit` already asks this model the "
        "neighbouring question (picogram versus microgram) and that layer is in production, so "
        "the prompt shape and the model are both already exercised on unit adjudication."),
    evidence_chars=1200,
)

_VALUE_SPEC = FieldSpec(
    field="value_raw",
    scope="activity",
    needle_fields=("value_raw", "peptide_name", "target_organism"),
    settles_it=("this number appearing in the text or a table cell, on the row for this peptide "
                "and the column for this target"),
    question="Does the document report this number for this peptide, assay and target?",
    failure_modes=(
        FailureMode("range_without_upper",
                    "A range whose upper bound was never stored.",
                    ("5 to 50", "0.3-0.5"), count=882, deterministic=True),
        FailureMode("mangled_operator",
                    "A comparison or plus-minus operator lost to PDF decoding, leaving the "
                    "number's meaning unsettled -- an upper bound and a lower bound read the "
                    "same once the character is gone.",
                    ("�50", "(�1.78 to �0.76)"), count=54),
        FailureMode("fabricated",
                    "A number that appears nowhere in the document. The extraction model has "
                    "been caught emitting its own prompt example.",
                    ("12.5 with MCF-7 and IC50",), count=685),
        FailureMode("stock_solution_captured_as_potency",
                    "The concentration the peptide was DISSOLVED at, recorded as the "
                    "concentration at which it acted. 2 mg/mL is a perfectly ordinary stock "
                    "solution and an impossible MIC, so no change of unit repairs it -- the "
                    "measurement does not belong on this assay at all. The recorded sentence "
                    "gives it away: it describes a preparation, not a result.",
                    ("MIC 2 mg/mL from 'solution (2 mg/ml, dissolved in sterile deionized "
                     "water) was incubated'",), count=113),
        FailureMode("contradicts_its_own_context_sentence",
                    "The sentence the extractor saved states a different quantity from the one "
                    "it recorded, after allowing for an honest unit conversion. The unit is "
                    "right and the number was misread.",
                    ("value_lower 0 from 'The MICs ... are 4.69 and 0.01 mg/ml'",), count=288),
        FailureMode("wrong_row",
                    "A real number from the right table but the wrong row -- another peptide's "
                    "value, or another organism's column.", (), count=None, measured=False),
    ),
    model_rationale=(
        "The pipeline model. The fabrication mode is the one `document_quality."
        "activity_supported_by_text` already screens deterministically; what reaches the model is "
        "wrong-row attribution, which needs the table read as prose."),
    evidence_chars=2000,
)

_TARGET_ORGANISM_SPEC = FieldSpec(
    field="target_organism",
    scope="activity",
    needle_fields=("target_organism", "value_raw", "peptide_name"),
    settles_it=("the organism named in the sentence reporting this value, or in the row or column "
                "header of its table"),
    question="Which organism was this measurement made against, as the document states it?",
    failure_modes=(
        FailureMode("missing",
                    "No organism recorded although the assay must have had one. This is the "
                    "largest single loss of usable data in the corpus.",
                    (), count=7971),
        FailureMode("abbreviation_split",
                    "The same organism written two ways, so it counts as two. Give the full "
                    "binomial.",
                    ("E. coli vs Escherichia coli", "S. aureus vs Staphylococcus aureus"),
                    count=None),
        FailureMode("strain_suffix",
                    "A strain designation fused to the species name. Keep the species; the strain "
                    "belongs elsewhere.",
                    ("E. coli ATCC25922", "S. aureus ATCC 29213"), count=None),
        FailureMode("placeholder", "A placeholder instead of an organism.",
                    ("N/A", "not specified"), count=None),
        FailureMode("cell_line_in_organism_field",
                    "A human cell line where an organism belongs.",
                    ("MCF-7", "HeLa"), count=None),
    ),
    model_rationale=(
        "The pipeline model. This is the field `llm.normalize_organisms_llm` already handles in "
        "production; the spec exists so the same judgement is reachable per record rather than "
        "only per batch of names."),
    evidence_chars=1600,
)

_TARGET_CELL_LINE_SPEC = FieldSpec(
    field="target_cell_line",
    scope="activity",
    needle_fields=("target_cell_line", "value_raw", "peptide_name"),
    settles_it=("the cell line named in the sentence reporting this value, or in the row or column "
                "header of its table"),
    question="Which cell line was this measurement made against, as the document states it?",
    failure_modes=(
        FailureMode("missing",
                    "No cell line recorded for a cytotoxicity or antiproliferative measurement, "
                    "which must have had one. Only 2.9% of activities carry this field at all.",
                    (), count=16979),
        FailureMode("organism_in_cell_line_field",
                    "A bacterium or fungus where a cell line belongs.",
                    ("Escherichia coli", "Candida albicans"), count=None),
        FailureMode("tissue_not_line",
                    "A tissue or cell type rather than a named line.",
                    ("erythrocytes", "fibroblasts"), count=None),
    ),
    model_rationale="The pipeline model, for the same reason as target_organism.",
    evidence_chars=1400,
)

#: Every reviewable field, keyed by field name.
FIELD_SPECS: Dict[str, FieldSpec] = {
    spec.field: spec for spec in (
        _NAME_SPEC, _SEQUENCE_SPEC, _ACTIVITY_TYPE_SPEC, _ACTIVITY_UNIT_SPEC,
        _VALUE_SPEC, _TARGET_ORGANISM_SPEC, _TARGET_CELL_LINE_SPEC,
    )
}

#: Fields living on an activity rather than on the peptide record.
ACTIVITY_FIELDS: Tuple[str, ...] = tuple(
    name for name, spec in FIELD_SPECS.items() if spec.scope == "activity")
PEPTIDE_FIELDS: Tuple[str, ...] = tuple(
    name for name, spec in FIELD_SPECS.items() if spec.scope == "peptide")


# --------------------------------------------------------------------------- deterministic path

@dataclass(frozen=True)
class Verdict:
    """
    One field's review result.

    :param field: Field judged.
    :param verdict: One of :data:`VERDICTS`.
    :param reason: Short justification.
    :param failure_mode: Which named mode fired, or ``""``.
    :param corrected: Proposed replacement, or ``""``.
    :param decided_by: ``"rule"`` or the model tag, so a later audit can tell them apart.
    """
    field: str
    verdict: str
    reason: str = ""
    failure_mode: str = ""
    corrected: str = ""
    decided_by: str = "rule"


#: The novelty catalogue, loaded once. Step 12 judges tens of thousands of fields and the
#: catalogue costs ~3s to build, so a per-call load would dominate the step. ``False`` means
#: "tried and unavailable", which is distinct from ``None`` meaning "not tried yet".
_CATALOGUE: Any = None


def _catalogue() -> Any:
    """
    :returns: The shared :class:`sample.novelty.Catalogue`, or ``None`` when it cannot be built.

    A missing catalogue makes the name rule silent rather than failing the step: it is one rule
    among several, and step 12 running without it is better than step 12 not running.
    """
    global _CATALOGUE
    if _CATALOGUE is None:
        from sample import novelty
        try:
            _CATALOGUE = novelty.Catalogue.load()
        except Exception as error:                        # noqa: BLE001 -- see the docstring
            logger.warning("Name-conflict rule disabled: catalogue unavailable (%s).", error)
            _CATALOGUE = False
    return _CATALOGUE or None


def _name_conflict(sequence: Any, name: str) -> Optional[Verdict]:
    """
    :param sequence: The extracted sequence.
    :param name: The extracted peptide name.
    :returns: A :class:`Verdict` when an adjudicating name contradicts the sequence, else
        ``None``.

    Only a name the catalogue says identifies something is allowed to contradict a sequence --
    :data:`sample.novelty.ADJUDICATING`. A name attached to twenty or more sequences is a class
    label or an organism that some source filed in a name column, and letting it overrule a
    sequence would turn ``Mouse-ear cress`` into a verdict on 300 peptides.
    """
    from sample import novelty

    catalogue = _catalogue()
    if catalogue is None or not name:
        return None
    verdict = catalogue.judge(sequence, name)
    if verdict.name_agreement != novelty.NAME_DISAGREES:
        return None
    found = (f"; this sequence is catalogued as {verdict.parent}"
             if verdict.parent and verdict.parent != verdict.named else "")
    return Verdict("peptide_sequence", VERDICT_WRONG,
                   f"the name '{name}' identifies {verdict.named} in this project's catalogue, "
                   f"which this sequence does not match{found}",
                   failure_mode="sequence_of_another_peptide")


#: Sibling measurements shown to the model. Enough to establish what unit the paper works in
#: without spending the context rung on a table: a reviewer needs the pattern, not the data.
_MAX_SIBLINGS: int = 8

#: Characters of the recorded context sentence passed through. Median length is 66 and the third
#: quartile 194, so this truncates only the long tail -- one sentence reached 9,364 characters,
#: which is a page of text that the extractor mislabelled as a sentence.
_MAX_CONTEXT_SENTENCE: int = 400


def record_context(record: Dict[str, Any], activity: Optional[Dict[str, Any]] = None,
                   field: str = "") -> str:
    """
    The evidence this record carries about itself, as text for a prompt.

    :param record: The peptide record.
    :param activity: The activity under review, when the field is activity-scoped.
    :param field: The field under review, used to leave out the value being judged.
    :returns: A compact block, or ``""`` when the record carries nothing useful.

    Three things, all already in the JSON and none requiring the document to be reopened:

    * **The context sentence.** Extraction saves the sentence each value came from, on 53.2% of
      activities, 32.8% of them with a readable value and unit. For a scale question that sentence
      is usually decisive and it is three orders of magnitude cheaper than locating the passage in
      the document again -- its median length is 66 characters.
    * **The sibling measurements.** A paper reports a table of MICs in one unit, so when one row
      disagrees with the others the others are the evidence. This is also what distinguishes a
      paper that genuinely works in mg/mL from a row whose micro sign was dropped.
    * **The deterministic scale diagnosis**, when :mod:`sample.scale` found something it could not
      prove. Naming the suspicion tells the model what to check; leaving it out means asking an
      open question about a field that looks perfectly well formed.

    The sentence is labelled as what the extractor recorded rather than as the document's own
    words, because it is extraction output and can itself be the thing that went wrong -- one
    measured record carries a stock-solution sentence under an MIC.
    """
    lines: List[str] = []
    if activity is not None:
        sentence = str(activity.get("context_sentence") or "").strip()
        if sentence:
            lines.append("sentence this value was taken from: "
                         + json.dumps(sentence[:_MAX_CONTEXT_SENTENCE], ensure_ascii=False))
        siblings = []
        for other in record.get("biological_activities") or ():
            if not isinstance(other, dict) or other is activity:
                continue
            siblings.append(f"{other.get('activity_type') or '?'}"
                            f" = {other.get('value_raw') or '?'}"
                            f" {other.get('activity_unit') or '?'}"
                            f" vs {other.get('target_organism') or other.get('target_cell_line') or '?'}")
            if len(siblings) >= _MAX_SIBLINGS:
                break
        if siblings:
            lines.append("other measurements on this same peptide in this same document:")
            lines += [f"  - {entry}" for entry in siblings]

        from sample import scale as scale_module
        finding = scale_module.diagnose(record, activity)
        if finding.status != scale_module.CLEAN:
            lines.append(f"automated check flagged this measurement: {finding.fault}"
                         f" ({finding.why})")
            if finding.proposed_unit:
                lines.append(f"  a rule would read the unit as: {finding.proposed_unit}")
            if not finding.rescalable:
                lines.append("  note: changing the unit would NOT fix this; the number itself or "
                             "the assay label is what is in doubt")
            lines.append(scale_module.describe_bands())
    elif field != "peptide_sequence":
        names = [str(record.get("peptide_name") or "").strip()]
        lines += [f"peptide name recorded: {name}" for name in names if name]
    return "\n".join(lines)


#: Strain and resistance-phenotype abbreviations measured in the unit column. Not organisms in
#: general: only tokens that cannot be anything else, so a real unit is never condemned.
_STRAIN_TOKENS: frozenset = frozenset({"mrsa", "mssa", "vre", "vrsa", "esbl", "acre", "are",
                                       "sds", "cre", "mdr"})


def _unit_is_not_a_unit(value: str) -> str:
    """
    :param value: The unit field's contents.
    :returns: ``"assay name"`` or ``"strain"`` when the unit is one of those, else ``""``.

    Deliberately exact-match on a token, not a substring test: `M` is a unit and `mM` is a unit,
    and a substring rule looking for `MIC` inside a unit string would also hit nothing useful
    while risking real ones.
    """
    from sample.validation_rules import VALID_ACTIVITY_TYPES

    token = (value or "").strip().lower()
    if not token:
        return ""
    if token in _STRAIN_TOKENS:
        return "strain"
    if token in {t.strip().lower() for t in VALID_ACTIVITY_TYPES if t}:
        return "assay name"
    if token in {"mic", "mbc", "mfc", "mlc", "mic50", "mic90", "zoi", "ic50"}:
        return "assay name"
    return ""


def deterministic_verdict(field: str, value: Any, record: Dict[str, Any],
                          activity: Optional[Dict[str, Any]] = None) -> Optional[Verdict]:
    """
    The answer for this field when a rule already settles it, else ``None``.

    :param field: Field to judge.
    :param value: Its current value.
    :param record: The peptide record.
    :param activity: The activity, for activity-scoped fields.
    :returns: A :class:`Verdict` decided by rule, or ``None`` to escalate to the model.

    Consulted before every model call, and it is the reason this system is affordable. Measured
    over the refined output on disk, the rules settle 1,423 incoherent units, 1,163 mojibake
    units, 882 unbounded ranges and 68 prose types with no model call at all -- 3,536 defects
    that a per-record LLM pass would otherwise have spent calls rediscovering, at a measured
    ~38s per document of model time.

    Returning ``None`` is not "the field is fine"; it means "a rule cannot tell", which is
    precisely the population the prompts above are written for.
    """
    text = "" if value is None else str(value)

    if field == "peptide_sequence":
        from sample.sequence import validate_extracted_sequence
        ok, reason = validate_extracted_sequence(value, record.get("peptide_name") or "")
        if not ok:
            return Verdict(field, VERDICT_WRONG, f"structural rule: {reason}",
                           failure_mode=reason)
        # The name catalogue settles `sequence_of_another_peptide`, which until it was wired in
        # here was the one failure mode on this field that nothing could detect -- the sequence
        # is structurally perfect, so no rule above has anything to object to, and a model shown
        # only the document cannot know that the name belongs to a different peptide. The
        # catalogue can: where the extracted name identifies exactly one known sequence and the
        # extracted sequence bears no relation to it, the pairing is wrong. Measured over the
        # refined corpus, this fires on 63 records whose sequence matches the catalogue exactly
        # under another peptide's name, plus 29 where it matches nothing -- and on the 40 cases
        # checkable by hand it was right 40 times.
        verdict = _name_conflict(value, record.get("peptide_name") or "")
        if verdict is not None:
            return verdict
        return None

    if field == "activity_type":
        from sample.preprocessing import _MAX_ACTIVITY_TYPE_CHARS
        if len(text) > _MAX_ACTIVITY_TYPE_CHARS:
            return Verdict(field, VERDICT_WRONG,
                           "longer than any real activity type; this is prose",
                           failure_mode="prose_instead_of_type")
        return None

    if field == "activity_unit":
        from sample.concentration import activity_unit_is_coherent
        # A unit that is the assay's own name, or an organism, is not a unit. The failure mode was
        # declared deterministic and counted at 100, but nothing implemented it: measured over the
        # corpus, `MIC` appears as a unit 622 times, `MBC` 111, `MFC` 28, `MRSA` 89 and `VRE` 25 --
        # 1,028 in all, every one of them escalating to a model that cannot recover what the unit
        # should have been either. Flagged, never guessed: the real unit is not in the record.
        leaked = _unit_is_not_a_unit(text)
        if leaked:
            return Verdict(field, VERDICT_WRONG,
                           f"the {leaked} '{text.strip()}' is in the unit field; a unit cannot be "
                           f"the assay's own name or its target",
                           failure_mode="metric_as_unit")
        if "�" in text:
            from sample.preprocessing import canonicalize_unit
            repaired = canonicalize_unit(text)
            if "�" not in repaired:
                return Verdict(field, VERDICT_WRONG, "micro sign lost by PDF decoding",
                               failure_mode="mojibake", corrected=repaired)
            # Still corrupt after repair: the lost character was not a micro sign, so no rule
            # can name it. Escalating is right -- the document says what it was.
            return None
        kind = (activity or {}).get("activity_type")
        coherent, why = activity_unit_is_coherent(kind, text)
        if not coherent:
            return Verdict(field, VERDICT_WRONG, f"a {kind} cannot be measured in {text}",
                           failure_mode="non_concentration_unit")
        return None

    if field == "value_raw":
        if activity is not None and "�" not in text:
            import re as _re
            has_range = bool(_re.search(r"\d\s*(?:to|[-–])\s*\d", text))
            if has_range and activity.get("value_upper") in (None, ""):
                return Verdict(field, VERDICT_WRONG,
                               "a range whose upper bound was never stored",
                               failure_mode="range_without_upper")
        return None

    if field in ("target_organism", "target_cell_line"):
        if not text.strip() or text.strip().lower() in (
                "n/a", "na", "none", "nd", "unknown", "not specified", "-", "unspecified"):
            # A rule can see that it is absent but not what it should be; that needs the text.
            return None
        return None

    return None


# --------------------------------------------------------------------------- the reviewer

def review_field(field: str, record: Dict[str, Any],
                 activity: Optional[Dict[str, Any]] = None, text: str = "",
                 model: Optional[str] = None, use_llm: bool = True,
                 timeout: Optional[float] = None,
                 variant: Optional[str] = None) -> Verdict:
    """
    Reviews one field of one record.

    :param field: A key of :data:`FIELD_SPECS`.
    :param record: The peptide record.
    :param activity: The activity, required for activity-scoped fields.
    :param text: Document text, from which evidence is selected.
    :param model: Override the spec's model.
    :param use_llm: When false, return the rule's verdict or ``unsure``; makes the whole system
        runnable with no model server, which is how the tests exercise it.
    :param timeout: Seconds for the model call.
    :param variant: Prompt variant; :data:`PROMPT_VARIANT` when omitted.
    :returns: A :class:`Verdict`.
    :raises KeyError: On an unknown field, rather than silently reviewing nothing.
    """
    spec = FIELD_SPECS[field]
    source = activity if spec.scope == "activity" else record
    if spec.scope == "activity" and activity is None:
        raise ValueError(f"{field} is activity-scoped; an activity is required")
    value = (source or {}).get(field)

    ruled = deterministic_verdict(field, value, record, activity)
    if ruled is not None:
        return ruled
    if not use_llm:
        return Verdict(field, VERDICT_UNSURE, "no rule decided and the model was not consulted")

    evidence = select_evidence(text, spec.needles(record, activity), spec.evidence_chars)
    json_context = record_context(record, activity, field)

    from sample import llm as llm_module
    try:
        raw = llm_module.llm_call(
            spec.build_messages(value, record, activity, evidence, variant, json_context),
            temperature=0.0,
            response_schema=spec.schema(variant),
            model=model or spec.model,
            max_tokens=spec.max_output_tokens,
            timeout=timeout,
            layer=f"field_review:{field}",
        )
        payload = llm_module.clean_json(raw)
    except Exception as e:
        # Fails to `unsure`, never to a verdict. A review layer that failed OPEN would silently
        # confirm every field whenever the model server was busy -- the same shape of bug as the
        # empty-completion schema failure this module's `schema` docstring describes.
        logger.debug(f"field_review {field}: {type(e).__name__}: {e}")
        return Verdict(field, VERDICT_UNSURE, f"review unavailable: {type(e).__name__}")

    if not isinstance(payload, dict):
        return Verdict(field, VERDICT_UNSURE, "model returned no object")
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        return Verdict(field, VERDICT_UNSURE, f"unrecognised verdict {verdict!r}")
    return Verdict(field, verdict,
                   reason=str(payload.get("reason") or "")[:300],
                   failure_mode=str(payload.get("failure_mode") or ""),
                   corrected=str(payload.get("corrected") or ""),
                   decided_by=model or spec.model)


def review_record(record: Dict[str, Any], text: str = "", fields: Optional[Sequence[str]] = None,
                  use_llm: bool = True, model: Optional[str] = None,
                  timeout: Optional[float] = None) -> List[Verdict]:
    """
    Reviews every requested field of one peptide record and each of its activities.

    :param record: The peptide record.
    :param text: Document text.
    :param fields: Fields to review; all of :data:`FIELD_SPECS` when omitted.
    :param use_llm: Passed through to :func:`review_field`.
    :param model: Override every spec's model.
    :param timeout: Seconds per model call.
    :returns: One verdict per field reviewed, peptide-scoped first.
    """
    wanted = tuple(fields) if fields else tuple(FIELD_SPECS)
    out: List[Verdict] = []
    for name in wanted:
        if FIELD_SPECS[name].scope != "peptide":
            continue
        out.append(review_field(name, record, None, text, model, use_llm, timeout))
    for activity in record.get("biological_activities") or []:
        if not isinstance(activity, dict):
            continue
        for name in wanted:
            if FIELD_SPECS[name].scope != "activity":
                continue
            out.append(review_field(name, record, activity, text, model, use_llm, timeout))
    return out


def group_by_model(fields: Sequence[str]) -> Dict[str, List[str]]:
    """
    The requested fields grouped by the model that reviews them.

    :param fields: Field names.
    :returns: ``{model_tag: [field, ...]}``.

    The grouping exists so a caller can run all of one model's fields before loading the next.
    Ollama evicts one model to load another, and this pipeline has already paid for ignoring
    that: 31 llama-server restarts and 23 evictions in 20 minutes when two stages disagreed
    about the model. Per-field model choice is only affordable batched this way.
    """
    groups: Dict[str, List[str]] = {}
    for name in fields:
        groups.setdefault(FIELD_SPECS[name].model, []).append(name)
    return groups


def describe_specs() -> str:
    """
    :returns: A human-readable table of every field spec, for `sample-cli review-fields --list`.

    Printed rather than documented in prose so the prompts, schemas and budgets a run will
    actually use can be read without opening the source.
    """
    lines = [f"{'field':18s} {'scope':9s} {'model':16s} {'evid':>5s} {'out':>4s}  "
             f"{'modes':>5s} {'rules':>5s}"]
    lines.append("-" * 78)
    for name, spec in FIELD_SPECS.items():
        rules = sum(1 for m in spec.failure_modes if m.deterministic)
        lines.append(f"{name:18s} {spec.scope:9s} {spec.model:16s} "
                     f"{spec.evidence_chars:>5d} {spec.max_output_tokens:>4d}  "
                     f"{len(spec.failure_modes):>5d} {rules:>5d}")
    lines.append("")
    lines.append(f"evidence budget for the bottom context rung: {EVIDENCE_BUDGET_CHARS:,} chars "
                 f"(num_ctx {DEFAULT_NUM_CTX})")
    lines.append("'rules' counts failure modes settled without a model call.")
    return "\n".join(lines)
