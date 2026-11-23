import os
import glob
import json
import random
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger("sample.validation_extras")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Injection(NamedTuple):
    """
    One corruption, and what it proves.

    :param name: Identifier used in the report.
    :param apply: Mutates a peptide record in place.
    :param tier: ``structural``, ``semantic`` or ``blind`` -- see :data:`INJECTIONS`.
    :param probes: The contract promise this corruption tests, so a miss names the rule that is
        not actually enforced rather than just a failing label.
    """
    name: str
    apply: Callable[[Dict[str, Any]], None]
    tier: str
    probes: str


# Corruptions are graded, because a suite that scores 100 % measures nothing. The previous set was
# 13 structural faults -- a DNA string in a peptide field, a negative concentration -- all of which
# any schema check catches, so its perfect score said only that the contract has a schema.
#
#   structural  malformed values. A type or range check catches these. Floor, not evidence.
#   semantic    well-formed and plausible, but contradicting a rule the contract publishes in
#               validation_rules: an assay paired with a target group it forbids, a unit whose
#               quantity that assay cannot use, an organism whose NCBI domain contradicts the
#               activity class. These are what real extraction errors look like -- an LLM reading
#               a table rarely invents "about ten", it misreads which column an assay came from.
#   blind       plausible corruptions no schema can see from the record alone, because detecting
#               them needs the source document. Scored separately and NEVER counted as failures;
#               they document the contract's blind spots, which is a result in itself.
#
# The headline sensitivity covers structural + semantic only. Mixing the blind tier in would make
# the number look worse the more honestly it was measured.
INJECTIONS: Tuple[Injection, ...] = (
    # --- structural -----------------------------------------------------------------------------
    Injection("noncanonical_residue",
              lambda p: p.update(
                  {"peptide_sequence": (p.get("peptide_sequence") or "GIGKFLHSA") + "Z1"}),
              "structural", "sequence alphabet"),
    Injection("dna_as_peptide",
              lambda p: p.update({"peptide_sequence": "ACGTACGTACGTACGTACGT"}),
              "structural", "sequence is protein, not nucleotide"),
    Injection("oversized_sequence", lambda p: p.update({"peptide_sequence": "A" * 120}),
              "structural", "length bound"),
    Injection("placeholder_name", lambda p: p.update({"peptide_name": "[insert peptide name]"}),
              "structural", "no template placeholders"),
    Injection("no_activities", lambda p: p.update({"biological_activities": []}),
              "structural", "a peptide record carries measurements"),
    Injection("unknown_activity_type",
              lambda p: _first_activity(p, {"activity_type": "unknown"}),
              "structural", "assay is from the known set"),
    Injection("noncanonical_unit", lambda p: _first_activity(p, {"activity_unit": "lM"}),
              "structural", "unit canonicalisation"),
    Injection("invented_organism",
              lambda p: _first_activity(p, {"target_organism": "Fakus organismus"}),
              "structural", "organism resolves in NCBI taxonomy"),
    Injection("negative_value", lambda p: _first_activity(p, {"value_lower": -5.0}),
              "structural", "concentrations are positive"),
    Injection("absurd_value",
              lambda p: _first_activity(p, {"value_lower": 1e9, "activity_unit": "uM"}),
              "structural", "unit range ceiling"),
    Injection("nonnumeric_value", lambda p: _first_activity(p, {"value_lower": "about ten"}),
              "structural", "value is numeric"),
    Injection("invalid_target_group", lambda p: _first_activity(p, {"target_group": "antigravity"}),
              "structural", "target group is from the known set"),
    Injection("unknown_cell_line",
              lambda p: _first_activity(
                  p, {"target_cell_line": "NotACellLine-99", "target_organism": ""}),
              "structural", "cell line resolves in Cellosaurus"),

    # --- semantic -------------------------------------------------------------------------------
    # Each of these is a rule validation_rules states explicitly, so a miss is a real gap.
    Injection("mic_with_anticancer_group",
              lambda p: _first_activity(p, {"activity_type": "MIC", "target_group": "anticancer",
                                            "target_organism": "Escherichia coli",
                                            "target_cell_line": ""}),
              "semantic", "ACTIVITY_RULES: MIC does not allow the anticancer group"),
    Injection("mic_against_cell_line",
              lambda p: _first_activity(p, {"activity_type": "MIC", "target_organism": "",
                                            "target_cell_line": "MCF-7",
                                            "target_group": "antibacterial"}),
              "semantic", "ACTIVITY_RULES: MIC target_kind is organism"),
    Injection("ic50_against_organism",
              lambda p: _first_activity(p, {"activity_type": "IC50",
                                            "target_organism": "Escherichia coli",
                                            "target_cell_line": "", "target_group": "anticancer",
                                            "activity_unit": "uM", "value_lower": 12.0}),
              "semantic", "anticancer group requires an animal-domain target"),
    Injection("antifungal_against_bacterium",
              lambda p: _first_activity(p, {"target_group": "antifungal",
                                            "target_organism": "Escherichia coli",
                                            "target_cell_line": ""}),
              "semantic", "TARGET_GROUP_DOMAINS: antifungal requires the fungi domain"),
    Injection("antibacterial_against_fungus",
              lambda p: _first_activity(p, {"target_group": "antibacterial",
                                            "target_organism": "Candida albicans",
                                            "target_cell_line": ""}),
              "semantic", "TARGET_GROUP_DOMAINS: antibacterial requires bacteria or archaea"),
    Injection("zone_with_molar_unit",
              lambda p: _first_activity(p, {"activity_type": "Zone of Inhibition",
                                            "activity_unit": "uM", "value_lower": 10.0}),
              "semantic", "ACTIVITY_RULES: a zone is a length, not a concentration"),
    Injection("mic_with_length_unit",
              lambda p: _first_activity(p, {"activity_type": "MIC", "activity_unit": "mm",
                                            "value_lower": 12.0}),
              "semantic", "ACTIVITY_RULES: MIC quantities exclude length"),
    Injection("zone_exceeds_plate",
              lambda p: _first_activity(p, {"activity_type": "Zone of Inhibition",
                                            "activity_unit": "mm", "value_lower": 500.0}),
              "semantic", "UNIT_RANGES: a zone cannot exceed the plate"),
    Injection("percentage_above_range",
              lambda p: _first_activity(p, {"activity_unit": "%", "value_lower": 950.0,
                                            "activity_type": "Hemolysis"}),
              "semantic", "UNIT_RANGES: proportion ceiling"),
    Injection("inverted_range",
              lambda p: _first_activity(p, {"value_lower": 64.0, "value_upper": 2.0,
                                            "activity_unit": "uM"}),
              "semantic", "a range runs low to high"),
    Injection("organism_in_cell_line_field",
              lambda p: _first_activity(p, {"target_cell_line": "Staphylococcus aureus",
                                            "target_organism": ""}),
              "semantic", "cell line field holds a cell line"),
    Injection("cell_line_in_organism_field",
              lambda p: _first_activity(p, {"target_organism": "HeLa", "target_cell_line": ""}),
              "semantic", "organism field holds an organism"),
    Injection("homoglyph_unit",
              lambda p: _first_activity(p, {"activity_unit": "μM"}),  # Greek mu, not micro
              "semantic", "unit canonicalisation across homoglyphs"),
    Injection("homoglyph_residue",
              lambda p: p.update(
                  {"peptide_sequence": "А" + (p.get("peptide_sequence") or "GIGKFLHSA")[1:]}),
              "semantic", "sequence alphabet across homoglyphs"),  # Cyrillic A
    Injection("whitespace_activity_type",
              lambda p: _first_activity(p, {"activity_type": "  mic  "}),
              "semantic", "assay name is trimmed and matched case-insensitively"),
    Injection("comma_decimal", lambda p: _first_activity(p, {"value_lower": "0,5"}),
              "semantic", "European decimal comma is not read as a thousands separator"),

    # --- blind ----------------------------------------------------------------------------------
    # Undetectable without the source document. Reported, never scored as failures.
    Injection("magnitude_shift_in_range",
              lambda p: _first_activity(p, {"value_lower": 8000.0, "activity_unit": "uM"}),
              "blind", "a 1000x transcription error that stays inside the valid range"),
    Injection("plausible_unit_swap",
              lambda p: _first_activity(p, {"activity_unit": "nM", "value_lower": 4.0}),
              "blind", "uM read as nM, both valid for the assay"),
    Injection("truncated_sequence",
              lambda p: p.update(
                  {"peptide_sequence": (p.get("peptide_sequence") or "GIGKFLHSAKKFGKA")[:8]}),
              "blind", "a sequence cut short is still a valid sequence"),
    Injection("swapped_valid_organism",
              lambda p: _first_activity(p, {"target_organism": "Staphylococcus aureus",
                                            "target_cell_line": "", "target_group": "antibacterial"}),
              "blind", "the wrong organism, but a real one consistent with the assay"),
)

#: Name -> injection, for callers that address one by name.
_INJECTIONS: Dict[str, Injection] = {inj.name: inj for inj in INJECTIONS}

#: Tiers whose misses count against the headline sensitivity.
SCORED_TIERS: Tuple[str, ...] = ("structural", "semantic")


def _hard_negatives(seed: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Builds valid records that superficially look wrong.

    Specificity is the half of this measurement that catches over-eager validation, and ordinary
    records are a weak control because they are unremarkable. These are all things the contract
    documents as legitimate, so any rejection here is the contract destroying real data.

    :param seed: A contract-valid record to derive variants from.
    :returns: ``(name, record)`` pairs.
    """
    def variant(updates: Dict[str, Any], activity: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
        record = json.loads(json.dumps(seed))
        record.update(updates)
        if activity:
            _first_activity(record, activity)
        return record

    # Each variant sets EVERY field the assay constrains, not just the interesting one. Setting
    # only the interesting field leaves the seed's own group and target in place, which produces a
    # genuinely invalid combination -- and then the contract's correct rejection is scored as a
    # false positive. Three of these were wrong that way on first writing: a Hemolysis variant that
    # kept `antibacterial` (Hemolysis allows only `other`), and a ratio variant naming an assay
    # that does not exist.
    cases: List[Tuple[str, Dict[str, Any]]] = [
        # strict_schema publishes a measurement with no stated target on purpose: incomplete, not
        # malformed.
        ("no_stated_target", variant({}, {"target_organism": "", "target_cell_line": ""})),
        # UNIT_RANGES admits >100 % deliberately: normalised hemolysis assays really do report it.
        # Hemolysis allows only the 'other' group, and the target is left empty on purpose --
        # these assays run on primary erythrocytes, which have no Cellosaurus accession, so naming
        # a "cell line" here is itself a violation. That is what made this control fail before.
        ("hemolysis_115_percent",
         variant({}, {"activity_type": "Hemolysis", "activity_unit": "%", "value_lower": 115.0,
                      "target_group": "other", "target_organism": "", "target_cell_line": ""})),
        # A ratio of two concentrations has no unit; the empty unit is a first-class UNIT_RANGES
        # row. TI takes no target group and either target kind.
        ("dimensionless_ratio",
         variant({}, {"activity_type": "TI", "activity_unit": "", "value_lower": 12.0,
                      "value_upper": None, "target_group": "other"})),
        # Short peptides are real; several natural AMPs are under 10 residues.
        ("very_short_peptide", variant({"peptide_sequence": "GIGKF"})),
    ]

    # Only legitimate when something else identifies the record. With neither name nor sequence
    # there is nothing to publish, and rejection is correct.
    if str(seed.get("peptide_name") or "").strip():
        cases.append(("name_only_no_sequence", variant({"peptide_sequence": ""})))

    # Deliberately NOT included: a sequence containing X. The contract restricts sequences to the
    # 20 canonical residues and clears anything else, which is a consistent policy rather than a
    # defect -- `noncanonical_residue` above tests the same rule from the other side. Whether X
    # should be admitted is a contract decision, not something this suite can settle, and asserting
    # it here would manufacture a false finding.
    return cases


def _first_activity(peptide: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Applies `updates` to the record's first activity, creating one if it has none."""
    acts = peptide.get("biological_activities") or []
    if not acts:
        acts = [{"activity_type": "MIC", "activity_unit": "uM", "value_lower": 4.0,
                 "target_organism": "Escherichia coli", "target_cell_line": "",
                 "target_group": "antibacterial"}]
        peptide["biological_activities"] = acts
    acts[0].update(updates)


def _contract_rejects(record: Dict[str, Any]) -> bool:
    """
    Whether the published contract ACTS on this record: refuses it, drops an activity, clears a
    field, or rewrites one into canonical form. The single detector used for BOTH the injected
    trials and the untouched controls, so sensitivity and specificity are measured the same way.

    Normalization counts as acting, not as a miss. A corrupted unit like "lM" is not rejected --
    it is canonicalized to "uM", which is the better outcome and exactly what the contract is for.
    Counting only outright rejection scored that as a failure to detect, when the corruption had in
    fact been neutralized. What matters is that the published record differs from the corrupted
    one; an untouched valid record is already canonical, so nothing changes and it stays clean.

    Deliberately does NOT consult validate_extraction_logic. That function rejects any empty string
    anywhere in the record, and the contract writes "" on purpose for an unstated organism, cell
    line or operator -- so including it marked every record as failing and reported 100% detection
    no matter what was injected. Measuring the contract means running the contract.
    """
    from sample.strict_schema import enforce_peptide

    clean, violations = enforce_peptide(record)
    if clean is None or violations:
        return True
    before = record.get("biological_activities") or []
    after = clean.get("biological_activities") or []
    if len(after) < len(before):
        return True
    # Any contract-governed field rewritten on the way through.
    for field in ("peptide_sequence", "peptide_name"):
        if str(record.get(field) or "") != str(clean.get(field) or ""):
            return True
    for original, published in zip(before, after):
        for field in ("activity_type", "activity_unit", "value_raw_unit", "value_lower",
                      "target_organism", "target_cell_line", "target_group", "value_operator"):
            if str(original.get(field) or "") != str(published.get(field) or ""):
                return True
    return False


def inject_error(peptide_data: Dict[str, Any],
                 error_type: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Injects one corruption into a copy of a PEPTIDE record.

    :param peptide_data: A contract-valid peptide record to corrupt. Copied, never mutated.
    :param error_type: Which corruption to apply; a key of ``_INJECTIONS``. ``None`` picks one
        at random.
    :returns: ``(mutated_record, error_type)``.

    Operates on a peptide record, not a document envelope. The previous version was handed whole
    `{"metadata": ..., "peptides": [...]}` documents and wrote `peptide_sequence` at the top level,
    where no validator reads it -- so the suite measured detection of a field that was never
    published, and its result said nothing about the real records.
    """
    error_type = error_type or random.choice(sorted(_INJECTIONS))
    mutated = json.loads(json.dumps(peptide_data))
    _INJECTIONS[error_type].apply(mutated)
    return mutated, error_type

def validate_error_injection(input_dir: str, output_report: str,
                             num_samples: int = 2000) -> Optional[Dict[str, Any]]:
    """
    Measures whether the published contract actually catches the faults it claims to.

    Injects every corruption in :data:`INJECTIONS` into contract-valid records and reports
    detection rates per type and per tier, then runs the same detector over untouched records and
    over deliberately awkward valid ones -- a detector that rejects everything scores 100 %
    sensitivity and is useless.

    The headline number covers the ``structural`` and ``semantic`` tiers. The ``blind`` tier is
    reported separately and never counted as failure: those corruptions cannot be detected from
    the record alone, and folding them in would make the score fall the more honestly the suite
    was written.

    :param input_dir: Directory of extraction JSONs to draw seed records from.
    :param output_report: Path for the JSON report.
    :param num_samples: Trials to run, distributed evenly across injection types.
    :returns: The report dict, or ``None`` when no contract-valid seed records were found.
    """
    logger.info("Starting error injection validation test...")
    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir} for error injection.")
        return

    # Collect PEPTIDE records, unwrapping the document envelope. Only records that already satisfy
    # the contract are used as seeds: injecting into an already-invalid record cannot show whether
    # the injected fault is the one being caught.
    from sample.strict_schema import enforce_peptide

    samples = []
    unreadable_files = 0
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            unreadable_files += 1
            continue
        peptides = data.get("peptides") if isinstance(data, dict) else data
        for pep in (peptides or []):
            if isinstance(pep, dict):
                clean, _ = enforce_peptide(pep)
                if clean is not None:
                    samples.append(clean)

    if unreadable_files:
        logger.warning(f"Error injection seed collection: {unreadable_files} of {len(json_files)} "
                       f"JSON files unreadable.")

    if not samples:
        logger.warning(
            f"No contract-valid peptide records found in {input_dir}; nothing to inject into. "
            "Run `sample-cli revise` first."
        )
        return

    # Every corruption is tested against every sample slot in rotation, so each type gets an equal
    # and adequately-powered number of trials. Sampling types at random left rare types with one or
    # two trials and a detection rate that was mostly noise.
    types = [inj.name for inj in INJECTIONS]
    trial_count = max(num_samples, len(types))
    trials = [(random.choice(samples), types[i % len(types)]) for i in range(trial_count)]

    results: Dict[str, Any] = {
        "total_tested": len(trials),
        "trials_per_type": trial_count // len(types),
        "errors_detected": 0,
        "errors_missed": 0,
        "detections_by_type": {},
        "misses_by_type": {},
    }
    by_tier: Dict[str, Dict[str, int]] = {
        tier: {"tested": 0, "detected": 0} for tier in ("structural", "semantic", "blind")}

    for record, err_type in trials:
        tier = _INJECTIONS[err_type].tier
        by_tier[tier]["tested"] += 1
        mutated, err_type = inject_error(record, err_type)
        try:
            caught = _contract_rejects(mutated)
        except Exception as e:
            # A crash is a defect in the validator, not a detection. Counting it as one is how the
            # previous version could report high sensitivity while catching nothing: it wrapped the
            # whole block in `except (ContentValidationError, Exception)`.
            caught = False
            logger.error(f"Validator raised {type(e).__name__} on '{err_type}': {e}")
            results.setdefault("validator_errors", {})
            results["validator_errors"][err_type] = results["validator_errors"].get(err_type, 0) + 1

        if caught:
            by_tier[tier]["detected"] += 1
            results["errors_detected"] += 1
            results["detections_by_type"][err_type] = results["detections_by_type"].get(err_type, 0) + 1
        else:
            results["errors_missed"] += 1
            results["misses_by_type"][err_type] = results["misses_by_type"].get(err_type, 0) + 1

    for tier, counts in by_tier.items():
        counts["detection_rate_percentage"] = (
            100.0 * counts["detected"] / counts["tested"] if counts["tested"] else 0.0)
    results["by_tier"] = by_tier

    # The headline covers only the tiers the contract claims to police. Blind-tier corruptions
    # cannot be seen without the source document, so counting them would penalise the suite for
    # testing honestly -- and would let anyone raise the score by deleting the hard cases.
    scored_tested = sum(by_tier[t]["tested"] for t in SCORED_TIERS)
    scored_detected = sum(by_tier[t]["detected"] for t in SCORED_TIERS)
    rate = 100.0 * scored_detected / scored_tested if scored_tested else 0.0
    results["scored_tested"] = scored_tested
    results["scored_detected"] = scored_detected
    results["detection_rate_percentage"] = rate

    # Named so a reader sees which promises are unenforced rather than a bare percentage. A
    # structural or semantic miss is a gap in the contract; a blind miss is expected.
    results["unenforced_rules"] = sorted(
        {_INJECTIONS[name].probes for name in results["misses_by_type"]
         if _INJECTIONS[name].tier in SCORED_TIERS})
    results["known_blind_spots"] = sorted(
        {_INJECTIONS[name].probes for name in results["misses_by_type"]
         if _INJECTIONS[name].tier == "blind"})

    # Specificity control. A detector that rejects everything scores 100% sensitivity and is
    # useless, so the same detector is run over untouched records: any hit here is a false positive
    # and means the contract is destroying valid data. Without this the headline rate is
    # uninterpretable -- which is exactly how the previous suite reported high sensitivity while
    # its detector was flagging every record it saw.
    control_size = min(len(samples), max(num_samples // 2, 200))
    controls = [samples[i % len(samples)] for i in range(control_size)]
    false_positives = 0
    for record in controls:
        try:
            if _contract_rejects(record):
                false_positives += 1
        except Exception as e:
            # Fails safe rather than silent: a check that cannot even run is treated as a rejection
            # and counted below, not dropped. The debug line exists so a spike in false_positives
            # can be traced back to "the checker crashed" rather than "the contract is stricter".
            logger.debug(f"_contract_rejects raised on a control record: {e}")
            false_positives += 1
    results["controls_tested"] = len(controls)
    results["false_positives"] = false_positives
    results["false_positive_rate_percentage"] = (
        100 * false_positives / len(controls) if controls else 0.0)

    # Hard negatives: valid records that look wrong. Untouched records are a weak control because
    # they are unremarkable -- these are the cases where an over-eager rule would silently delete
    # real data, and each is something the contract documents as legitimate.
    hard_negative_failures: Dict[str, int] = {}
    hard_negative_trials = 0
    for seed in random.sample(samples, min(len(samples), 50)):
        for name, record in _hard_negatives(seed):
            hard_negative_trials += 1
            try:
                rejected = _contract_rejects(record)
            except Exception as e:
                # Same fail-safe reasoning as the false-positive control above: a crashed check
                # counts as a failure rather than being dropped from the trial count.
                logger.debug(f"_contract_rejects raised on hard negative {name!r}: {e}")
                rejected = True
            if rejected:
                hard_negative_failures[name] = hard_negative_failures.get(name, 0) + 1
    results["hard_negatives_tested"] = hard_negative_trials
    results["hard_negative_failures"] = hard_negative_failures
    results["hard_negative_failure_rate_percentage"] = (
        100.0 * sum(hard_negative_failures.values()) / hard_negative_trials
        if hard_negative_trials else 0.0)

    for err_type, missed in sorted(results["misses_by_type"].items(), key=lambda kv: -kv[1]):
        tier = _INJECTIONS[err_type].tier
        message = (f"Contract did not catch '{err_type}' in {missed} trial(s) "
                   f"-- {_INJECTIONS[err_type].probes}")
        if tier == "blind":
            logger.info(f"[blind spot, expected] {message}")
        else:
            logger.warning(f"[{tier} GAP] {message}")

    if false_positives:
        logger.warning(
            f"Contract rejected {false_positives}/{len(controls)} UNCORRUPTED records -- "
            "it is removing valid data."
        )
    for name, count in sorted(hard_negative_failures.items(), key=lambda kv: -kv[1]):
        logger.warning(
            f"Contract rejected the valid case '{name}' in {count} trial(s) -- "
            "this is documented as legitimate, so rejecting it destroys real data.")

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    tiers = results["by_tier"]
    logger.info(
        f"Error injection complete over {results['total_tested']} trials "
        f"({results['trials_per_type']} per corruption type).")
    logger.info(
        f"  Sensitivity (structural+semantic): {rate:.2f}% "
        f"({scored_detected}/{scored_tested} caught)")
    logger.info(
        f"    structural: {tiers['structural']['detection_rate_percentage']:.2f}%  "
        f"semantic: {tiers['semantic']['detection_rate_percentage']:.2f}%  "
        f"blind (not scored): {tiers['blind']['detection_rate_percentage']:.2f}%")
    logger.info(
        f"  False positives: {results['false_positive_rate_percentage']:.2f}% on untouched "
        f"records, {results['hard_negative_failure_rate_percentage']:.2f}% on hard negatives")
    if results["unenforced_rules"]:
        logger.warning(
            f"  {len(results['unenforced_rules'])} published rule(s) are NOT enforced: "
            + "; ".join(results["unenforced_rules"]))
    logger.info(f"  Report saved to {output_report}")
    return results

def run_manual_curation(*args: Any, **kwargs: Any) -> None:
    """
    REMOVED. This stage is now sample.revision.run_revision (CLI: `sample-cli revise`).

    The previous implementation copied the first `limit` files (default: 10 of ~46,000) and stamped
    each record with `curated: true`, `curator: "Expert Curator"` and a fixed
    `curation_timestamp` -- an assertion that a named human expert had reviewed data nobody had
    looked at. Any consumer of the database, or any reviewer of a paper built on it, would read
    those fields as evidence of manual curation. Its output directory was also read by no later
    step, so the claim was the only thing it produced.

    The replacement does the real work the step was named for -- enforcing the output contract, with
    an LLM repair pass -- and records what it actually is: an automated revision stage. If genuine
    human curation is added later it must be a separate, honestly-labelled provenance field.
    """
    raise NotImplementedError(
        "run_manual_curation has been replaced by sample.revision.run_revision "
        "(`sample-cli revise`), which records automated-revision provenance rather than "
        "asserting expert curation that did not happen."
    )
