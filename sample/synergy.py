"""
Peptide--antibiotic synergy: extraction from DBAASP and per-pathogen models.

Synergy was previously out of reach. It is measured on *combinations* -- a checkerboard assay
varying a peptide and an antibiotic together -- while the release stored only single-agent
activities, and ``FIC`` appeared in :mod:`sample.validation_rules` with nothing populating it.
Predicting synergy looked like a data-acquisition project rather than a modelling one.

It is not: DBAASP already ships the combination data, in a ``synergies`` block that the loader
never read. **5,507 measurements across 815 peptides, 149 antibiotics and 514 target strains**,
each with the peptide's MIC alone and in combination, the antibiotic's MIC alone and in
combination, and the fractional inhibitory concentration index.

Two properties make this unusually good training material:

* **The negatives are real.** Every other model in this project separates peptides from shuffled
  decoys, which answers "does this look like a peptide at all". Here a non-synergistic combination
  is an actual experiment that was run and found not to synergise, so the model learns the
  question that was asked.
* **The classes are nearly balanced** -- 2,702 synergistic against 2,548 not, at the conventional
  FICI cutoff -- so accuracy is meaningful without resampling.

The FICI cutoffs follow the standard checkerboard interpretation: synergy at or below 0.5,
additive up to 1, indifferent up to 4, antagonistic above. The 0.5 boundary is the one the
literature treats as a claim of synergy, so it is the classification target.
"""
import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("sample.synergy")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: FICI at or below this is reported as synergy. The conventional checkerboard cutoff.
SYNERGY_CUTOFF: float = 0.5

#: FICI band edges, as ``(upper_bound, label)`` in ascending order.
FICI_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.5, "synergy"),
    (1.0, "additive"),
    (4.0, "indifferent"),
    (float("inf"), "antagonism"),
)

#: Minimum measurements before a pathogen gets its own model. Below this, cross-validated scores
#: are too unstable to mean anything and the model would mostly memorise its few peptides.
MIN_ROWS_PER_PATHOGEN: int = 80

#: Antibiotics rare enough to be pooled into an ``other`` indicator rather than given a column of
#: their own; below this count a one-hot column is a peptide identifier in disguise.
MIN_ANTIBIOTIC_COUNT: int = 15


def _slug(value: str) -> str:
    """
    Filesystem-safe lowercase slug for a label.

    Delegates to :func:`sample.ml_models._slug` rather than reimplementing it: model namespaces
    are directory paths shared with that module, and two slug functions would eventually disagree
    about one label and split a model's namespace in two. Imported inside the call because
    ``ml_models`` pulls in scikit-learn, which this module should not require merely to parse a
    DBAASP card.

    :param value: Any label going into a directory or column name.
    :returns: The slug.
    """
    from sample.ml_models import _slug as _shared_slug

    return _shared_slug(value)


def parse_measurement(value: Any) -> Optional[float]:
    """
    Parses a concentration or index that may carry a comparison operator.

    DBAASP writes censored measurements as ``<0.5`` or ``>128``. The bound is kept as the value:
    it is the only number available, and discarding the row would drop exactly the strongest and
    weakest results, biasing the training set toward the middle.

    :param value: The raw field.
    :returns: The number, or ``None`` when the field holds no usable value.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("na", "n/a", "-", "nd", "none"):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text.replace(",", "."))
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if number == number else None  # reject NaN


def classify_fici(fici: Optional[float]) -> Optional[str]:
    """
    :param fici: A fractional inhibitory concentration index.
    :returns: The interaction band, or ``None`` when the index is missing.
    """
    if fici is None:
        return None
    for upper, label in FICI_BANDS:
        if fici <= upper:
            return label
    return "antagonism"


def extract_synergy_records(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Reads every synergy measurement out of the DBAASP cards.

    :param directory: DBAASP JSON directory; the packaged location when omitted.
    :returns: One record per measurement, with the peptide sequence attached so the rows can be
        featurised without a second pass over the cards.
    """
    directory = directory or os.path.join(DATA_DIR, "antimicrobial_databases", "dbaasp")
    records: List[Dict[str, Any]] = []

    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                card = json.load(handle)
        except (OSError, ValueError) as e:
            logger.debug(f"Skipping unreadable DBAASP card {path}: {e}")
            continue

        sequence = (card.get("sequence") or "").strip().upper()
        for entry in card.get("synergies") or []:
            fici = parse_measurement(entry.get("fici"))
            records.append({
                "dbaasp_id": card.get("dbaaspId"),
                "sequence": sequence,
                "target_organism": ((entry.get("targetSpecie") or {}).get("name") or "").strip(),
                "antibiotic": (entry.get("antibioticName") or "").strip(),
                "measure": ((entry.get("activityMeasureGroup") or {}).get("name") or "").strip(),
                "unit": ((entry.get("unit") or {}).get("name") or "").strip(),
                "peptide_alone": parse_measurement(entry.get("peptideActivity")),
                "peptide_combined": parse_measurement(entry.get("peptideAntibioticActivity")),
                "antibiotic_alone": parse_measurement(entry.get("antibioticActivity")),
                "antibiotic_combined": parse_measurement(entry.get("antibioticPeptideActivity")),
                "fici": fici,
                "interaction": classify_fici(fici),
                "is_synergistic": None if fici is None else int(fici <= SYNERGY_CUTOFF),
                # The source's 1-based pointer into its own article list, so a prediction can cite
                # the measurement it learned from.
                "reference_index": str(entry.get("reference") or "") or None,
            })

    logger.info(f"Extracted {len(records)} synergy measurements from DBAASP.")
    return records


def recompute_fici(record: Dict[str, Any]) -> Optional[float]:
    """
    Recomputes FICI from the four MICs, for cross-checking the stored value.

    ``FICI = MIC_peptide_combined / MIC_peptide_alone + MIC_antibiotic_combined /
    MIC_antibiotic_alone``. DBAASP stores a curated ``fici`` field, and this exists to find rows
    where the two disagree -- a disagreement means one of the five numbers was mis-transcribed,
    and the row should not be trained on.

    :param record: A record from :func:`extract_synergy_records`.
    :returns: The recomputed index, or ``None`` when any of the four MICs is missing or a
        denominator is zero.
    """
    pa, pc = record.get("peptide_alone"), record.get("peptide_combined")
    aa, ac = record.get("antibiotic_alone"), record.get("antibiotic_combined")
    if None in (pa, pc, aa, ac) or not pa or not aa:
        return None
    return pc / pa + ac / aa


def normalise_pathogen(name: str) -> str:
    """
    Collapses a strain designation to its species binomial.

    Synergy is reported against strains -- ``Escherichia coli ATCC 25922``, ``Pseudomonas
    aeruginosa 910``, ``Pseudomonas aeruginosa Ps4`` -- which fragments 514 labels across what are
    really a few dozen species. Grouping to the binomial is what makes a per-pathogen model
    possible: without it, no single target has enough measurements to train on.

    :param name: The target organism as DBAASP wrote it.
    :returns: ``Genus species``, or the cleaned original when it does not parse as a binomial.
    """
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        return ""
    parts = cleaned.split(" ")
    if len(parts) >= 2 and parts[0][:1].isupper() and parts[1][:1].islower():
        return f"{parts[0]} {parts[1]}"
    return cleaned


def build_dataframe(records: Optional[Sequence[Dict[str, Any]]] = None,
                    check_fici: bool = True, tolerance: float = 0.25) -> Any:
    """
    Assembles the synergy training table.

    :param records: Records from :func:`extract_synergy_records`; extracted when omitted.
    :param check_fici: Drop rows whose stored FICI disagrees with the value implied by their four
        MICs. Only applied where all four are present, so it removes transcription errors without
        discarding the many rows that report FICI alone.
    :param tolerance: Relative disagreement tolerated before a row is dropped.
    :returns: A :class:`pandas.DataFrame` with one row per usable measurement.
    """
    import pandas as pd

    frame = pd.DataFrame(list(records if records is not None else extract_synergy_records()))
    if frame.empty:
        return frame

    frame = frame[frame["fici"].notna()].copy()
    frame["pathogen"] = frame["target_organism"].map(normalise_pathogen)
    frame = frame[(frame["pathogen"] != "") & (frame["sequence"].str.len() >= 5)]

    if check_fici:
        recomputed = frame.apply(recompute_fici, axis=1)
        comparable = recomputed.notna()
        disagreement = (recomputed - frame["fici"]).abs() / frame["fici"].clip(lower=1e-6)
        drop = comparable & (disagreement > tolerance)
        if int(drop.sum()):
            logger.info(f"Dropping {int(drop.sum())} rows whose stored FICI disagrees with the "
                        f"value implied by their MICs (>{tolerance:.0%} apart).")
        frame = frame[~drop]

    logger.info(f"Synergy table: {len(frame)} rows, {frame['sequence'].nunique()} peptides, "
                f"{frame['pathogen'].nunique()} pathogens, "
                f"{frame['antibiotic'].nunique()} antibiotics.")
    return frame.reset_index(drop=True)


def featurise(frame: Any, antibiotics: Optional[Sequence[str]] = None,
              use_network: bool = True) -> Tuple[Any, List[str]]:
    """
    Builds the feature matrix: peptide descriptors plus the antibiotic's Morgan fingerprint.

    The peptide is described by its sequence-derived descriptors, deliberately -- the measured
    single-agent MICs are more predictive, but a user asking "will this peptide synergise with
    gentamicin against *P. aeruginosa*" has a sequence and nothing else. The antibiotic is
    represented by its ECFP (:mod:`sample.antibiotics`), not a one-hot indicator: a fingerprint
    carries chemical structure, so the model can score an antibiotic it never trained on by its
    similarity to ones it did, which a per-antibiotic indicator column can never do.

    :param frame: The table from :func:`build_dataframe`.
    :param antibiotics: Retained for signature compatibility; no longer used to pick indicator
        columns, since the fingerprint block is the same width for every antibiotic.
    :param use_network: Passed to :func:`sample.antibiotics.build_fingerprints` -- ``False`` uses
        only the cached SMILES for a reproducible offline run.
    :returns: ``(features, column_names)``.
    """
    import pandas as pd

    from sample.antibiotics import build_fingerprints, fingerprint_columns
    from sample.descriptors import compute_all_descriptors

    descriptors = pd.DataFrame(
        [compute_all_descriptors(seq) for seq in frame["sequence"]], index=frame.index)
    descriptors = descriptors.select_dtypes(include=["number"])

    fingerprints = build_fingerprints(frame["antibiotic"], use_network=use_network)
    fp_columns = fingerprint_columns()
    zero = [0.0] * len(fp_columns)
    fp_rows = [list(fingerprints.get(str(name or "").strip(), zero)) for name in frame["antibiotic"]]
    fp_frame = pd.DataFrame(fp_rows, columns=fp_columns, index=frame.index)

    features = pd.concat([descriptors, fp_frame], axis=1)
    features = features.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    return features, list(features.columns)


def _model_root(model_dir: Optional[str] = None) -> str:
    """
    Resolves the synergy namespace, which lives under ``models_retired`` rather than ``models``.

    These models are retired: none of the nine regression scopes reached the held-out R2 gate
    (every one scored <= 0.073, most negative), and seven of the nine classification scopes
    missed theirs too -- S. aureus landed at ROC-AUC 0.482, below chance. Only A. baumannii and
    K. pneumoniae classification ever passed. They are kept on disk for provenance and possible
    re-training, but deliberately outside ``model_registry.default_model_dir()`` so the web
    server and REST API cannot offer them. See KNOWN_ISSUES.md.

    :param model_dir: Model directory; the retired-model default when omitted.
    :returns: The directory holding the synergy namespace.
    """
    return os.path.join(
        model_dir or os.path.join(DATA_DIR, "sample_final_release", "models_retired"),
        "synergy")


def available_models(model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lists the trained synergy models.

    :param model_dir: Model directory; the release default when omitted.
    :returns: One entry per model, with its pathogen, metrics and the antibiotics it was trained
        on. Sorted with the pooled model first, then by pathogen.
    """
    import glob as _glob

    import joblib

    models: List[Dict[str, Any]] = []
    for path in sorted(_glob.glob(os.path.join(_model_root(model_dir), "**", "model.pkl"),
                                  recursive=True)):
        try:
            bundle = joblib.load(path)
        except Exception as e:
            logger.warning(f"Could not read synergy model {path}: {e}")
            continue
        pathogen = bundle.get("pathogen", "")
        models.append({
            "model_id": os.path.relpath(os.path.dirname(path),
                                        os.path.dirname(_model_root(model_dir))).replace(
                                            os.sep, "/"),
            "pathogen": pathogen,
            "cutoff": bundle.get("cutoff", SYNERGY_CUTOFF),
            "antibiotics": bundle.get("antibiotics", []),
            "metrics": bundle.get("metrics", {}),
        })
    models.sort(key=lambda m: (m["pathogen"] != "all_pathogens", m["pathogen"]))
    return models


def predict_synergy(sequences: Sequence[str], antibiotic: str, pathogen: str = "all_pathogens",
                    model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scores peptide/antibiotic combinations for synergy.

    :param sequences: Peptide sequences in one-letter code.
    :param antibiotic: The partner antibiotic, by name.
    :param pathogen: Species binomial, or ``all_pathogens`` for the pooled model. A pathogen with
        no model of its own falls back to the pooled one, and the response says which answered.
    :param model_dir: Model directory; the release default when omitted.
    :returns: One record per sequence: the probability of synergy, the label at the FICI cutoff,
        and which model produced it.
    :raises FileNotFoundError: When no synergy model has been trained.
    """
    import joblib
    import pandas as pd

    root = _model_root(model_dir)
    wanted = _slug(pathogen or "all_pathogens")
    scope = "general" if wanted == "all_pathogens" else "organism_specific"
    path = os.path.join(root, scope, wanted, "fici", "classification", "model.pkl")

    if not os.path.exists(path):
        # Fall back to the pooled model rather than refusing: a pathogen without its own model is
        # the common case, and the pooled one is a defensible answer as long as the response says
        # that is what happened.
        path = os.path.join(root, "general", "all_pathogens", "fici", "classification",
                            "model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No synergy model under {root}. Run `sample-cli train-synergy` (not a numbered pipeline step).")

    bundle = joblib.load(path)
    model = bundle["model"]
    columns = bundle["feature_columns"]
    used_pathogen = bundle.get("pathogen", "all_pathogens")

    cleaned = [re.sub(r"[^A-Za-z]", "", str(s or "")).upper() for s in sequences]
    frame = pd.DataFrame({
        "sequence": cleaned,
        "antibiotic": [antibiotic] * len(cleaned),
    })

    features, _ = featurise(frame, antibiotics=bundle.get("antibiotics") or [])
    # Column order and membership must match training exactly; a descriptor that failed to compute
    # for an odd sequence would otherwise shift every column after it.
    features = features.reindex(columns=columns, fill_value=0.0)
    probabilities = model.predict_proba(features.to_numpy())[:, 1]

    cutoff = bundle.get("cutoff", SYNERGY_CUTOFF)
    return [{
        "sequence": sequence,
        "antibiotic": antibiotic,
        "pathogen": used_pathogen,
        "requested_pathogen": pathogen,
        "model_used": os.path.relpath(os.path.dirname(path),
                                      os.path.dirname(root)).replace(os.sep, "/"),
        "probability_synergy": float(probability),
        "predicted_interaction": "synergy" if probability >= 0.5 else "not synergy",
        "fici_cutoff": cutoff,
        "error": None if sequence else "empty sequence",
    } for sequence, probability in zip(cleaned, probabilities)]


def train_synergy_models(model_dir: Optional[str] = None, report_path: Optional[str] = None,
                         min_rows: int = MIN_ROWS_PER_PATHOGEN) -> Dict[str, Any]:
    """
    Trains, per pathogen (plus a pooled model across all of them), a synergy CLASSIFIER (is FICI
    at or below the cutoff) AND a FICI REGRESSOR (predict the continuous index). The regressor is
    the more informative model -- "how synergistic", not just "synergistic or not" -- and is also
    scored as a classifier at the same cutoff so the two can be compared directly.

    Both go through :func:`sample.ml_models.fit_and_validate` -- the same validated-fit core that
    steps 45-46's own models use: a fixed holdout carved out before any model selection (never a full-data
    `cross_val_predict`, which reports in-sample-flavoured numbers), CV-based hyperparameter
    selection on the train/selection portion only, XGBoost as the base estimator, bootstrap 95% CI,
    y-randomization, a null-significance test, and :func:`sample.ml_models.assess_quality`'s same
    quality gate -- a model that does not clear it is reported but never written to disk.

    Grouping is by near-identity peptide cluster (:func:`sample.ml_models._cluster_groups`), not
    exact sequence. Without that, an analog peptide (a point mutant, tested against a different
    antibiotic) can land one member in train and its sibling in test, and the model scores well by
    recognising the analog rather than by predicting synergy -- the same leakage steps 45-46 guard
    against for its own tasks.

    :param model_dir: Directory to write models into; ``data/sample_final_release/models`` when
        omitted, so they join the existing registry namespace.
    :param report_path: Where to write the metrics report.
    :param min_rows: Minimum measurements for a pathogen to get its own model.
    :returns: The report: one entry per (pathogen, learning_task) attempted, kept or not.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from sample import ml_models

    model_dir = model_dir or os.path.join(DATA_DIR, "sample_final_release", "models")
    report_path = report_path or os.path.join(
        DATA_DIR, "sample_final_release", "synergy_report.json")
    # Distinct from ml_models.py's own default (``data/families``, step 35's working directory) and
    # from any other caller's isolated directory -- see _cluster_groups' own docstring on why two
    # concurrent callers must not share PepMatch scratch files.
    cluster_work_dir = os.path.join(DATA_DIR, "synergy_family_scratch")

    frame = build_dataframe()
    if frame.empty:
        logger.warning("No synergy measurements available; nothing to train.")
        return {}

    counts = frame["antibiotic"].value_counts()
    antibiotics = sorted(counts[counts >= MIN_ANTIBIOTIC_COUNT].index)

    groups_to_train: List[Tuple[str, Any]] = [("all_pathogens", frame)]
    for pathogen, subset in frame.groupby("pathogen"):
        if len(subset) >= min_rows and subset["is_synergistic"].nunique() > 1:
            groups_to_train.append((pathogen, subset))

    def _write_if_passed(record: Dict[str, Any], namespace: str, model: Any,
                         columns: List[str], learning_task: str) -> None:
        bundle: Dict[str, Any] = {
            "model": model, "feature_columns": columns, "metrics": record,
            "pathogen": pathogen, "antibiotics": list(antibiotics), "cutoff": SYNERGY_CUTOFF,
        }
        if learning_task == "regression":
            bundle["learning_task"] = "regression"
        metric = ml_models._primary_metric(learning_task == "regression")
        status = "keep" if record["quality_passed"] else "drop"
        logger.info(f"  [{status}] {namespace}  n={record['n_samples']:>5} "
                   f"{metric}={record['holdout'].get(metric, float('nan')):.3f}"
                   + ("" if record["quality_passed"] else f"  <- {record['quality_failures'][0]}"))
        if not record["quality_passed"]:
            return
        destination = os.path.join(model_dir, *namespace.split("/"))
        os.makedirs(destination, exist_ok=True)
        joblib.dump(bundle, os.path.join(destination, "model.pkl"))
        record["model_path"] = os.path.join(destination, "model.pkl")

    report: Dict[str, Any] = {}
    pbar = tqdm(groups_to_train, desc="Training synergy models", unit="pathogen")
    for pathogen, subset in pbar:
        pbar.set_description(f"{pathogen} (classifier + regressor)")
        features, columns = featurise(subset, antibiotics=antibiotics)
        X = features.to_numpy()
        # A composite id, not a bare sequence: the same peptide legitimately repeats across several
        # rows (once per antibiotic partner), so a sequence alone would not identify a row.
        entry_ids = (subset["sequence"].astype(str) + "::" + subset["antibiotic"].astype(str)
                    ).to_numpy()

        family_frame = pd.DataFrame({"sequence": subset["sequence"].to_numpy(),
                                     "is_decoy": 0})
        try:
            groups = ml_models._cluster_groups(family_frame, work_dir=cluster_work_dir)
        except Exception as e:
            logger.info(f"Skipping {pathogen}: family clustering failed ({e}).")
            continue

        scope = "organism_specific" if pathogen != "all_pathogens" else "general"

        # Classifier: is FICI at or below the cutoff.
        y_cls = subset["is_synergistic"].astype(int).to_numpy()
        if len(np.unique(y_cls)) < 2:
            logger.info(f"Skipping {pathogen}: one class only.")
            continue
        try:
            # calibrate_threshold=False: predict_synergy serves this classifier at a hardcoded
            # 0.5, deliberately aligned with the domain's own FICI synergy cutoff (see
            # fit_and_validate's docstring) -- an F1-calibrated threshold would describe holdout
            # performance at an operating point the served predictions never actually use.
            cls_record = ml_models.fit_and_validate(X, y_cls, groups, entry_ids,
                                                     is_regression=False,
                                                     calibrate_threshold=False)
        except Exception as e:
            logger.info(f"Skipping {pathogen} (classifier): {e}")
            cls_record = None

        if cls_record is not None:
            model = cls_record.pop("_model")
            cls_record.pop("_scaler")
            cls_record.pop("_applicability_domain")
            cls_record.pop("_train_entry_ids")
            test_ids = cls_record.pop("_test_entry_ids")
            cls_record.update({
                "n_rows": int(len(subset)),
                "n_peptides": int(subset["sequence"].nunique()),
                "n_antibiotics": int(subset["antibiotic"].nunique()),
                "positive_rate": float(y_cls.mean()),
            })
            passed, reasons = ml_models.assess_quality(cls_record)
            cls_record["quality_passed"], cls_record["quality_failures"] = passed, reasons
            namespace = f"synergy/{scope}/{_slug(pathogen)}/fici/classification"
            _write_if_passed(cls_record, namespace, model, columns, "classification")
            report[namespace] = cls_record

        # Regressor: predict the continuous FICI. FICI is a continuous index, so "how synergistic"
        # is a more informative question than the thresholded "is it synergistic" above -- both are
        # trained and independently validated, mirroring how steps 45-46 treat a potency assay's
        # classification and regression versions as two separate tasks.
        fici_true = subset["fici"].to_numpy(dtype=float)
        try:
            reg_record = ml_models.fit_and_validate(X, fici_true, groups, entry_ids,
                                                     is_regression=True)
        except Exception as e:
            logger.info(f"Skipping {pathogen} (regressor): {e}")
            reg_record = None

        if reg_record is not None:
            reg_model = reg_record.pop("_model")
            reg_record.pop("_scaler")
            reg_record.pop("_applicability_domain")
            reg_record.pop("_train_entry_ids")
            reg_test_ids = reg_record.pop("_test_entry_ids")

            # The regression model is ALSO scored as a classifier by mapping its predicted FICI
            # through the same SYNERGY_CUTOFF, so its practical usefulness for the yes/no decision
            # can be read next to its R^2 -- computed only on the regressor's OWN holdout rows, not
            # the full dataset, so this derived metric carries the same no-leakage guarantee as
            # every other number here.
            test_mask = np.isin(entry_ids, reg_test_ids)
            fici_pred_holdout = reg_model.predict(X[test_mask])
            y_cls_holdout = y_cls[test_mask]
            # A synergy score in (0, 1) that crosses 0.5 exactly at the FICI cutoff, so _scores'
            # own 0.5-threshold metrics correspond to the cutoff, and its ROC-AUC (rank-based)
            # reads "lower predicted FICI => more synergistic" correctly.
            synergy_score = SYNERGY_CUTOFF / (SYNERGY_CUTOFF + np.clip(fici_pred_holdout, 0.0, None))
            reg_record["classification_at_cutoff"] = (
                ml_models._scores(y_cls_holdout, synergy_score)
                if len(np.unique(y_cls_holdout)) > 1 else {})
            reg_record.update({
                "n_rows": int(len(subset)),
                "n_peptides": int(subset["sequence"].nunique()),
                "n_antibiotics": int(subset["antibiotic"].nunique()),
            })
            passed, reasons = ml_models.assess_quality(reg_record)
            reg_record["quality_passed"], reg_record["quality_failures"] = passed, reasons
            reg_namespace = f"synergy/{scope}/{_slug(pathogen)}/fici/regression"
            _write_if_passed(reg_record, reg_namespace, reg_model, columns, "regression")
            report[reg_namespace] = reg_record

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    n_kept = sum(1 for r in report.values() if r.get("quality_passed"))
    logger.info(f"Evaluated {len(report)} synergy model(s); kept {n_kept} meeting the quality "
               f"gate. Report at {report_path}.")
    return report
