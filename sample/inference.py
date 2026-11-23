"""
Featurization and scoring of user-submitted peptides.

Training features are computed by steps 21-27 over the whole dataset at once. Inference has to
reproduce them for a handful of sequences typed into a form, and reproduce them EXACTLY -- a
column computed a slightly different way at inference time than at training time yields a
confident, wrong answer with nothing to indicate anything went wrong. So every block below is
computed by the same definition as :mod:`sample.enrichment`, and the column list comes from the
model's own pickle rather than being re-derived here.

Where the two necessarily differ, it is documented at the point of difference: the dataset's
molecular descriptors are computed from a stored SMILES string, which a user submitting a bare
sequence does not have, so the molecule is built from the sequence with RDKit instead.

Anything this module cannot compute is an error, never a silently-filled default. A zero in a
feature column is a real value -- "this peptide contains no tryptophan" -- and cannot be reused to
mean "not computed" without corrupting the prediction.
"""
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sample.sequence import get_peptide_charge, sequence_to_1_letter

logger = logging.getLogger("sample.inference")

#: Residue order for the composition block. Must match :func:`sample.enrichment.enrich_with_embeddings`.
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"

#: Residue groupings for the composition block. Order matters -- the columns are positional in the
#: legacy ``emb_dim_*`` naming, so reordering this dict would silently remap trained features.
AA_GROUPS: Dict[str, str] = {
    "aliphatic": "AILV",
    "aromatic": "FWY",
    "acidic": "DE",
    "basic": "RHK",
    "polar_uncharged": "STNQCM",
    "small_tiny": "GAS",
    "sulfur": "CM",
    "proline": "P",
}

#: Maximum sequences one job may submit. A bound exists so a paste of an entire proteome cannot
#: occupy a worker indefinitely; jobs above it are rejected at submission with a clear message
#: rather than accepted and left to time out.
MAX_PEPTIDES_PER_JOB: int = 1000

_VALID_RESIDUES = set(AMINO_ACIDS)


def parse_peptide_input(text: str) -> List[Dict[str, str]]:
    """
    Reads the submission box, accepting either FASTA or one sequence per line.

    :param text: Raw textarea contents.
    :returns: ``{"peptide_name", "sequence"}`` dicts in submission order. Unnamed sequences are
        named ``peptide_1``, ``peptide_2``, … so every result row can be traced back to its input.
    """
    peptides: List[Dict[str, str]] = []
    current_name: Optional[str] = None
    current_seq: List[str] = []

    def _flush() -> None:
        if current_seq:
            name = current_name or f"peptide_{len(peptides) + 1}"
            peptides.append({"peptide_name": name, "sequence": "".join(current_seq).upper()})

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            current_name, current_seq = line[1:].strip() or None, []
            continue
        if current_name is None and not current_seq:
            # Plain sequence-per-line mode: each line is its own peptide.
            cleaned = re.sub(r"[\s\d]", "", line).upper()
            if cleaned:
                peptides.append({"peptide_name": f"peptide_{len(peptides) + 1}",
                                 "sequence": cleaned})
            continue
        current_seq.append(re.sub(r"[\s\d]", "", line).upper())
    _flush()
    return peptides


def validate_sequence(sequence: str) -> Optional[str]:
    """
    :param sequence: One-letter sequence to check.
    :returns: ``None`` when usable, otherwise a message naming the problem. Checked before the job
        is queued so a typo comes back immediately instead of as a failed job minutes later.
    """
    seq = sequence_to_1_letter(str(sequence or "")).upper()
    if not seq:
        return "empty sequence"
    if len(seq) < 2:
        return "sequence too short to score"
    unknown = sorted(set(seq) - _VALID_RESIDUES)
    if unknown:
        return f"non-standard residue(s): {', '.join(unknown)}"
    return None


def _composition_row(seq: str) -> List[float]:
    """
    :param seq: One-letter sequence.
    :returns: The 28-value composition vector -- 20 single-residue frequencies followed by the 8
        group frequencies, in :data:`AMINO_ACIDS` then :data:`AA_GROUPS` order.
    """
    n = len(seq)
    if n == 0:
        return [0.0] * (len(AMINO_ACIDS) + len(AA_GROUPS))
    single = [seq.count(aa) / n for aa in AMINO_ACIDS]
    grouped = [sum(seq.count(r) for r in residues) / n for residues in AA_GROUPS.values()]
    return single + grouped


#: Column names for the dipeptide-composition block, in the order :func:`_dpc_row` emits them.
DPC_COLUMNS: Tuple[str, ...] = tuple(f"dpc_{a}{b}" for a in AMINO_ACIDS for b in AMINO_ACIDS)


def _dpc_row(seq: str) -> List[float]:
    """
    Dipeptide composition: the frequency of each of the 400 ordered residue pairs.

    Adds the adjacency that plain amino-acid composition throws away. That matters for this
    problem specifically: an antimicrobial peptide's mechanism is amphipathic -- the ALTERNATION of
    cationic and hydrophobic residues along the chain, not merely how many of each are present --
    and AAC is invariant to every rearrangement of a sequence, which is exactly why the
    composition block's ROC-AUC collapsed against shuffled decoys in this project's own
    decoy-strategy comparison. DPC is not shuffle-invariant, so it can express some of that
    ordering while staying a cheap fixed-length count vector.

    :param seq: One-letter sequence.
    :returns: 400 frequencies over the ``n-1`` adjacent pairs, in :data:`DPC_COLUMNS` order. All
        zeros for a sequence shorter than two residues, which has no pairs. Pairs containing a
        non-standard residue are counted in the denominator but match no column, so a sequence full
        of them yields a legitimately near-zero vector rather than a renormalised one.
    """
    n = len(seq)
    if n < 2:
        return [0.0] * len(DPC_COLUMNS)
    counts: Dict[str, int] = {}
    for i in range(n - 1):
        pair = seq[i:i + 2]
        counts[pair] = counts.get(pair, 0) + 1
    total = float(n - 1)
    return [counts.get(f"{a}{b}", 0) / total for a in AMINO_ACIDS for b in AMINO_ACIDS]


#: The seven physicochemical attributes of the CTD descriptor (Dubchak et al. 1995), each
#: partitioning the 20 standard residues into three classes. The partitions are the published ones;
#: changing a partition silently redefines every ``ctd_*`` column a model was trained on.
CTD_PROPERTIES: Dict[str, Tuple[str, str, str]] = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}

#: The distribution descriptor's percentile cut points, as fractions of a class's total count.
_CTD_DISTRIBUTION_POINTS: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.00)


def _ctd_columns() -> List[str]:
    """
    :returns: The 147 CTD column names -- 21 composition, 21 transition, 105 distribution -- in the
        order :func:`_ctd_row` emits them.
    """
    names: List[str] = []
    for prop in CTD_PROPERTIES:
        names.extend(f"ctd_comp_{prop}_g{g}" for g in (1, 2, 3))
    for prop in CTD_PROPERTIES:
        names.extend(f"ctd_tran_{prop}_{pair}" for pair in ("12", "13", "23"))
    for prop in CTD_PROPERTIES:
        for g in (1, 2, 3):
            names.extend(f"ctd_dist_{prop}_g{g}_p{int(p * 100)}"
                         for p in _CTD_DISTRIBUTION_POINTS)
    return names


#: Column names for the CTD block, in the order :func:`_ctd_row` emits them.
CTD_COLUMNS: Tuple[str, ...] = tuple(_ctd_columns())


def _ctd_row(seq: str) -> List[float]:
    """
    Composition/Transition/Distribution descriptor over :data:`CTD_PROPERTIES`.

    Each property recodes the sequence into a 3-letter class string, then summarises it three ways:
    **composition** (what fraction of residues is in each class), **transition** (how often adjacent
    residues cross between two classes) and **distribution** (how far along the chain you must read
    to have seen 0/25/50/75/100% of a class). The distribution part is the one that earns CTD its
    place beside AAC and DPC here -- it encodes WHERE in the peptide a class sits, so an N-terminal
    cationic patch and a C-terminal one describe differently even at identical composition.

    :param seq: One-letter sequence.
    :returns: 147 values in :data:`CTD_COLUMNS` order. All zeros for an empty sequence. Residues
        outside the 20 standard ones belong to no class and are skipped, so they dilute the
        composition rather than being forced into a class they do not belong to.
    """
    if not seq:
        return [0.0] * len(CTD_COLUMNS)

    encoded: Dict[str, List[int]] = {}
    for prop, groups in CTD_PROPERTIES.items():
        member = {r: i + 1 for i, residues in enumerate(groups) for r in residues}
        encoded[prop] = [member[r] for r in seq if r in member]

    composition: List[float] = []
    for prop in CTD_PROPERTIES:
        classes = encoded[prop]
        n = len(classes)
        composition.extend(classes.count(g) / n if n else 0.0 for g in (1, 2, 3))

    transition: List[float] = []
    for prop in CTD_PROPERTIES:
        classes = encoded[prop]
        # n-1 adjacent pairs, and a transition is unordered: 1->2 and 2->1 are the same crossing.
        pairs = float(len(classes) - 1)
        for first, second in ((1, 2), (1, 3), (2, 3)):
            if pairs <= 0:
                transition.append(0.0)
                continue
            crossings = sum(1 for a, b in zip(classes, classes[1:])
                            if (a, b) == (first, second) or (a, b) == (second, first))
            transition.append(crossings / pairs)

    distribution: List[float] = []
    for prop in CTD_PROPERTIES:
        classes = encoded[prop]
        n = len(classes)
        for g in (1, 2, 3):
            positions = [i + 1 for i, c in enumerate(classes) if c == g]
            total = len(positions)
            if total == 0 or n == 0:
                distribution.extend([0.0] * len(_CTD_DISTRIBUTION_POINTS))
                continue
            for point in _CTD_DISTRIBUTION_POINTS:
                # The 0% point is the FIRST occurrence, not position zero -- index 0 of the
                # occurrence list. Higher points use the ceiling so that e.g. 25% of 3 occurrences
                # resolves to the 1st, never to a fractional index.
                rank = 0 if point == 0.0 else int(math.ceil(point * total)) - 1
                distribution.append(positions[min(rank, total - 1)] / n)

    return composition + transition + distribution


#: The sequence-defined descriptors this module reproduces from `peptides.py` at scoring time,
#: beyond the RDKit block. Each is a column :func:`sample.enrichment.enrich_with_descriptors` wrote
#: into the training set, so the name here and the name there must stay in step.
PEPTIDE_DESCRIPTOR_COLUMNS: Tuple[str, ...] = (
    "boman_index", "aliphatic_index", "hydrophobic_moment", "isoelectric_point", "net_charge_ph7",
)

#: The full descriptor block a model may be trained on: the RDKit/charge columns plus the
#: `peptides.py` ones above.
DESCRIPTOR_COLUMNS: Tuple[str, ...] = (
    "molecular_weight", "logp", "hydrogen_bond_donors", "hydrogen_bond_acceptors",
    "charge", "sequence_length",
) + PEPTIDE_DESCRIPTOR_COLUMNS


def _descriptor_row(seq: str) -> Dict[str, float]:
    """
    Computes the molecular descriptor block for one sequence.

    Step 27 derives these from the dataset's stored SMILES. A user submitting a bare sequence has
    no SMILES, so the molecule is built with ``Chem.MolFromSequence`` and the same RDKit
    descriptors are read off it. For a peptide of standard residues the two routes describe the
    same neutral linear molecule, so the values are comparable with the training set.

    The `peptides.py` descriptors in :data:`PEPTIDE_DESCRIPTOR_COLUMNS` are delegated to
    :func:`sample.enrichment.peptide_descriptors` -- the SAME function that computed the training
    set's columns, called rather than reimplemented. Two independent routes to a name like
    ``aliphatic_index`` already exist in this codebase (:mod:`sample.admet` has its own, over
    Biopython, rounded to 2dp), and scoring a model through the other one would feed it a feature
    that is subtly not the feature it was fit on.

    :param seq: One-letter sequence.
    :returns: Descriptor name -> value, exactly the keys in :data:`DESCRIPTOR_COLUMNS`. Values
        neither library can produce are ``NaN``, which the gradient-boosted models handle natively
        as missing.
    """
    from rdkit.Chem import Descriptors

    from sample.enrichment import molecule_from_sequence

    row: Dict[str, float] = {
        "molecular_weight": float("nan"), "logp": float("nan"),
        "hydrogen_bond_donors": float("nan"), "hydrogen_bond_acceptors": float("nan"),
    }
    try:
        # `molecule_from_sequence`, NOT a bare `Chem.MolFromSequence`: enrichment built the
        # training columns through the topology-aware route, which closes a known peptide's
        # disulfides and head-to-tail bond. Scoring a cysteine-rich peptide through the open-chain
        # route instead reports its free thiols as hydrogen-bond donors -- measured at 94 against
        # the stored 86 for one defensin, with LogP off by 2.1 -- so the model receives a feature
        # that is numerically not the feature it was fit on. A sequence absent from the topology
        # index is built linear by both routes, so this only ever narrows the gap.
        mol = molecule_from_sequence(seq)
        if mol is not None:
            row["molecular_weight"] = float(Descriptors.MolWt(mol))
            row["logp"] = float(Descriptors.MolLogP(mol))
            row["hydrogen_bond_donors"] = float(Descriptors.NumHDonors(mol))
            row["hydrogen_bond_acceptors"] = float(Descriptors.NumHAcceptors(mol))
    except Exception as e:
        logger.warning(f"RDKit descriptors unavailable for a {len(seq)}-mer: {e}")

    try:
        row["charge"] = float(get_peptide_charge(seq))
    except Exception:
        row["charge"] = float("nan")
    row["sequence_length"] = float(len(seq))

    from sample.enrichment import peptide_descriptors

    peptide = peptide_descriptors(seq)
    for column in PEPTIDE_DESCRIPTOR_COLUMNS:
        value = peptide.get(column)
        row[column] = float(value) if value is not None and value == value else float("nan")
    # Same precedence as enrichment's own: peptides.py's topology-corrected mass wins, RDKit's
    # fills in only where peptides.py had no answer. Scoring must not cross those two sources in
    # the opposite order from the column it is being compared against.
    weight = peptide.get("molecular_weight")
    if weight is not None and weight == weight:
        row["molecular_weight"] = float(weight)
    return row


def _embedding_frame(sequences: Sequence[str], model_name: str, prefix: str,
                     spaced: bool) -> pd.DataFrame:
    """
    Mean-pooled transformer embeddings, matching :func:`sample.enrichment.enrich_with_embeddings`.

    :param sequences: One-letter sequences.
    :param model_name: HuggingFace model id.
    :param prefix: Column prefix, e.g. ``esm2_dim_``.
    :param spaced: Insert spaces between residues. ESM-2 does not need it; retained for
        models that do.
    :returns: One row per sequence.
    :raises RuntimeError: When the model cannot be loaded, so a request for a model that needs
        this block fails loudly instead of scoring against an empty feature matrix.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(f"{prefix} features need torch and transformers: {e}") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Could not load {model_name}: {e}") from e

    texts = [" ".join(list(s)) for s in sequences] if spaced else list(sequences)
    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            inputs = tokenizer(texts[i:i + 16], padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = torch.sum(outputs.last_hidden_state.float() * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            vectors.extend((summed / counts).cpu().numpy())

    frame = pd.DataFrame(np.vstack(vectors))
    frame.columns = [f"{prefix}{i}" for i in range(frame.shape[1])]
    return frame


def build_feature_frame(sequences: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    """
    Builds the feature matrix a model needs, in the model's own column order.

    Only the blocks the requested columns actually reference are computed, so scoring against a
    descriptor-only model does not load a transformer.

    :param sequences: One-letter sequences, already validated.
    :param columns: The model's training columns, from its pickle.
    :returns: A frame with exactly `columns`, in that order, one row per sequence.
    :raises RuntimeError: When a requested block cannot be computed.
    """
    seqs = [sequence_to_1_letter(str(s)).upper() for s in sequences]
    wanted = list(columns)
    frame = pd.DataFrame(index=range(len(seqs)))

    if wanted == ["sequence"]:
        # A fine-tuned transformer's own feature contract: the raw sequence string, not a
        # precomputed numeric column -- passed through unmodified rather than routed through any
        # of the numeric blocks below, every one of which computes a named scalar column.
        frame["sequence"] = seqs
        return frame[wanted]

    if any(c in wanted for c in DESCRIPTOR_COLUMNS):
        frame = frame.join(pd.DataFrame([_descriptor_row(s) for s in seqs]))

    needs_named_composition = any(c.startswith(("aa_freq_", "aa_group_freq_")) for c in wanted)
    needs_legacy_composition = any(c.startswith("emb_dim_") for c in wanted)
    if needs_named_composition or needs_legacy_composition:
        rows = [_composition_row(s) for s in seqs]
        named = [f"aa_freq_{aa}" for aa in AMINO_ACIDS] + \
                [f"aa_group_freq_{g}" for g in AA_GROUPS]
        composition = pd.DataFrame(rows, columns=named)
        if needs_named_composition:
            frame = frame.join(composition)
        if needs_legacy_composition:
            # The legacy block is the 20 single-residue frequencies under positional names, which
            # is what the pre-rewrite enrichment wrote. Verified against the shipped dataset:
            # final_dataset.csv carries exactly emb_dim_0..emb_dim_19.
            legacy = composition[[f"aa_freq_{aa}" for aa in AMINO_ACIDS]].copy()
            legacy.columns = [f"emb_dim_{i}" for i in range(len(AMINO_ACIDS))]
            frame = frame.join(legacy)

    if any(c.startswith("dpc_") for c in wanted):
        frame = frame.join(pd.DataFrame([_dpc_row(s) for s in seqs], columns=list(DPC_COLUMNS)))

    if any(c.startswith("ctd_") for c in wanted):
        frame = frame.join(pd.DataFrame([_ctd_row(s) for s in seqs], columns=list(CTD_COLUMNS)))

    # ProtBERT was removed from the pipeline (transformers 5.x will not load its
    # non-safetensors checkpoint below torch 2.6, which this project's CUDA pin rules out), so no
    # model trained from here on carries protbert_dim_ columns. A legacy model that does will
    # report them as missing features rather than silently scoring on zeros.
    if any(c.startswith("esm2_dim_") for c in wanted):
        frame = frame.join(_embedding_frame(seqs, "facebook/esm2_t6_8M_UR50D", "esm2_dim_", False))

    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Cannot build {len(missing)} of the model's {len(wanted)} feature columns "
            f"(first missing: {', '.join(missing[:5])}). The model was trained on features this "
            "server cannot reproduce.")
    return frame[wanted]


def _in_domain_flags(features: pd.DataFrame, entry: Dict[str, Any]) -> List[Optional[bool]]:
    """
    Checks each row against a model's applicability domain, when it has one.

    :param features: The model's own feature frame (:func:`build_feature_frame`'s output),
        unscaled -- the model's ``"scaler"`` (fit on the same column set during training) is
        applied here, not by the caller.
    :param entry: The registry entry from :func:`sample.model_registry.load_model`, carrying
        ``"scaler"``/``"applicability_domain"`` when the model was trained by steps 45-46.
    :returns: One flag per row: ``True``/``False`` from :func:`sample.ml_models.in_domain`, or
        ``None`` for every row when the model predates the applicability domain (a legacy model,
        or an older release) -- "not computed", never silently reported as "out of domain".
    """
    scaler = entry.get("scaler")
    ad = entry.get("applicability_domain")
    if scaler is None or not ad:
        return [None] * len(features)
    try:
        from sample.ml_models import in_domain
        flags = in_domain(scaler.transform(features), ad)
        return [bool(f) for f in flags]
    except Exception as e:
        logger.warning(f"Applicability-domain check failed, reporting unknown rather than "
                       f"failing the prediction: {e}")
        return [None] * len(features)


def predict(sequences: Sequence[str], model_id: str,
            model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scores sequences with one registered model.

    :param sequences: One-letter sequences.
    :param model_id: Registry id, from :func:`sample.model_registry.list_models`.
    :param model_dir: Namespace root; the registry default when omitted.
    :returns: One dict per sequence with ``probability`` and ``predicted_label`` for a classifier,
        or ``value`` for a regressor; plus ``in_domain`` (``True``/``False``, or ``None`` when the
        model carries no applicability domain to check against -- see :func:`_in_domain_flags`).
        A peptide outside the domain is still scored: this project's policy is to warn, not refuse,
        since the model may still generalize and the alternative is refusing a real question.
    :raises FileNotFoundError: When the model id is unknown.
    """
    from sample.model_registry import load_model

    estimator, columns, entry = load_model(model_id, model_dir)
    features = build_feature_frame(sequences, columns)
    in_domain_flags = _in_domain_flags(features, entry)

    is_regression = str(entry.get("learning_task", "")).lower() == "regression"
    results: List[Dict[str, Any]] = []

    if is_regression or not hasattr(estimator, "predict_proba"):
        values = np.asarray(estimator.predict(features)).ravel()
        for i, value in enumerate(values):
            results.append({"position": i, "sequence": sequences[i],
                            "value": float(value), "probability": None,
                            "predicted_label": None, "in_domain": in_domain_flags[i]})
        return results

    # The model's own F1-calibrated decision cutoff (sample.ml_models._best_f1_threshold,
    # computed at training time from an internal split of the training data only -- never this
    # model's holdout). Falls back to the conventional 0.5 for a model trained before calibration
    # existed. Reported alongside the probability rather than replacing it, because the
    # probability is the model's actual output and the label is only one reading of it.
    threshold = entry.get("metrics", {}).get("threshold")
    if threshold is None:
        threshold = 0.5

    probabilities = np.asarray(estimator.predict_proba(features))[:, 1]
    for i, probability in enumerate(probabilities):
        results.append({
            "position": i,
            "sequence": sequences[i],
            "probability": float(probability),
            "predicted_label": "active" if probability >= threshold else "inactive",
            "value": None,
            "in_domain": in_domain_flags[i],
        })
    return results
