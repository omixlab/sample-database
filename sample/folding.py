"""
Structure prediction with ColabFold, and the SDF and SMILES derived from it.

Step 27 previously built 3D structures the only way it could without a folding model: parse the
stored SMILES, embed it with distance geometry, minimize with a force field. That produces a
valid low-energy conformer and says nothing about how the peptide actually folds -- ETKDG has no
notion of a helix or a disulfide-stapled loop, and for these molecules the fold is the interesting
part. It also inherited whatever was in the SMILES column, which is unvalidated and in thousands
of rows is not a molecule at all.

Predicting the structure inverts the dependency, and fixes both problems at once::

    sequence -> ColabFold (AlphaFold2) -> PDB -> RDKit -> SDF + SMILES

The sequence is the one field this dataset is confident about, so deriving everything downstream
from it means every peptide gets a consistent structure and a syntactically valid SMILES, whatever
the source database happened to record.

**MSA mode.** ColabFold normally builds an MSA with MMseqs2 against ColabFoldDB (~2 TB). That is
not available here, and querying the public MMseqs2 API 35,411 times would be an abuse of a shared
service. Predictions run in ``single_sequence`` mode, which is the usual choice for short peptides
-- an MSA over 20 residues carries little signal -- and keeps the work genuinely local. The
trade-off is real and is recorded per structure: single-sequence AlphaFold2 is less accurate than
MSA-backed AlphaFold2, and pLDDT is stored so a consumer can filter on confidence rather than
assume it.

**This is a long job.** 35,411 peptides at a few seconds each is measured in days of GPU time, so
folding is resumable by construction: a peptide whose PDB already exists is skipped, and the
sequence set is processed shortest-first so the cheap majority lands early.
"""
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("sample.folding")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Where predicted structures and their derivatives are written.
STRUCTURE_DIR = os.path.join(DATA_DIR, "sample_enriched_release", "structures")

#: Longest sequence to fold. AlphaFold2 memory scales with the square of the length, and a 12 GB
#: card runs out well before this dataset's 1,388-residue maximum. The 2,218 sequences above this
#: are recorded as skipped rather than allowed to fail one at a time deep inside a batch.
MAX_FOLD_LENGTH: int = 400

#: Shortest sequence sent to ColabFold. Anything at or below :data:`SHORT_PEPTIDE_LENGTH` goes to
#: distance geometry instead -- see that constant for the reasoning.
#:
#: A dipeptide has no fold, and asking AlphaFold2 for one produces confident-looking nonsense: a
#: first trial run folded 300 entries of 2-3 residues and got a median pLDDT of 34.5 with residues
#: placed 30 A apart -- further than any covalent bond, in a chain that should be 3.8 A per step.
#: The failure is not AlphaFold2's; it is being asked a question with no answer.
MIN_FOLD_LENGTH: int = 16

#: Longest peptide handled by distance geometry rather than by folding.
#:
#: Below about 15 residues a peptide has no tertiary structure to predict -- it is a short chain
#: whose conformation is set by local geometry, which is exactly what ETKDG plus a force field
#: models well and what a co-evolution model has nothing to say about. AlphaFold2 in
#: single-sequence mode returns low-confidence output for these regardless, so routing them to
#: RDKit is both faster and more honest: an energy-minimised conformer does not claim to be a
#: predicted fold.
SHORT_PEPTIDE_LENGTH: int = 15

#: Distance between consecutive alpha carbons in a real chain, in angstroms. Peptide geometry
#: fixes this at about 3.8; anything far outside the tolerance below is a broken prediction.
CA_CA_DISTANCE: float = 3.8

#: Tolerance on that distance before a gap is called a chain break.
CA_CA_TOLERANCE: float = 1.2

#: Sequences per ColabFold invocation. Batching amortises model loading, which dominates for short
#: peptides -- loading AlphaFold2 weights takes longer than folding a 20-residue peptide.
BATCH_SIZE: int = 50

#: AlphaFold2 models to run. ColabFold's default of 5 gives an ensemble; 1 is a fifth of the
#: runtime for a per-structure confidence that is already reported by pLDDT. Raise it when
#: accuracy matters more than throughput.
NUM_MODELS: int = 1

#: Recycles through the network. 3 is ColabFold's default and the accuracy knee for short chains.
NUM_RECYCLE: int = 3

#: The 20 standard residues. ColabFold accepts nothing else, and this dataset carries sequences
#: with modification codes and ambiguity characters that must be filtered out before they reach it.
_STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")


#: Environment holding ``colabfold_batch``. ColabFold's JAX/AlphaFold2 stack cannot be resolved
#: against this project's own dependencies -- pip gives up trying -- so it lives in a tool
#: environment that the orchestration shells into, the same arrangement
#: :mod:`sample.peptide_predictors` uses for ``sample-predictors``.
FOLD_ENV = os.environ.get("SAMPLE_FOLD_ENV", "sample-fold").strip()


def colabfold_command() -> Optional[List[str]]:
    """
    Resolves how to invoke ColabFold.

    :returns: The command prefix -- ``["colabfold_batch"]`` when it is directly on ``PATH``, or a
        ``conda run`` into :data:`FOLD_ENV` otherwise -- or ``None`` when neither is available.
    """
    if shutil.which("colabfold_batch"):
        return ["colabfold_batch"]

    if shutil.which("conda") and FOLD_ENV:
        probe = subprocess.run(
            ["conda", "run", "-n", FOLD_ENV, "which", "colabfold_batch"],
            capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip():
            return ["conda", "run", "--no-capture-output", "-n", FOLD_ENV, "colabfold_batch"]
    return None


def colabfold_available() -> bool:
    """
    :returns: Whether ColabFold can be invoked, directly or through its environment.
    """
    return colabfold_command() is not None


def _cuda_environment() -> Dict[str, str]:
    """
    Builds the environment ColabFold needs to find its CUDA libraries.

    JAX's ``cuda12`` plugin ships the CUDA runtime as ``nvidia-*`` pip packages, whose shared
    objects live under ``site-packages/nvidia/*/lib`` -- a location the dynamic loader does not
    search. Without this, the plugin aborts with ``The cuSPARSE library was not found``, JAX falls
    back to CPU **silently**, and folding still runs: minutes per peptide instead of seconds,
    which turns a multi-day job into an impossible one. The failure looks like slowness rather
    than an error, so it is worth setting explicitly rather than relying on the environment.

    :returns: A copy of the current environment with ``LD_LIBRARY_PATH`` extended.
    """
    environment = dict(os.environ)

    prefix = os.path.join(
        os.path.expanduser("~"), "miniconda3", "envs", FOLD_ENV) if FOLD_ENV else ""
    candidates: List[str] = []
    for base in (prefix, os.environ.get("CONDA_PREFIX", "")):
        if not base:
            continue
        for python_dir in sorted(glob.glob(os.path.join(base, "lib", "python3.*"))):
            candidates.extend(
                sorted(glob.glob(os.path.join(python_dir, "site-packages", "nvidia", "*", "lib"))))

    if candidates:
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(candidates + ([existing] if existing else []))
    return environment


def is_foldable(sequence: Any, max_length: int = MAX_FOLD_LENGTH,
                min_length: int = MIN_FOLD_LENGTH) -> bool:
    """
    :param sequence: A peptide sequence.
    :param max_length: Longest sequence to accept.
    :param min_length: Shortest sequence to accept. Peptides at or below
        :data:`SHORT_PEPTIDE_LENGTH` are handled by distance geometry instead, which suits a chain
        with no tertiary structure better than a co-evolution model does.
    :returns: Whether ColabFold can usefully fold it. Rejects non-standard residues, which
        AlphaFold2 has no representation for, and sequences outside the useful length range.
    """
    text = str(sequence or "").strip().upper()
    if not text or not (min_length <= len(text) <= max_length):
        return False
    return set(text) <= _STANDARD_RESIDUES


def pdb_path(entry_id: str, structure_dir: Optional[str] = None) -> str:
    """
    :param entry_id: Peptide identifier.
    :param structure_dir: Output directory; :data:`STRUCTURE_DIR` when omitted.
    :returns: Where this peptide's predicted structure lives.
    """
    return os.path.join(structure_dir or STRUCTURE_DIR, f"{entry_id}.pdb")


def folded_entry_ids(structure_dir: Optional[str] = None) -> set:
    """
    :param structure_dir: Structure directory; :data:`STRUCTURE_DIR` when omitted.
    :returns: Entries that already have a predicted structure. The distance-geometry fallback
        skips these, so a prediction is never overwritten by an embedded conformer.
    """
    structure_dir = structure_dir or STRUCTURE_DIR
    if not os.path.isdir(structure_dir):
        return set()
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(structure_dir, "*.pdb"))}


def _run_colabfold(fasta_path: str, out_dir: str, num_models: int = NUM_MODELS,
                   num_recycle: int = NUM_RECYCLE, timeout: float = 7200.0) -> bool:
    """
    Invokes ``colabfold_batch`` on one FASTA.

    :param fasta_path: Input FASTA.
    :param out_dir: Directory ColabFold writes into.
    :param num_models: AlphaFold2 models to run.
    :param num_recycle: Recycles per model.
    :param timeout: Seconds before the batch is abandoned.
    :returns: Whether the invocation succeeded.
    """
    prefix = colabfold_command()
    if prefix is None:
        logger.error("ColabFold is not available; cannot fold.")
        return False

    command = prefix + [
        "--msa-mode", "single_sequence",
        "--num-models", str(num_models),
        "--num-recycle", str(num_recycle),
        # Ranked output only; the unrelaxed per-model PDBs multiply the file count for no gain
        # once a single model is being run.
        "--rank", "plddt",
        fasta_path, out_dir,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                env=_cuda_environment())
    except subprocess.TimeoutExpired:
        logger.error(f"colabfold_batch exceeded {timeout:.0f}s on {fasta_path}; abandoning batch.")
        return False
    except OSError as e:
        logger.error(f"Could not run colabfold_batch: {e}")
        return False

    if result.returncode != 0:
        logger.error(f"colabfold_batch failed ({result.returncode}): "
                     f"{(result.stderr or '').strip()[-600:]}")
        return False
    return True


def _plddt_from_pdb(path: str) -> Optional[float]:
    """
    Reads the mean pLDDT from a predicted structure.

    AlphaFold2 writes per-residue confidence into the B-factor column, so the mean over CA atoms
    is the structure's overall confidence. Stored because single-sequence predictions vary widely
    in quality and a consumer must be able to filter rather than assume.

    :param path: A predicted PDB.
    :returns: Mean pLDDT (0-100), or ``None`` when it cannot be read.
    """
    values: List[float] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        values.append(float(line[60:66]))
                    except ValueError:
                        continue
    except OSError:
        return None
    return sum(values) / len(values) if values else None


def chain_breaks(pdb_file: str, tolerance: float = CA_CA_TOLERANCE) -> List[Tuple[int, float]]:
    """
    Finds consecutive residues whose alpha carbons are too far apart to be bonded.

    Peptide geometry fixes the CA-CA distance of adjacent residues at about 3.8 A. A prediction
    that places them further apart is not a fold, whatever its pLDDT says -- a first trial run
    produced a PRO-GLY "structure" with its two residues 30 A apart, which no covalent bond spans.
    Checking the geometry catches that class of failure directly, rather than inferring it from a
    confidence score that is itself unreliable on degenerate input.

    :param pdb_file: A predicted structure.
    :param tolerance: Allowed deviation from :data:`CA_CA_DISTANCE`, in angstroms.
    :returns: ``(residue_index, distance)`` for each break, empty when the chain is continuous.
    """
    import math

    coordinates: List[Tuple[float, float, float]] = []
    try:
        with open(pdb_file, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        coordinates.append((float(line[30:38]), float(line[38:46]),
                                            float(line[46:54])))
                    except ValueError:
                        continue
    except OSError:
        return []

    breaks: List[Tuple[int, float]] = []
    for index in range(1, len(coordinates)):
        a, b = coordinates[index - 1], coordinates[index]
        distance = math.dist(a, b)
        if abs(distance - CA_CA_DISTANCE) > tolerance:
            breaks.append((index, distance))
    return breaks


def _collect_predictions(out_dir: str, structure_dir: str,
                         wanted: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Moves ColabFold's ranked output into the release layout.

    ColabFold names files ``<id>_relaxed_rank_001_*.pdb`` or ``<id>_unrelaxed_rank_001_*.pdb``
    depending on whether Amber relaxation ran. The best-ranked file for each id is taken and
    renamed to ``<entry_id>.pdb``.

    :param out_dir: ColabFold's output directory.
    :param structure_dir: Release structure directory.
    :param wanted: Mapping from the FASTA header used to the entry id.
    :returns: Per-entry metadata for the structures collected.
    """
    os.makedirs(structure_dir, exist_ok=True)
    collected: Dict[str, Dict[str, Any]] = {}

    for header, entry_id in wanted.items():
        matches = sorted(glob.glob(os.path.join(out_dir, f"{header}_*rank_001*.pdb")))
        if not matches:
            matches = sorted(glob.glob(os.path.join(out_dir, f"{header}_*.pdb")))
        if not matches:
            continue

        # Prefer the Amber-relaxed structure when ColabFold produced one.
        best = next((m for m in matches if "relaxed" in os.path.basename(m)
                     and "unrelaxed" not in os.path.basename(m)), matches[0])

        # Validate before it reaches the release directory: a broken prediction that lands there
        # is indistinguishable from a good one to everything downstream.
        breaks = chain_breaks(best)
        if breaks:
            first, distance = breaks[0]
            logger.warning(
                f"{entry_id}: discarding prediction with {len(breaks)} chain break(s) -- "
                f"residues {first} and {first + 1} are {distance:.1f} A apart, against the "
                f"{CA_CA_DISTANCE} A a peptide bond spans.")
            continue

        destination = pdb_path(entry_id, structure_dir)
        shutil.copyfile(best, destination)
        collected[entry_id] = {
            "pdb": destination,
            "plddt": _plddt_from_pdb(destination),
            "relaxed": "unrelaxed" not in os.path.basename(best),
            "chain_breaks": 0,
        }

    return collected


def fold_sequences(peptides: Iterable[Tuple[str, str]], structure_dir: Optional[str] = None,
                   batch_size: int = BATCH_SIZE, num_models: int = NUM_MODELS,
                   num_recycle: int = NUM_RECYCLE,
                   limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Predicts structures for peptides that do not already have one.

    :param peptides: ``(entry_id, sequence)`` pairs.
    :param structure_dir: Where structures are written; :data:`STRUCTURE_DIR` when omitted.
    :param batch_size: Sequences per ColabFold invocation.
    :param num_models: AlphaFold2 models per sequence.
    :param num_recycle: Recycles per model.
    :param limit: Stop after this many new structures, for a bounded trial run.
    :returns: Per-entry metadata for everything folded in this call.
    :raises RuntimeError: When ``colabfold_batch`` is not installed.
    """
    if not colabfold_available():
        raise RuntimeError(
            "colabfold_batch is not on PATH. Create the folding environment with "
            "`make env-fold` (environments/sample-fold.yml).")

    structure_dir = structure_dir or STRUCTURE_DIR
    os.makedirs(structure_dir, exist_ok=True)

    pending: List[Tuple[str, str]] = []
    skipped_unfoldable = existing = 0
    for entry_id, sequence in peptides:
        if os.path.exists(pdb_path(entry_id, structure_dir)):
            existing += 1
            continue
        if not is_foldable(sequence):
            skipped_unfoldable += 1
            continue
        pending.append((entry_id, str(sequence).strip().upper()))

    # Shortest first: runtime scales steeply with length, so this puts the cheap majority of the
    # dataset on disk early and leaves the expensive tail for later runs.
    pending.sort(key=lambda pair: len(pair[1]))
    if limit is not None:
        pending = pending[:limit]

    logger.info(f"Folding: {len(pending)} to predict, {existing} already present, "
                f"{skipped_unfoldable} not foldable (non-standard residues or over "
                f"{MAX_FOLD_LENGTH} residues).")
    if not pending:
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        with tempfile.TemporaryDirectory(prefix="colabfold_") as work_dir:
            fasta_path = os.path.join(work_dir, "batch.fasta")
            wanted: Dict[str, str] = {}
            with open(fasta_path, "w", encoding="utf-8") as handle:
                for index, (entry_id, sequence) in enumerate(batch):
                    # ColabFold builds filenames from the header, so a short unique token avoids
                    # both filesystem-hostile characters and truncation collisions.
                    header = f"s{index:04d}"
                    wanted[header] = entry_id
                    handle.write(f">{header}\n{sequence}\n")

            out_dir = os.path.join(work_dir, "out")
            os.makedirs(out_dir, exist_ok=True)
            if _run_colabfold(fasta_path, out_dir, num_models=num_models,
                              num_recycle=num_recycle):
                results.update(_collect_predictions(out_dir, structure_dir, wanted))

        done = start + len(batch)
        logger.info(f"  ...{done}/{len(pending)} folded ({len(results)} structures written).")

    return results


# --------------------------------------------------------------------------------------------
# PDB -> SDF and SMILES
# --------------------------------------------------------------------------------------------

#: Minimum bonds per atom for a molecule to be considered connected. A peptide is a covalent
#: chain, so its bond count is close to its atom count; anything near zero means connectivity was
#: never perceived.
_MIN_BOND_RATIO: float = 0.5


def structure_to_molecule(pdb_file: str) -> Optional[Any]:
    """
    Reads a predicted PDB into an RDKit molecule with explicit bonds.

    ``MolFromPDBFile`` infers connectivity from residue templates and interatomic distances, which
    is what a predicted structure needs: AlphaFold2 writes no ``CONECT`` records, so **proximity
    bonding is the only source of connectivity there is**. Disabling it was tried as a way around
    valence errors and is a trap -- the parse succeeds, every atom comes back isolated, and the
    derived SMILES is ``C.C.C.C...``: syntactically valid, chemically meaningless, and silent.
    That is why the connectivity check below exists rather than trusting a non-``None`` return.

    Sanitization is relaxed rather than abandoned on failure. A predicted structure occasionally
    puts an oxygen at a valence RDKit rejects, usually at the C-terminus; skipping the property
    check keeps the geometry and the bonds while dropping only the assertion that tripped.

    :param pdb_file: A predicted structure.
    :returns: The molecule, or ``None`` when it cannot be read as a connected molecule.
    """
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=True)

    if mol is None:
        # Keep proximity bonding; relax only the checks that failed.
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(
                mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception as e:
            logger.debug(f"Could not sanitize {pdb_file}: {e}")
            return None

    atoms, bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    if atoms and bonds / atoms < _MIN_BOND_RATIO:
        logger.warning(
            f"{os.path.basename(pdb_file)}: only {bonds} bonds for {atoms} atoms, so connectivity "
            f"was not perceived. Discarding rather than emitting a disconnected SMILES.")
        return None
    return mol


def _smiles_for(mol: Any, sequence: str = "") -> Optional[str]:
    """
    Produces the canonical SMILES for a folded peptide.

    The **sequence** is the authority, not the structure. Bond perception over a predicted PDB is
    geometric: AlphaFold2 emits no ``CONECT`` records, so RDKit infers bonds from interatomic
    distances, and two atoms that land slightly too close become bonded. In practice that puts a
    third bond on a terminal oxygen and the molecule then fails valence checks -- a defect in the
    *perception*, not in the peptide, which is an ordinary chain of standard residues.

    ``MolFromSequence`` builds that chain from the sequence with correct topology and valences, so
    it is tried first. The structure-derived SMILES is the fallback for peptides whose sequence
    RDKit will not build, and it is only accepted if it survives sanitization -- an unsanitizable
    molecule would otherwise yield a SMILES asserting chemistry that cannot exist.

    :param mol: The molecule read from the predicted structure.
    :param sequence: The peptide sequence, in one-letter codes.
    :returns: Canonical SMILES, or ``None`` when neither route produces a valid one.
    """
    from rdkit import Chem

    text = str(sequence or "").strip().upper()
    if text and set(text) <= _STANDARD_RESIDUES:
        from_sequence = Chem.MolFromSequence(text)
        if from_sequence is not None:
            return Chem.MolToSmiles(from_sequence)

    try:
        # Hydrogens are dropped: they are implicit in SMILES, and keeping them explicit produces a
        # string no other tool would generate for the same molecule.
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception as e:
        logger.debug(f"Could not derive SMILES from the structure: {e}")
        return None


def derive_sdf_and_smiles(entry_id: str, sequence: str = "",
                          structure_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Writes an SDF and derives a SMILES from a peptide's predicted structure.

    The SMILES is *derived*, not copied. The stored SMILES column is unvalidated and holds
    sentinels, truncated generations and residue letters left in place of atoms; a SMILES read off
    a predicted structure is syntactically valid by construction and consistent with the
    coordinates shipped beside it.

    :param entry_id: Peptide identifier.
    :param sequence: Sequence, recorded as an SDF property.
    :param structure_dir: Structure directory; :data:`STRUCTURE_DIR` when omitted.
    :returns: ``{"entry_id", "sdf", "smiles", "plddt", "n_atoms"}``, or ``None`` when there is no
        usable structure.
    """
    from rdkit import Chem

    structure_dir = structure_dir or STRUCTURE_DIR
    source = pdb_path(entry_id, structure_dir)
    if not os.path.exists(source):
        return None

    mol = structure_to_molecule(source)
    if mol is None:
        return None

    plddt = _plddt_from_pdb(source)
    mol.SetProp("_Name", entry_id)
    if sequence:
        mol.SetProp("Sequence", str(sequence))
    mol.SetProp("Source", "ColabFold (AlphaFold2, single-sequence MSA)")

    # AlphaFold2 predicts a linear chain. For a peptide whose backbone is closed head-to-tail, or
    # braced by disulfides, the prediction is missing exactly the bonds that define its shape --
    # a cyclotide's cystine knot is why it resists proteases, and none of it is in the model. The
    # topology is recorded on the structure so a reader is not left to infer it from coordinates
    # that do not contain it.
    try:
        from sample.enrichment import topology_for

        topology = topology_for(sequence) if sequence else None
        if topology is not None:
            mol.SetProp("Topology", topology.kind)
            mol.SetProp("Disulfides", str(topology.n_disulfides))
            if topology.is_cyclic:
                mol.SetProp(
                    "TopologyNote",
                    "Predicted as a linear chain; the cyclic bonds this peptide carries are not "
                    "modelled by AlphaFold2.")
    except Exception as e:
        logger.debug(f"Could not annotate topology for {entry_id}: {e}")

    mol.SetProp("Dimensionality", "3D")
    if plddt is not None:
        mol.SetProp("pLDDT", f"{plddt:.2f}")

    sdf_file = os.path.join(structure_dir, f"{entry_id}.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(mol)
    writer.close()

    smiles = _smiles_for(mol, sequence)
    if not smiles:
        logger.debug(f"No usable SMILES for {entry_id}.")
        return None

    return {
        "entry_id": entry_id,
        "sdf": sdf_file,
        "smiles": smiles,
        "plddt": plddt,
        "n_atoms": mol.GetNumAtoms(),
    }


def build_structures(peptides: Iterable[Tuple[str, str]], structure_dir: Optional[str] = None,
                     output_path: Optional[str] = None, limit: Optional[int] = None,
                     **fold_kwargs: Any) -> Dict[str, Any]:
    """
    Folds peptides and derives their SDF and SMILES in one pass.

    :param peptides: ``(entry_id, sequence)`` pairs.
    :param structure_dir: Structure directory; :data:`STRUCTURE_DIR` when omitted.
    :param output_path: Where the per-peptide record is written;
        ``data/predicted_structures.json`` when omitted.
    :param limit: Stop after this many new structures.
    :param fold_kwargs: Passed through to :func:`fold_sequences`.
    :returns: A summary of what was produced.
    """
    structure_dir = structure_dir or STRUCTURE_DIR
    output_path = output_path or os.path.join(DATA_DIR, "predicted_structures.json")

    peptides = list(peptides)
    fold_sequences(peptides, structure_dir=structure_dir, limit=limit, **fold_kwargs)

    # Derive for everything with a structure on disk, not just this run's output, so an
    # interrupted job still produces a complete record of what exists.
    records: Dict[str, Dict[str, Any]] = {}
    failed = 0
    for entry_id, sequence in peptides:
        if not os.path.exists(pdb_path(entry_id, structure_dir)):
            continue
        record = derive_sdf_and_smiles(entry_id, sequence, structure_dir)
        if record is None:
            failed += 1
            continue
        records[entry_id] = record

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)

    confident = sum(1 for r in records.values()
                    if r.get("plddt") is not None and r["plddt"] >= 70)
    summary = {
        "path": output_path,
        "structures": len(records),
        "unreadable": failed,
        "confident_plddt_70": confident,
    }
    logger.info(f"Derived SDF and SMILES for {len(records)} predicted structures "
                f"({confident} with pLDDT >= 70); {failed} could not be read.")
    return summary
