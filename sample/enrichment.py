import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import logging
import concurrent.futures
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger("sample.enrichment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from sample.sequence import get_peptide_charge, sequence_to_1_letter

@lru_cache(maxsize=1)
def _topology_index() -> Dict[str, Any]:
    """
    Loads the peptide topology harvested by ``sample-cli build-topology``.

    :returns: Sequence -> topology record, empty when the file has not been built. Absent topology
        means every peptide is treated as linear, which is what the pipeline did before -- so a
        missing file degrades to the old behaviour rather than failing.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "peptide_topology.json")

    # Built on demand rather than made a numbered step of its own. It reads only the source
    # databases, so it has no ordering constraint worth a pipeline slot, and deriving it here
    # means enrichment cannot silently treat cyclic peptides as linear just because a step was
    # skipped. `sample-cli build-topology` refreshes it explicitly.
    if not os.path.exists(path):
        try:
            from sample.cyclization import build_topologies

            logger.info("No peptide topology file; harvesting it from the source databases.")
            build_topologies(output_path=path)
        except Exception as e:
            logger.warning(f"Could not build peptide topology ({e}); treating every peptide as "
                           f"linear.")
            return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read peptide topology: {e}")
        return {}


def topology_for(sequence: Any) -> Optional[Any]:
    """
    :param sequence: One-letter sequence.
    :returns: A :class:`~sample.cyclization.Topology` for it, or ``None`` when unknown.
    """
    from sample.cyclization import Bond, Topology

    text = re.sub(r"[^A-Za-z]", "", str(sequence or "")).upper()
    record = _topology_index().get(text)
    if not record:
        return None
    return Topology(
        kind=record.get("kind", "unknown"),
        bonds=tuple(Bond(b["position1"], b["position2"], b["bond_type"], b["cycle_type"])
                    for b in record.get("bonds") or []),
        source=record.get("source", ""))


def molecule_from_sequence(sequence: Any) -> Optional[Any]:
    """
    Builds a peptide molecule from its sequence.

    **The sequence is the only input this step trusts.** The stored SMILES column is unvalidated
    and holds sentinels, truncated generations, unbalanced parentheses and -- in 79 rows -- the
    peptide's own sequence pasted in verbatim. Every tool here consumed it, so a corrupt cell
    became a missing descriptor, a failed structure, and several lines of RDKit output apiece.

    ``MolFromSequence`` builds the same peptide from the field the dataset is confident about,
    with correct topology and valences by construction. A peptide that cannot be built this way
    has non-standard residues, which is a real property of the peptide rather than a data fault.

    :param sequence: One-letter sequence.
    :returns: The molecule, or ``None`` when the sequence is empty or non-standard.
    """
    from sample.cyclization import molecule_for

    # Cyclic where the sources say so: a head-to-tail backbone bond and any disulfide bridges are
    # applied, so a cyclotide is built as the macrocycle it is rather than an open chain several
    # daltons too heavy.
    mol, _ = molecule_for(sequence, topology_for(sequence))
    return mol


#: Peptide descriptors taken from `peptides.py`, as ``column -> (method, kwargs)``.
#:
#: These are the properties that actually characterise an antimicrobial peptide -- net charge at
#: physiological pH, amphipathicity, the Boman index of protein-binding potential -- and they are
#: defined on the SEQUENCE. RDKit's generic cheminformatics descriptors are computed alongside
#: them from the same sequence-built molecule, but they describe a molecule rather than a peptide:
#: LogP of a 23-residue cation is not a number anyone reasons with, while its hydrophobic moment
#: is.
PEPTIDE_DESCRIPTORS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "isoelectric_point": ("isoelectric_point", {}),
    "boman_index": ("boman", {}),
    "aliphatic_index": ("aliphatic_index", {}),
    "instability_index": ("instability_index", {}),
    "hydrophobicity": ("hydrophobicity", {"scale": "KyteDoolittle"}),
    "hydrophobic_moment": ("hydrophobic_moment", {"angle": 100}),
}


def peptide_descriptors(sequence: Any, failure_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Computes sequence-defined peptide descriptors with `peptides.py`.

    :param sequence: One-letter sequence.
    :param failure_counts: When given, incremented in place per descriptor column that fell back
        to ``nan`` for this sequence -- a non-standard residue commonly defeats one descriptor's
        lookup table without affecting the others, so failure is per-column, not per-sequence.
        Callers processing many sequences pass a shared dict and log one summary afterward rather
        than a line per sequence; omit it to leave the per-sequence NaN as the only record, as
        every caller before this one did.
    :returns: Descriptor name -> value, with ``nan`` for anything the library declines to compute.
        An empty mapping when the sequence is unusable.
    :raises ImportError: Never -- a missing library yields an empty mapping and a warning, so the
        rest of the enrichment still runs.
    """
    text = re.sub(r"[^A-Za-z]", "", str(sequence or "")).upper()
    if not text:
        return {}

    try:
        import peptides
    except ImportError:
        logger.warning("peptides.py is not installed; peptide descriptors will be empty. "
                       "Install it with `pip install peptides`.")
        return {}

    try:
        peptide = peptides.Peptide(text)
    except Exception as e:
        logger.debug(f"peptides.py could not read {text[:20]!r}: {e}")
        return {}

    values: Dict[str, Any] = {}
    # Taken from peptides.py rather than RDKit: it is computed from the residue composition, so it
    # is available for sequences carrying non-standard residues that RDKit cannot build into a
    # molecule at all. Same number where both can answer -- 2466.93 against 2466.94 for magainin-2.
    #
    # Corrected for topology. peptides.py computes a LINEAR chain, so a head-to-tail cyclic
    # peptide comes out one water too heavy and each disulfide two hydrogens too heavy. On a
    # cyclotide that is 22 Da -- far more than enough to break an identity match against PubChem
    # or ChEMBL, which is exactly what step 12 does with these numbers.
    try:
        from sample.cyclization import mass_correction

        weight = float(peptide.molecular_weight())
        topology = topology_for(text)
        values["molecular_weight"] = weight - (mass_correction(topology) if topology else 0.0)
        values["topology"] = topology.kind if topology else "unknown"
        values["n_disulfides"] = topology.n_disulfides if topology else 0
        values["is_cyclic"] = int(bool(topology and topology.is_cyclic))
    except Exception:
        values["molecular_weight"] = np.nan
        values["topology"] = "unknown"
        values["n_disulfides"] = 0
        values["is_cyclic"] = 0
        if failure_counts is not None:
            failure_counts["molecular_weight"] = failure_counts.get("molecular_weight", 0) + 1

    for column, (method, kwargs) in PEPTIDE_DESCRIPTORS.items():
        try:
            values[column] = float(getattr(peptide, method)(**kwargs))
        except Exception:
            # One descriptor failing -- an instability index needs a dipeptide table lookup that
            # a non-standard residue defeats -- must not cost the others.
            values[column] = np.nan
            if failure_counts is not None:
                failure_counts[column] = failure_counts.get(column, 0) + 1
    try:
        from sample.cyclization import charge_correction

        # Corrected for a closed backbone, which has no free termini to titrate.
        values["net_charge_ph7"] = float(peptide.charge(pH=7.4)) - charge_correction(
            topology_for(text), ph=7.4)
    except Exception:
        values["net_charge_ph7"] = np.nan
        if failure_counts is not None:
            failure_counts["net_charge_ph7"] = failure_counts.get("net_charge_ph7", 0) + 1
    return values


def enrich_with_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates peptide and molecular descriptors from the sequences.

    Two sources, both keyed on the sequence and neither on the stored SMILES:
    `peptides.py` for the sequence-defined peptide properties (see :data:`PEPTIDE_DESCRIPTORS`),
    and RDKit over :func:`molecule_from_sequence` for the generic chemistry columns the released
    schema already carries.

    :param df: Rows carrying ``sequence``.
    :returns: The same rows with descriptor columns added.
    """
    logger.info("Enriching peptide dataset with peptide and molecular descriptors...")

    # RDKit's logger is per-process and this stage runs first, so the log has to be quietened
    # here as well as in the structure stage.
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    unusable = 0
    charge_failures = 0
    descriptor_failures: Dict[str, int] = {}
    peptide_rows: List[Dict[str, Any]] = []
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []
    charge_list = []
    seq_len_list = []
    
    for idx, row in df.iterrows():
        seq = row.get("sequence")
        smiles = row.get("SMILES")
        
        mw, logp, hbd, hba = np.nan, np.nan, np.nan, np.nan
        charge = np.nan
        seq_len = len(str(seq)) if pd.notna(seq) else 0
        
        if pd.notna(seq) and seq:
            try:
                mol = molecule_from_sequence(seq)
                if mol:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                else:
                    unusable += 1
            except Exception:
                unusable += 1
                
        if pd.notna(seq) and seq:
            try:
                charge = get_peptide_charge(seq)
            except Exception:
                charge_failures += 1

        mw_list.append(mw)
        logp_list.append(logp)
        hbd_list.append(hbd)
        hba_list.append(hba)
        charge_list.append(charge)
        seq_len_list.append(seq_len)
        peptide_rows.append(peptide_descriptors(seq, descriptor_failures)
                            if pd.notna(seq) and seq else {})

    peptide_frame = pd.DataFrame(peptide_rows, index=df.index)

    df_enriched = df.copy()
    # peptides.py first, RDKit only where it has no answer.
    df_enriched["molecular_weight"] = peptide_frame.get(
        "molecular_weight", pd.Series(np.nan, index=df.index)).fillna(pd.Series(mw_list,
                                                                                index=df.index))
    df_enriched["logp"] = logp_list
    df_enriched["hydrogen_bond_donors"] = hbd_list
    df_enriched["hydrogen_bond_acceptors"] = hba_list
    df_enriched["charge"] = charge_list
    df_enriched["sequence_length"] = seq_len_list

    # peptides.py columns, added beside the existing ones rather than replacing them: `charge`
    # stays as sample.sequence computes it so the released column keeps its meaning, and
    # `net_charge_ph7` carries peptides.py's value alongside for comparison.
    for column in list(PEPTIDE_DESCRIPTORS) + ["net_charge_ph7"]:
        df_enriched[column] = peptide_frame.get(column, np.nan)
    # Topology travels with the descriptors: a consumer comparing masses or charges needs to know
    # whether the termini exist.
    for column, default in (("topology", "unknown"), ("n_disulfides", 0), ("is_cyclic", 0)):
        df_enriched[column] = peptide_frame.get(column, default)

    computed = int(peptide_frame.notna().any(axis=1).sum()) if len(peptide_frame) else 0
    logger.info(f"Descriptors complete: {computed}/{len(df_enriched)} sequences carry peptide "
                f"descriptors; {unusable} could not be built into a molecule (non-standard "
                f"residues).")
    if charge_failures:
        logger.warning(f"sample.sequence charge computation failed for {charge_failures}/"
                       f"{len(df_enriched)} sequences; `charge` is left NaN for those rows.")
    if descriptor_failures:
        logger.info(f"peptides.py descriptor computation: {descriptor_failures} (column -> "
                   f"failure count) across {len(df_enriched)} sequences; NaN filled in per "
                   f"column, one failing descriptor does not block the others for that sequence.")
    return df_enriched

def enrich_with_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates peptide embeddings.
    Since loading large transformer models (like ESM) requires heavy downloads and RAM,
    we compute standard amino acid frequency composition vectors (20 dimensions),
    which act as excellent sequence embeddings for classical ML classification.
    """
    logger.info("Enriching peptide dataset with frequency embeddings...")
    
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aa_groups = {
        "aliphatic": list("AILV"),
        "aromatic": list("FWY"),
        "acidic": list("DE"),
        "basic": list("RHK"),
        "polar_uncharged": list("STNQCM"),
        "small_tiny": list("GAS"),
        "sulfur": list("CM"),
        "proline": list("P")
    }
    
    embedding_rows = []
    
    for idx, row in df.iterrows():
        seq = sequence_to_1_letter(str(row.get("sequence", "")))
        seq_len = len(seq)
        
        vector = []
        # Single AA frequencies
        for aa in amino_acids:
            freq = (seq.count(aa) / seq_len) if seq_len > 0 else 0.0
            vector.append(freq)
            
        # Grouped AA frequencies
        for group, residues in aa_groups.items():
            count = sum(seq.count(r) for r in residues)
            freq = (count / seq_len) if seq_len > 0 else 0.0
            vector.append(freq)
            
        embedding_rows.append(vector)
        
    df_emb = df.copy()
    emb_cols = [f"aa_freq_{aa}" for aa in amino_acids] + [f"aa_group_freq_{group}" for group in aa_groups.keys()]

    # Wide blocks are collected and concatenated once at the end rather than assigned into the
    # frame as they are produced. Each assignment of a 320- or 1024-column block into an already
    # wide frame copies it, which pandas reports as "DataFrame is highly fragmented" -- once per
    # block, repeatedly, on a 37,000-row frame.
    blocks: List[pd.DataFrame] = [
        pd.DataFrame(embedding_rows, columns=emb_cols, index=df_emb.index)]

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    # Both models below are small (ProtBERT ~420M params, ESM-2 8M) relative to a 12GB GPU;
    # a larger batch keeps the GPU fed instead of bottlenecking on tokenizer/host->device transfer.
    batch_size = 64 if device.type == "cuda" else 16

    # Computed once and reused for both models instead of re-materializing per model.
    raw_sequences = [sequence_to_1_letter(str(seq)) for seq in df.get("sequence", pd.Series(dtype=str)).fillna("")]

    def _mean_pooled_embeddings(model: Any, tokenizer: Any, sequences: List[str],
                                spaced: bool = False) -> Any:
        """
        Embeds sequences and mean-pools over residues to one vector each.

        :param model: Loaded transformer.
        :param tokenizer: Matching tokenizer.
        :param sequences: One-letter sequences to embed.
        :param spaced: Insert spaces between residues, which some protein models require.
        :returns: One row per sequence.
        """
        embeddings = []
        texts = [" ".join(list(seq)) for seq in sequences] if spaced else sequences
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                try:
                    outputs = model(**inputs)
                except torch.cuda.OutOfMemoryError:
                    # Fall back to a smaller sub-batch rather than aborting the whole run.
                    torch.cuda.empty_cache()
                    half = max(1, len(batch) // 2)
                    outputs = None
                    for j in range(0, len(batch), half):
                        sub_inputs = tokenizer(batch[j:j + half], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                        sub_outputs = model(**sub_inputs)
                        _accumulate_mean_pooled(embeddings, sub_outputs, sub_inputs)
                    continue

                _accumulate_mean_pooled(embeddings, outputs, inputs)
        return embeddings

    def _accumulate_mean_pooled(out_list: List[Any], outputs: Any, inputs: Any) -> None:
        """
        Mean-pools one batch, masking padding, and appends the rows.

        :param out_list: Accumulator appended to in place.
        :param outputs: Model outputs for the batch.
        :param inputs: Tokenizer inputs, whose attention mask excludes padding from the mean.
        :returns: ``None``.
        """
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_embeds = outputs.last_hidden_state.float() * attention_mask
        sum_embeds = torch.sum(masked_embeds, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_embeds = (sum_embeds / sum_mask).cpu().numpy()
        out_list.extend(mean_embeds)

    # ESM-2 (8M) is the only protein language model used here. ProtBERT was removed: it ships
    # only a `pytorch_model.bin` with no safetensors, and transformers 5.x refuses to `torch.load`
    # such a checkpoint below torch 2.6 -- while this project pins torch 2.5.1+cu121 deliberately,
    # for the CUDA 12.2 driver. The combination is unresolvable without giving up the CUDA pin,
    # and ESM-2 is the stronger protein language model regardless.
    #
    # Note for anyone restoring it: the failure was caught and logged, so runs completed while
    # silently producing zero of the 1,024 ProtBERT columns.
    try:
        logger.info(f"Loading ESM-2 model on {device} (batch_size={batch_size}, fp16={use_fp16})...")
        tokenizer_esm2 = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model_esm2 = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D", torch_dtype=torch.float16 if use_fp16 else None).to(device)
        model_esm2.eval()

        esm2_embeddings = _mean_pooled_embeddings(model_esm2, tokenizer_esm2, raw_sequences, spaced=False)

        # Append to df_emb
        esm2_dim = len(esm2_embeddings[0]) if esm2_embeddings else 320
        esm2_cols = [f"esm2_dim_{i}" for i in range(esm2_dim)]
        blocks.append(pd.DataFrame(esm2_embeddings, columns=esm2_cols, index=df_emb.index))
        logger.info("ESM-2 embeddings added successfully.")

        del model_esm2
        del tokenizer_esm2
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Failed to generate ESM-2 embeddings: {e}")

    # One concatenation for every wide block, so the frame is built once rather than copied per
    # block.
    df_emb = pd.concat([df_emb] + blocks, axis=1)

    logger.info(f"Embeddings enrichment complete "
                f"({sum(b.shape[1] for b in blocks)} embedding columns added).")
    return df_emb



def enrich_with_2D_and_3D_structures(df: pd.DataFrame, out_dir: str,
                                     skip_entry_ids: Optional[set] = None,
                                     max_length: Optional[int] = None) -> pd.DataFrame:
    """
    Builds 3D structures by distance-geometry embedding and force-field minimisation.

    This is the path for **short peptides**, not merely a fallback. Below about 15 residues there
    is no tertiary structure to predict: the conformation is set by local geometry, which is what
    ETKDG plus MMFF94 models well and what a co-evolution model has nothing to say about.
    AlphaFold2 returns low-confidence output for these regardless, so an energy-minimised
    conformer is both the faster answer and the more honest one -- it does not claim to be a
    predicted fold.

    Longer peptides go to ColabFold (see :mod:`sample.folding`), and those entries arrive in
    ``skip_entry_ids`` so a prediction is never overwritten by an embedding. Sequences too long
    for either route get no structure rather than a meaningless one: embedding a 700-residue chain
    with distance geometry is slow and produces a conformer with no relation to the real fold.

    :param df: Rows carrying ``entry_id`` and ``sequence``.
    :param out_dir: Directory for the SDF files.
    :param skip_entry_ids: Entries that already have a predicted structure.
    :param max_length: Longest sequence to embed; :data:`~sample.folding.SHORT_PEPTIDE_LENGTH`
        when omitted.
    :returns: ``None``; structures are written to disk.
    """
    from sample.folding import SHORT_PEPTIDE_LENGTH

    max_length = SHORT_PEPTIDE_LENGTH if max_length is None else max_length
    logger.info(f"Generating 2D/3D structures and writing to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)

    # RDKit logs every parse failure to stderr, and this dataset has enough malformed SMILES to
    # bury the run's own output in tens of thousands of multi-line reports. They are counted and
    # summarised below instead, which is the actionable form.
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    skip_entry_ids = skip_entry_ids or set()
    jobs = []
    too_long = 0
    for _, row in df.iterrows():
        entry_id = row.get("entry_id")
        sequence = row.get("sequence")
        if pd.isna(entry_id) or not entry_id or pd.isna(sequence) or not sequence:
            continue
        if str(entry_id) in skip_entry_ids:
            continue
        if len(re.sub(r"[^A-Za-z]", "", str(sequence))) > max_length:
            too_long += 1
            continue
        jobs.append((str(entry_id), str(sequence), out_dir))

    if skip_entry_ids:
        logger.info(f"{len(skip_entry_ids)} peptides already have a predicted structure and are "
                    f"left alone; distance geometry only fills the remainder.")
    if too_long:
        logger.info(
            f"{too_long} sequences longer than {max_length} residues were left without a "
            f"structure: they are ColabFold's job, and distance geometry over a long chain "
            f"produces a conformer unrelated to the real fold.")

    if not jobs:
        logger.info(f"Structure generation complete. Wrote 0 2D models and 0 3D models to {out_dir}.")
        return

    success_2d = 0
    success_3d = 0
    workers = max(1, min(os.cpu_count() or 1, len(jobs)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for is_2d, is_3d in executor.map(_generate_structure_for_row, jobs, chunksize=32):
            success_2d += int(is_2d)
            success_3d += int(is_3d)

    unbuildable = len(jobs) - success_2d
    logger.info(
        f"Structure generation complete. Wrote {success_2d} structures to {out_dir}, "
        f"{success_3d} of them with a converged 3D geometry.")
    if unbuildable:
        logger.info(
            f"{unbuildable} sequences could not be built into a molecule -- they contain "
            f"non-standard residues, which is a property of the peptide rather than a data fault.")
    if success_2d and success_3d < success_2d:
        logger.info(
            f"{success_2d - success_3d} molecules were written as 2D or unconverged 3D: large "
            f"flexible peptides do not always embed, and an unconverged geometry is recorded as "
            f"such in the SDF rather than presented as a minimized structure.")


#: Force-field iterations per optimization pass. RDKit's default of 200 rarely converges a
#: peptide: these molecules run to hundreds of rotatable bonds, and an unconverged geometry is a
#: strained one, which is worse than useless for anything downstream that reads the coordinates.
OPTIMIZE_MAX_ITERATIONS: int = 2000

#: Optimization passes before giving up on convergence. Each pass restarts from where the last
#: finished, so this is a cheap way to give hard molecules more iterations without paying for them
#: on easy ones.
OPTIMIZE_MAX_PASSES: int = 5


def _embed_3d(mol_3d: "Chem.Mol") -> bool:
    """
    Generates a 3D conformer.

    Two attempts, because plain distance-geometry embedding fails outright on large flexible
    peptides. ETKDGv3 first -- it uses torsion-angle preferences from experimental structures and
    is markedly better on peptide backbones -- then a retry with random starting coordinates,
    which is the documented escape hatch for molecules whose bounds matrix cannot be satisfied by
    the ordinary route.

    :param mol_3d: A molecule with explicit hydrogens, modified in place.
    :returns: Whether a conformer was produced.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 1000
    params.useSmallRingTorsions = True

    if AllChem.EmbedMolecule(mol_3d, params) == 0:
        return True

    params.useRandomCoords = True
    return AllChem.EmbedMolecule(mol_3d, params) == 0


def _optimize_3d(mol_3d: "Chem.Mol") -> Tuple[bool, str]:
    """
    Minimizes a conformer's energy with a force field.

    MMFF94 where the molecule is fully parameterized, UFF otherwise -- MMFF is the better field
    for organic molecules but has no parameters for some of the modified residues and metal
    coordination in this dataset, and asking for it anyway returns a null force field rather than
    raising.

    Convergence is checked rather than assumed. Both optimizers return 0 for converged and 1 for
    "hit the iteration cap", and the previous version discarded that return value and recorded
    every molecule as a success -- including ones that never embedded, since a failed embed raises
    nothing and simply leaves the molecule without coordinates.

    :param mol_3d: A molecule with a conformer, modified in place.
    :returns: ``(converged, force_field_used)``.
    """
    if AllChem.MMFFHasAllMoleculeParams(mol_3d):
        field, optimize = "MMFF94", AllChem.MMFFOptimizeMolecule
    else:
        field, optimize = "UFF", AllChem.UFFOptimizeMolecule

    for _ in range(OPTIMIZE_MAX_PASSES):
        result = optimize(mol_3d, maxIters=OPTIMIZE_MAX_ITERATIONS)
        if result == 0:
            return True, field
        if result < 0:
            # No force field could be set up; the geometry is the raw embedding.
            return False, "none"
    return False, field


def _generate_structure_for_row(job: tuple) -> Tuple[bool, bool]:
    """
    Runs in a worker process: builds one molecule and writes its own SDF file.

    Embarrassingly parallel over rows, so each embed and minimization runs on its own core.

    A molecule is only reported as 3D when it actually has an optimized conformer. When the embed
    or the force field fails the 2D depiction is still written -- it is a usable structure -- but
    it is not counted as a 3D model, so the reported totals mean what they say.

    :param job: ``(entry_id, sequence, out_dir)``.
    :returns: ``(wrote_2d, wrote_optimized_3d)``.
    """
    entry_id, sequence, out_dir = job
    try:
        # Built from the sequence, never from the stored SMILES: this step takes one input, and
        # it is the field the dataset is confident about.
        mol = molecule_from_sequence(sequence)
        if mol is None:
            return False, False

        AllChem.Compute2DCoords(mol)

        mol_3d = Chem.AddHs(mol)
        converged = False
        field = "none"
        if _embed_3d(mol_3d):
            converged, field = _optimize_3d(mol_3d)
            mol_to_write = mol_3d
        else:
            mol_to_write = mol

        mol_to_write.SetProp("_Name", entry_id)
        mol_to_write.SetProp("Sequence", sequence)
        # Recorded in the file itself so a consumer can tell a minimized geometry from a raw
        # embedding or a flat depiction without re-deriving it.
        mol_to_write.SetProp("Dimensionality", "3D" if mol_to_write is mol_3d else "2D")
        mol_to_write.SetProp("ForceField", field)
        mol_to_write.SetProp("Converged", "true" if converged else "false")

        sdf_path = os.path.join(out_dir, f"{entry_id}.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol_to_write)
        writer.close()

        return True, converged
    except Exception as e:
        logger.debug(f"Structure generation failed for {entry_id}: {e}")
        return False, False
