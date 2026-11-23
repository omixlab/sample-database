"""
Peptide topology: cyclic backbones, disulfide bridges, and the molecules they imply.

The pipeline treated every peptide as a linear chain. It is not: **1,315 DRAMP entries are marked
cyclic**, and DBAASP records exact bond positions for roughly a fifth of its cards -- 1,259
disulfide bonds, 532 amide bonds, 149 thioethers, 106 esters. That information was read by nothing,
so a cyclotide and a linear peptide of the same composition were indistinguishable downstream.

Treating a cyclic peptide as linear is wrong in ways that propagate:

* **Mass.** Head-to-tail cyclization condenses the termini and loses a water (18.011 Da); each
  disulfide loses two hydrogens (2.016 Da). A cyclotide reported as linear is several daltons
  heavy, which is more than enough to break an identity match against PubChem or ChEMBL.
* **Charge.** A head-to-tail cycle has no free alpha-amino or alpha-carboxyl group, so the two
  ionisable termini are not there to titrate. At pH 7.4 the correction is small -- the protonated
  N-terminus and deprotonated C-terminus very nearly cancel, leaving about 0.03 charge units --
  but it grows at the extremes of the pH range, and it is exactly zero effort to get right.
* **Structure.** The backbone ring and the disulfide bridges are the structure for a cyclotide --
  the cystine knot is why it resists proteases. A linear model of one is not a worse model, it is
  a different molecule.

DBAASP distinguishes the cases precisely, which is what makes this tractable: ``cycleType`` is
``NCB`` for a true N-to-C backbone bond and ``CST`` for a cystine bridge, so a head-to-tail cycle
is never confused with a peptide that merely contains disulfides.
"""
import glob
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

logger = logging.getLogger("sample.cyclization")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Mass lost when two termini condense into an amide bond, in daltons.
WATER_MASS: float = 18.0106

#: Mass lost when two thiols oxidise into a disulfide, in daltons.
DISULFIDE_MASS_LOSS: float = 2.0157

#: DBAASP bond-type codes worth modelling, mapped to a readable name. Types outside this set are
#: recorded but not built into the molecule -- naming them is useful, guessing their chemistry is
#: not.
BOND_TYPES: Dict[str, str] = {
    "DSB": "disulfide",
    "AMD": "amide",
    "TIE": "thioether",
    "EST": "ester",
    "DCB": "dicarba",
    "CAR": "carbon",
    "ETH": "ether",
    "AMN": "amine",
}

#: DBAASP cycle-type codes. ``NCB`` is the one that changes the backbone: a bond between the N and
#: C termini, i.e. a genuinely circular peptide.
HEAD_TO_TAIL_CODE = "NCB"
CYSTINE_CODE = "CST"


class Bond(NamedTuple):
    """
    One intrachain bond.

    :param position1: 1-based residue position of the first partner.
    :param position2: 1-based residue position of the second partner.
    :param bond_type: Chemistry, from :data:`BOND_TYPES`.
    :param cycle_type: The source's classification, e.g. ``NCB`` or ``CST``.
    """
    position1: int
    position2: int
    bond_type: str
    cycle_type: str

    @property
    def is_head_to_tail(self) -> bool:
        """:returns: Whether this bond closes the backbone into a ring."""
        return self.cycle_type.upper() == HEAD_TO_TAIL_CODE

    @property
    def is_disulfide(self) -> bool:
        """:returns: Whether this is a cystine bridge."""
        return self.bond_type == "disulfide" or self.cycle_type.upper() == CYSTINE_CODE


class Topology(NamedTuple):
    """
    A peptide's shape.

    :param kind: ``linear``, ``head_to_tail``, ``disulfide``, ``branched`` or ``unknown``.
    :param bonds: Intrachain bonds, where the source records them.
    :param source: Which database said so.
    """
    kind: str
    bonds: Tuple[Bond, ...]
    source: str

    @property
    def is_cyclic(self) -> bool:
        """
        :returns: Whether the peptide is cyclic by any route -- a closed backbone or at least one
            disulfide bridge. Both make the molecule a ring; only the first removes the termini.
        """
        return self.kind in ("head_to_tail", "disulfide") or bool(self.bonds)

    @property
    def n_disulfides(self) -> int:
        """:returns: How many cystine bridges the source records."""
        return sum(1 for b in self.bonds if b.is_disulfide)


def _parse_bonds(raw_bonds: Iterable[Dict[str, Any]]) -> List[Bond]:
    """
    Reads DBAASP's ``intrachainBonds`` block.

    :param raw_bonds: The block as stored.
    :returns: Bonds with usable positions, in source order.
    """
    bonds: List[Bond] = []
    for entry in raw_bonds or []:
        if not isinstance(entry, dict):
            continue
        try:
            position1 = int(entry.get("position1"))
            position2 = int(entry.get("position2"))
        except (TypeError, ValueError):
            continue
        if position1 <= 0 or position2 <= 0 or position1 == position2:
            continue

        type_block = entry.get("type") or {}
        cycle_block = entry.get("cycleType") or {}
        code = str(type_block.get("name") or "").strip().upper()
        bonds.append(Bond(
            position1=min(position1, position2),
            position2=max(position1, position2),
            bond_type=BOND_TYPES.get(code, code.lower() or "unknown"),
            cycle_type=str(cycle_block.get("name") or "").strip().upper(),
        ))
    return bonds


def classify(bonds: Iterable[Bond], declared: str = "") -> str:
    """
    Decides a peptide's topology from its bonds and any declared label.

    Bonds outrank the label. DRAMP's ``Linear/Cyclic/Branched`` column says *that* a peptide is
    cyclic without saying how, and 4,157 of its rows say only "Not included yet"; DBAASP's bond
    list says exactly how. Where both exist the bonds win, and the label is the fallback.

    :param bonds: Intrachain bonds.
    :param declared: A source's own topology label, if any.
    :returns: ``head_to_tail``, ``disulfide``, ``branched``, ``linear`` or ``unknown``.
    """
    bonds = list(bonds)
    if any(b.is_head_to_tail for b in bonds):
        return "head_to_tail"
    if any(b.is_disulfide for b in bonds):
        return "disulfide"

    label = str(declared or "").strip().lower()
    if label.startswith("cyclic"):
        # Cyclic by declaration, with no bond list to say how. Recorded honestly rather than
        # guessed into head_to_tail, which would assert a backbone bond nobody stated.
        return "cyclic_unspecified"
    if label.startswith("branched"):
        return "branched"
    if label.startswith("linear"):
        return "linear"
    if bonds:
        return "disulfide" if any(b.is_disulfide for b in bonds) else "branched"
    return "unknown"


def harvest_dbaasp(directory: Optional[str] = None) -> Dict[str, Topology]:
    """
    Reads topology from DBAASP cards, keyed by sequence.

    :param directory: DBAASP JSON directory; the packaged location when omitted.
    :returns: Sequence -> topology.
    """
    directory = directory or os.path.join(DATA_DIR, "antimicrobial_databases", "dbaasp")
    topologies: Dict[str, Topology] = {}

    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                card = json.load(handle)
        except (OSError, ValueError):
            continue

        sequence = str(card.get("sequence") or "").strip().upper()
        if not sequence:
            continue
        bonds = _parse_bonds(card.get("intrachainBonds") or [])
        kind = classify(bonds)
        if kind == "unknown" and not bonds:
            continue
        topologies[sequence] = Topology(kind=kind, bonds=tuple(bonds), source="DBAASP")

    logger.info(f"DBAASP: topology for {len(topologies)} sequences "
                f"({sum(1 for t in topologies.values() if t.kind == 'head_to_tail')} head-to-tail, "
                f"{sum(1 for t in topologies.values() if t.kind == 'disulfide')} disulfide-bonded).")
    return topologies


def harvest_dramp(path: Optional[str] = None) -> Dict[str, Topology]:
    """
    Reads DRAMP's declared topology column, keyed by sequence.

    :param path: DRAMP dataset; the packaged location when omitted.
    :returns: Sequence -> topology. Rows labelled "Not included yet" are omitted rather than
        recorded as linear -- absence of a statement is not a statement of absence.
    """
    import csv

    path = path or os.path.join(
        DATA_DIR, "antimicrobial_databases", "dramp", "Datasets", "general_amps.txt")
    topologies: Dict[str, Topology] = {}
    if not os.path.exists(path):
        return topologies

    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequence = str(row.get("Sequence") or "").strip().upper()
            declared = str(row.get("Linear/Cyclic/Branched") or "").strip()
            if not sequence or not declared or declared.lower().startswith("not included"):
                continue
            kind = classify([], declared)
            if kind == "unknown":
                continue
            topologies[sequence] = Topology(kind=kind, bonds=(), source="DRAMP")

    logger.info(f"DRAMP: topology for {len(topologies)} sequences "
                f"({sum(1 for t in topologies.values() if t.kind == 'cyclic_unspecified')} cyclic).")
    return topologies


def harvest_topologies() -> Dict[str, Topology]:
    """
    Merges topology from every source that records it.

    DBAASP wins on conflict because it states the bonds rather than a label, and a bond list is
    checkable against the sequence while a label is not.

    :returns: Sequence -> topology.
    """
    merged = dict(harvest_dramp())
    merged.update(harvest_dbaasp())
    logger.info(f"Topology known for {len(merged)} sequences.")
    return merged


def mass_correction(topology: Topology) -> float:
    """
    Mass to subtract from a linear peptide's weight, in daltons.

    A linear-chain calculation -- which is what ``peptides.py`` and a sequence-built RDKit molecule
    both give -- overstates a cyclic peptide's mass. Head-to-tail closure condenses the termini and
    releases water; each disulfide releases two hydrogens.

    :param topology: The peptide's topology.
    :returns: The correction, ``0.0`` for a linear peptide.
    """
    correction = 0.0
    if topology.kind == "head_to_tail":
        correction += WATER_MASS
    correction += DISULFIDE_MASS_LOSS * topology.n_disulfides
    return correction


#: Terminal pKa values, matching the scale `peptides.py` uses by default (Lehninger).
_N_TERMINUS_PKA: float = 9.69
_C_TERMINUS_PKA: float = 2.34


def terminal_charge(ph: float = 7.4) -> float:
    """
    Net charge contributed by a peptide's free termini at a given pH.

    Henderson-Hasselbalch on both: the alpha-amino group carries ``+1 / (1 + 10^(pH - pKa))`` and
    the alpha-carboxyl ``-1 / (1 + 10^(pKa - pH))``. A head-to-tail cycle has neither, so this is
    what must be subtracted from a linear-chain charge calculation.

    :param ph: The pH to evaluate at.
    :returns: The termini's combined contribution, in charge units.
    """
    positive = 1.0 / (1.0 + 10 ** (ph - _N_TERMINUS_PKA))
    negative = -1.0 / (1.0 + 10 ** (_C_TERMINUS_PKA - ph))
    return positive + negative


def charge_correction(topology: Optional["Topology"], ph: float = 7.4) -> float:
    """
    Charge to subtract from a linear-chain calculation for this topology.

    :param topology: The peptide's topology.
    :param ph: The pH the charge was computed at.
    :returns: The correction, ``0.0`` unless the backbone is closed.
    """
    if topology is None or topology.kind != "head_to_tail":
        return 0.0
    return terminal_charge(ph)


def _residue_atom_indices(mol: Any) -> Dict[int, Dict[str, int]]:
    """
    Maps residue number to the backbone and side-chain atoms cyclization needs.

    :param mol: A molecule built by ``Chem.MolFromSequence``, whose atoms carry PDB residue info.
    :returns: ``residue_number -> {"N", "C", "OXT", "SG"}`` for the atoms present.
    """
    indices: Dict[int, Dict[str, int]] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        name = (info.GetName() or "").strip()
        number = info.GetResidueNumber()
        if name in ("N", "C", "OXT", "SG"):
            indices.setdefault(number, {})[name] = atom.GetIdx()
    return indices


def cyclize(mol: Any, topology: Topology) -> Tuple[Any, List[str]]:
    """
    Applies a peptide's bonds to a linear molecule.

    Head-to-tail closure removes the C-terminal hydroxyl and bonds that carbonyl carbon to the
    N-terminal nitrogen -- the condensation that makes the peptide circular. Disulfides bond the
    two cysteine sulfurs directly. Both are done on the molecule built from the sequence, so the
    result is a real cyclic molecule rather than a linear one wearing a label.

    Bond types other than amide and disulfide are recorded but not built: a thioether or ester
    bridge needs to know which side-chain atoms participate, which the source does not state, and
    inventing that would produce a confidently wrong structure.

    :param mol: A molecule from ``Chem.MolFromSequence``.
    :param topology: The peptide's topology.
    :returns: ``(molecule, applied)`` where ``applied`` names the bonds actually formed. The
        molecule is returned unchanged when nothing could be applied.
    """
    from rdkit import Chem

    if mol is None or not topology.bonds:
        return mol, []

    editable = Chem.RWMol(mol)
    residues = _residue_atom_indices(editable)
    applied: List[str] = []
    to_remove: List[int] = []

    for bond in topology.bonds:
        if bond.is_head_to_tail:
            first = residues.get(bond.position1, {})
            last = residues.get(bond.position2, {})
            nitrogen, carbon = first.get("N"), last.get("C")
            if nitrogen is None or carbon is None:
                continue
            if editable.GetBondBetweenAtoms(nitrogen, carbon) is not None:
                continue
            # The terminal hydroxyl leaves as part of the water condensed out.
            oxt = last.get("OXT")
            if oxt is not None:
                to_remove.append(oxt)
            editable.AddBond(nitrogen, carbon, Chem.BondType.SINGLE)
            applied.append(f"head_to_tail:{bond.position1}-{bond.position2}")

        elif bond.is_disulfide:
            sulfur1 = residues.get(bond.position1, {}).get("SG")
            sulfur2 = residues.get(bond.position2, {}).get("SG")
            if sulfur1 is None or sulfur2 is None:
                # The stated positions are not both cysteines; the bond is recorded but the
                # structure is left alone rather than bonded to whatever happens to be there.
                continue
            if editable.GetBondBetweenAtoms(sulfur1, sulfur2) is not None:
                continue
            editable.AddBond(sulfur1, sulfur2, Chem.BondType.SINGLE)
            applied.append(f"disulfide:{bond.position1}-{bond.position2}")

    if not applied:
        return mol, []

    # Removed last, and in descending order, so earlier deletions do not shift the indices of
    # later ones.
    for index in sorted(set(to_remove), reverse=True):
        editable.RemoveAtom(index)

    try:
        result = editable.GetMol()
        Chem.SanitizeMol(result)
    except Exception as e:
        logger.debug(f"Could not sanitize cyclized molecule: {e}")
        return mol, []
    return result, applied


def molecule_for(sequence: str, topology: Optional[Topology] = None) -> Tuple[Any, List[str]]:
    """
    Builds the molecule a peptide actually is, cyclic or not.

    :param sequence: One-letter sequence.
    :param topology: The peptide's topology; treated as linear when omitted.
    :returns: ``(molecule, applied_bonds)``, or ``(None, [])`` when the sequence cannot be built.
    """
    from rdkit import Chem

    text = re.sub(r"[^A-Za-z]", "", str(sequence or "")).upper()
    if not text:
        return None, []

    mol = Chem.MolFromSequence(text)
    if mol is None or topology is None:
        return mol, []
    return cyclize(mol, topology)


def build_topologies(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Harvests topology and writes it for the enrichment and load steps.

    :param output_path: Where to write; ``data/peptide_topology.json`` when omitted.
    :returns: A summary of what was written.
    """
    topologies = harvest_topologies()
    output_path = output_path or os.path.join(DATA_DIR, "peptide_topology.json")

    payload = {
        sequence: {
            "kind": topology.kind,
            "source": topology.source,
            "n_disulfides": topology.n_disulfides,
            "mass_correction": round(mass_correction(topology), 4),
            "bonds": [{"position1": b.position1, "position2": b.position2,
                       "bond_type": b.bond_type, "cycle_type": b.cycle_type}
                      for b in topology.bonds],
        }
        for sequence, topology in topologies.items()
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)

    counts: Dict[str, int] = {}
    for topology in topologies.values():
        counts[topology.kind] = counts.get(topology.kind, 0) + 1

    summary = {"path": output_path, "sequences": len(payload), "by_kind": counts}
    logger.info(f"Wrote topology for {len(payload)} sequences to {output_path}: {counts}")
    return summary
