"""
Structure-derived analysis of a folded peptide: secondary structure, Ramachandran, disorder.

Once a peptide has a real predicted structure (step 27's ColabFold PDB), three things it cannot say
from sequence alone become answerable directly from the coordinates, and each drives a figure on the
peptide page:

* **Secondary structure via DSSP.** DSSP assigns each residue a state (helix / strand / turn / coil)
  from the actual hydrogen-bond geometry, which is what the fold reports -- a sequence propensity
  (:mod:`sample.secondary_structure`) is a guess at the same thing and is kept only as a fallback
  when no fold exists or DSSP is not installed.
* **The Ramachandran plot.** The backbone phi/psi of every residue, scored against the Top500
  reference distributions (Lovell et al. 2003, as packaged by `pyrama`) at MolProbity's cutoffs --
  favored at or above 2% density, allowed at or above 0.05% (0.2% for Gly/Pro/pre-Pro). Crucially
  the score is per residue CLASS: (60, -120) is sparse-but-observed for a general residue, well
  populated for glycine, and forbidden for proline, and one set of basins cannot say all three.
  A coarse box approximation remains as a fallback, so a deployment without the tables gets a
  coarser call rather than no figure.
* **Disorder from pLDDT.** ColabFold writes per-residue pLDDT into the PDB B-factor column, and a
  low pLDDT over a stretch is the standard AlphaFold-era proxy for intrinsic disorder. The
  per-residue confidence is turned into a disorder track and a summary fraction.

DSSP is reached by shell-out (its `mkdssp` binary), so the environment that serves the page need
not embed it. The phi/psi angles and the pLDDT track come straight from BioPython and need nothing
further; only the Ramachandran *classification* reads an external resource, and it reads `pyrama`'s
data files directly rather than importing the package, whose ``__init__`` pulls in
``matplotlib.pyplot`` -- a request path should not be initialising a plotting backend.
"""
import functools
import logging
import math
import os
import shutil
import sys
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.structure_analysis")

#: pLDDT at or below this is treated as likely disordered -- the conventional AlphaFold cut for
#: "very low confidence", which over a run tracks intrinsic disorder well.
DISORDER_PLDDT: float = 50.0

#: DSSP's eight states collapsed to the three the page shows. DSSP letters: H/G/I are helices,
#: E/B are strands, everything else is loop/turn/coil.
_DSSP_THREE_STATE: Dict[str, str] = {
    "H": "helix", "G": "helix", "I": "helix", "E": "sheet", "B": "sheet",
    "T": "coil", "S": "coil", "-": "coil", " ": "coil", "P": "coil",
}


def _first_model(pdb_path: str) -> Any:
    """
    :param pdb_path: A PDB file.
    :returns: The first model of the parsed structure.
    :raises FileNotFoundError: When the file does not exist.
    """
    from Bio.PDB import PDBParser

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(pdb_path)
    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    return next(iter(structure))


#: FALLBACK ONLY: a coarse box approximation of the three classic Ramachandran basins, used when
#: the reference tables below cannot be loaded. It is kept because an absent optional dependency
#: should coarsen the call, not remove the figure -- but it is no longer what the page reports.
#: Its boxes are visibly too generous: on a 33-residue helix it called 6 residues "allowed" that
#: the real distribution scores as favored, and it missed an outlier at phi=-21 psi=-27 whose
#: reference density is 1e-05.
#:
#: Each basin is a list of (phi_min, phi_max, psi_min, psi_max) boxes in degrees -- beta/extended
#: gets two, since its psi values cluster near +-180 and BioPython reports psi on a -180..180
#: scale, so the basin is split across the wrap point rather than expressed with one contiguous box.
_RAMA_FAVORED: Dict[str, List[tuple]] = {
    "alpha_r": [(-100.0, -30.0, -67.0, -7.0)],
    "beta": [(-180.0, -45.0, 90.0, 180.0), (-180.0, -45.0, -180.0, -170.0)],
    "alpha_l": [(30.0, 100.0, -7.0, 80.0)],
}
_RAMA_ALLOWED: Dict[str, List[tuple]] = {
    "alpha_r": [(-130.0, -15.0, -90.0, 30.0)],
    "beta": [(-180.0, -40.0, 60.0, 180.0), (-180.0, -40.0, -180.0, -150.0)],
    "alpha_l": [(15.0, 130.0, -30.0, 90.0)],
}


def _in_any_box(phi: float, psi: float, boxes: List[tuple]) -> bool:
    return any(phi_min <= phi <= phi_max and psi_min <= psi <= psi_max
               for phi_min, phi_max, psi_min, psi_max in boxes)


#: MolProbity's contour cutoffs against the Top500 reference densities: at or above 2% of the
#: reference maximum is "favored"; below the per-class allowed cutoff is an outlier.
RAMA_FAVORED_CUTOFF: float = 0.02

#: The allowed cutoff is looser for glycine, proline and pre-proline because their distributions
#: are broader and sparser; these are pyrama's own `RAMA_PREFERENCES[...]["bounds"][1]` values.
RAMA_ALLOWED_CUTOFF: Dict[str, float] = {
    "General": 0.0005, "GLY": 0.002, "PRO": 0.002, "PRE-PRO": 0.002,
}

#: The four Top500 tables (Lovell et al. 2003, B<30), one per residue class, as packaged by
#: `pyrama`. Only the data files are read -- never `import pyrama`, whose ``__init__`` pulls in
#: ``matplotlib.pyplot``, and a request path has no business initialising a plotting backend.
_REFERENCE_TABLES: Dict[str, str] = {
    "General": "pref_general.data", "GLY": "pref_glycine.data",
    "PRO": "pref_proline.data", "PRE-PRO": "pref_preproline.data",
}

#: How the classification is described wherever it is reported, so the page can never claim a
#: rigour the classifier did not use.
RAMA_REFERENCE_NAME: str = "Top500 reference distributions (Lovell et al. 2003), via pyrama"


@functools.lru_cache(maxsize=1)
def _reference_tables() -> Optional[Dict[str, Any]]:
    """
    Loads the four Top500 tables into 180x180 grids indexed ``[psi_bin, phi_bin]`` at 2 degrees.

    :returns: ``{residue_class: numpy grid}``, or ``None`` when pyrama or numpy is unavailable --
        the signal for :func:`classify_ramachandran` to fall back to the box approximation rather
        than fail. Cached: the four tables parse once per process, not once per request.
    """
    try:
        import importlib.metadata as metadata
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dependency in every served env
        return None
    try:
        files = list(metadata.files("pyrama") or [])
    except Exception:
        logger.info("pyrama is not installed; Ramachandran regions fall back to the box test.")
        return None

    grids: Dict[str, Any] = {}
    for kind, stem in _REFERENCE_TABLES.items():
        match = next((f for f in files if str(f).endswith(stem)), None)
        if match is None:
            logger.warning(f"pyrama is installed but {stem} is missing; using the box test.")
            return None
        rows = np.loadtxt(match.locate(), comments="#")
        grid = np.zeros((180, 180))
        for phi, psi, value in rows:
            grid[int((psi + 180) // 2), int((phi + 180) // 2)] = value
        grids[kind] = grid
    return grids


def _reference_bin(angle: float) -> int:
    """
    :param angle: An angle in degrees.
    :returns: Its 2-degree bin index. The modulo makes the grid wrap, so +180 and -180 land in the
        same bin -- they are the same conformation, and a table that stopped dead at one end would
        call one of them an outlier.
    """
    return int(((angle + 180.0) % 360.0) / 2.0)


def residue_type_for(resname: str, next_resname: Optional[str] = None) -> str:
    """
    The MolProbity residue class a phi/psi pair must be scored against.

    Glycine and proline take precedence over pre-proline: the Top500 pre-proline table is
    explicitly "not Gly or Pro".

    :param resname: Three-letter residue name.
    :param next_resname: The following residue's name, or ``None`` at a chain end.
    :returns: ``"GLY"``, ``"PRO"``, ``"PRE-PRO"`` or ``"General"``.
    """
    name = (resname or "").strip().upper()
    if name == "GLY":
        return "GLY"
    if name == "PRO":
        return "PRO"
    if (next_resname or "").strip().upper() == "PRO":
        return "PRE-PRO"
    return "General"


def reference_density(phi: Optional[float], psi: Optional[float],
                      residue_type: str = "General") -> Optional[float]:
    """
    :param phi: Backbone phi, in degrees.
    :param psi: Backbone psi, in degrees.
    :param residue_type: One of :func:`residue_type_for`'s classes.
    :returns: The reference density at that point, or ``None`` when an angle is missing or the
        tables are unavailable.
    """
    tables = _reference_tables()
    if tables is None or phi is None or psi is None:
        return None
    grid = tables.get(residue_type)
    if grid is None:
        # An unrecognised class scores against the general distribution rather than nothing.
        # Compared explicitly against None: `or` on a numpy array raises.
        grid = tables["General"]
    return float(grid[_reference_bin(psi), _reference_bin(phi)])


def classify_ramachandran(phi: Optional[float], psi: Optional[float],
                          residue_type: str = "General") -> Optional[str]:
    """
    Scores one phi/psi pair against the Top500 distribution for its residue class.

    Residue class is the part the box approximation could not express at all. (60, -120) is an
    outlier for a general residue, sits at density 0.127 -- comfortably favored -- for glycine, and
    is flatly forbidden for proline. One set of boxes cannot say all three.

    :param phi: Backbone phi, in degrees.
    :param psi: Backbone psi, in degrees.
    :param residue_type: From :func:`residue_type_for`; defaults to the general case.
    :returns: ``"favored"``, ``"allowed"`` or ``"outlier"``; ``None`` when either angle is
        undefined (a chain terminus has no neighbour to define the dihedral).
    """
    if phi is None or psi is None:
        return None

    density = reference_density(phi, psi, residue_type)
    if density is None:
        # No reference tables: the coarse boxes, which ignore residue type entirely.
        if any(_in_any_box(phi, psi, boxes) for boxes in _RAMA_FAVORED.values()):
            return "favored"
        if any(_in_any_box(phi, psi, boxes) for boxes in _RAMA_ALLOWED.values()):
            return "allowed"
        return "outlier"

    if density >= RAMA_FAVORED_CUTOFF:
        return "favored"
    return ("allowed" if density >= RAMA_ALLOWED_CUTOFF.get(residue_type, 0.0005)
            else "outlier")


def ramachandran(pdb_path: str) -> List[Dict[str, Any]]:
    """
    Backbone phi/psi angles per residue, scored against the reference for that residue's class.

    :param pdb_path: A PDB file.
    :returns: ``[{"residue", "phi", "psi", "region", "residue_type", "density"}]`` in chain order,
        in degrees. The first residue has no phi and the last no psi (no neighbour to define the
        dihedral); ``region`` and ``density`` are ``None`` for those. ``residue_type`` is the
        distribution the point was scored against, which the page uses to mark glycine and proline
        distinctly -- otherwise a favored glycine sitting outside the general-case shading reads as
        a mistake.
    """
    from Bio.PDB.Polypeptide import PPBuilder

    model = _first_model(pdb_path)
    out: List[Dict[str, Any]] = []
    for chain in model:
        for polypeptide in PPBuilder().build_peptides(chain):
            angles = polypeptide.get_phi_psi_list()
            for index, (residue, (phi, psi)) in enumerate(zip(polypeptide, angles)):
                # Pre-proline is a property of the NEXT residue, so the successor is looked up
                # here rather than inferred from the angles.
                following = (polypeptide[index + 1].get_resname()
                             if index + 1 < len(polypeptide) else None)
                kind = residue_type_for(residue.get_resname(), following)
                phi_deg = round(math.degrees(phi), 1) if phi is not None else None
                psi_deg = round(math.degrees(psi), 1) if psi is not None else None
                density = reference_density(phi_deg, psi_deg, kind)
                out.append({
                    "residue": residue.get_resname(),
                    "phi": phi_deg,
                    "psi": psi_deg,
                    "region": classify_ramachandran(phi_deg, psi_deg, kind),
                    "residue_type": kind,
                    "density": None if density is None else round(density, 5),
                })
    return out


@functools.lru_cache(maxsize=4)
def ramachandran_reference_contours(residue_type: str = "General") -> Optional[Dict[str, Any]]:
    """
    The favored and allowed regions of a reference distribution, as polygons the page can draw.

    Contours rather than an image on purpose: the polygons are ~24 KB of vector paths against
    ~300 KB for the equivalent rendered SVG, they scale with the plot, and they stay legible in
    both themes instead of baking one theme's colours into a bitmap. `contourpy` does the tracing
    -- it ships with matplotlib but is imported directly, since this is the part actually used.

    :param residue_type: Which distribution to outline; the general case is what the page shades.
    :returns: ``{"favored": [[[phi, psi], ...]], "allowed": [...]}`` with coordinates in degrees,
        or ``None`` when the tables or contourpy are unavailable.
    """
    tables = _reference_tables()
    if tables is None:
        return None
    try:
        import contourpy
        import numpy as np
    except Exception:
        logger.info("contourpy unavailable; the plot omits its reference regions.")
        return None

    grid = tables.get(residue_type)
    if grid is None:
        grid = tables["General"]
    centres = np.arange(-179.0, 181.0, 2.0)[:180]
    generator = contourpy.contour_generator(x=centres, y=centres, z=grid,
                                            fill_type="OuterOffset")
    allowed_floor = RAMA_ALLOWED_CUTOFF.get(residue_type, 0.0005)
    bands = {"allowed": (allowed_floor, RAMA_FAVORED_CUTOFF),
             "favored": (RAMA_FAVORED_CUTOFF, float(grid.max()) + 1.0)}

    contours: Dict[str, Any] = {}
    for band, (low, high) in bands.items():
        points, offsets = generator.filled(low, high)
        polygons = []
        for chunk, offset in zip(points, offsets):
            for start, end in zip(offset[:-1], offset[1:]):
                # Whole degrees: the plot is ~200px across, so a half-degree is well under a
                # pixel, and rounding cuts the payload by a third.
                polygons.append([[round(float(x)), round(float(y))]
                                 for x, y in chunk[start:end]])
        contours[band] = polygons
    return contours


def ramachandran_summary(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Corpus-quality-style rollup of a :func:`ramachandran` call, the number a reader actually wants:
    "is this structure's backbone geometry plausible?" rather than one point at a time.

    :param points: The output of :func:`ramachandran`.
    :returns: ``{"n_scored", "favored_pct", "allowed_pct", "outlier_pct"}``. ``n_scored`` excludes
        chain-terminal residues, which have no phi/psi to classify.
    """
    regions = [p["region"] for p in points if p.get("region") is not None]
    n = len(regions)
    if n == 0:
        return {"n_scored": 0, "favored_pct": 0.0, "allowed_pct": 0.0, "outlier_pct": 0.0}
    counts = {"favored": 0, "allowed": 0, "outlier": 0}
    for region in regions:
        counts[region] += 1
    return {"n_scored": n,
            "favored_pct": round(100.0 * counts["favored"] / n, 1),
            "allowed_pct": round(100.0 * counts["allowed"] / n, 1),
            "outlier_pct": round(100.0 * counts["outlier"] / n, 1)}


def per_residue_plddt(pdb_path: str) -> List[float]:
    """
    :param pdb_path: A ColabFold PDB, whose B-factor column holds pLDDT.
    :returns: One pLDDT per residue (its CA atom's B-factor), in chain order.
    """
    model = _first_model(pdb_path)
    values: List[float] = []
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                values.append(float(residue["CA"].get_bfactor()))
    return values


def disorder(pdb_path: str) -> Dict[str, Any]:
    """
    :param pdb_path: A ColabFold PDB.
    :returns: ``{"plddt", "disordered", "disorder_fraction", "mean_plddt"}`` -- the per-residue
        pLDDT, a per-residue boolean disorder flag (pLDDT <= :data:`DISORDER_PLDDT`), the fraction
        of residues flagged, and the mean pLDDT.
    """
    plddt = per_residue_plddt(pdb_path)
    if not plddt:
        return {"plddt": [], "disordered": [], "disorder_fraction": 0.0, "mean_plddt": 0.0}
    flags = [p <= DISORDER_PLDDT for p in plddt]
    return {"plddt": [round(p, 1) for p in plddt], "disordered": flags,
            "disorder_fraction": round(sum(flags) / len(flags), 3),
            "mean_plddt": round(sum(plddt) / len(plddt), 1)}


def _mkdssp_command() -> Optional[List[str]]:
    """
    :returns: How to invoke DSSP (``mkdssp``, or the older ``dssp``), as a single-element list
        holding an ABSOLUTE path where possible, or ``None`` when neither binary can be found.
    """
    # The running interpreter's own environment is searched first, by absolute path, before PATH.
    # `dssp` is a conda dependency of the serving environment, so the binary sits in the same
    # bin/ directory as the python executing this line -- but PATH does not reliably contain that
    # directory: on this host `conda run -n <env>` leaves `envs/sample/bin` first for the
    # subprocess it launches (see KNOWN_ISSUES.md). A `shutil.which` lookup alone therefore
    # reported "not installed" for an environment that had it, and the peptide page's DSSP track
    # silently fell back to nothing -- while Ramachandran and pLDDT disorder, which shell out to
    # no binary at all, rendered from the same PDB perfectly well.
    for binary in ("mkdssp", "dssp"):
        candidate = os.path.join(os.path.dirname(sys.executable), binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return [candidate]
    for binary in ("mkdssp", "dssp"):
        found = shutil.which(binary)
        if found:
            return [found]
    return None


def dssp_secondary_structure(pdb_path: str) -> Optional[Dict[str, Any]]:
    """
    Per-residue secondary structure from DSSP.

    :param pdb_path: A PDB file.
    :returns: ``{"residues": [{"index", "aa", "ss", "state"}], "summary": {helix, sheet, coil}}``
        where ``ss`` is DSSP's raw letter and ``state`` its three-state collapse; or ``None`` when
        DSSP (``mkdssp``) is not installed, so the caller can fall back to the sequence propensity.
    """
    from Bio.PDB import DSSP

    command = _mkdssp_command()
    if command is None:
        logger.info("mkdssp not found; secondary structure falls back to sequence propensity.")
        return None
    try:
        model = _first_model(pdb_path)
        # `dssp=` is passed explicitly: Biopython execs the binary by bare name otherwise, which
        # fails for exactly the PATH reason _mkdssp_command works around above -- so resolving the
        # path without handing it over would find DSSP and still not run it.
        dssp = DSSP(model, pdb_path, dssp=command[0])
    except Exception as e:
        logger.warning(f"DSSP failed on {pdb_path}: {e}")
        return None

    residues: List[Dict[str, Any]] = []
    counts = {"helix": 0, "sheet": 0, "coil": 0}
    for i, key in enumerate(dssp.keys()):
        aa, ss = dssp[key][1], dssp[key][2]
        state = _DSSP_THREE_STATE.get(ss, "coil")
        counts[state] += 1
        residues.append({"index": i, "aa": aa, "ss": ss, "state": state})
    total = len(residues) or 1
    return {"residues": residues,
            "summary": {k: round(v / total, 3) for k, v in counts.items()}}


def analyze_structure(pdb_path: str) -> Optional[Dict[str, Any]]:
    """
    All three structure-derived analyses for one peptide's PDB.

    :param pdb_path: The peptide's predicted structure, e.g.
        ``data/sample_enriched_release/structures/<entry_id>.pdb``.
    :returns: ``{"ramachandran", "ramachandran_summary", "disorder", "dssp"}``; ``None`` when the
        PDB does not exist (the peptide was never folded), so the page can hide the figures rather
        than error.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    try:
        rama = ramachandran(pdb_path)
        contours = ramachandran_reference_contours("General")
        return {"ramachandran": rama, "ramachandran_summary": ramachandran_summary(rama),
                "disorder": disorder(pdb_path), "dssp": dssp_secondary_structure(pdb_path),
                # The page states the method it was given, so it can never advertise a rigour the
                # classifier did not use: this names the reference only when it was really read.
                "ramachandran_reference": {
                    "method": (f"Scored against the {RAMA_REFERENCE_NAME}, per residue class, at "
                               f"MolProbity's cutoffs (favored >= 2%, allowed >= 0.05%, or 0.2% "
                               f"for Gly/Pro/pre-Pro)." if contours is not None else
                               "Coarse box approximation of the textbook basins -- reference "
                               "tables unavailable."),
                    "reference": RAMA_REFERENCE_NAME if contours is not None else "",
                    "residue_type": "General",
                    "contours": contours,
                }}
    except Exception as e:
        logger.warning(f"Structure analysis failed for {pdb_path}: {e}")
        return None
