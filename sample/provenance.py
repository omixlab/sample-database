"""
Where a peptide came from: the organism that makes it, or the bench that designed it.

The release stored no source organism at all. It exists as ``source_organism`` in
``activities.csv``, but that column describes the *measurement's* provenance and is empty for most
rows; nothing recorded the peptide's own origin, so a natural defensin and a designed analogue
were indistinguishable on the page.

Three answers are possible and they are not interchangeable:

* **A named organism.** DBAASP records it per gene (``sourceGenes[].source``): *Tityus obscurus*,
  *Phyllomedusa bicolor*. 4,031 peptides.
* **Synthetic.** No natural gene, and the source states the peptide was made rather than isolated
  (``synthesisType`` of ``Synthetic``). 20,586 peptides. This is a real fact about the molecule --
  a designed sequence has no organism, and saying so is different from not knowing.
* **Unknown.** Neither is recorded: 131 peptides. Stored as ``NULL`` and not rendered at all,
  because an empty field invites the reader to supply their own assumption, and "Unknown" printed
  in a table reads as a claim that somebody looked.

Ribosomal and non-ribosomal peptides are *not* treated as synthetic: those describe how the
organism builds it, not that a chemist did.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

from sample.config import DATA_DIR

logger = logging.getLogger("sample.provenance")

#: ``synthesisType`` values that mean the peptide has no natural source. ``Ribosomal`` and
#: ``Nonribosomal`` describe an organism's own machinery and are deliberately absent.
_SYNTHETIC_TYPES = frozenset({"synthetic"})

#: What is stored when a peptide was designed rather than isolated.
SYNTHETIC = "Synthetic"


def source_organism_for(record: Dict[str, Any]) -> Optional[str]:
    """
    Decides one peptide's origin.

    :param record: One DBAASP peptide record.
    :returns: The organism's name, :data:`SYNTHETIC`, or ``None`` when neither is recorded. A
        named organism always wins: a peptide can be chemically synthesised *and* have a natural
        source, and in that case the organism is the more informative answer.
    """
    for gene in record.get("sourceGenes") or []:
        if not isinstance(gene, dict):
            continue
        source = str(gene.get("source") or "").strip()
        if source:
            return source

    synthesis = record.get("synthesisType")
    name = str((synthesis or {}).get("name") or "").strip().lower() \
        if isinstance(synthesis, dict) else ""
    if name in _SYNTHETIC_TYPES:
        return SYNTHETIC
    return None


def harvest_source_organisms(directory: Optional[str] = None) -> Dict[str, str]:
    """
    Reads the origin of every peptide DBAASP describes.

    :param directory: DBAASP JSON directory; the packaged location when omitted.
    :returns: Sequence -> origin. Peptides whose origin cannot be determined are omitted rather
        than mapped to a placeholder, so the caller stores ``NULL`` for them.
    """
    import glob

    directory = directory or os.path.join(str(DATA_DIR), "antimicrobial_databases", "dbaasp")
    if not os.path.isdir(directory):
        logger.warning(f"DBAASP directory not found at {directory}; no source organisms.")
        return {}

    from sample.sequence import sequence_to_1_letter

    origins: Dict[str, str] = {}
    natural = 0
    unreadable = 0
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            # Counted rather than silently dropped: a bare `except: continue` in exactly this
            # shape once hid a 60% loss of the activity table with no error in the log -- see
            # DECISIONS.md.
            unreadable += 1
            continue
        if not isinstance(record, dict):
            continue
        sequence = sequence_to_1_letter(str(record.get("sequence") or "")).upper().strip()
        if not sequence:
            continue
        origin = source_organism_for(record)
        if not origin:
            continue
        # A named organism beats a previously stored "Synthetic" for the same sequence: the same
        # peptide is often filed both as an isolate and as a synthesised analogue.
        if origin != SYNTHETIC or sequence not in origins:
            origins[sequence] = origin
        if origin != SYNTHETIC:
            natural += 1

    logger.info(f"Source organisms: {len(origins)} sequences "
                f"({natural} named organisms, {len(origins) - natural} synthetic).")
    if unreadable:
        logger.warning(f"{unreadable} DBAASP files could not be read and were skipped.")
    return origins


def build_source_organisms(output_path: Optional[str] = None,
                           directory: Optional[str] = None) -> Dict[str, Any]:
    """
    Harvests peptide origins and writes them for the loader.

    :param output_path: Where to write; ``data/source_organisms.json`` when omitted.
    :param directory: DBAASP JSON directory.
    :returns: A summary of what was written.
    """
    origins = harvest_source_organisms(directory)
    output_path = output_path or os.path.join(str(DATA_DIR), "source_organisms.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"source_organisms": origins}, handle, ensure_ascii=False, indent=1)

    synthetic = sum(1 for v in origins.values() if v == SYNTHETIC)
    summary = {
        "path": output_path,
        "sequences": len(origins),
        "named_organisms": len(origins) - synthetic,
        "synthetic": synthetic,
    }
    logger.info(f"Wrote {summary['sequences']} source organisms to {output_path}.")
    return summary
