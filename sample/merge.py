import os
import glob
import json
import logging
import re
import pandas as pd

from sample.preprocessing import raw_value_and_unit
from sample.config import iter_extraction_files
from sample.external_links import normalize_identifier
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from Bio import SeqIO
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("sample.merge")

# The canonical release path. Writing the backward-compatibility copy of
# merged_source.csv is gated on the caller actually producing this file.
_DEFAULT_MERGED_OUTPUT = "data/sample_raw_release/merged_from_database.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sequence_to_helm(sequence: str) -> str:
    """
    Translates a standard/modified amino acid sequence to HELM string representation.
    """
    import re
    # 1. Input Sanitization
    if not sequence or sequence in ["None", "/", "O", "OO", "OOO", "OOOO", "OOOOO"]:
        return ""
    sequence = str(sequence).strip()
    
    # Auto-correct malformed multipliers (e.g., missing opening parenthesis)
    if sequence.count(')') > sequence.count('('):
        sequence = '(' + sequence

    # 2. Multiplier Expansion
    def expand_match(match: "re.Match") -> str:
        """
        Expands one ``(BLOCK)n`` multiplier in a HELM-style sequence.

        :param match: Match with the repeated block as group 1 and the count as group 2.
        :returns: The block repeated `n` times.
        """
        seq_block = match.group(1)
        multiplier = int(match.group(2))
        return "".join([seq_block] * multiplier)

    while re.search(r"\(([^)]+)\)(\d+)", sequence):
        sequence = re.sub(r"\(([^)]+)\)(\d+)", expand_match, sequence)
        
    # Strip residual formatting parentheses
    sequence = sequence.replace('(', '').replace(')', '')

    # 3. Topology Splitting
    chain_strings = re.split(r'([,*])', sequence)
    
    helm_chains = []
    connections = []
    parsed_chains = []
    operators = []
    
    chain_idx = 1
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    token_regex = re.compile(r"(C\d+R?|RC\d+|[A-Z]\d+|-NH2|-OH|NH2|OH|[A-Z]|\?|-)")
    
    # 4. Monomer Parsing & Chain Construction
    for part in chain_strings:
        if part in ['*', ',']:
            operators.append(part)
            continue
        if not part.strip():
            continue
            
        tokens = token_regex.findall(part)
        monomers = []
        monomer_idx = 1
        chain_id = f"PEPTIDE{chain_idx}"
        
        for token in tokens:
            if token == '-':
                # Bypass staple logic if hyphen follows an N-terminal lipid (C followed by numbers only)
                if monomer_idx == 2 and re.fullmatch(r"C\d+", tokens[0]):
                    continue
                # Standard side-chain staple logic
                if monomer_idx > 1:
                    connections.append(f"{chain_id},{chain_id},{monomer_idx-1}:R3-{monomer_idx}:R3")
                continue
            
            # Map terminal modifications and un-natural amino acids
            if token in ["-NH2", "NH2"]:
                monomers.append("[am]")
            elif token in ["-OH", "OH"]:
                monomers.append("[OH]")
            elif token in standard_aa:
                monomers.append(token)
            else:
                monomers.append(f"[{token}]")
            monomer_idx += 1
            
        if monomers:
            helm_chains.append(f"{chain_id}{{{'.'.join(monomers)}}}")
            parsed_chains.append(chain_id)
            chain_idx += 1

    # 5. Cross-Chain Linkage Assembly
    for i, op in enumerate(operators):
        if i + 1 < len(parsed_chains):
            chain_a = parsed_chains[i]
            chain_b = parsed_chains[i+1]
            if op == '*':
                connections.append(f"{chain_a},{chain_b},3:R3-1:R1")

    # 6. Final String Compilation
    helm_str = "|".join(helm_chains)
    conn_str = "|".join(connections)
    
    if not helm_str:
        return ""
        
    return f"{helm_str}${conn_str}$$$"

#: A plain one-letter peptide sequence, with the ambiguity codes real dumps use. DRAMP's
#: `Sequence` column can carry Unicode-decorated residues marking modification sites, which this
#: rejects.
#:
#: Re-exported from database_activities rather than defined here: the activity loader has to make
#: the same judgement on the same rows, and when only this module made it, strings that could not
#: enter the consensus entered the activity table unchallenged.
from sample.database_activities import DRAMP_PLAIN_RESIDUES as _DRAMP_RESIDUES


def merge_databases(output_path: str, base_dir: str = "data/antimicrobial_databases") -> None:
    """
    Reads ADP, CAMP, DBAMP, DRAMP, DBAASP raw data files, normalizes their schemas,
    generates consensus sequence mappings, converts to SMILES, and outputs a consensus CSV.
    """
    logger.info("Merging antimicrobial peptide databases...")
    
    # 1. ADP
    adp_fasta = os.path.join(base_dir, "adp", "peptides.fasta")
    df_adp = pd.DataFrame(columns=['internal_id', 'source', 'sequence'])
    if os.path.exists(adp_fasta):
        df_adp = pd.DataFrame([
            {
                'internal_id': record.id,
                'source': 'ADP',
                'sequence': str(record.seq).upper().strip()
            } for record in SeqIO.parse(adp_fasta, 'fasta')
        ])
    else:
        logger.warning(f"ADP fasta not found at {adp_fasta}")

    # Per-file failures in each ingest loop below used to disappear into a bare `except: pass`.
    # That is the exact shape that hid 152,308 DBAASP activities from step 19 earlier this
    # session -- a total failure looked identical to a partial one, and the log said nothing. Each
    # loop now counts what it could not parse and reports it once, after the loop, rather than
    # once per file.
    ingest_errors: Dict[str, int] = {}

    # 2. CAMP
    camp_peptides = []
    camp_files = glob.glob(os.path.join(base_dir, "camp", "*.csv"))
    camp_failed = 0
    for csv_file in camp_files:
        try:
            camp_id = os.path.splitext(os.path.basename(csv_file))[0]
            df_entry = pd.read_csv(csv_file).T
            data_dict = dict(zip(df_entry.iloc[0], df_entry.iloc[1]))
            sequence = data_dict.get('Sequence')
            if pd.notna(sequence) and sequence:
                camp_peptides.append({
                    'internal_id': camp_id,
                    'source': 'CAMP',
                    'sequence': str(sequence).upper().strip(),
                    'reference': str(data_dict.get('Pubmed Id', '')),
                    'biological_activity': str(data_dict.get('Activity', ''))
                })
        except Exception:
            camp_failed += 1
    if camp_failed:
        ingest_errors['CAMP'] = camp_failed
    df_camp = pd.DataFrame(camp_peptides)

    # 3. DBAMP
    #
    # Two sources, and the big one used to be ignored entirely. This block read ONLY
    # `dbamp/extracted_metadata/*/1.csv` -- 394 scraped entries -> 381 peptides -- while the dump
    # also carries dbAMP's actual releases: `dbAMP2024/dbAMP3.fasta` (35,599 records) plus 25
    # activity-labelled FASTA files. 35,628 unique sequences were on disk and 381 reached the
    # consensus, a 99% loss that logged nothing because nothing counted what was never opened.
    #
    # The FASTAs are read first (they are the release), then the metadata rows are merged over
    # them: metadata carries the PubMed id and target species, which the FASTA headers do not, so
    # an entry present in both keeps the richer record.
    dbamp_peptides = []
    dbamp_seen = set()
    dbamp_fastas = sorted(glob.glob(os.path.join(base_dir, "dbamp", "*.fasta"))
                          + glob.glob(os.path.join(base_dir, "dbamp", "dbAMP2024", "*.fasta")))
    for fasta_path in dbamp_fastas:
        # The activity-labelled files name their activity in the filename
        # (dbAMP_Antibacterial.fasta); dbAMP3.fasta is the whole release and names none.
        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        activity = stem[len("dbAMP_"):] if stem.startswith("dbAMP_") and stem != "dbAMP3" else ""
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequence = str(record.seq).upper().strip()
                if not sequence:
                    continue
                key = (str(record.id).strip(), sequence)
                if key in dbamp_seen:
                    continue
                dbamp_seen.add(key)
                dbamp_peptides.append({
                    'internal_id': str(record.id).strip(),
                    'source': 'DBAMP',
                    'sequence': sequence,
                    'reference': '',
                    'biological_activity': activity,
                })
        except Exception as e:
            logger.warning(f"dbAMP FASTA {fasta_path} could not be read: {e}")
    logger.info(f"dbAMP: {len(dbamp_peptides)} records from {len(dbamp_fastas)} FASTA file(s).")

    dbamp_files = glob.glob(os.path.join(base_dir, "dbamp", "extracted_metadata", "*", "1.csv"))
    dbamp_failed = 0
    for csv_file in dbamp_files:
        try:
            dbamp_id = os.path.basename(os.path.dirname(csv_file))
            df_entry = pd.read_csv(csv_file).T
            data_dict = dict(zip(df_entry.iloc[0], df_entry.iloc[1]))
            sequence = data_dict.get('Sequence')
            if pd.notna(sequence) and sequence:
                dbamp_peptides.append({
                    'internal_id': dbamp_id,
                    'source': 'DBAMP',
                    'sequence': str(sequence).upper().strip(),
                    'reference': str(data_dict.get('Pubmed_Id', '')),
                    'biological_activity': str(data_dict.get('Target_Species', ''))
                })
        except Exception:
            dbamp_failed += 1
    if dbamp_failed:
        ingest_errors['DBAMP'] = dbamp_failed
    df_dbamp = pd.DataFrame(dbamp_peptides)
    if not df_dbamp.empty:
        # keep="last" deliberately: the metadata rows were appended after the FASTA rows and carry
        # the PubMed id and target species the FASTA headers lack, so where an id appears in both
        # the richer record is the one kept.
        df_dbamp = df_dbamp.drop_duplicates(subset=['internal_id'], keep='last')
    logger.info(f"dbAMP: {len(df_dbamp)} unique records after merging FASTA and metadata.")

    # 4. DRAMP
    #
    # Rows come from database_activities._dramp_rows, NOT a private glob here. This block used to
    # read only `dramp/Datasets/*.txt` and take the `Original_Sequence` column -- and that column
    # exists in exactly ONE of the five Datasets files (specific_amps.txt). Every row of
    # general_amps.txt (14,370) and patent_amps.txt (19,388) was silently skipped, and the two
    # other DRAMP directories (Activity/, Subclass/) were never opened at all. The result was 96
    # DRAMP peptides in the consensus against the 36,366 records the activity loader reads from
    # the same dump -- a 99.7% loss, invisible because a skipped row logs nothing.
    #
    # Sharing the row iterator is the actual fix: two readers of one dump drifting apart is what
    # caused this, so there is now one definition of "a DRAMP row" for both paths.
    from sample.database_activities import _dramp_rows, dramp_sequence

    dramp_peptides = []
    dramp_skipped = 0
    dramp_nonstandard = 0
    for row in _dramp_rows(base_dir):
        dramp_id = row.get('DRAMP_ID')
        # database_activities.dramp_sequence, not a local column choice. The Original_Sequence-
        # first rule used to be implemented HERE and nowhere else, so the activity loader -- the
        # other reader of this same dump -- took the stapling-annotated `Sequence` column and the
        # same 366 specific_amps peptides entered the consensus clean and the activity table
        # decorated with Unicode circled letters. One definition, both readers.
        sequence = dramp_sequence(row)
        if not (dramp_id and sequence):
            dramp_skipped += 1
            continue
        # Whatever column it came from, it must be a plain one-letter sequence. This is what
        # actually catches the decorated form if a future dump moves the annotation elsewhere,
        # rather than trusting the column name to keep meaning what it means today.
        if not _DRAMP_RESIDUES.match(sequence):
            dramp_nonstandard += 1
            continue
        # Only build the label when there is something in it: with both fields empty this
        # produced the literal string "Target:  MIC:" for 5,279 consensus rows -- a value that
        # looks like data but carries none, and that any consumer then has to learn to ignore.
        target = str(row.get('Target_Organism', '') or '').strip()
        mic = str(row.get('MIC', '') or '').strip()
        act = " ".join(part for part in
                       (f"Target: {target}" if target else "", f"MIC: {mic}" if mic else "")
                       if part)
        dramp_peptides.append({
            'internal_id': str(dramp_id).strip(),
            'source': 'DRAMP',
            'sequence': sequence,
            'reference': str(row.get('Pubmed_ID', '') or ''),
            'biological_activity': act
        })
    # Counted and reported, never silent -- see DECISIONS.md on the DBAASP loop that hid a 60% loss.
    if dramp_skipped:
        logger.info(f"DRAMP: {dramp_skipped} rows carried no id/sequence and were skipped.")
    if dramp_nonstandard:
        logger.info(f"DRAMP: {dramp_nonstandard} rows had a non-standard residue string and were "
                    f"skipped.")
    df_dramp = pd.DataFrame(dramp_peptides)
    if not df_dramp.empty:
        df_dramp = df_dramp.drop_duplicates(subset=['internal_id'])

    # 5. DBAASP
    dbaasp_peptides = []
    dbaasp_files = glob.glob(os.path.join(base_dir, "dbaasp", "*.json"))
    dbaasp_failed = 0
    for json_file in dbaasp_files:
        try:
            dbaasp_id = os.path.splitext(os.path.basename(json_file))[0]
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            sequence = json_data.get('sequence')
            if pd.notna(sequence) and sequence:
                ref = json_data.get('citation', {}).get('title', '')
                act = str(json_data.get('targetGroups', '')) + " " + str(json_data.get('targetObjects', ''))
                dbaasp_peptides.append({
                    'internal_id': dbaasp_id,
                    'source': 'DBAASP',
                    'sequence': str(sequence).upper().strip(),
                    'reference': ref,
                    'biological_activity': act
                })
        except Exception:
            dbaasp_failed += 1
    if dbaasp_failed:
        ingest_errors['DBAASP'] = dbaasp_failed
    df_dbaasp = pd.DataFrame(dbaasp_peptides)

    if ingest_errors:
        logger.warning(f"Consensus ingest: files that failed to parse, by source: {ingest_errors}")

    # 6. AntiCP2 anticancer peptides (positives only), when imported by `import-anticancer`.
    # Including them here is the features side of the anticancer integration: a novel anticancer
    # sequence gets an entry_id/structure/features and reaches final_dataset.csv, while one already
    # present from an antimicrobial source merges into its existing consensus row rather than
    # duplicating it. Empty (and harmless) on a checkout that never imported them.
    # Read the anticancer import from a sibling of THIS base_dir (data/anticancer_databases when
    # base_dir is data/antimicrobial_databases), not the packaged default -- so a build against a
    # different data root, including a test's isolated base_dir, does not pull in the global import.
    from sample.anticancer import consensus_frame as _anticancer_consensus_frame
    _anticancer_csv = os.path.join(os.path.dirname(os.path.abspath(base_dir)),
                                   "anticancer_databases", "anticancer_peptides.csv")
    df_anticp2 = _anticancer_consensus_frame(output_csv=_anticancer_csv)
    if not df_anticp2.empty:
        logger.info(f"Merging {len(df_anticp2)} AntiCP2 anticancer peptides into the consensus.")

    # Combine
    dfs = [df for df in [df_adp, df_camp, df_dbamp, df_dramp, df_dbaasp, df_anticp2] if not df.empty]
    if not dfs:
        logger.error("No database sequence data available to merge.")
        return

    df_consensus_raw = pd.concat(dfs, ignore_index=True)
    from sample.sequence import sequence_to_1_letter
    df_consensus_raw['sequence'] = df_consensus_raw['sequence'].apply(lambda x: sequence_to_1_letter(x) if pd.notna(x) else None)
    df_consensus_raw = df_consensus_raw.dropna(subset=['sequence'])
    df_consensus_raw = df_consensus_raw.drop_duplicates(subset=['sequence', 'source'])

    # Consensus mapping by sequence
    consensus_dataset_rows = []
    ID_PREFIX = 'SAMPLE'
    ID_ZFILL = 10
    entry_count = 0
    
    unique_sequences = df_consensus_raw['sequence'].unique()
    logger.info(f"Grouping and normalising {len(unique_sequences)} unique sequences using RDKit...")

    dropped_x_residue = 0
    dropped_unparseable = 0

    for sequence in tqdm(unique_sequences, desc="Consensus Building"):
        if 'X' in sequence or not sequence:
            dropped_x_residue += 1
            continue

        df_sources = df_consensus_raw[df_consensus_raw['sequence'] == sequence]
        
        entry_data = {
            'entry_id': None,
            'sequence': sequence,
            'ADP': None,
            'CAMP': None,
            'DBAMP': None,
            'DRAMP': None,
            'DBAASP': None,
            'AntiCP2': None,
            'Reference': None,
            'Biological_Activity': None
        }
        
        refs = []
        acts = []
        
        pmids = []
        for _, row in df_sources.iterrows():
            entry_data[row.source] = row.internal_id
            if 'reference' in row and pd.notna(row['reference']) and row['reference']:
                ref_val = str(row['reference'])
                refs.append(ref_val)
                if ref_val.isdigit():
                    pmids.append(ref_val)
            if 'biological_activity' in row and pd.notna(row['biological_activity']) and row['biological_activity']:
                acts.append(str(row['biological_activity']))
                
        entry_data['Reference'] = " | ".join(set(refs)) if refs else None
        entry_data['pubmed_ids'] = " | ".join(set(pmids)) if pmids else None
        entry_data['dois'] = None
        entry_data['patent_ids'] = None
        entry_data['Biological_Activity'] = " | ".join(set(acts)) if acts else None
            
        mol = Chem.MolFromSequence(sequence)
        if not mol:
            # Fallback to HELM
            helm = sequence_to_helm(sequence)
            if helm:
                try:
                    mol = Chem.MolFromHELM(helm)
                except Exception as e:
                    logger.debug(f"HELM parse failed for {sequence[:20]!r}: {e}")

        if not mol:
            dropped_unparseable += 1
            continue

        entry_count += 1
        entry_id = f"{ID_PREFIX}{str(entry_count).zfill(ID_ZFILL)}"
        entry_data['entry_id'] = entry_id
        entry_data['SMILES'] = Chem.MolToSmiles(mol)
        
        consensus_dataset_rows.append(entry_data)
        
    df_result = pd.DataFrame(consensus_dataset_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_result.to_csv(output_path, index=False)
    
    # Save a merged copy under databases for backward compatibility -- but ONLY when this run is
    # producing the canonical release. The path below is hard-coded and relative, so without this
    # guard any run with a custom --output (a test, a subset, a side-by-side comparison) silently
    # overwrote the project's baseline with a partial result. That is not hypothetical: the file
    # was found holding a single ADP row whose accession was `real_one`, a test fixture's FASTA
    # header, which made step 13 compute "99.89% new peptides" against a baseline of one sequence.
    # `merge_database_and_extractions` below has carried this guard for a while; this call site
    # was missed.
    compat_path = None
    if os.path.abspath(output_path) == os.path.abspath(_DEFAULT_MERGED_OUTPUT):
        compat_path = "data/antimicrobial_databases/merged_source.csv"
        os.makedirs(os.path.dirname(compat_path), exist_ok=True)
        df_result.to_csv(compat_path, index=False)
    else:
        logger.info(
            f"Non-default --output ({output_path}); leaving "
            "data/antimicrobial_databases/merged_source.csv untouched."
        )

    logger.info(f"Merge complete. Saved {entry_count} consensus peptide entries to {output_path}"
               + (f" and {compat_path}." if compat_path else ".")
               + f" Of {len(unique_sequences)} unique sequences: {dropped_x_residue} "
               f"had a non-standard 'X' residue or were empty, {dropped_unparseable} could not be "
               f"built into a molecule by RDKit or its HELM fallback.")

def _sequence_key(seq: Any) -> str:
    """Canonical one-letter form used to match a peptide across sources."""
    from sample.sequence import sequence_to_1_letter
    if not seq:
        return ""
    return (sequence_to_1_letter(str(seq).upper().strip()) or "").strip()


def _unify_units(activity: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Applies the source-database unit equivalences to a SAMPLE-extracted activity too.

    Both layers land in one table, so both have to express a quantity the same way. The
    third-party loader already collapses "umol/L" to "uM" (identical quantities, see
    database_activities._EQUIVALENT_UNITS); the extraction path treats both as canonical, so
    without this the merged table carried molar concentrations under two unit strings depending on
    which layer a row came from -- the exact fragmentation the unit work set out to remove, showing
    up again at the join.

    Returns (value, unit).
    """
    from sample.database_activities import _EQUIVALENT_UNITS

    unit = (activity.get("activity_unit") or "").strip()
    value = activity.get("value_lower")
    convert = _EQUIVALENT_UNITS.get(unit)
    if convert is None or value is None:
        return value, unit
    try:
        return convert(float(value))
    except (TypeError, ValueError):
        return value, unit


def _publication_columns(publications: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Flattens a record's publication list into the columns the activity table publishes."""
    pubs = [p for p in (publications or []) if isinstance(p, dict)]
    if not pubs:
        return {"pubmed_id": "", "doi": "", "publication_title": "",
                "journal": "", "publication_year": "", "authors": ""}
    primary = pubs[0]
    return {
        "pubmed_id": " | ".join(sorted({p.get("pubmed_id", "") for p in pubs if p.get("pubmed_id")})),
        "doi": " | ".join(sorted({p.get("doi", "") for p in pubs if p.get("doi")})),
        "publication_title": primary.get("title", ""),
        "journal": primary.get("journal", ""),
        "publication_year": primary.get("year") or "",
        "authors": "; ".join(primary.get("authors") or []),
    }


# Two measurements are "the same finding" when they report the same assay against the same target
# for the same peptide. Deliberately NOT including the value: two labs measuring the same MIC
# rarely print the same number, and treating 8 and 16 ug/mL against E. coli as unrelated findings
# would let this pipeline claim as novel a result that is already in DBAASP.
def _finding_key(activity: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    :param activity: An activity record from either provenance layer.
    :returns: ``(assay, organism, cell_line)`` -- the identity of a FINDING, deliberately without
        the measured value; see the comment above.
    """
    return (
        (activity.get("activity_type") or "").strip(),
        (activity.get("target_organism") or "").strip().lower(),
        (activity.get("target_cell_line") or "").strip().lower(),
    )


def build_integrated_activity_table(
        base_dir: str = "data/antimicrobial_databases",
        literature_dir: str = "data/literature_data_json_repaired",
        patent_dir: str = "data/patent_data_json_repaired",
        output_path: str = "data/sample_raw_release/activities.csv",
        databases: Optional[Iterable[str]] = None,
        limit: Optional[int] = None) -> "pd.DataFrame":
    """
    Builds the long-form activity table: one row per measurement, from the source databases AND
    from this project's own extraction, with the provenance of each row recorded.

    Why long-form. The consensus CSV carries one row per sequence and joins everything else into
    pipe-delimited strings, which cannot express "this MIC came from DBAASP record X and that one
    from PMID Y". Per-measurement provenance needs a per-measurement row, and that is also the
    shape any downstream analysis of potency actually wants.

    Every row states:
      * provenance      -- "Third Party" for anything read out of an external database,
                           "SAMPLE" for a measurement this pipeline extracted from primary text.
      * source_database -- which one, and its own record id, so a row can be traced back.
      * sequence_novelty / activity_novelty -- see _mark_novelty. These are computed by comparison,
        not asserted, so the claim that SAMPLE contributes new data is checkable.
    """
    from sample.database_activities import (
        load_all, PROVENANCE_THIRD_PARTY, PROVENANCE_SAMPLE,
    )

    rows = []
    third_party_findings = {}     # sequence -> {(assay, organism, cell line), ...}
    third_party_sequences = set()
    peptide_meta = {}             # sequence -> identity/provenance gathered from the databases

    logger.info("Reading structured activity data from the source databases...")
    for record in load_all(base_dir, databases=databases, limit=limit):
        seq = _sequence_key(record.get("sequence"))
        if not seq:
            continue
        third_party_sequences.add(seq)
        meta = peptide_meta.setdefault(seq, {
            "peptide_name": "", "source_organism": "", "databases": set(),
            "cross_references": set(), "activity_labels": set(),
        })
        meta["databases"].add(record.get("database", ""))
        meta["peptide_name"] = meta["peptide_name"] or record.get("name", "")
        meta["source_organism"] = meta["source_organism"] or record.get("source_organism", "")
        for label in record.get("activity_labels") or []:
            meta["activity_labels"].add(str(label).strip())
        # Validated on the way in. The sources write sentinels ("Not available"), prose
        # ("1MAG resolved by NMR."), comma-separated lists and stringified dicts into these
        # fields; storing them verbatim produced a column nothing could join on -- 59% of the
        # 165,554 tokens in the previous release were unusable. Anything that does not match its
        # namespace's accession format is dropped rather than guessed at.
        for kind, values in (record.get("cross_references") or {}).items():
            for value in (values if isinstance(values, list) else [values]):
                for accession in normalize_identifier(kind, value):
                    meta["cross_references"].add(f"{kind}:{accession}")

        pub_cols = _publication_columns(record.get("publications"))
        for act in record.get("activities") or []:
            third_party_findings.setdefault(seq, set()).add(_finding_key(act))
            rows.append({
                "sequence": seq,
                "peptide_name": record.get("name", ""),
                "activity_type": act.get("activity_type", ""),
                "value_lower": act.get("value_lower"),
                "activity_unit": act.get("activity_unit", ""),
                "value_operator": act.get("value_operator", ""),
                "value_raw": act.get("value_raw", ""),
                "value_raw_unit": act.get("value_raw_unit", ""),
                "target_organism": act.get("target_organism", ""),
                "target_cell_line": act.get("target_cell_line", ""),
                "target_group": act.get("target_group", ""),
                "assay_medium": act.get("assay_medium", ""),
                "assay_ph": act.get("assay_ph", ""),
                "source_organism": record.get("source_organism", ""),
                "provenance": PROVENANCE_THIRD_PARTY,
                "source_database": record.get("source_database") or record.get("database", ""),
                "source_record_id": act.get("source_record_id") or record.get("record_id", ""),
                "source_url": record.get("url", ""),
                **pub_cols,
            })

    logger.info(f"Third-party layer: {len(rows)} measurements over "
                f"{len(third_party_sequences)} sequences.")

    # SAMPLE's own extraction. Read from the revised/repaired set so the same contract governs both
    # layers of the table.
    # Recognized source subdirectories only -- see config.iter_extraction_files.
    sample_files = iter_extraction_files(literature_dir) + iter_extraction_files(patent_dir)
    literature_index = _literature_index()

    n_sample = 0
    n_cited = 0
    unreadable = 0
    for path in tqdm(sample_files, desc="Reading SAMPLE extractions"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            unreadable += 1
            continue
        peptides = doc.get("peptides") if isinstance(doc, dict) else doc
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}

        # The extraction's own metadata carries only `source_file`, so `pmid`, `doi` and `title`
        # below were always empty and every SAMPLE-extracted measurement shipped uncitable --
        # 4,333 activities with no way back to the paper they were read from, while the
        # third-party rows beside them carried a PMID 87 % of the time. The identifiers are one
        # join away: the extraction filename IS the `internal_id` recorded by the download step.
        citation = literature_index.get(_extraction_key(path, meta), {})
        if citation:
            n_cited += 1
        for pep in (peptides or []):
            if not isinstance(pep, dict):
                continue
            seq = _sequence_key(pep.get("peptide_sequence"))
            if not seq:
                continue

            # Step 12 resolves PubChem CIDs and ChEMBL ids for these peptides -- behind four
            # layers of rate limiting, and gated on an exact-mass identity check -- and writes
            # them here as "PubChem:123" / "ChEMBL:CHEMBL456". Until now nothing read the key, so
            # the entire output of that step was discarded: 2,860 ChEMBL and 606 PubChem hits sat
            # in data/external_db_cache.sqlite while the released table contained no ChEMBL
            # reference at all. Folded into the same per-sequence metadata the third-party
            # cross-references use, so both reach the release by one path.
            for token in pep.get("database_cross_reference") or []:
                namespace, _, value = str(token).partition(":")
                for accession in normalize_identifier(namespace, value):
                    peptide_meta.setdefault(seq, {
                        "peptide_name": "", "source_organism": "", "databases": set(),
                        "cross_references": set(), "activity_labels": set(),
                    })["cross_references"].add(f"{namespace.strip().lower()}:{accession}")

            for act in pep.get("biological_activities") or []:
                if not isinstance(act, dict):
                    continue
                value, unit = _unify_units(act)
                rows.append({
                    "sequence": seq,
                    "peptide_name": pep.get("peptide_name", ""),
                    "activity_type": act.get("activity_type", ""),
                    "value_lower": value,
                    "activity_unit": unit,
                    "value_operator": act.get("value_operator", ""),
                    "value_raw": act.get("value_raw", ""),
                    "value_raw_unit": act.get("value_raw_unit", ""),
                    "target_organism": act.get("target_organism", ""),
                    "target_cell_line": act.get("target_cell_line", ""),
                    "target_group": act.get("target_group", ""),
                    "assay_medium": "", "assay_ph": "",
                    "source_organism": "",
                    "provenance": PROVENANCE_SAMPLE,
                    "source_database": "",
                    "source_record_id": os.path.basename(path),
                    "source_url": "",
                    "pubmed_id": str(meta.get("pmid") or citation.get("pmid") or ""),
                    "doi": str(meta.get("doi") or citation.get("doi") or ""),
                    "publication_title": str(
                        meta.get("title") or citation.get("title") or ""),
                    "journal": "", "publication_year": "", "authors": "",
                })
                n_sample += 1
    logger.info(f"SAMPLE layer: {n_sample} measurements from {n_cited} documents resolved to a "
                f"source publication.")
    if unreadable:
        logger.warning(f"SAMPLE layer: {unreadable} of {len(sample_files)} extraction files "
                       f"could not be read and were skipped.")

    if not rows:
        logger.error("No activity rows assembled; nothing to write.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    _attach_taxonomy_ids(df)
    _mark_novelty(df, third_party_sequences, third_party_findings, peptide_meta)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Integrated activity table written to {output_path} ({len(df)} rows).")
    return df


def _extraction_key(path: str, meta: dict) -> str:
    """
    The download step's ``internal_id`` for one extraction document.

    The extraction filename is that id, and so is the stem of the ``source_file`` its metadata
    records; either is used, since a document renamed on the way through still carries the other.

    :param path: Path to the extraction JSON.
    :param meta: The document's ``metadata`` block.
    :returns: The identifier, or ``""`` when neither is available.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem:
        return stem
    source_file = str((meta or {}).get("source_file") or "")
    return os.path.splitext(os.path.basename(source_file))[0] if source_file else ""


def _literature_index(directory: str = None) -> dict:
    """
    Maps each downloaded document's ``internal_id`` to its publication identifiers.

    This is what makes a SAMPLE-extracted measurement citable. Step 1 and step 2 record, per
    document, the DOI and title they searched and the ``internal_id`` they filed it under; the
    extraction step names its output after that id. Nothing had ever joined the two, so activities
    read out of a specific paper reached the release with no reference to it.

    The ``_downloaded`` files are preferred over ``_metadata`` because they describe documents that
    were actually retrieved and therefore actually extracted, but both are read -- a document can
    be present in the metadata sweep and absent from the download log.

    :param directory: Literature directory; the packaged one when omitted.
    :returns: ``internal_id -> {"doi", "pmid", "title"}``.
    """
    directory = directory or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "literature_data")
    index: dict = {}

    for name in ("pubmed_metadata.csv", "scopus_metadata.csv",
                 "pubmed_downloaded.csv", "scopus_downloaded.csv"):
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue
        if "internal_id" not in frame.columns:
            continue

        for row in frame.itertuples(index=False):
            key = str(getattr(row, "internal_id", "") or "").strip()
            if not key:
                continue
            record = index.setdefault(key, {"doi": "", "pmid": "", "title": ""})
            for field, column in (("doi", "doi"), ("pmid", "pmid"), ("title", "title")):
                value = getattr(row, column, None)
                if value is not None and str(value).strip().lower() not in ("", "nan"):
                    record[field] = record[field] or str(value).strip()

    logger.info(f"Literature index: {len(index)} documents with a resolvable publication.")
    return index


def _attach_taxonomy_ids(df: "pd.DataFrame") -> None:
    """
    Adds ``target_taxid`` and ``source_taxid`` beside the organism-name columns.

    Every organism lookup in this pipeline already resolves through an NCBI tax_id and then keeps
    only the name, so the released tables carried names with no stable identifier: nothing could
    link to NCBI Taxonomy, and two sources spelling a species differently could not be joined.
    Resolved per distinct name rather than per row -- 185k rows share a few thousand organisms,
    and :func:`sample.nlp.resolve_taxon` is cached on top of that.

    :param df: Activity table, modified in place.
    :returns: ``None``.
    """
    from sample.nlp import resolve_taxon

    for name_column, id_column in (("target_organism", "target_taxid"),
                                   ("source_organism", "source_taxid")):
        if name_column not in df.columns:
            continue
        distinct = df[name_column].dropna().astype(str).unique()
        mapping = {}
        for name in distinct:
            if not name.strip():
                continue
            _, tax_id = resolve_taxon(name)
            if tax_id is not None:
                mapping[name] = tax_id
        df[id_column] = df[name_column].map(mapping)
        resolved = int(df[id_column].notna().sum())
        logger.info(f"{id_column}: resolved {resolved}/{len(df)} rows "
                    f"({len(mapping)}/{len(distinct)} distinct names).")


def _mark_novelty(df: "pd.DataFrame", third_party_sequences: Set[str],
                  third_party_findings: Dict[str, Set[tuple]],
                  peptide_meta: Dict[str, Dict[str, Any]]) -> None:
    """
    Adds the novelty and cross-source columns, in place.

    `sequence_novelty` answers "is this peptide known to the field?":
      Third Party        -- present in a source database, and SAMPLE did not extract it
      SAMPLE (new)       -- extracted from primary text and new to the field, as judged by
                            :mod:`sample.novelty` rather than by string membership
      Confirmed          -- extracted, and the catalogue already holds this peptide
      Unverifiable       -- extracted, and the claim cannot be adjudicated either way: the
                            sequence carries an ambiguity code, or is too short for its absence
                            to mean anything, or its name identifies a different peptide

    `Unverifiable` is the column's one addition, and it exists because the old three values
    forced a guess. The test used to be ``seq not in third_party_sequences``, which is right
    about what it asks and wrong about what the column is read as meaning: over 32,408 refined
    documents it called 975 sequences new, of which roughly 230 survive scrutiny -- the rest were
    fragments of catalogued peptides, catalogued peptides with flanking text, three-letter
    notation that could never match a one-letter set, and sequences masked by an ambiguity code.
    :mod:`sample.novelty` holds that measurement and the graded test it motivated; this function
    maps its verdicts onto the published column.

    Three further columns carry the detail, so the refinement is auditable rather than asserted:
      novelty_basis      -- the verdict (``known_fragment``, ``analogue``, ``name_conflict``, ...)
      novelty_parent     -- the catalogued sequence it was matched against, when there is one
      name_agreement     -- whether an adjudicating name corroborated the sequence, contradicted
                            it, or had no standing to say

    `activity_novelty` answers the sharper question, "is this MEASUREMENT new?":
      New in SAMPLE      -- a SAMPLE row whose (assay, target) is reported by no source database
                            for this peptide, i.e. genuinely added information
      Corroborates       -- a SAMPLE row matching a third-party finding: not new, but independent
                            support, which is what makes the rest of the extraction believable
      Third Party        -- the row came from a database

    A peptide being novel does not make its measurements novel, and vice versa: both are recorded
    because they answer different questions about what this project contributes.
    """
    from sample import novelty as novelty_module

    sample_sequences = set(df.loc[df["provenance"] == "SAMPLE", "sequence"])

    # The catalogue is the databases this merge just read, not a file on disk, so novelty is
    # judged against the same sources the rest of the table came from. Names come from step 35's
    # harvest when it exists; on a first run it does not, and the name tests then stay silent --
    # a weaker judgment, logged, rather than a failed step.
    catalogue = novelty_module.Catalogue.load(
        merged_path="", extra_sequences=third_party_sequences)

    #: One judgment per distinct extracted sequence, not per row: the activity table repeats a
    #: sequence once per measurement, and the edit-distance test is the expensive part.
    names_by_sequence: Dict[str, str] = {}
    if "peptide_name" in df.columns:
        for sequence, name in zip(df["sequence"], df["peptide_name"]):
            if name and sequence not in names_by_sequence:
                names_by_sequence[sequence] = name
    verdicts = {sequence: catalogue.judge(sequence, names_by_sequence.get(sequence, ""))
                for sequence in sample_sequences}

    def sequence_novelty(seq: str) -> str:
        """
        :param seq: Canonical one-letter sequence.
        :returns: ``"Third Party"``, ``"SAMPLE (new)"``, ``"Confirmed"`` or ``"Unverifiable"``.
        """
        in_db, in_sample = seq in third_party_sequences, seq in sample_sequences
        if not in_sample:
            return "Third Party"
        verdict = verdicts.get(seq)
        if verdict is None:
            return "Confirmed" if in_db else "SAMPLE (new)"
        if verdict.is_novel:
            return "SAMPLE (new)"
        if verdict.verdict in (novelty_module.NOT_COMPARABLE, novelty_module.NAME_CONFLICT):
            return "Unverifiable"
        return "Confirmed"

    df["sequence_novelty"] = df["sequence"].map(sequence_novelty)
    df["novelty_basis"] = df["sequence"].map(
        lambda s: verdicts[s].verdict if s in verdicts else "")
    df["novelty_parent"] = df["sequence"].map(
        lambda s: verdicts[s].parent if s in verdicts else "")
    df["name_agreement"] = df["sequence"].map(
        lambda s: verdicts[s].name_agreement if s in verdicts else "")

    if verdicts:
        counts = novelty_module.summarise(verdicts.values())
        logger.info("Novelty of %d extracted sequences: %s.", len(verdicts),
                    ", ".join(f"{count} {name}" for name, count in counts.items() if count))
        defects = sum(1 for verdict in verdicts.values() if verdict.is_defect)
        if defects:
            logger.warning(
                "%d extracted sequences cannot support a novelty claim either way "
                "(ambiguity codes, too short to adjudicate, or a name that identifies a "
                "different peptide); published as Unverifiable or flagged by name_agreement.",
                defects)

    def activity_novelty(row: Any) -> str:
        """
        :param row: One row of the activity table.
        :returns: ``"Third Party"``, ``"Corroborates"`` or ``"New in SAMPLE"``.
        """
        if row["provenance"] != "SAMPLE":
            return "Third Party"
        known = third_party_findings.get(row["sequence"], set())
        return "Corroborates" if _finding_key(row) in known else "New in SAMPLE"

    df["activity_novelty"] = df.apply(activity_novelty, axis=1)

    # Carry the databases that know each sequence onto every row, so a SAMPLE measurement can be
    # read alongside where else that peptide appears without a second join.
    df["known_in_databases"] = df["sequence"].map(
        lambda s: " | ".join(sorted(d for d in peptide_meta.get(s, {}).get("databases", set()) if d)))
    df["cross_references"] = df["sequence"].map(
        lambda s: " | ".join(sorted(peptide_meta.get(s, {}).get("cross_references", set()))))
    df["source_organism"] = df.apply(
        lambda r: r["source_organism"] or peptide_meta.get(r["sequence"], {}).get("source_organism", ""),
        axis=1)


def merge_database_and_extractions(consensus_csv: str = "data/sample_raw_release/merged_from_database.csv",
                                   literature_dir: str = "data/literature_data_json_repaired",
                                   patent_dir: str = "data/patent_data_json_repaired",
                                   output_path: str = "data/sample_raw_release/merged_from_database.csv",
                                   database_dir: str = "data/antimicrobial_databases",
                                   build_activity_table: bool = True) -> bool:
    """
    Merges antimicrobial database peptides in the consensus CSV with the LLM-extracted peptides
    from literature and patent JSON files. Overwrites the consensus CSV with the combined set, and
    (unless `build_activity_table` is False) writes the long-form activity table beside it.
    """
    logger.info("Merging database consensus peptides with LLM-extracted peptides...")
    
    # 1. Load baseline
    if os.path.exists(consensus_csv):
        # low_memory=False deliberately: pandas otherwise infers dtypes per CHUNK, and on a
        # consensus this size the ADP/CAMP id columns hold values that infer differently in
        # different chunks ("Columns (0: ADP, 1: CAMP) have mixed types"). That is not a cosmetic
        # warning in a step that decides what the release contains -- an id inferred as a float in
        # one chunk and a string in another stops comparing equal to itself. Reading the whole
        # column at once makes the inference consistent.
        df = pd.read_csv(consensus_csv, low_memory=False)
    else:
        logger.error(f"Consensus CSV not found at {consensus_csv}")
        return False
        
    # Ensure columns exist
    for col in ['LITERATURE', 'PATENT', 'Reference', 'Biological_Activity', 'pubmed_ids', 'dois', 'patent_ids']:
        if col not in df.columns:
            df[col] = None

    # Track sequence to index/row mapping for fast lookup
    df['sequence_clean'] = df['sequence'].str.replace(' ', '', regex=False).str.replace('\t', '', regex=False).str.upper().str.strip()
    seq_to_idx = {seq: idx for idx, seq in enumerate(df['sequence_clean'])}
    
    # Track next available ID
    # Find max ID number in the baseline
    id_nums = []
    for entry_id in df['entry_id']:
        if isinstance(entry_id, str) and entry_id.startswith('SAMPLE'):
            try:
                id_nums.append(int(entry_id[6:]))
            except ValueError:
                pass
    next_id_num = max(id_nums) + 1 if id_nums else 1
    
    # Helper to check sequence validity (only standard amino acid single-letter characters)
    def is_valid_peptide(seq: str) -> bool:
        """
        Function is_valid_peptide.
        """
        if not seq:
            return False
        import re
        pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
        return bool(pattern.match(seq))

    new_rows = []
    
    # 2. Process Literature & Database Papers JSONs
    lit_files = (iter_extraction_files(literature_dir)
                 + iter_extraction_files("data/antimicrobial_databases/papers_json"))
    # Resolves each extraction back to the paper it was read from. The extraction's own metadata
    # carries only `source_file`, so the `doi`/`pmid` lookups below always came back empty and
    # every one of the 12,395 populated `Reference` cells was the string "File: <uuid>.json" -- an
    # internal filename presented to readers as a citation. The filename IS the download step's
    # `internal_id`, so the identifiers are one join away.
    literature_index = _literature_index()
    for file_path in tqdm(lit_files, desc="Processing Literature Extractions"):
        source_folder = os.path.basename(os.path.dirname(file_path)) # e.g. 'pubmed' or 'scopus'
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
            if isinstance(raw_json, dict) and "peptides" in raw_json:
                peptides_list = raw_json["peptides"]
                meta = raw_json.get("metadata", {})
            elif isinstance(raw_json, list):
                peptides_list = raw_json
                meta = {}
            else:
                peptides_list = [raw_json]
                meta = {}
                
            for peptide_data in peptides_list:
                seq = peptide_data.get('peptide_sequence')
                if not seq:
                    continue
                from sample.sequence import sequence_to_1_letter
                seq_clean = sequence_to_1_letter(seq)
                if not is_valid_peptide(seq_clean):
                    continue
                    
                val_str = f"{source_folder}:{filename}"
                citation = literature_index.get(_extraction_key(file_path, meta), {})
                doi = meta.get('doi') or peptide_data.get('DOI') or citation.get('doi')
                pmid = meta.get('pmid') or peptide_data.get('PMID') or citation.get('pmid')
                title = str(meta.get('title') or citation.get('title') or '').strip()

                # Built from the paper, not the file it was parsed out of. The filename is kept
                # only as a last resort: a row that resolved to nothing still has to say which
                # extraction it came from, or the provenance is lost entirely.
                ref_parts = []
                if title:
                    ref_parts.append(title)
                if doi:
                    ref_parts.append(f"DOI: {doi}")
                if pmid:
                    ref_parts.append(f"PMID: {pmid}")
                ref_str = ", ".join(ref_parts) if ref_parts else f"File: {filename}"
                
                act_parts = []
                b_acts = peptide_data.get('biological_activities', [])
                if isinstance(b_acts, list) and b_acts:
                    for act in b_acts:
                        org = act.get('target_organism', '')
                        atype = act.get('activity_type', 'MIC')
                        # value_raw belongs with value_raw_unit, not with the canonical
                        # activity_unit -- see preprocessing.raw_value_and_unit.
                        val, unit = raw_value_and_unit(act)
                        val = val or act.get('value_lower') or ''
                        act_parts.append(f"{atype}:{val}{unit}({org})")
                elif 'target_organism' in peptide_data:
                    act_parts.append(f"Target:{peptide_data['target_organism']} MIC:{peptide_data.get('minimum_inhibitory_concentration', '')}")
                act_str = "; ".join(act_parts).strip()
                
                def append_to_col(current: Any, new_val: str) -> str:
                    """
                    Appends a value to a pipe-delimited cell, without duplicating it.

                    :param current: Existing cell contents; ``NaN`` or empty starts a new cell.
                    :param new_val: Value to add.
                    :returns: The updated cell contents.
                    """
                    if pd.isna(current) or not current: return new_val
                    if new_val not in str(current).split(' | '): return f"{current} | {new_val}"
                    return current
                
                if seq_clean in seq_to_idx:
                    # Update existing row
                    row_idx = seq_to_idx[seq_clean]
                    df.at[row_idx, 'LITERATURE'] = append_to_col(df.at[row_idx, 'LITERATURE'], val_str)
                    if ref_str: df.at[row_idx, 'Reference'] = append_to_col(df.at[row_idx, 'Reference'], ref_str)
                    if act_str: df.at[row_idx, 'Biological_Activity'] = append_to_col(df.at[row_idx, 'Biological_Activity'], act_str)
                else:
                    # Check if already added in new_rows
                    found = False
                    for nr in new_rows:
                        if nr['sequence'] == seq_clean:
                            nr['LITERATURE'] = append_to_col(nr['LITERATURE'], val_str)
                            if ref_str: nr['Reference'] = append_to_col(nr.get('Reference'), ref_str)
                            if act_str: nr['Biological_Activity'] = append_to_col(nr.get('Biological_Activity'), act_str)
                            found = True
                            break
                    if not found:
                        new_rows.append({
                            'sequence': seq_clean,
                            'LITERATURE': val_str,
                            'PATENT': None,
                            'Reference': ref_str if ref_str else None,
                            'Biological_Activity': act_str if act_str else None,
                            'SMILES': peptide_data.get('peptide_sequence_SMILES_format')
                        })
        except Exception as e:
            logger.debug(f"Error parsing literature extraction {file_path}: {e}")

    # 3. Process Patent JSONs
    pat_files = glob.glob(os.path.join(patent_dir, "google_patents", "*.json")) + glob.glob(os.path.join(patent_dir, "*.json"))
    for file_path in tqdm(pat_files, desc="Processing Patent Extractions"):
        source_folder = "google_patents"
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
            if isinstance(raw_json, dict) and "peptides" in raw_json:
                peptides_list = raw_json["peptides"]
                meta = raw_json.get("metadata", {})
            elif isinstance(raw_json, list):
                peptides_list = raw_json
                meta = {}
            else:
                peptides_list = [raw_json]
                meta = {}
                
            for peptide_data in peptides_list:
                seq = peptide_data.get('peptide_sequence')
                if not seq:
                    continue
                from sample.sequence import sequence_to_1_letter
                seq_clean = sequence_to_1_letter(seq)
                if not is_valid_peptide(seq_clean):
                    continue
                    
                val_str = f"{source_folder}:{filename}"
                ref_str = f"File: {filename}"
                pat_id = str(peptide_data.get('patent_id', '')).strip() if peptide_data.get('patent_id') else None
                if not pat_id:
                    pat_id = str(peptide_data.get('patent_number', '')).strip() if peptide_data.get('patent_number') else None
                
                if pat_id: ref_str += f", Patent_ID: {pat_id}"
                
                act_parts = []
                b_acts = peptide_data.get('biological_activities', [])
                if isinstance(b_acts, list) and b_acts:
                    for act in b_acts:
                        org = act.get('target_organism', '')
                        atype = act.get('activity_type', 'MIC')
                        # value_raw belongs with value_raw_unit, not with the canonical
                        # activity_unit -- see preprocessing.raw_value_and_unit.
                        val, unit = raw_value_and_unit(act)
                        val = val or act.get('value_lower') or ''
                        act_parts.append(f"{atype}:{val}{unit}({org})")
                elif 'target_organism' in peptide_data:
                    act_parts.append(f"Target:{peptide_data['target_organism']} MIC:{peptide_data.get('minimum_inhibitory_concentration', '')}")
                act_str = "; ".join(act_parts).strip()
                
                if seq_clean in seq_to_idx:
                    # Update existing row
                    row_idx = seq_to_idx[seq_clean]
                    df.at[row_idx, 'PATENT'] = append_to_col(df.at[row_idx, 'PATENT'], val_str)
                    if ref_str: df.at[row_idx, 'Reference'] = append_to_col(df.at[row_idx, 'Reference'], ref_str)
                    if act_str: df.at[row_idx, 'Biological_Activity'] = append_to_col(df.at[row_idx, 'Biological_Activity'], act_str)
                    if pat_id: df.at[row_idx, 'patent_ids'] = append_to_col(df.at[row_idx, 'patent_ids'], pat_id)
                else:
                    # Check if already added in new_rows
                    found = False
                    for nr in new_rows:
                        if nr['sequence'] == seq_clean:
                            nr['PATENT'] = append_to_col(nr['PATENT'], val_str)
                            if ref_str: nr['Reference'] = append_to_col(nr.get('Reference'), ref_str)
                            if act_str: nr['Biological_Activity'] = append_to_col(nr.get('Biological_Activity'), act_str)
                            if pat_id: nr['patent_ids'] = append_to_col(nr.get('patent_ids'), pat_id)
                            found = True
                            break
                    if not found:
                        new_rows.append({
                            'sequence': seq_clean,
                            'LITERATURE': None,
                            'PATENT': val_str,
                            'Reference': ref_str if ref_str else None,
                            'Biological_Activity': act_str if act_str else None,
                            'SMILES': peptide_data.get('peptide_sequence_SMILES_format'),
                            'pubmed_ids': None,
                            'dois': None,
                            'patent_ids': pat_id
                        })
        except Exception as e:
            logger.debug(f"Error parsing patent extraction {file_path}: {e}")

    # 4. Process new rows, calculate missing SMILES, assign IDs
    from rdkit import Chem
    final_new_rows = []
    
    dropped_no_structure = 0
    for nr in tqdm(new_rows, desc="Constructing Chemical structures for new peptides"):
        mol = None
        smiles = nr['SMILES']
        if pd.isna(smiles) or not smiles:
            mol = Chem.MolFromSequence(nr['sequence'])
            if mol:
                smiles = Chem.MolToSmiles(mol)

        # If still no SMILES, try sequence_to_helm fallback
        if not smiles:
            helm = sequence_to_helm(nr['sequence'])
            if helm:
                try:
                    mol = Chem.MolFromHELM(helm)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                except Exception as e:
                    logger.debug(f"HELM parse failed for {nr['sequence'][:20]!r}: {e}")

        # Skip if we couldn't resolve any structure
        if not smiles:
            dropped_no_structure += 1
            continue

        entry_id = f"SAMPLE{str(next_id_num).zfill(10)}"
        next_id_num += 1
        
        final_new_rows.append({
            'entry_id': entry_id,
            'sequence': nr['sequence'],
            'ADP': None,
            'CAMP': None,
            'DBAMP': None,
            'DRAMP': None,
            'DBAASP': None,
            'LITERATURE': nr['LITERATURE'],
            'PATENT': nr['PATENT'],
            'Reference': nr.get('Reference'),
            'Biological_Activity': nr.get('Biological_Activity'),
            'SMILES': smiles,
            'pubmed_ids': nr.get('pubmed_ids'),
            'dois': nr.get('dois'),
            'patent_ids': nr.get('patent_ids')
        })
        
    if dropped_no_structure:
        logger.warning(f"SAMPLE-literature merge: {dropped_no_structure} of {len(new_rows)} new "
                       f"peptides had no resolvable structure (RDKit and HELM both failed) and "
                       f"were dropped.")

    if final_new_rows:
        df_new = pd.DataFrame(final_new_rows)
        # Drop sequence_clean helper from df to align columns
        df = df.drop(columns=['sequence_clean'])
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df.drop(columns=['sequence_clean'])
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Backward-compatibility copy under the databases directory. Written ONLY when this run is
    # producing the canonical release, i.e. output_path is the default. It used to be written
    # unconditionally, which meant any run with a custom --output -- a test, a subset, a
    # side-by-side comparison -- silently overwrote the project's baseline
    # merged_source.csv with a partial result. That file is an input to step 13, so
    # damaging it is not recoverable from within the pipeline.
    compat_path = None
    if os.path.abspath(output_path) == os.path.abspath(_DEFAULT_MERGED_OUTPUT):
        compat_path = "data/antimicrobial_databases/merged_source.csv"
        os.makedirs(os.path.dirname(compat_path), exist_ok=True)
        df.to_csv(compat_path, index=False)
    else:
        logger.info(
            f"Non-default --output ({output_path}); leaving "
            "data/antimicrobial_databases/merged_source.csv untouched."
        )

    logger.info(f"Successfully merged. Total peptides: {len(df)} ({len(final_new_rows)} new extracted peptides added). Saved to {output_path}"
                + (f" and {compat_path}." if compat_path else "."))

    # Long-form activity table. The wide CSV above carries one row per sequence with everything
    # else joined into pipe-delimited strings, which cannot express which database a given MIC came
    # from -- so the measured activity data in the source databases (~194,500 rows in DBAASP alone,
    # against ~29,900 extracted here) had nowhere to go and was dropped at parse time. This emits
    # one row per measurement, from both layers, each stamped with its provenance.
    if build_activity_table:
        try:
            activities_path = os.path.join(os.path.dirname(os.path.abspath(output_path)),
                                           "activities.csv")
            df_acts = build_integrated_activity_table(
                base_dir=database_dir,
                literature_dir=literature_dir,
                patent_dir=patent_dir,
                output_path=activities_path,
            )
            if not df_acts.empty:
                _summarize_provenance(df, df_acts, output_path, compat_path)
        except Exception as e:
            # A failure here must not lose the merge above, which is already written to disk.
            logger.error(f"Integrated activity table could not be built: {e}", exc_info=True)

    return True


def _summarize_provenance(df_wide: "pd.DataFrame", df_acts: "pd.DataFrame",
                          output_path: str, compat_path: Optional[str]) -> None:
    """
    Adds per-sequence provenance and novelty summary columns to the wide consensus CSV, so the
    headline table answers "where did this peptide come from, and what is new here?" without a
    join against the long-form table.
    """
    by_sequence = df_acts.groupby("sequence")
    known_in = by_sequence["known_in_databases"].first()
    novelty = by_sequence["sequence_novelty"].first()
    n_third_party = df_acts[df_acts["provenance"] == "Third Party"].groupby("sequence").size()
    n_sample = df_acts[df_acts["provenance"] == "SAMPLE"].groupby("sequence").size()
    n_new = (df_acts[df_acts["activity_novelty"] == "New in SAMPLE"]
             .groupby("sequence").size())

    key = df_wide["sequence"].astype(str).str.upper().str.strip()
    df_wide["known_in_databases"] = key.map(known_in).fillna("")
    df_wide["sequence_novelty"] = key.map(novelty).fillna("Third Party")
    df_wide["n_activities_third_party"] = key.map(n_third_party).fillna(0).astype(int)
    df_wide["n_activities_sample"] = key.map(n_sample).fillna(0).astype(int)
    df_wide["n_activities_new_in_sample"] = key.map(n_new).fillna(0).astype(int)

    df_wide.to_csv(output_path, index=False)
    if compat_path:
        df_wide.to_csv(compat_path, index=False)

    counts = df_acts["provenance"].value_counts().to_dict()
    novel = int((df_acts["activity_novelty"] == "New in SAMPLE").sum())
    corroborating = int((df_acts["activity_novelty"] == "Corroborates").sum())
    unique = df_acts.drop_duplicates("sequence")["sequence_novelty"]
    new_sequences = int((unique == "SAMPLE (new)").sum())
    unverifiable = int((unique == "Unverifiable").sum())
    logger.info(
        f"Activity provenance: {counts.get('Third Party', 0)} Third Party, "
        f"{counts.get('SAMPLE', 0)} SAMPLE ({novel} new findings, "
        f"{corroborating} corroborating a source database); "
        f"{new_sequences} sequences judged new to the field, "
        f"{unverifiable} unverifiable either way."
    )
