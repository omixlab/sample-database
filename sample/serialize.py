import os
import csv
import glob
import json
import pickle
import logging
import re
import sqlite3
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from sample.config import iter_extraction_files

logger = logging.getLogger("sample.serialize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def serialize_raw_datasets(consensus_csv: str = "data/sample_raw_release/merged_from_database.csv",
                            out_dir: str = "data/sample_raw_release") -> bool:
    """
    Serializes consensus database datasets into multiple structured file formats:
    Parquet, JSON, and Pickle (for fast model loader routines).
    """
    logger.info(f"Serializing raw consensus dataset from {consensus_csv}...")
    if not os.path.exists(consensus_csv):
        raise FileNotFoundError(f"Consensus CSV not found at {consensus_csv}")
        
    df = pd.read_csv(consensus_csv)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Parquet
    parquet_path = os.path.join(out_dir, "merged_from_database.parquet")
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Parquet format to {parquet_path}")
    
    # 2. JSON
    json_path = os.path.join(out_dir, "merged_from_database.json")
    df.to_json(json_path, orient="records", indent=4)
    logger.info(f"Saved JSON format to {json_path}")
    
    # 3. Pickle
    pickle_path = os.path.join(out_dir, "merged_from_database.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(df, f)
    logger.info(f"Saved Pickle format to {pickle_path}")
    
    return True

def _taxid_for(organism: object) -> object:
    """
    Resolves an organism name to its NCBI tax id for storage beside the name.

    :param organism: Target organism string, possibly strain-qualified or absent.
    :returns: The tax id, or ``None`` when the name is empty or does not resolve. Never raises --
        a taxonomy miss must not abort a database build.
    """
    if not organism or not str(organism).strip():
        return None
    try:
        from sample.nlp import resolve_taxon
        return resolve_taxon(str(organism))[1]
    except Exception as e:
        logger.debug(f"taxid lookup failed for {organism!r}: {e}")
        return None


def load_sql_database(final_dataset_csv: str, db_path: str = "data/sample.db") -> bool:
    """
    Creates and populates an SQLite database from the final assembled dataset,
    setting up normalized tables for peptides, activities, and descriptors.
    """
    logger.info(f"Loading final assembled dataset from {final_dataset_csv} into SQLite database {db_path}...")
    if not os.path.exists(final_dataset_csv):
        raise FileNotFoundError(f"Final dataset CSV not found at {final_dataset_csv}")
        
    df = pd.read_csv(final_dataset_csv)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    # entry_id -> molecular_weight, for the derived value_mM/value_ug_ml/potency columns every
    # activities-table INSERT below computes. Built once here rather than re-read per row.
    mw_by_entry_id: Dict[str, float] = {}
    if "molecular_weight" in df.columns:
        mw_mask = df["entry_id"].notna() & df["molecular_weight"].notna()
        mw_by_entry_id = dict(zip(
            df.loc[mw_mask, "entry_id"].astype(str),
            df.loc[mw_mask, "molecular_weight"].astype(float),
        ))

    def _concentration(entry_id: object, value: object, unit: object) -> Dict[str, Any]:
        """Derived mM/ug_ml/potency for one activity row, or all-``None`` when not computable."""
        from sample.concentration import concentration_conversions
        empty = {"mM": None, "ug_ml": None, "potency": None}
        if value is None or unit is None or (isinstance(value, float) and pd.isna(value)):
            return empty
        try:
            out = concentration_conversions(float(value), unit, mw_by_entry_id.get(str(entry_id)))
        except (TypeError, ValueError):
            return empty
        return {"mM": out["mM"], "ug_ml": out["ug_ml"], "potency": out["potency"]}

    # Built entirely in memory and only written to `db_path` once, at the very end (see the
    # `conn.backup(...)` call below). A run against the on-disk file directly was measured taking
    # 50+ CPU-minutes and tens of GB of read() syscalls on this same data -- SQLite's default 2MB
    # page cache thrashes against a file that grows past it, and every one of the many sequential
    # phases here (peptides, activities, bibliography, names, topology, modifications, source
    # organisms, cross-references, GO terms, families, bioactivity linking, two index-building
    # passes) paid that cost separately. An in-memory database has no file to page against, so the
    # cost disappears regardless of which specific phase was actually slow.
    conn = sqlite3.connect(":memory:")

    # Copy in whatever already exists at db_path, before any of the DROP/CREATE calls below. Some
    # tables here are never dropped -- `peptide_overrides` (see `_apply_peptide_overrides`) holds
    # admin corrections made through `/admin` and is deliberately durable across every rebuild.
    # Starting from a blank in-memory database would silently discard it, since nothing in this
    # function's own inputs (the CSV, the JSON/CAMP/DBAASP files) ever reconstructs it.
    if os.path.exists(db_path):
        disk_conn = sqlite3.connect(db_path)
        disk_conn.backup(conn)
        disk_conn.close()

    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    # Large page cache even in memory -- SQLite's pager still buffers by page count, not a hard
    # cap on the backing store, so a small cache_size can still force internal reallocation churn
    # under this much random-access insert traffic. 256 MB comfortably covers this dataset.
    cursor.execute("PRAGMA cache_size = -262144;")

    # Drop existing tables to apply schema migrations
    cursor.execute("DROP TABLE IF EXISTS activities;")
    cursor.execute("DROP TABLE IF EXISTS descriptors;")
    cursor.execute("DROP TABLE IF EXISTS peptide_references;")
    cursor.execute("DROP TABLE IF EXISTS papers;")
    cursor.execute("DROP TABLE IF EXISTS peptides;")
    
    # 0. Create papers and many-to-many tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_type TEXT,
        source_id TEXT,
        UNIQUE(source_type, source_id)
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS peptide_references (
        peptide_id TEXT,
        paper_id INTEGER,
        FOREIGN KEY (peptide_id) REFERENCES peptides(entry_id) ON DELETE CASCADE,
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
        UNIQUE(peptide_id, paper_id)
    );
    ''')
    
    # 1. Create peptides table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptides (
        entry_id TEXT PRIMARY KEY,
        sequence TEXT NOT NULL,
        SMILES TEXT,
        source TEXT,
        is_decoy INTEGER DEFAULT 0,
        LITERATURE TEXT,
        PATENT TEXT,
        Reference TEXT,
        Biological_Activity TEXT,
        -- The organism that produces the peptide, or 'Synthetic' for a designed sequence.
        -- NULL where neither is recorded: see sample.provenance for why that is left empty
        -- rather than filled with "Unknown".
        source_organism TEXT,
        -- Raw "uniprot:P81941 | pdb:1MAG" cell, parsed on read by
        -- sample.external_links.cross_reference_links -- see sample.identifiers for why this is
        -- not pre-split into its own table.
        cross_references TEXT
    );
    """)
    
    # 2. Create descriptors table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS descriptors (
        entry_id TEXT PRIMARY KEY,
        molecular_weight REAL,
        logp REAL,
        hydrogen_bond_donors INTEGER,
        hydrogen_bond_acceptors INTEGER,
        charge REAL,
        sequence_length INTEGER,
        -- Step 43's Isolation Forest verdict. Flagged, never dropped: an anomaly is still a real
        -- peptide with real evidence behind it, just compositionally unusual relative to the rest
        -- of the release -- worth a visible caveat on the page, not exclusion from the dataset.
        is_anomaly INTEGER,
        anomaly_score REAL,
        FOREIGN KEY (entry_id) REFERENCES peptides(entry_id) ON DELETE CASCADE
    );
    """)
    
    # 3. Create activities table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT,
        activity_group TEXT,
        value REAL,
        unit TEXT,
        -- Derived from (value, unit) and the peptide's own molecular_weight (already loaded into
        -- `df` above) via sample.concentration.concentration_conversions -- computed once here,
        -- at load time, rather than left for every consumer (the website, a future analysis) to
        -- reimplement its own unit math. NULL when `unit` is not a recognized concentration, or
        -- (for value_ug_ml specifically) when molecular_weight is unavailable for this peptide.
        value_mM REAL,
        value_ug_ml REAL,
        -- -log10(molar concentration): the standard pharmacological potency transform (pIC50-style)
        -- so activity spanning orders of magnitude compares on one linear scale.
        potency REAL,
        target_organism TEXT,
        -- The NCBI tax id the organism name resolves to. Every lookup in this pipeline already
        -- joins through it; storing it makes the row linkable to NCBI Taxonomy and joinable
        -- across sources that spell a species differently.
        target_taxid INTEGER,
        activity_type TEXT,
        mechanism_of_action TEXT,
        -- Which source record this measurement came from, and that record's own pointer to the
        -- paper backing it. DBAASP numbers the articles on each card and every measurement cites
        -- one by index; without both columns the bibliography can only be attached at peptide
        -- level, which would credit each paper with measurements it never reported.
        source_accession TEXT,
        source_reference_index TEXT,
        FOREIGN KEY (entry_id) REFERENCES peptides(entry_id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    
    # Populate database
    peptides_inserted = 0
    descriptors_inserted = 0
    activities_inserted = 0
    # Counted, not just logged at debug level. DECISIONS.md's first project rule exists because a
    # bare except in THIS file's DBAASP ingest loop hid a 60% loss of the activity table -- 152,308
    # rows -- with nothing in the log. These handlers were the same shape: debug-level and
    # uncounted, so a total failure and a clean run looked identical. The released database holds
    # 204,370 activities against 225,871 rows in activities.csv, and without these counters there
    # was no way to tell how much of that 21,501-row gap is by design (the two are derived
    # differently) and how much is silent insert failure.
    activities_failed = 0
    activity_json_failed = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Peptides, descriptors, activities"):
        entry_id = row.get("entry_id")
        seq = row.get("sequence")
        smiles = row.get("SMILES")
        source = row.get("source", "merged")
        is_decoy = int(row.get("is_decoy", 0))
        
        if pd.isna(entry_id) or not entry_id:
            continue
            
        # Insert peptide
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO peptides (entry_id, sequence, SMILES, source, is_decoy, LITERATURE, PATENT, Reference, Biological_Activity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, seq, smiles, source, is_decoy, row.get("LITERATURE"), row.get("PATENT"), row.get("Reference"), row.get("Biological_Activity"))
            )
            peptides_inserted += 1
            
            # Process many-to-many references
            def link_refs(val: object, stype: str) -> None:
                """
                Records a cross-reference cell as rows in the ``papers`` link table.

                :param val: Raw identifier(s) from the merged table, pipe-separated. ``NaN`` or
                    empty is ignored.
                :param stype: Source database name (``PubChem``, ``ChEMBL``, ...) stored alongside
                    the identifier so the pair is unambiguous.
                :returns: ``None``; rows are written through the enclosing cursor.
                """
                if pd.notna(val) and val:
                    for sid in str(val).split(" | "):
                        sid = sid.strip()
                        # PMIDs arrive from the merged frame in float form ("10075760.0") because
                        # pandas types any integer column containing a NaN as float64. Stored raw,
                        # that reaches the page as the accession text AND corrupts the PubMed URL
                        # built from it, so the fractional part is dropped at the boundary.
                        if stype == "pubmed":
                            sid = re.sub(r"\.\d+$", "", sid)
                        if sid:
                            cursor.execute("INSERT OR IGNORE INTO papers (source_type, source_id) VALUES (?, ?)", (stype, sid))
                            cursor.execute("SELECT id FROM papers WHERE source_type = ? AND source_id = ?", (stype, sid))
                            paper_id = cursor.fetchone()[0]
                            cursor.execute("INSERT OR IGNORE INTO peptide_references (peptide_id, paper_id) VALUES (?, ?)", (entry_id, paper_id))
            
            link_refs(row.get("pubmed_ids"), "pubmed")
            link_refs(row.get("dois"), "doi")
            link_refs(row.get("patent_ids"), "patent")
            link_refs(row.get("ADP"), "ADP")
            link_refs(row.get("CAMP"), "CAMP")
            link_refs(row.get("DBAMP"), "DBAMP")
            link_refs(row.get("DRAMP"), "DRAMP")
            link_refs(row.get("DBAASP"), "DBAASP")
            
        except Exception as e:
            logger.warning(f"Error inserting peptide {entry_id}: {e}")
            continue
            
        # Insert descriptors
        mw = row.get("molecular_weight")
        logp = row.get("logp")
        hbd = row.get("hydrogen_bond_donors")
        hba = row.get("hydrogen_bond_acceptors")
        charge = row.get("charge")
        seq_len = row.get("sequence_length")
        is_anomaly = row.get("is_anomaly")
        anomaly_score = row.get("anomaly_score")

        if pd.notna(mw):
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO descriptors (entry_id, molecular_weight, logp, hydrogen_bond_donors, hydrogen_bond_acceptors, charge, sequence_length, is_anomaly, anomaly_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry_id, float(mw) if pd.notna(mw) else None,
                     float(logp) if pd.notna(logp) else None,
                     int(hbd) if pd.notna(hbd) else None,
                     int(hba) if pd.notna(hba) else None,
                     float(charge) if pd.notna(charge) else None,
                     int(seq_len) if pd.notna(seq_len) else None,
                     int(is_anomaly) if pd.notna(is_anomaly) else None,
                     float(anomaly_score) if pd.notna(anomaly_score) else None)
                )
                descriptors_inserted += 1
            except Exception as e:
                logger.debug(f"Error inserting descriptors for {entry_id}: {e}")
                
        # Insert activities (from row if present)
        activity_group = row.get("activity_group")
        val = row.get("value")
        unit = row.get("unit")
        target = row.get("target_organism")
        
        if pd.notna(activity_group) and activity_group:
            try:
                conc = _concentration(entry_id, val if pd.notna(val) else None,
                                      unit if pd.notna(unit) else None)
                cursor.execute(
                    "INSERT INTO activities (entry_id, activity_group, value, unit, value_mM, "
                    "value_ug_ml, potency, target_organism) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry_id, str(activity_group), float(val) if pd.notna(val) else None,
                     str(unit) if pd.notna(unit) else None, conc["mM"], conc["ug_ml"],
                     conc["potency"], str(target) if pd.notna(target) else None)
                )
                activities_inserted += 1
            except Exception as e:
                activities_failed += 1
                logger.debug(f"Error inserting activity for {entry_id}: {e}")
                
    # Build sequence-to-ID map for JSON extraction loading.
    #
    # Vectorized rather than a second `df.iterrows()` pass over the same frame the main loop above
    # already walked -- iterrows() reconstructs a Series per row, and on the ~107k-row release
    # table that pass alone measured 18s against 0.65s here (28x), on top of the row loop this
    # duplicated.
    mask = df["entry_id"].notna() & df["sequence"].notna()
    seq_to_id = dict(zip(
        df.loc[mask, "sequence"].astype(str).str.upper().str.strip(),
        df.loc[mask, "entry_id"].astype(str),
    ))
            
    # SAMPLE's own extracted activities.
    #
    # Two things were wrong here and they compounded. The scan used `os.walk`, which descends into
    # every sibling directory -- including the `pubmed.bak`, `pubmed_old`, `scopus.bak` and
    # `scopus_old` copies left beside the live output. Those are byte-for-byte duplicates of each
    # other, so everything they matched was inserted twice: 57,748 surplus rows, 22.7 % of the
    # table. And the field names it read (`activity_group`, `target_organism_or_cell`,
    # `activity_value`) predate the current extraction contract, so the *live* directories matched
    # nothing at all. The table was therefore built entirely from stale backups in a schema the
    # pipeline no longer produces.
    #
    # `iter_extraction_files` applies the recognized-subdirectory allowlist that the rest of the
    # pipeline uses (see config.iter_extraction_files), which is what keeps the backups out. Only
    # the repaired set is read -- the raw `literature_data_json` holds the same documents before
    # the contract was enforced, and reading both would double-count them again.
    json_files = (iter_extraction_files("data/literature_data_json_repaired")
                  + iter_extraction_files("data/patent_data_json_repaired"))

    logger.info(f"Scanning {len(json_files)} LLM JSON files to populate activities table...")
    for jf in tqdm(json_files, desc="LLM JSON extraction files"):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in (data.get("peptides") or []) if isinstance(data, dict) else []:
                if not isinstance(item, dict):
                    continue
                seq = item.get("peptide_sequence")
                if not seq:
                    continue
                seq_clean = str(seq).upper().strip()
                target_id = seq_to_id.get(seq_clean)
                if not target_id:
                    continue

                moa = item.get("mechanism_of_action")
                activities = item.get("biological_activities", [])
                if not isinstance(activities, list):
                    activities = [activities]

                for act in activities:
                    if not isinstance(act, dict):
                        continue
                    act_group = act.get("target_group")
                    # A cell-line assay names the line, not a species; either can be the target.
                    target_org = act.get("target_organism") or act.get("target_cell_line")
                    act_type = act.get("activity_type")
                    # `value_lower` is the normalised numeric the contract produces; `value_raw` is
                    # the string it was parsed from and may carry a range or an operator.
                    act_val = act.get("value_lower")
                    act_unit = act.get("activity_unit")

                    if act_group or target_org or act_val is not None:
                        try:
                            conc = _concentration(target_id, act_val, act_unit)
                            cursor.execute(
                                "INSERT INTO activities (entry_id, activity_group, value, unit, "
                                "value_mM, value_ug_ml, potency, target_organism, target_taxid, "
                                "activity_type, mechanism_of_action) VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (target_id,
                                 str(act_group) if act_group else None,
                                 float(act_val) if act_val is not None else None,
                                 str(act_unit) if act_unit else None,
                                 conc["mM"], conc["ug_ml"], conc["potency"],
                                 str(target_org) if target_org else None,
                                 _taxid_for(target_org),
                                 str(act_type) if act_type else None,
                                 str(moa) if moa else None)
                            )
                            activities_inserted += 1
                        except Exception as ex:
                            activities_failed += 1
                            logger.debug(f"Error inserting activities details: {ex}")
        except Exception as e:
            activity_json_failed += 1
            logger.debug(f"Error parsing json file {jf}: {e}")

    # Parse CAMP database activity files
    camp_dir = "data/antimicrobial_databases/camp"
    if os.path.exists(camp_dir):
        logger.info("Scanning CAMP database files for activities...")
        camp_files = glob.glob(os.path.join(camp_dir, "*.csv"))
        camp_errors: Dict[str, int] = {}
        for csv_file in tqdm(camp_files, desc="CAMP activity files"):
            try:
                # Each file is a two-column `key,value` table (one CAMP record's fields as rows),
                # not tabular data -- pandas' per-call setup cost (DataFrame + dtype inference)
                # measured at ~4ms/file here, 99s for the 23,718 files on this corpus alone. A
                # plain csv.reader gives the same dict in the same read, verified against every
                # file in this corpus: none has a row with anything but exactly 2 fields, so this
                # is not a quoting/escaping behavior change from pandas' parser.
                with open(csv_file, newline='', encoding='utf-8') as fh:
                    reader = csv.reader(fh)
                    next(reader, None)  # header row: "key,value"
                    data_dict = {row[0]: row[1] for row in reader if len(row) >= 2}
                sequence = data_dict.get('Sequence')
                if not sequence:
                    continue
                seq_clean = str(sequence).upper().strip()
                target_id = seq_to_id.get(seq_clean)
                if not target_id:
                    continue
                
                act_group = data_dict.get('Activity')
                if act_group:
                    act_group = str(act_group).lower()
                    if "anti" in act_group:
                        if "bacterial" in act_group:
                            act_group = "antibacterial"
                        elif "fungal" in act_group:
                            act_group = "antifungal"
                        elif "viral" in act_group:
                            act_group = "antiviral"
                        elif "tumor" in act_group or "cancer" in act_group:
                            act_group = "anticancer"
                            
                    cursor.execute(
                        "INSERT INTO activities (entry_id, activity_group, value, unit, target_organism, target_taxid, activity_type, mechanism_of_action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (target_id, act_group, None, None, None, None, None, "CAMP database record")
                    )
                    activities_inserted += 1
            except Exception as e:
                camp_errors[type(e).__name__] = camp_errors.get(type(e).__name__, 0) + 1
        if camp_errors:
            logger.warning(f"CAMP activity scan: {sum(camp_errors.values())} of {len(camp_files)} "
                           f"files failed: {camp_errors}")

    # Parse DBAASP database activity files
    dbaasp_dir = "data/antimicrobial_databases/dbaasp"
    if os.path.exists(dbaasp_dir):
        logger.info("Scanning DBAASP database files for activities...")
        dbaasp_files = glob.glob(os.path.join(dbaasp_dir, "*.json"))
        dbaasp_errors: Dict[str, int] = {}
        for json_file in tqdm(dbaasp_files, desc="DBAASP activity files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                seq = data.get("sequence")
                if not seq:
                    continue
                seq_clean = str(seq).upper().strip()
                target_id = seq_to_id.get(seq_clean)
                if not target_id:
                    continue
                
                target_activities = data.get("targetActivities", [])
                for act in target_activities:
                    target_species = act.get("targetSpecies", {}) or {}
                    target_name = target_species.get("name")
                    
                    act_measure = act.get("activityMeasureGroup")
                    act_type = act_measure.get("name") if act_measure else act.get("activityMeasureValue")
                    
                    act_val = act.get("activity")
                    if act_val is None:
                        act_val = act.get("concentration")
                    try:
                        act_val = float(act_val) if act_val is not None else None
                    except ValueError:
                        act_val = None
                        
                    unit_info = act.get("unit") or {}
                    unit_name = unit_info.get("name")
                    
                    # Derived from THIS activity's own target organism.
                    #
                    # It used to be read from `data["targetGroups"]`, which is a peptide-level
                    # field listing every group the peptide was ever tested against, with
                    # "cancer" checked first. So a peptide assayed against both tumour cells and
                    # microbes had every one of its activities stamped `anticancer` -- including
                    # its Escherichia coli MICs and its Candida albicans MICs. DBAASP records no
                    # per-activity group (an activity carries only `targetSpecies.name`), so the
                    # group has to come from the organism, which is what infer_activity_group
                    # does via the NCBI taxonomy.
                    from sample.nlp import infer_activity_group
                    act_group = infer_activity_group(target_name or "", act_type or "")

                    conc = _concentration(target_id, act_val, unit_name)
                    cursor.execute(
                        "INSERT INTO activities (entry_id, activity_group, value, unit, value_mM, "
                        "value_ug_ml, potency, target_organism, target_taxid, activity_type, "
                        "mechanism_of_action, source_accession, source_reference_index) VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (target_id, act_group, act_val, unit_name, conc["mM"], conc["ug_ml"],
                         conc["potency"], target_name,
                         _taxid_for(target_name), act_type, "DBAASP database record",
                         data.get("dbaaspId"), str(act.get("reference") or "") or None)
                    )
                    activities_inserted += 1
            except Exception as e:
                # This used to be a bare `except: pass`. When the DBAASP layer silently stopped
                # contributing -- 152,308 activities, 60 % of the table -- there was nothing in the
                # log to say so, and the count looked plausible enough to pass unnoticed. Failures
                # are counted and reported below rather than swallowed one file at a time.
                dbaasp_errors[type(e).__name__] = dbaasp_errors.get(type(e).__name__, 0) + 1
        if dbaasp_errors:
            logger.warning(f"DBAASP scan: {sum(dbaasp_errors.values())} of {len(dbaasp_files)} "
                           f"files failed: {dbaasp_errors}")

    conn.commit()

    # Everything past this point is a sequence of named phases rather than one big loop, so
    # progress is shown per phase instead of per row -- each of these can individually take
    # anywhere from under a second to tens of seconds, and with no output between them a stalled
    # run and a merely slow one look identical from the terminal.
    phases: List[Tuple[str, Callable[[], None]]] = [
        ("Peptide overrides", lambda: _apply_peptide_overrides(cursor)),
        ("Bibliography", lambda: _load_bibliography(cursor)),
        ("Peptide names + topology", lambda: (_load_peptide_names(cursor, seq_to_id),
                                               _load_topology(cursor, seq_to_id))),
        ("Chemical modifications", lambda: _load_modifications(cursor, seq_to_id)),
        ("Source organisms", lambda: _load_source_organisms(cursor, seq_to_id)),
        ("Cross-references", lambda: _load_cross_references(cursor, seq_to_id)),
        ("GO terms + families + bioactivity",
         lambda: (_load_gene_ontology(cursor, seq_to_id), _load_families(cursor),
                  _load_bioactivity(cursor), _link_peptides_to_bioactivity(cursor, seq_to_id))),
        ("Indexes", lambda: (_create_indexes(cursor), _create_fulltext_index(cursor))),
        # Query-planner statistics for the indexes just built. Space reclamation is no longer this
        # step's job -- the destination file is written fresh below, never DROP-and-reused, so
        # there is no freed space to reclaim.
        ("Query planner statistics", lambda: cursor.execute("ANALYZE;")),
    ]
    with tqdm(phases, desc="Loading SQLite database", unit="phase") as pbar:
        for label, action in pbar:
            pbar.set_description(label)
            action()
            conn.commit()

    # Write the completed in-memory database out to disk in one pass, replacing whatever was at
    # db_path. Stale sidecar files from an interrupted prior run (rollback journal, WAL) would
    # otherwise sit next to the new file and confuse the next open.
    for suffix in ("", "-journal", "-wal", "-shm"):
        stale = db_path + suffix
        if os.path.exists(stale):
            os.remove(stale)
    disk_conn = sqlite3.connect(db_path)
    conn.backup(disk_conn)
    disk_conn.close()
    conn.close()

    logger.info(f"SQL Load complete. Inserted {peptides_inserted} peptides, "
                f"{descriptors_inserted} descriptors, and {activities_inserted} activities records.")
    if activities_failed or activity_json_failed:
        logger.warning(
            f"SQL load dropped {activities_failed} activity row(s) on INSERT and skipped "
            f"{activity_json_failed} unparseable activity JSON payload(s). These are NOT in "
            f"data/sample.db and so are invisible to the web server and the API. Re-run with "
            f"the logger at DEBUG to see each one.")
    else:
        logger.info("SQL load: no activity rows were dropped on INSERT.")
    return True


def _apply_peptide_overrides(cursor: sqlite3.Cursor) -> int:
    """
    Reapplies durable admin corrections onto the just-rebuilt `peptides` table.

    `peptides` above is dropped and repopulated from the CSVs on every run, which is right for
    everything the pipeline itself derives -- and wrong for a curator's hand correction, which the
    pipeline knows nothing about and would otherwise silently discard. `peptide_overrides`
    (:class:`sample.models.PeptideOverride`) is written by `/admin`'s entry-edit form and has the
    opposite lifecycle: never dropped here, only read.

    Table and column both come from the same source (`sample/config.EDITABLE_PEPTIDE_FIELDS`) that
    the admin form validates against before ever writing a row, so `field` is trustworthy -- it is
    still checked again here rather than interpolated on faith, because this function is the one
    place that turns it into a raw SQL column name.

    :param cursor: Open cursor on the release database, after `peptides` has been repopulated.
    :returns: How many overrides were reapplied. Skips (and reports in the summary log line) rows
        whose `entry_id` no longer exists in this release -- a peptide legitimately dropping out
        between releases is not an error.
    """
    from sample.config import EDITABLE_PEPTIDE_FIELDS

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_overrides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT NOT NULL,
        field TEXT NOT NULL,
        value TEXT,
        admin_username TEXT,
        changed_at TEXT,
        UNIQUE(entry_id, field)
    );
    """)

    cursor.execute("SELECT entry_id, field, value FROM peptide_overrides")
    rows = cursor.fetchall()
    if not rows:
        return 0

    applied = 0
    skipped_field = 0
    skipped_missing = 0
    for entry_id, field, value in rows:
        if field not in EDITABLE_PEPTIDE_FIELDS:
            # Only reachable if EDITABLE_PEPTIDE_FIELDS was narrowed after the row was written --
            # the admin form already refuses to write anything outside it.
            skipped_field += 1
            continue
        cursor.execute(f"UPDATE peptides SET {field} = ? WHERE entry_id = ?", (value, entry_id))
        if cursor.rowcount:
            applied += 1
        else:
            skipped_missing += 1

    logger.info(f"Reapplied {applied} admin override(s) onto the rebuilt peptides table"
               + (f" ({skipped_missing} skipped: entry_id no longer present)"
                  if skipped_missing else "")
               + (f" ({skipped_field} skipped: field no longer editable)"
                  if skipped_field else "") + ".")
    return applied


def _load_bibliography(cursor: sqlite3.Cursor, references_json: str = None) -> int:
    """
    Loads the resolved bibliography and attaches it to peptides and activities.

    The ``papers`` table is not a bibliography: its ``source_type`` values are source-database
    names and its ``source_id`` values are those databases' accessions, so it records where a
    peptide was copied from, not which paper measured it. This adds the citations themselves --
    PMID, DOI, authors, journal and a preformatted APA string -- built by
    ``sample-cli build-references``.

    Two link tables, because the sources differ in what they can support. DBAASP states which
    article backs each individual measurement, so those become ``activity_references``. The other
    databases attach a reference to the peptide as a whole, which becomes ``peptide_bibliography``.
    Keeping them apart matters: presenting a peptide-level citation as though it supported a
    specific MIC would overstate what the source actually says.

    :param cursor: Open cursor on the release database.
    :param references_json: Bibliography file; ``data/references.json`` when omitted. Absent means
        the step has not been run, which is a warning rather than an error -- the rest of the
        database is still valid without it.
    :returns: The number of references loaded.
    """
    references_json = references_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "references.json")

    cursor.execute("DROP TABLE IF EXISTS activity_references;")
    cursor.execute("DROP TABLE IF EXISTS peptide_bibliography;")
    cursor.execute("DROP TABLE IF EXISTS bibliography;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bibliography (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ref_key TEXT UNIQUE,
        pmid TEXT,
        doi TEXT,
        scopus_eid TEXT,
        title TEXT,
        authors TEXT,
        journal TEXT,
        year INTEGER,
        volume TEXT,
        issue TEXT,
        pages TEXT,
        apa TEXT,
        url TEXT,
        source_db TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_bibliography (
        peptide_id TEXT,
        reference_id INTEGER,
        UNIQUE(peptide_id, reference_id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity_references (
        activity_id INTEGER,
        reference_id INTEGER,
        UNIQUE(activity_id, reference_id)
    );
    """)

    if not os.path.exists(references_json):
        logger.warning(
            f"No bibliography at {references_json}; peptides will have no citations. "
            f"Run `sample-cli build-references` (step 36) to create it.")
        return 0

    with open(references_json, encoding="utf-8") as handle:
        payload = json.load(handle)

    reference_ids: Dict[str, int] = {}
    for key, record in tqdm(list((payload.get("references") or {}).items()),
                            desc="Bibliography: references"):
        cursor.execute(
            "INSERT OR IGNORE INTO bibliography "
            "(ref_key, pmid, doi, scopus_eid, title, authors, journal, year, volume, issue, "
            " pages, apa, url, source_db) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, record.get("pmid"), record.get("doi"), record.get("scopus_eid"),
             record.get("title"), "; ".join(record.get("authors") or []),
             record.get("journal"), record.get("year"), record.get("volume"),
             record.get("issue"), record.get("pages"), record.get("apa"),
             record.get("url"), record.get("source_db")))
        cursor.execute("SELECT id FROM bibliography WHERE ref_key = ?", (key,))
        row = cursor.fetchone()
        if row:
            reference_ids[key] = row[0]

    # A source accession (DBAASPN_10055, AP00001, CAMPSQ1, DRAMP00005) reaches its peptides
    # through the papers table, which already records exactly that mapping.
    #
    # Both lookups below used to be a query inside this loop, one per link. Neither is served by an
    # index at this point in the load: `papers` only has the composite UNIQUE(source_type,
    # source_id), which can't answer a source_id-only filter, and the index on
    # `activities(source_accession, source_reference_index)` doesn't exist yet -- index creation is
    # a later phase of this same load. That made both queries a full table scan per link; measured
    # at 16 items/s over 218k links, a ~3h45m projected runtime for this loop alone. Building both
    # maps once up front turns every iteration into a dict lookup instead.
    accession_to_peptides: Dict[str, List[str]] = {}
    for source_id, peptide_id in cursor.execute(
            "SELECT p.source_id, pr.peptide_id FROM peptide_references pr "
            "JOIN papers p ON p.id = pr.paper_id"):
        accession_to_peptides.setdefault(source_id, []).append(peptide_id)

    # DBAASP's per-measurement pointer: the index is 1-based over that card's own article list, and
    # each measurement stores the index it cites. Matching on both the accession and that index
    # attaches a paper only to the measurements that actually cite it. Joining on the peptide alone
    # instead produced 351,243 links across 254,248 activities -- every paper credited with every
    # measurement of every peptide it touched.
    activity_lookup: Dict[Tuple[str, str], List[int]] = {}
    for accession, ref_index, activity_id in cursor.execute(
            "SELECT source_accession, source_reference_index, id FROM activities "
            "WHERE source_accession IS NOT NULL AND source_reference_index IS NOT NULL"):
        activity_lookup.setdefault((accession, ref_index), []).append(activity_id)

    peptide_links = 0
    peptide_bib_rows: List[Tuple[str, int]] = []
    activity_ref_rows: List[Tuple[int, int]] = []
    for link in tqdm(payload.get("links") or [], desc="Bibliography: peptide/activity links"):
        reference_id = reference_ids.get(link.get("reference"))
        if reference_id is None:
            continue
        peptides = accession_to_peptides.get(link.get("accession"))
        if not peptides:
            continue

        for peptide_id in peptides:
            peptide_bib_rows.append((peptide_id, reference_id))
            peptide_links += 1

        if link.get("activity_index"):
            key = (link.get("accession"), str(link.get("activity_index")))
            for activity_id in activity_lookup.get(key, []):
                activity_ref_rows.append((activity_id, reference_id))

    cursor.executemany(
        "INSERT OR IGNORE INTO peptide_bibliography (peptide_id, reference_id) VALUES (?, ?)",
        peptide_bib_rows)

    before = cursor.execute("SELECT COUNT(*) FROM activity_references").fetchone()[0]
    cursor.executemany(
        "INSERT OR IGNORE INTO activity_references (activity_id, reference_id) VALUES (?, ?)",
        activity_ref_rows)
    activity_links = cursor.execute(
        "SELECT COUNT(*) FROM activity_references").fetchone()[0] - before

    logger.info(f"Bibliography: {len(reference_ids)} references, {peptide_links} peptide links, "
                f"{activity_links} activity links.")
    return len(reference_ids)


def _load_peptide_names(cursor: sqlite3.Cursor, seq_to_id: Dict[str, str],
                        names_json: str = None) -> int:
    """
    Loads peptide names and synonyms, and makes them searchable.

    Peptides had no name in the release at all: ``peptides`` carries an id, a sequence and a
    SMILES string, so searching "magainin" returned nothing while magainin's sequence sat in the
    table. Names arrive keyed by sequence, because source accessions differ per database while the
    sequence is common to all of them.

    Synonyms are stored beside primary names and marked as such. They are what makes the search
    usable across the literature -- the same peptide is ``Dermaseptin-B2``, ``DRS-B2``, ``DS bII``
    and ``ADENOREGULIN`` in different papers.

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :param names_json: Name file; ``data/peptide_names.json`` when omitted.
    :returns: The number of names loaded.
    """
    names_json = names_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "peptide_names.json")

    cursor.execute("DROP TABLE IF EXISTS peptide_names;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_names (
        entry_id TEXT,
        name TEXT,
        name_kind TEXT,     -- 'primary' or 'synonym'
        source_db TEXT,
        UNIQUE(entry_id, name)
    );
    """)

    if not os.path.exists(names_json):
        logger.warning(
            f"No peptide names at {names_json}; peptides will not be searchable by name. "
            f"Run `sample-cli build-names` to create it.")
        return 0

    with open(names_json, encoding="utf-8") as handle:
        by_sequence = json.load(handle)

    rows = []
    for sequence, names in by_sequence.items():
        entry_id = seq_to_id.get(str(sequence).upper().strip())
        if not entry_id:
            continue
        for record in names:
            rows.append((entry_id, record.get("name"), record.get("kind"),
                         record.get("source")))

    cursor.executemany(
        "INSERT OR IGNORE INTO peptide_names (entry_id, name, name_kind, source_db) "
        "VALUES (?,?,?,?)", rows)
    logger.info(f"Names: {len(rows)} loaded for {len({r[0] for r in rows})} peptides "
                f"({sum(1 for r in rows if r[2] == 'synonym')} synonyms).")
    return len(rows)


def _load_topology(cursor: sqlite3.Cursor, seq_to_id: Dict[str, str],
                   topology_json: str = None) -> int:
    """
    Loads peptide topology: cyclic backbones and disulfide bridges.

    The pipeline treated every peptide as a linear chain, so a cyclotide and a linear peptide of
    the same composition were indistinguishable in the release. Topology is what separates them,
    and it changes the mass, the charge and the structure -- see :mod:`sample.cyclization`.

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :param topology_json: Topology file; ``data/peptide_topology.json`` when omitted.
    :returns: The number of peptides with a recorded topology.
    """
    topology_json = topology_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "peptide_topology.json")

    cursor.execute("DROP TABLE IF EXISTS peptide_topology;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_topology (
        entry_id TEXT PRIMARY KEY,
        kind TEXT,              -- linear | head_to_tail | disulfide | cyclic_unspecified | branched
        is_cyclic INTEGER,
        n_disulfides INTEGER,
        mass_correction REAL,   -- daltons a linear calculation overstates by
        bonds TEXT,             -- JSON: the source's bond list, with positions
        source_db TEXT
    );
    """)

    if not os.path.exists(topology_json):
        logger.warning(f"No peptide topology at {topology_json}; cyclic peptides will be "
                       f"indistinguishable from linear ones. Run `sample-cli build-topology`.")
        return 0

    with open(topology_json, encoding="utf-8") as handle:
        topologies = json.load(handle)

    rows = []
    for sequence, record in topologies.items():
        entry_id = seq_to_id.get(str(sequence).upper().strip())
        if not entry_id:
            continue
        kind = record.get("kind", "unknown")
        rows.append((
            entry_id, kind,
            int(kind in ("head_to_tail", "disulfide", "cyclic_unspecified")),
            int(record.get("n_disulfides") or 0),
            float(record.get("mass_correction") or 0.0),
            json.dumps(record.get("bonds") or []),
            record.get("source"),
        ))

    cursor.executemany(
        "INSERT OR REPLACE INTO peptide_topology "
        "(entry_id, kind, is_cyclic, n_disulfides, mass_correction, bonds, source_db) "
        "VALUES (?,?,?,?,?,?,?)", rows)

    cyclic = sum(1 for r in rows if r[2])
    logger.info(f"Topology: {len(rows)} peptides ({cyclic} cyclic, "
                f"{sum(1 for r in rows if r[1] == 'head_to_tail')} head-to-tail).")
    return len(rows)


def _load_modifications(cursor: sqlite3.Cursor, seq_to_id: Dict[str, str],
                        modifications_json: str = None) -> int:
    """
    Loads chemical modifications: capped termini and non-canonical residues.

    The sequence alone does not describe the molecule. 44 % of DBAASP's peptides are C-terminally
    amidated, which removes a negative charge -- so a net charge computed from the sequence is
    wrong by +1 for each of them, and net charge is the property most often quoted for an
    antimicrobial peptide. See :mod:`sample.modifications` for why the descriptor is published
    unaltered beside the modification rather than silently corrected.

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :param modifications_json: Modification file; ``data/modifications.json`` when omitted.
    :returns: The number of modification rows loaded.
    """
    modifications_json = modifications_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "modifications.json")

    cursor.execute("DROP TABLE IF EXISTS peptide_modifications;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_modifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT,
        terminus TEXT,          -- N | C, NULL for a residue modification
        position INTEGER,       -- 1-based residue, NULL for a terminal modification
        code TEXT,              -- the source's short code: AMD, ACT, C16, DHF
        description TEXT,
        kind TEXT,              -- amidation | acylation | unusual_residue | other
        source_db TEXT,
        source_accession TEXT,
        UNIQUE (entry_id, terminus, position, code)
    );
    """)

    if not os.path.exists(modifications_json):
        logger.warning(f"No modifications at {modifications_json}; amidation and other terminal "
                       f"chemistry will be missing. Run `sample-cli build-modifications`.")
        return 0

    with open(modifications_json, encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = []
    for record in payload.get("modifications", []):
        entry_id = seq_to_id.get(str(record.get("sequence") or "").upper().strip())
        if not entry_id:
            continue
        rows.append((
            entry_id, record.get("terminus"), record.get("position"),
            record.get("code"), record.get("description"), record.get("kind"),
            record.get("source_db"), record.get("source_accession"),
        ))

    cursor.executemany(
        "INSERT OR IGNORE INTO peptide_modifications "
        "(entry_id, terminus, position, code, description, kind, source_db, source_accession) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_modifications_entry "
                   "ON peptide_modifications(entry_id);")

    amidated = len({r[0] for r in rows if r[5] == "amidation"})
    logger.info(f"Modifications: {len(rows)} rows for {len({r[0] for r in rows})} peptides "
                f"({amidated} amidated).")
    return len(rows)


def _load_source_organisms(cursor: sqlite3.Cursor, seq_to_id: Dict[str, str],
                           organisms_json: str = None) -> int:
    """
    Fills in each peptide's origin: a named organism, or ``Synthetic``.

    An ``UPDATE`` rather than its own table, because this is a property of the peptide and there
    is exactly one value per row. Peptides whose origin is unrecorded keep ``NULL`` and the page
    omits the field entirely -- printing "Unknown" would read as though somebody had checked.

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :param organisms_json: Origin file; ``data/source_organisms.json`` when omitted.
    :returns: The number of peptides given an origin.
    """
    organisms_json = organisms_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "source_organisms.json")

    if not os.path.exists(organisms_json):
        logger.warning(f"No source organisms at {organisms_json}; the origin field will be empty "
                       f"for every peptide. Run `sample-cli build-source-organisms`.")
        return 0

    with open(organisms_json, encoding="utf-8") as handle:
        origins = (json.load(handle) or {}).get("source_organisms", {})

    rows = []
    for sequence, origin in origins.items():
        entry_id = seq_to_id.get(str(sequence).upper().strip())
        if entry_id and origin:
            rows.append((origin, entry_id))

    cursor.executemany("UPDATE peptides SET source_organism = ? WHERE entry_id = ?", rows)
    synthetic = sum(1 for origin, _ in rows if origin == "Synthetic")
    logger.info(f"Source organisms: {len(rows)} peptides "
                f"({len(rows) - synthetic} named, {synthetic} synthetic).")
    return len(rows)


def _load_cross_references(cursor: sqlite3.Cursor, seq_to_id: Dict[str, str],
                           refs_json: str = None) -> int:
    """
    Fills in each peptide's UniProt/PDB cross-references.

    An ``UPDATE`` rather than its own table, for the same reason as
    :func:`_load_source_organisms`: one raw cell per peptide, parsed on read by
    :func:`sample.external_links.cross_reference_links` rather than pre-split here.

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :param refs_json: Cross-reference file; ``data/cross_references.json`` when omitted.
    :returns: The number of peptides given a cross-reference cell.
    """
    refs_json = refs_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "cross_references.json")

    if not os.path.exists(refs_json):
        logger.warning(f"No cross-references at {refs_json}; UniProt/PDB links will be missing. "
                       f"Run `sample-cli build-cross-references`.")
        return 0

    with open(refs_json, encoding="utf-8") as handle:
        refs = (json.load(handle) or {}).get("cross_references", {})

    rows = []
    for sequence, blob in refs.items():
        entry_id = seq_to_id.get(str(sequence).upper().strip())
        if entry_id and blob:
            rows.append((blob, entry_id))

    cursor.executemany("UPDATE peptides SET cross_references = ? WHERE entry_id = ?", rows)
    logger.info(f"Cross-references: {len(rows)} peptides given a UniProt/PDB cell.")
    return len(rows)


def _load_gene_ontology(cursor: sqlite3.Cursor, seq_to_id: Dict[str, str],
                        go_json: str = None) -> int:
    """
    Loads Gene Ontology annotations harvested from CAMP.

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :param go_json: GO term file; ``data/gene_ontology.json`` when omitted.
    :returns: The number of GO term rows loaded.
    """
    go_json = go_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "gene_ontology.json")

    cursor.execute("DROP TABLE IF EXISTS peptide_go_terms;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_go_terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT,
        go_id TEXT,          -- GO:0005576
        category TEXT,       -- Molecular function | Biological process | Cellular component | NULL
        term TEXT,           -- free-text definition, e.g. "Extracellular region"
        evidence TEXT,       -- GO evidence code, e.g. IEA, IDA
        source_db TEXT,
        UNIQUE (entry_id, go_id, evidence)
    );
    """)

    if not os.path.exists(go_json):
        logger.warning(f"No Gene Ontology terms at {go_json}; the GO panel will be empty. "
                       f"Run `sample-cli build-gene-ontology`.")
        return 0

    with open(go_json, encoding="utf-8") as handle:
        by_sequence = (json.load(handle) or {}).get("gene_ontology", {})

    rows = []
    for sequence, terms in by_sequence.items():
        entry_id = seq_to_id.get(str(sequence).upper().strip())
        if not entry_id:
            continue
        for term in terms:
            rows.append((
                entry_id, term.get("go_id"), term.get("category"), term.get("term"),
                term.get("evidence"), "CAMP",
            ))

    cursor.executemany(
        "INSERT OR IGNORE INTO peptide_go_terms "
        "(entry_id, go_id, category, term, evidence, source_db) VALUES (?,?,?,?,?,?)", rows)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_go_terms_entry ON peptide_go_terms(entry_id);")

    peptides_annotated = len({r[0] for r in rows})
    logger.info(f"Gene Ontology: {len(rows)} terms for {peptides_annotated} peptides.")
    return len(rows)


def _load_families(cursor: sqlite3.Cursor, families_json: str = None) -> int:
    """
    Loads the family assignment built by ``sample-cli build-families``.

    Two provenance levels share one table and are distinguished by ``source``: ``pfam`` for the
    10,728 peptides matched to a curated Pfam family at its gathering threshold, and ``cluster``
    for the remainder grouped by near-identity. That distinction has to survive into the database,
    because a curated family assignment and a clustering artefact are not equally strong claims
    and a reader must be able to tell them apart.

    :param cursor: Open cursor on the release database.
    :param families_json: Assignment file; ``data/families/families.json`` when omitted.
    :returns: The number of peptides assigned a family.
    """
    families_json = families_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "families", "families.json")

    cursor.execute("DROP TABLE IF EXISTS peptide_families;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_families (
        entry_id TEXT PRIMARY KEY,
        family_id TEXT,
        family_name TEXT,
        source TEXT,          -- 'pfam' (curated) or 'cluster' (derived here)
        evalue REAL,
        all_domains TEXT
    );
    """)

    if not os.path.exists(families_json):
        logger.warning(
            f"No family assignment at {families_json}; peptides will have no family. "
            f"Run `sample-cli build-families` (step 37) to create it.")
        return 0

    with open(families_json, encoding="utf-8") as handle:
        assignment = json.load(handle)

    rows = [(entry_id, record.get("family_id"), record.get("family_name"),
             record.get("source"), record.get("evalue"),
             ",".join(record.get("all_domains") or []))
            for entry_id, record in assignment.items()
            if record.get("family_id")]
    cursor.executemany(
        "INSERT OR REPLACE INTO peptide_families "
        "(entry_id, family_id, family_name, source, evalue, all_domains) VALUES (?,?,?,?,?,?)",
        rows)

    pfam = sum(1 for r in rows if r[3] == "pfam")
    logger.info(f"Families: {len(rows)} peptides assigned "
                f"({pfam} from Pfam, {len(rows) - pfam} from clustering).")
    return len(rows)


def _link_peptides_to_bioactivity(cursor: sqlite3.Cursor,
                                  seq_to_id: Dict[str, str]) -> int:
    """
    Bridges peptides to the external identifiers their bioactivity is filed under.

    The bridge is indirect and lossy, and it is worth being explicit about why. Step 12 resolved
    ChEMBL and PubChem identifiers by peptide *name*, so the only route from a released peptide to
    its assay records is name -> identifier (``external_db_cache``) joined to name -> sequence
    (``activities.csv``) joined to sequence -> entry_id. Names are not identifiers: only 755 of
    21,089 activity names were ever resolved, reaching 768 sequences. So this table covers a small
    fraction of the release by construction, and its absence for a peptide means "never looked up",
    not "no assays exist".

    :param cursor: Open cursor on the release database.
    :param seq_to_id: Sequence -> entry_id, as built by the caller.
    :returns: The number of peptide/identifier links written.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(base, "data", "external_db_cache.sqlite")
    activities_path = os.path.join(base, "data", "sample_raw_release", "activities.csv")

    cursor.execute("DROP TABLE IF EXISTS peptide_bioactivity;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peptide_bioactivity (
        entry_id TEXT,
        archive TEXT,
        identifier TEXT,
        UNIQUE(entry_id, archive, identifier)
    );
    """)

    if not (os.path.exists(cache_path) and os.path.exists(activities_path)):
        logger.warning("External database cache or activity table missing; peptides will not be "
                       "linked to bioactivity archives.")
        return 0

    by_name: Dict[str, List[Tuple[str, str]]] = {}
    cache = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        for source, query, ext_id in cache.execute(
                "SELECT source, query, ext_id FROM lookup_cache "
                "WHERE ext_id IS NOT NULL AND ext_id != ''"):
            by_name.setdefault(str(query).strip().lower(), []).append((source, str(ext_id)))
    finally:
        cache.close()

    frame = pd.read_csv(activities_path, low_memory=False,
                        usecols=["peptide_name", "sequence"])
    frame = frame[frame["peptide_name"].notna() & frame["sequence"].notna()]

    links = set()
    for name, sequence in zip(frame["peptide_name"], frame["sequence"]):
        entries = by_name.get(str(name).strip().lower())
        if not entries:
            continue
        entry_id = seq_to_id.get(str(sequence).upper().strip())
        if not entry_id:
            continue
        for archive, identifier in entries:
            links.add((entry_id, archive, identifier))

    cursor.executemany(
        "INSERT OR IGNORE INTO peptide_bioactivity (entry_id, archive, identifier) "
        "VALUES (?,?,?)", sorted(links))
    logger.info(f"Bioactivity: linked {len({l[0] for l in links})} peptides to "
                f"{len(links)} external identifiers.")
    return len(links)


def _load_bioactivity(cursor: sqlite3.Cursor, bioactivity_json: str = None) -> int:
    """
    Loads the ChEMBL / PubChem BioAssay / BindingDB linkage.

    Stored keyed by the external identifier, with :func:`_link_peptides_to_bioactivity` providing
    the peptide bridge separately. Keeping them apart is deliberate: the archive records are exact,
    the peptide bridge is a name match, and merging them would let the second's uncertainty be read
    as the first's.

    :param cursor: Open cursor on the release database.
    :param bioactivity_json: Linkage file; ``data/bioactivity.json`` when omitted.
    :returns: The number of identifiers with data.
    """
    bioactivity_json = bioactivity_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bioactivity.json")

    cursor.execute("DROP TABLE IF EXISTS bioactivity;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bioactivity (
        archive TEXT NOT NULL,      -- chembl | pubchem | bindingdb
        identifier TEXT NOT NULL,   -- ChEMBL id, PubChem CID, or UniProt accession
        n_records INTEGER,
        payload TEXT,               -- the archive's own records, as JSON
        PRIMARY KEY (archive, identifier)
    );
    """)

    if not os.path.exists(bioactivity_json):
        logger.warning(
            f"No bioactivity linkage at {bioactivity_json}. "
            f"Run `sample-cli link-bioactivity` (step 39) to create it.")
        return 0

    with open(bioactivity_json, encoding="utf-8") as handle:
        linkage = json.load(handle)

    total = 0
    for archive, entries in linkage.items():
        rows = [(archive, identifier, len(records), json.dumps(records))
                for identifier, records in (entries or {}).items() if records]
        cursor.executemany(
            "INSERT OR REPLACE INTO bioactivity (archive, identifier, n_records, payload) "
            "VALUES (?,?,?,?)", rows)
        total += len(rows)
        logger.info(f"Bioactivity: {archive} -> {len(rows)} identifiers, "
                    f"{sum(r[2] for r in rows)} records.")
    return total


def _create_indexes(cursor: sqlite3.Cursor) -> None:
    """
    Creates the indexes the query layer needs.

    Without these every search is a full scan: the peptide browser filters activities by organism,
    assay and group across ~228k rows, and joins papers through `peptide_references` for each hit.
    The `is_decoy` index matters for a different reason -- it is on the hot path of *every*
    collection query, because decoys share this table with real peptides and must be excluded.

    :param cursor: Open cursor on the database being built.
    :returns: ``None``.
    """
    statements = (
        "CREATE INDEX IF NOT EXISTS idx_activities_entry     ON activities(entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_activities_organism  ON activities(target_organism)",
        "CREATE INDEX IF NOT EXISTS idx_activities_taxid     ON activities(target_taxid)",
        "CREATE INDEX IF NOT EXISTS idx_activities_type      ON activities(activity_type)",
        "CREATE INDEX IF NOT EXISTS idx_activities_group     ON activities(activity_group)",
        "CREATE INDEX IF NOT EXISTS idx_activities_unit      ON activities(unit)",
        "CREATE INDEX IF NOT EXISTS idx_peptides_is_decoy    ON peptides(is_decoy)",
        "CREATE INDEX IF NOT EXISTS idx_peptides_sequence    ON peptides(sequence)",
        "CREATE INDEX IF NOT EXISTS idx_peptides_source      ON peptides(source)",
        "CREATE INDEX IF NOT EXISTS idx_pepref_peptide       ON peptide_references(peptide_id)",
        "CREATE INDEX IF NOT EXISTS idx_pepref_paper         ON peptide_references(paper_id)",
        "CREATE INDEX IF NOT EXISTS idx_papers_source        ON papers(source_type, source_id)",
        "CREATE INDEX IF NOT EXISTS idx_bib_pmid             ON bibliography(pmid)",
        "CREATE INDEX IF NOT EXISTS idx_bib_doi              ON bibliography(doi)",
        "CREATE INDEX IF NOT EXISTS idx_pepbib_peptide       ON peptide_bibliography(peptide_id)",
        "CREATE INDEX IF NOT EXISTS idx_actref_activity      ON activity_references(activity_id)",
        "CREATE INDEX IF NOT EXISTS idx_pepfam_family        ON peptide_families(family_id)",
        "CREATE INDEX IF NOT EXISTS idx_pepfam_source        ON peptide_families(source)",
        "CREATE INDEX IF NOT EXISTS idx_bioact_archive       ON bioactivity(archive)",
        "CREATE INDEX IF NOT EXISTS idx_pepnames_entry       ON peptide_names(entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_topology_kind        ON peptide_topology(kind)",
        "CREATE INDEX IF NOT EXISTS idx_topology_cyclic      ON peptide_topology(is_cyclic)",
        # Name search is a `lower(name) LIKE` scan, so the lowered expression is what must be
        # indexed; an index on `name` alone would not be used.
        "CREATE INDEX IF NOT EXISTS idx_pepnames_lower       ON peptide_names(lower(name))",
    )
    for statement in statements:
        cursor.execute(statement)
    logger.info(f"Created {len(statements)} indexes.")


def _create_fulltext_index(cursor: sqlite3.Cursor) -> None:
    """
    Builds an FTS5 index over the free-text columns.

    Created here rather than when full-text search is actually wired up, because it is one
    statement inside the build that already holds every row in hand; adding it later would mean
    rebuilding a 111 MB database for a single virtual table.

    Decoys are excluded at build time -- they are generated strings with no provenance, and nobody
    should be able to reach one through a text search.

    Indexes `sequence`, `LITERATURE` and `PATENT`, which is what actually carries text. It is
    deliberately NOT `Reference` or `Biological_Activity`: those columns exist in the schema but
    are always NULL, because `final_dataset.csv` -- the only input to this function -- does not
    carry them. They are present in `merged_from_database.csv` and dropped by step 29's assembly,
    so restoring them is an assembly fix, not a fix here. Indexing them anyway would produce an
    index that looks populated (35,411 rows) and matches nothing.

    `LITERATURE` and `PATENT` hold `source:filename.json` tokens
    (`google_patents:JP2021054842A.json`), so the unicode61 tokenizer splits the patent number out
    as its own term and a search for "JP2021054842A" finds the peptide.

    :param cursor: Open cursor on the database being built.
    :returns: ``None``. Degrades to a logged warning when the SQLite build lacks FTS5, since the
        rest of the database is still usable without it.
    """
    try:
        cursor.execute("DROP TABLE IF EXISTS peptides_fts;")
        cursor.execute("""
            CREATE VIRTUAL TABLE peptides_fts USING fts5(
                entry_id UNINDEXED,
                sequence,
                names,
                literature,
                patent,
                tokenize = "unicode61"
            );
        """)
        # Names and synonyms are collapsed into one indexed column, so a search for "magainin"
        # finds the peptide whether the corpus called it Magainin 2, Mag2 or Mag-2. Without this
        # the database held magainin's sequence and could not be asked for it by name.
        cursor.execute("""
            INSERT INTO peptides_fts (entry_id, sequence, names, literature, patent)
            SELECT p.entry_id, p.sequence,
                   COALESCE((SELECT group_concat(n.name, ' ') FROM peptide_names n
                             WHERE n.entry_id = p.entry_id), ''),
                   COALESCE(p.LITERATURE, ''), COALESCE(p.PATENT, '')
            FROM peptides p WHERE COALESCE(p.is_decoy, 0) = 0;
        """)
        indexed = cursor.execute("SELECT COUNT(*) FROM peptides_fts").fetchone()[0]
        logger.info(f"Created FTS5 full-text index over {indexed} real (non-decoy) peptides.")
    except sqlite3.OperationalError as e:
        logger.warning(f"FTS5 unavailable, skipping full-text index: {e}")
