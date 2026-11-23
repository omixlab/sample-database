import os
import re
import glob
import json
import logging
import pandas as pd

logger = logging.getLogger("sample.validation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from sample.sequence import is_valid_peptide, AA_3_TO_1
from sample.config import iter_extraction_files
from sample.rf_filter import filter_sequence

def validate_extractions(literature_dir: str = "data/literature_data_json",
                         patent_dir: str = "data/patent_data_json",
                         merged_source_csv: str = "data/antimicrobial_databases/merged_source.csv",
                         report_path: str = "data/validation_report.json",
                         run_llm: bool = True) -> dict[str, float]:
    """
    Validates LLM-extracted peptide sequences against single-letter standard amino acid formats,
    calculates comparative overlap against established databases, and outputs statistics.

    :param run_llm: When ``False``, skip the LLM field audit and compute the report from the
        ``*_validated`` trees a previous run already produced. The report's novelty figure depends
        on `merged_source_csv`, so a wrong or damaged baseline has to be fixable without paying for
        a full re-validation -- which is tens of thousands of LLM calls, and which would REPLACE
        good validated output with unvalidated copies if the backend happened to be down (an
        unreachable model makes every verdict ``None``, which is a keep-everything no-op).
    """
    logger.info("Starting validation of LLM-extracted peptide sequences...")
    
    json_files = iter_extraction_files(literature_dir) + iter_extraction_files(patent_dir)
                 
    if not json_files:
        logger.warning("No LLM extraction JSON files found to validate.")
        return {}
        
    # Run the deep LLM validation over all extracted activities against their original context.
    try:
        from sample.llm_validator import run_llm_validation
        
        lit_out = literature_dir.rstrip("/") + "_validated"
        pat_out = patent_dir.rstrip("/") + "_validated"
        
        tasks = []
        for f in json_files:
            if literature_dir in f:
                out = f.replace(literature_dir, lit_out, 1)
            elif patent_dir in f:
                out = f.replace(patent_dir, pat_out, 1)
            else:
                out = f
            tasks.append((f, out))
            
        if run_llm:
            run_llm_validation(tasks, workers=10)
        else:
            logger.info("run_llm=False: reusing the existing validated trees, no LLM calls.")
        
        # Point to the validated files for downstream baseline comparison. Done whether or not the
        # audit just ran: the statistics describe the validated corpus either way, and a report
        # computed over the RAW tree would silently describe a different population.
        existing = [t[1] for t in tasks if os.path.exists(t[1])]
        if run_llm or existing:
            missing = len(tasks) - len(existing)
            if not run_llm and missing:
                logger.warning(
                    f"{missing} of {len(tasks)} validated file(s) are absent; the report covers "
                    f"the {len(existing)} that exist.")
            json_files = [t[1] for t in tasks] if run_llm else existing
        
    except ImportError:
        logger.warning("llm_validator not found, skipping deep context validation.")

    from tqdm import tqdm
    from sample.rf_filter import filter_sequences
    
    rows = []
    peptide_count = 0
    activity_count = 0
    
    peptides_to_check = []
    peptide_metadata = []
    
    for json_file in tqdm(json_files, desc="Parsing extractions"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                document = json.load(f)

            # Unwrap the document envelope. Extraction writes {"metadata": ..., "peptides": [...]},
            # and treating that dict as a single peptide record -- which is what wrapping it in a
            # list did -- meant `peptide_data.get('peptide_sequence')` was None for every file, so
            # this stage skipped every record and reported an empty corpus. A bare list is still
            # accepted for older files that predate the envelope.
            if isinstance(document, dict):
                peptides_list = document.get("peptides") or []
            elif isinstance(document, list):
                peptides_list = document
            else:
                peptides_list = []

            for peptide_data in peptides_list:
                if not isinstance(peptide_data, dict):
                    continue
                seq = peptide_data.get('peptide_sequence')
                if not seq or not is_valid_peptide(seq):
                    continue
                peptides_to_check.append(str(seq).upper())
                peptide_metadata.append(peptide_data)
        except Exception as e:
            logger.debug(f"Error parsing {json_file}: {e}")

    if not peptides_to_check:
        logger.warning("No valid peptides were extracted from the JSON files.")
        return {}
        
    logger.info(f"Loaded {len(peptides_to_check)} valid sequences. Running RF filter...")
    filter_results = filter_sequences(peptides_to_check)
    
    for seq, peptide_data, passed in zip(peptides_to_check, peptide_metadata, filter_results):
        if not passed:
            logger.debug(f"RF Filter rejected sequence: {seq}")
            continue
            
        peptide_count += 1
        activities = peptide_data.get('biological_activities', [])
        if not activities:
            activities = []
        for activity in activities:
            if not isinstance(activity, dict):
                continue
            rows.append({
                'sequence': seq,
                **activity
            })
            activity_count += 1

    if not rows:
        logger.warning("No valid peptides were extracted from the JSON files.")
        return {}

    df_data = pd.DataFrame(rows)
    unique_extracted_peptides = set(df_data['sequence'].dropna().unique())
    
    # Read database baseline if it exists
    databases_peptides = set()
    if os.path.exists(merged_source_csv):
        try:
            df_db = pd.read_csv(merged_source_csv)
            if 'sequence' in df_db.columns:
                valid_db_seqs = df_db['sequence'].dropna().apply(lambda x: str(x).upper().strip())
                databases_peptides = set(valid_db_seqs[valid_db_seqs.map(is_valid_peptide)])
        except Exception as e:
            logger.error(f"Error reading merged database baseline: {e}")
    else:
        logger.warning(f"Baseline merged database file not found at {merged_source_csv}")

    total_peptides = len(databases_peptides.union(unique_extracted_peptides))
    
    # New peptides are those extracted by LLM that are NOT present in standard databases
    new_peptides = len(unique_extracted_peptides - databases_peptides)
    overlapping_peptides = len(unique_extracted_peptides.intersection(databases_peptides))

    stats = {
        "total_extracted_peptides": len(unique_extracted_peptides),
        "total_activities_extracted": activity_count,
        "database_baseline_peptides": len(databases_peptides),
        "new_peptides_discovered": new_peptides,
        "overlapping_peptides": overlapping_peptides,
        "total_combined_peptides": total_peptides,
        "new_peptides_percentage": (new_peptides / total_peptides * 100) if total_peptides > 0 else 0
    }

    logger.info(f"Validation completed successfully:")
    logger.info(f" - Baseline peptides from database: {stats['database_baseline_peptides']}")
    logger.info(f" - Unique extracted peptides by LLM: {stats['total_extracted_peptides']}")
    logger.info(f" - Overlapping peptides: {stats['overlapping_peptides']}")
    logger.info(f" - New peptides discovered (not in database): {stats['new_peptides_discovered']}")
    logger.info(f" - Total combined library: {stats['total_combined_peptides']}")

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Validation report saved to {report_path}")
    return stats



def clean_and_repair_peptide(peptide_data: dict) -> dict:
    """
    Cleans peptide_sequence but preserves three-letter code format per new directives.
    Only attempts SMILES generation if sequence is strictly single-letter.
    """
    from rdkit import Chem
    
    seq = peptide_data.get("peptide_sequence")
    if seq:
        peptide_data["peptide_sequence"] = str(seq).strip()
        
    # Generate or correct SMILES from single-letter sequence using RDKit (only if single-letter)
    current_seq = peptide_data.get("peptide_sequence")
    if current_seq:
        current_seq_clean = str(current_seq).upper().strip()
        
        # Check against size filter: must not exceed 50 amino acids
        from sample.sequence import sequence_to_1_letter
        seq_1l = sequence_to_1_letter(current_seq_clean)
        if len(seq_1l) > 50:
            logger.warning(f"Size Filter rejected sequence (>{len(seq_1l)} AAs): '{current_seq}'")
            peptide_data["peptide_sequence"] = None
            return peptide_data
        
        # Check against RF filter if it's purely alphabetical
        if re.match(r'^[A-Z]+$', current_seq_clean):
            if not filter_sequence(current_seq_clean):
                logger.warning(f"RF Filter rejected sequence as English word: '{current_seq}'")
                peptide_data["peptide_sequence"] = None
                return peptide_data

        # Only attempt if it looks strictly like a standard single-letter sequence
        if re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', current_seq_clean):
            try:
                mol = Chem.MolFromSequence(current_seq_clean)
                if mol:
                    correct_smiles = Chem.MolToSmiles(mol)
                    old_smiles = peptide_data.get("peptide_sequence_SMILES_format")
                    if old_smiles != correct_smiles:
                        peptide_data["peptide_sequence_SMILES_format"] = correct_smiles
                        logger.info(f"Corrected SMILES for '{current_seq_clean}' (was: '{old_smiles}', now: '{correct_smiles}')")
            except Exception as e:
                logger.warning(f"Could not generate SMILES for peptide sequence '{current_seq_clean}': {e}")
                
    return peptide_data

def extract_repair_extractions(input_dir: str = "data/literature_data_json_revised",
                              schema_path: str = "data/prompts/extraction_record_schema.json",
                              output_dir: str = None,
                              log_dir: str = "data/repaired_logs") -> bool:
    """
    Validates and repairs LLM extraction JSON records against the schema, nullifying fields
    causing validation errors to repair the JSON, saving output and error logs.
    """
    import json
    import glob
    from jsonschema import Draft7Validator
    
    if output_dir is None:
        output_dir = input_dir.rstrip("/") + "_repaired"
        
    logger.info(f"Extract-repairing JSONs in {input_dir} against schema {schema_path}...")
    
    if not os.path.exists(schema_path):
        logger.error(f"Schema file not found at {schema_path}")
        return False
        
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)
        
    validator = Draft7Validator(schema_data)
    
    json_files = glob.glob(os.path.join(input_dir, "*", "*.json")) + glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    repaired_count = 0
    total_errors = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Perform sequence and SMILES post-processing repairs before validation
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        clean_and_repair_peptide(item)
            elif isinstance(data, dict):
                clean_and_repair_peptide(data)
                
            errors = list(validator.iter_errors(data))
            if not errors:
                # Save cleaned data
                rel_path = os.path.relpath(file_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                continue
                
            error_log = []
            for error in errors:
                path_list = list(error.path)
                error_log.append({
                    "field_path": " -> ".join(map(str, path_list)) if path_list else "root",
                    "error_message": error.message,
                    "validator": error.validator,
                    "invalid_value": error.instance
                })

            # Repair by DROPPING the records that fail, not by nulling the path the error points at.
            #
            # The previous version walked to error.path and assigned None. For a missing required
            # property the path is the object itself (["peptides", 0]), so it replaced an entire
            # peptide record with null -- and the next error for that same record then traversed
            # into the null and raised TypeError, which the outer handler logged at DEBUG and
            # swallowed, leaving the file never written at all. Combined with a schema that no real
            # record could satisfy, that silently emptied this stage's output, and step 19 merges
            # from it.
            #
            # Dropping the offending record keeps the rest of the document, cannot corrupt a record
            # into a shape nothing downstream expects, and is honest about what happened.
            bad_indices = set()
            for error in errors:
                path_list = list(error.path)
                if len(path_list) >= 2 and path_list[0] == "peptides":
                    try:
                        bad_indices.add(int(path_list[1]))
                    except (TypeError, ValueError):
                        continue
            if bad_indices and isinstance(data, dict) and isinstance(data.get("peptides"), list):
                kept = [p for i, p in enumerate(data["peptides"]) if i not in bad_indices]
                logger.debug(
                    f"{os.path.basename(file_path)}: dropped {len(bad_indices)} record(s) "
                    f"failing the published schema, kept {len(kept)}"
                )
                data["peptides"] = kept

            # Save repaired JSON
            rel_path = os.path.relpath(file_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            # Save log
            log_path = os.path.join(log_dir, os.path.basename(file_path))
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(error_log, f, indent=4)
                
            repaired_count += 1
            total_errors += len(errors)
        except Exception as e:
            logger.debug(f"Error validating/repairing file {file_path}: {e}")
            
    logger.info(f"Completed extract-repair. Processed {len(json_files)} files. Repaired {repaired_count} files containing a total of {total_errors} errors. Outputs saved to {output_dir}.")
    return True

# Keep alias for backward compatibility or direct imports if any
validate_and_repair_extractions = extract_repair_extractions
