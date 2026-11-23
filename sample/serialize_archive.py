"""
Serialize sAMPle Dataset Records, PDB Structures, and SDF Conformers into a unified ZIP archive.
Produces data/sample_final_release/sample_records_json_pdb_sdf.zip and updates zenodo_dump.
"""
import os
import sys
import time
import json
import sqlite3
import zipfile
from collections import defaultdict

def create_records_archive(output_path: str = "data/sample_final_release/sample_records_json_pdb_sdf.zip"):
    print(f"Creating serialized dataset records archive at: {output_path}...")
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    db_path = "data/sample.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("Pre-fetching descriptors, activities, and references from SQLite...")
    # Load descriptors
    descriptors_by_id = {}
    for row in cursor.execute("SELECT entry_id, molecular_weight, logp, hydrogen_bond_donors, hydrogen_bond_acceptors, charge, sequence_length, is_anomaly, anomaly_score FROM descriptors"):
        descriptors_by_id[row["entry_id"]] = {
            "molecular_weight": row["molecular_weight"],
            "logp": row["logp"],
            "hydrogen_bond_donors": row["hydrogen_bond_donors"],
            "hydrogen_bond_acceptors": row["hydrogen_bond_acceptors"],
            "charge": row["charge"],
            "sequence_length": row["sequence_length"],
            "is_anomaly": row["is_anomaly"],
            "anomaly_score": row["anomaly_score"],
        }

    # Load activities
    activities_by_id = defaultdict(list)
    for row in cursor.execute("SELECT entry_id, activity_group, value, unit, value_mM, value_ug_ml, potency, target_organism, activity_type, mechanism_of_action FROM activities"):
        activities_by_id[row["entry_id"]].append({
            "activity_group": row["activity_group"],
            "value": row["value"],
            "unit": row["unit"],
            "value_mM": row["value_mM"],
            "value_ug_ml": row["value_ug_ml"],
            "potency": row["potency"],
            "target_organism": row["target_organism"],
            "activity_type": row["activity_type"],
            "mechanism_of_action": row["mechanism_of_action"],
        })

    # Load papers/references
    refs_by_id = defaultdict(list)
    for row in cursor.execute("SELECT pr.peptide_id, p.source_type, p.source_id FROM peptide_references pr JOIN papers p ON pr.paper_id = p.id"):
        refs_by_id[row["peptide_id"]].append({
            "source_type": row["source_type"],
            "source_id": row["source_id"],
        })

    # Fetch all peptides (active curated peptides)
    cursor.execute("SELECT entry_id, sequence, SMILES, source, is_decoy, LITERATURE, PATENT, Reference, Biological_Activity, source_organism, cross_references FROM peptides WHERE COALESCE(is_decoy, 0) = 0")
    peptides = cursor.fetchall()
    total_peptides = len(peptides)
    print(f"Loaded {total_peptides} curated peptides for archiving.")

    struct_dir = "data/sample_enriched_release/structures"
    n_json = 0
    n_pdb = 0
    n_sdf = 0

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write README
        readme_text = (
            "sAMPle Dataset: Serialized Records, PDB Structures, and SDF Conformers\n"
            "======================================================================\n\n"
            "This archive contains complete serialized data for all curated bioactive peptides in sAMPle:\n"
            "  - json/<entry_id>.json : Complete metadata record including biophysical descriptors, quantitative bioactivities, and literature provenance\n"
            "  - pdb/<entry_id>.pdb   : 3D atomic structures predicted via AlphaFold2 / ColabFold with per-residue pLDDT confidence scores in the B-factor column\n"
            "  - sdf/<entry_id>.sdf   : 3D energy-minimized small-molecule conformers with complete bond orders and formal charges\n\n"
            "Repository: https://github.com/firas-bioinfo/sAMPle\n"
            "Web Portal: https://sample-database.org\n"
            "Permanent DOI: https://doi.org/10.5281/zenodo.14885834\n"
        )
        zf.writestr("README.txt", readme_text)

        for i, p in enumerate(peptides):
            eid = p["entry_id"]
            record = {
                "entry_id": eid,
                "sequence": p["sequence"],
                "smiles": p["SMILES"],
                "source": p["source"],
                "source_organism": p["source_organism"],
                "biological_activity": p["Biological_Activity"],
                "literature": p["LITERATURE"],
                "patent": p["PATENT"],
                "cross_references": p["cross_references"],
                "descriptors": descriptors_by_id.get(eid, {}),
                "activities": activities_by_id.get(eid, []),
                "references": refs_by_id.get(eid, []),
            }
            zf.writestr(f"json/{eid}.json", json.dumps(record, indent=2))
            n_json += 1

            pdb_file = os.path.join(struct_dir, f"{eid}.pdb")
            if os.path.exists(pdb_file):
                zf.write(pdb_file, arcname=f"pdb/{eid}.pdb")
                n_pdb += 1

            sdf_file = os.path.join(struct_dir, f"{eid}.sdf")
            if os.path.exists(sdf_file):
                zf.write(sdf_file, arcname=f"sdf/{eid}.sdf")
                n_sdf += 1

            if (i + 1) % 10000 == 0 or (i + 1) == total_peptides:
                print(f"  Processed {i+1}/{total_peptides} peptides (JSON: {n_json}, PDB: {n_pdb}, SDF: {n_sdf})...")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"Archive completed in {elapsed:.1f}s!")
    print(f"Output: {output_path} ({size_mb:.2f} MB)")
    print(f"Contents: {n_json} JSON records, {n_pdb} PDB structures, {n_sdf} SDF conformers.")

    # Also sync to zenodo_dump if zenodo_dump exists
    zenodo_path = "data/zenodo_dump/sample_records_json_pdb_sdf.zip"
    if os.path.isdir("data/zenodo_dump"):
        import shutil
        print(f"Copying archive to {zenodo_path}...")
        shutil.copy2(output_path, zenodo_path)
        print("Zenodo dump updated.")

if __name__ == "__main__":
    create_records_archive()
