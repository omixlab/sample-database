"""
Tests for the DRAMP 5.0 update: release identification, the shared field accessors, and the
curated-SMILES trust tier.

The thing being guarded here is a *provenance* property, not a parsing one. DRAMP has no version
in its download path and updates the files in place, so the URL cannot say what you have:
verified 2026-10-16, ``DRAMP3.0_new/general_amps.txt`` answers 200 with a 28-column DRAMP 5.0
header while ``DRAMP5.0/`` and ``DRAMP4.0_new/`` both 404. On this host that had already gone
wrong silently -- the URLs name 3.0, the files fetched through them in March 2026 are 4.0
(general 11,612 / patent 17,886, matching 4.0's published table exactly), and the same URLs now
serve 5.0.
"""
import json
import os

import pytest

from sample import databases as DB
from sample import database_activities as DA


def _dump(base, general_rows=0, patent_rows=0, smiles=False, extra=None):
    """Writes a minimal DRAMP tree with the given row counts and schema."""
    datasets = os.path.join(base, "Datasets")
    os.makedirs(datasets, exist_ok=True)
    for name, count in (("general_amps.txt", general_rows), ("patent_amps.txt", patent_rows)):
        header = ["DRAMP_ID", "Sequence", "Target_Organism"]
        if smiles:
            header.append("SMILES")
        lines = ["\t".join(header)]
        for i in range(count):
            row = [f"DRAMP{i:05d}", "GIGKFLHSAKKF", "E. coli"]
            if smiles:
                row.append("CC(N)C(=O)O")
            lines.append("\t".join(row))
        with open(os.path.join(datasets, name), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    for name, header, rows in (extra or []):
        with open(os.path.join(datasets, name), "w", encoding="utf-8") as handle:
            handle.write(header + "\n" + "\n".join(rows) + "\n")
    return base


class TestReleaseDetection:
    def test_published_counts_plus_the_smiles_column_identify_5_0(self, tmp_path):
        base = _dump(str(tmp_path / "dramp"), general_rows=12784, patent_rows=18715, smiles=True)
        result = DB.detect_dramp_release(base)
        assert result["release"] == "5.0"
        assert result["confidence"] == "counts+schema"

    def test_the_4_0_dump_this_host_actually_had_is_identified_as_4_0(self, tmp_path):
        """
        The real case. These counts are what `general_amps.txt` and `patent_amps.txt` hold on
        this machine, downloaded through URLs that say "DRAMP3.0_new".
        """
        base = _dump(str(tmp_path / "dramp"), general_rows=11612, patent_rows=17886, smiles=False)
        result = DB.detect_dramp_release(base)
        assert result["release"] == "4.0"

    def test_a_release_publishing_only_a_total_does_not_match_everything(self, tmp_path):
        """
        Regression. DRAMP 2.0 and 3.0 publish a total and no per-category counts, so an `all()`
        over an empty comparison set returned True and they matched every dump vacuously -- which
        made a dump whose counts identify 4.0 exactly come back "unidentified" on a three-way tie.
        """
        base = _dump(str(tmp_path / "dramp"), general_rows=11612, patent_rows=17886)
        result = DB.detect_dramp_release(base)
        assert result["release"] == "4.0", "a vacuous match has crept back in"

    def test_counts_and_schema_disagreeing_is_reported_not_resolved(self, tmp_path):
        """
        A dump whose counts say 4.0 but which carries the column 5.0 introduced is neither
        release. Saying so beats picking one -- the point of this function is provenance, and a
        confident wrong answer is worse here than an admitted unknown.
        """
        base = _dump(str(tmp_path / "dramp"), general_rows=11612, patent_rows=17886, smiles=True)
        result = DB.detect_dramp_release(base)
        assert result["release"] is None
        assert result["confidence"] == "conflicting"

    def test_an_unmatched_dump_with_smiles_is_bounded_not_named(self, tmp_path):
        """A dump taken between releases: the schema still puts a floor under it."""
        base = _dump(str(tmp_path / "dramp"), general_rows=999, patent_rows=999, smiles=True)
        result = DB.detect_dramp_release(base)
        assert result["release"] == "5.0+"
        assert result["confidence"] == "schema-only"

    def test_nothing_on_disk_is_never_guessed(self, tmp_path):
        result = DB.detect_dramp_release(str(tmp_path / "absent"))
        assert result["release"] is None
        assert result["confidence"] == "unidentified"


class TestManifest:
    def test_it_records_the_release_and_warns_about_the_url(self, tmp_path):
        """
        The note matters as much as the data. A reader who finds "DRAMP3.0_new" in the manifest
        and concludes the database was built from DRAMP 3.0 would be wrong, and this is the only
        place that says so.
        """
        base = _dump(str(tmp_path / "dramp"), general_rows=12784, patent_rows=18715, smiles=True)
        path = DB.write_dramp_manifest(base)
        manifest = json.load(open(path, encoding="utf-8"))
        assert manifest["release"]["release"] == "5.0"
        assert "does NOT indicate the release" in manifest["download_base_note"]

    def test_every_expected_file_is_listed_present_or_not(self, tmp_path):
        """
        Absent files are recorded as absent rather than omitted: "this file was never downloaded"
        is exactly the fact that hid Activity/ and Subclass/ going unfetched for months.
        """
        base = _dump(str(tmp_path / "dramp"), general_rows=1, patent_rows=1)
        manifest = json.load(open(DB.write_dramp_manifest(base), encoding="utf-8"))
        expected = {f"{sub}/{name}" for sub, names in DB.DRAMP_FILES.items() for name in names}
        assert set(manifest["files"]) == expected
        assert manifest["files"]["Activity/Antibacterial_amps.txt"]["present"] is False
        assert manifest["files"]["Datasets/general_amps.txt"]["present"] is True

    def test_a_digest_is_recorded_for_each_present_file(self, tmp_path):
        base = _dump(str(tmp_path / "dramp"), general_rows=2, patent_rows=2)
        manifest = json.load(open(DB.write_dramp_manifest(base), encoding="utf-8"))
        entry = manifest["files"]["Datasets/general_amps.txt"]
        assert len(entry["sha256"]) == 64
        assert entry["rows"] == 2

    def test_a_missing_manifest_reads_as_none(self, tmp_path):
        assert DB.read_dramp_manifest(str(tmp_path)) is None


class TestTheDownloadInventory:
    def test_all_three_read_directories_are_fetched(self):
        """
        `database_activities._dramp_rows` globs Datasets/, Activity/ and Subclass/, and
        `sample.merge` shares that iterator -- but only Datasets/ was ever downloaded. A fresh
        clone built its database from a third of the dump, silently.
        """
        assert set(DB.DRAMP_FILES) == {"Datasets", "Activity", "Subclass"}
        assert all(DB.DRAMP_FILES[tree] for tree in DB.DRAMP_FILES)

    def test_the_smiles_downloads_are_deliberately_absent(self):
        """
        DRAMP 5.0 offers separate `*_smiles.*` files, but the same strings are already the last
        column of the Datasets exports. Fetching them would duplicate ~100 MB and -- the real
        hazard -- drop extra .txt files into Datasets/, where `_dramp_rows` would read them as
        rows and, de-duplicating by DRAMP_ID on first-seen, could shadow a peptide's full record
        with a two-column SMILES stub.
        """
        every = [name for names in DB.DRAMP_FILES.values() for name in names]
        assert not [name for name in every if "smiles" in name.lower()]


class TestSharedFieldAccessors:
    def test_the_stapling_annotated_column_is_never_returned(self):
        """
        In specific_amps.txt, `Sequence` marks stapling positions with Unicode circled letters
        and `Original_Sequence` holds the clean form; measured over the March dump they differ in
        366 of 366 rows that populate both.
        """
        row = {"DRAMP_ID": "DRAMP21468", "Sequence": "KFFⓀKLKⓀAVKKGFKKFAKV",
               "Original_Sequence": "KFFKKLKKAVKKGFKKFAKV"}
        assert DA.dramp_sequence(row) == "KFFKKLKKAVKKGFKKFAKV"

    def test_files_carrying_only_sequence_still_work(self):
        """The fallback: four of the five exports have no Original_Sequence column at all."""
        assert DA.dramp_sequence({"Sequence": "GIGKFLHSAKKF"}) == "GIGKFLHSAKKF"

    def test_both_spellings_of_hemolytic_activity_are_found(self):
        """
        general_amps.txt says `Hemolytic_activity`, specific_amps.txt says `Hemolytic_Activity`.
        A reader keyed on one returns nothing for the other's 6,286 rows -- which in the output
        is indistinguishable from "this entry has no hemolytic data".
        """
        assert DA.dramp_field({"Hemolytic_activity": "HC50 100"}, "hemolytic") == "HC50 100"
        assert DA.dramp_field({"Hemolytic_Activity": "HC50 100"}, "hemolytic") == "HC50 100"

    def test_both_spellings_of_the_uniprot_column_are_found(self):
        assert DA.dramp_field({"Swiss_Prot_Entry": "P12345"}, "uniprot") == "P12345"
        assert DA.dramp_field({"Uniprot_Entry": "P12345"}, "uniprot") == "P12345"

    def test_an_unknown_key_raises_rather_than_returning_empty(self):
        """An empty string would look exactly like a legitimately empty field."""
        with pytest.raises(KeyError):
            DA.dramp_field({}, "not_a_field")

    @pytest.mark.parametrize("row,expected", [
        ({"Linear/Cyclic/Branched": "Linear", "N-terminal_Modification": "None",
          "C-terminal_Modification": "Free", "Other_Modifications": "None"}, False),
        ({"Linear/Cyclic/Branched": "Cyclic"}, True),
        ({"Linear/Cyclic": "Linear", "C-terminal_Modification": "Amidation"}, True),
        ({"Linear/Cyclic": "Linear",
          "Special_Amino_Acid_and_Stapling_Position": "Lys(N3) at 4,9"}, True),
        ({}, False),
    ])
    def test_modification_is_read_from_whichever_columns_exist(self, row, expected):
        assert DA.dramp_is_modified(row) is expected


class TestCuratedSmiles:
    def test_the_smiles_column_is_loaded_with_its_modification_status(self, tmp_path):
        base = str(tmp_path)
        _dump(os.path.join(base, "dramp"), extra=[
            ("general_amps.txt",
             "DRAMP_ID\tSequence\tSMILES\tLinear/Cyclic/Branched",
             ["DRAMP0001\tGIGKFL\tCC(N)C(=O)O\tLinear",
              "DRAMP0002\tKWKLFK\tC1CCCCC1\tCyclic"]),
        ])
        curated = DA.load_dramp_smiles(base)
        assert curated["DRAMP0001"] == {"smiles": "CC(N)C(=O)O", "modified": False,
                                        "sequence": "GIGKFL"}
        assert curated["DRAMP0002"]["modified"] is True

    def test_a_pre_5_0_dump_yields_nothing_rather_than_failing(self, tmp_path):
        """
        Which is a normal state, not an error: every dump taken before October 2026 has no
        SMILES column, and the caller must read an empty mapping as "no curated chemistry".
        """
        base = str(tmp_path)
        _dump(os.path.join(base, "dramp"), general_rows=5, patent_rows=5, smiles=False)
        assert DA.load_dramp_smiles(base) == {}


class TestTheCuratedTrustTier:
    """
    Where DRAMP's curated SMILES sits relative to a sequence-derived structure, and why that is
    not simply "the computed one wins".
    """

    @staticmethod
    def _frame(dramp_id="DRAMP0001", stored="C"):
        import pandas as pd
        return pd.DataFrame({"entry_id": ["SAMPLE0000000001"], "DRAMP": [dramp_id],
                             "SMILES": [stored]})

    def test_a_derived_structure_wins_for_an_unmodified_peptide(self, tmp_path):
        from sample import smiles_validation as SV

        structures = tmp_path / "s.json"
        structures.write_text(json.dumps({"SAMPLE0000000001": {"smiles": "DERIVED"}}),
                              encoding="utf-8")
        df = self._frame()
        summary = SV.validate_smiles_column(
            df, predicted_structures_json=str(structures),
            curated={"DRAMP0001": {"smiles": "CURATED", "modified": False}})
        assert df["SMILES"].iloc[0] == "DERIVED"
        assert summary["replaced_from_structure"] == 1

    def test_the_curated_structure_wins_for_a_modified_peptide(self, tmp_path):
        """
        The load-bearing case. A derived structure is built from the sequence, so it describes a
        linear unmodified molecule: for a cyclic, stapled or terminally-modified peptide it is
        wrong in the worst way available -- it parses, round-trips, and looks exactly as
        trustworthy as a right answer.
        """
        from sample import smiles_validation as SV

        structures = tmp_path / "s.json"
        structures.write_text(json.dumps({"SAMPLE0000000001": {"smiles": "DERIVED"}}),
                              encoding="utf-8")
        df = self._frame()
        summary = SV.validate_smiles_column(
            df, predicted_structures_json=str(structures),
            curated={"DRAMP0001": {"smiles": "CURATED", "modified": True}})
        assert df["SMILES"].iloc[0] == "CURATED"
        assert summary["replaced_from_curated"] == 1
        # Counted, so the log states how often this trust decision fired rather than hiding it.
        assert summary["curated_preferred_over_structure"] == 1

    def test_curated_beats_the_stored_value_when_there_is_no_structure(self, tmp_path):
        from sample import smiles_validation as SV

        df = self._frame(stored="Not available")
        SV.validate_smiles_column(df, predicted_structures_json=str(tmp_path / "absent.json"),
                                  curated={"DRAMP0001": {"smiles": "CURATED",
                                                         "modified": False}})
        assert df["SMILES"].iloc[0] == "CURATED"

    def test_a_dataset_without_a_dramp_column_is_not_an_error(self, tmp_path):
        """
        The join key is absent from any dataset built before DRAMP ids were carried through.
        That costs the curated tier and nothing else.
        """
        import pandas as pd
        from sample import smiles_validation as SV

        df = pd.DataFrame({"entry_id": ["SAMPLE0000000001"], "SMILES": ["CC(N)C(=O)O"]})
        summary = SV.validate_smiles_column(
            df, predicted_structures_json=str(tmp_path / "absent.json"),
            curated={"DRAMP0001": {"smiles": "CURATED", "modified": True}})
        assert summary["replaced_from_curated"] == 0
        assert summary["kept"] == 1

    def test_the_tier_can_be_switched_off(self, tmp_path):
        from sample import smiles_validation as SV

        df = self._frame(stored="CC(N)C(=O)O")
        summary = SV.validate_smiles_column(
            df, predicted_structures_json=str(tmp_path / "absent.json"), curated={})
        assert summary["replaced_from_curated"] == 0
        assert df["SMILES"].iloc[0] == "CC(N)C(=O)O"

class TestNonPlainSequencesAreRejectedByBothReaders:
    """
    507 rows of the 2026-03 dump carry stapling annotation in ``Sequence``; 366 have an
    ``Original_Sequence`` to fall back on and 141 do not. Those 141 were rejected from the
    consensus by ``sample.merge`` and accepted by the activity loader, so the two outputs
    disagreed about the same peptides.
    """

    def test_a_stapling_annotated_sequence_is_not_plain(self):
        assert DA.dramp_sequence_is_plain("KFFⓀKLKⓀAVKKGFKKFAKV") is False

    def test_a_normal_sequence_is_plain(self):
        assert DA.dramp_sequence_is_plain("KFFKKLKKAVKKGFKKFAKV") is True

    def test_ambiguity_codes_are_accepted(self):
        """Real dumps use B/X/Z/U/O; rejecting them would drop legitimate records."""
        assert DA.dramp_sequence_is_plain("GIGKFLXZBUO") is True

    def test_empty_is_not_plain(self):
        assert DA.dramp_sequence_is_plain("") is False

    def test_merge_and_the_activity_loader_share_one_predicate(self):
        """
        Not two regexes that happen to agree today. The drift between these two readers has now
        caused three separate defects in this dump.
        """
        import sample.merge as M

        assert M._DRAMP_RESIDUES is DA.DRAMP_PLAIN_RESIDUES

    def test_un_circling_is_not_offered_as_a_repair(self):
        """
        Documents a tested negative result. Stripping the circles looks like the obvious fix, but
        measured against the 366 rows carrying both columns it reproduces ``Original_Sequence`` in
        only 34: that column is canonicalised, not merely unmarked -- circled X marks a
        non-standard residue DRAMP substitutes, so ``KKKKKKAAFⓍAWAⓍFAA`` is published as
        ``KKKKKKAAFAAWAAFAA``, not ``KKKKKKAAFXAWAXFAA``. Un-circling would have shipped 332
        sequences that disagree with DRAMP.
        """
        annotated = "KKKKKKAAFⓍAWAⓍFAA"
        assert DA.dramp_sequence({"Sequence": annotated}) == annotated.upper()
        assert DA.dramp_sequence_is_plain(DA.dramp_sequence({"Sequence": annotated})) is False

    def test_the_activity_loader_skips_them(self, tmp_path):
        base = str(tmp_path)
        _dump(os.path.join(base, "dramp"), extra=[
            ("stapled_amps.txt", "DRAMP_ID\tSequence\tTarget_Organism",
             ["DRAMP9001\tKFFⓀKLKⓀAVKKGFKKFAKV\tE. coli",
              "DRAMP9002\tKFFKKLKKAVKKGFKKFAKV\tE. coli"]),
        ])
        loaded = {record["record_id"] for record in DA.load_dramp(base)}
        assert "DRAMP9002" in loaded
        assert "DRAMP9001" not in loaded, \
            "a sequence that cannot enter the consensus must not enter the activity table"

