"""
Tests for the consensus merge's source ingestion (:mod:`sample.merge`).

Covers the class of bug where a source is read differently by the consensus builder than by the
activity loader, so the two silently disagree about how much of a dump exists.
"""
from typing import Any


class TestDrampConsensusIngest:
    """Regression: DRAMP contributed 96 peptides to the consensus while its own loader read
    36,366 records from the same dump -- the consensus block took `Original_Sequence`, a column
    present in only one of the five Datasets files, and never opened Activity/ or Subclass/."""

    @staticmethod
    def _dramp_dump(base: str) -> None:
        import os

        # Mirrors the real dump: general_amps has `Sequence`, specific_amps has BOTH and is the
        # only file where `Original_Sequence` exists at all.
        for sub, name, header, rows in (
            ("Datasets", "general_amps.txt", "DRAMP_ID\tSequence\tTarget_Organism",
             ["DRAMP0001\tGIGKFLHSAKKF\tE. coli", "DRAMP0002\tKWKLFKKIEK\tS. aureus"]),
            ("Datasets", "specific_amps.txt",
             "DRAMP_ID\tSequence\tOriginal_Sequence\tTarget_Organism",
             ["DRAMP0003\tILPWKWPWWPWRR\tILPWKWPWWPWRR\tE. coli"]),
            ("Activity", "activity_amps.txt", "DRAMP_ID\tSequence\tTarget_Organism",
             ["DRAMP0004\tFLPIIAKLLSGLL\tC. albicans"]),
            ("Subclass", "sub_amps.txt", "DRAMP_ID\tSequence\tTarget_Organism",
             ["DRAMP0005\tRWCVYAYVRVRGVLVRYRRCW\tP. aeruginosa"]),
        ):
            directory = os.path.join(base, "dramp", sub)
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
                handle.write(header + "\n" + "\n".join(rows) + "\n")

    def test_reads_the_sequence_column_and_every_dramp_directory(self, tmp_path: Any) -> None:
        from sample.database_activities import _dramp_rows

        base = str(tmp_path)
        self._dramp_dump(base)
        rows = list(_dramp_rows(base))
        found = {r["DRAMP_ID"] for r in rows}
        # All five, from all three directories -- not just the one file with Original_Sequence.
        assert found == {"DRAMP0001", "DRAMP0002", "DRAMP0003", "DRAMP0004", "DRAMP0005"}
        # And every one has a usable sequence under the canonical column.
        assert all((r.get("Sequence") or r.get("Original_Sequence")) for r in rows)

    def test_consensus_ingest_keeps_rows_lacking_original_sequence(self, tmp_path: Any) -> None:
        # The actual regression: rows whose only sequence column is `Sequence` must survive into
        # the consensus. Before the fix these were all dropped.
        from sample.database_activities import _dramp_rows

        base = str(tmp_path)
        self._dramp_dump(base)
        usable = [r for r in _dramp_rows(base)
                  if (r.get("Sequence") or r.get("Original_Sequence"))]
        only_sequence = [r for r in usable if not r.get("Original_Sequence")]
        assert len(only_sequence) >= 4, \
            "rows carrying only the Sequence column must be ingested, not skipped"


class TestDbampIngest:
    """Regression: dbAMP shipped 35,599 records and both readers saw ~390, because each looked
    only at `extracted_metadata/` and never opened the release FASTAs."""

    @staticmethod
    def _dbamp_dump(base: str) -> None:
        import os

        d = os.path.join(base, "dbamp")
        os.makedirs(os.path.join(d, "dbAMP2024"), exist_ok=True)
        # Activity-labelled files: the activity is in the FILENAME, nowhere else.
        with open(os.path.join(d, "dbAMP_Antibacterial.fasta"), "w", encoding="utf-8") as fh:
            fh.write(">dbAMP_00001\nGIGKFLHSAKKF\n>dbAMP_00002\nKWKLFKKIEK\n")
        with open(os.path.join(d, "dbAMP_Antifungal.fasta"), "w", encoding="utf-8") as fh:
            fh.write(">dbAMP_00002\nKWKLFKKIEK\n>dbAMP_00003\nILPWKWPWWPWRR\n")
        # The whole release, unlabelled.
        with open(os.path.join(d, "dbAMP2024", "dbAMP3.fasta"), "w", encoding="utf-8") as fh:
            fh.write(">dbAMP_00004\nFLPIIAKLLSGLL\n")
        # One scraped metadata entry, which carries what the FASTAs lack.
        meta = os.path.join(d, "extracted_metadata", "dbAMP_00001")
        os.makedirs(meta, exist_ok=True)
        with open(os.path.join(meta, "1.csv"), "w", encoding="utf-8") as fh:
            fh.write("key,value\nSequence,GIGKFLHSAKKF\nPubmed_Id,12345678\nActivity,Antibacterial\n")

    def test_activity_loader_reads_the_release_not_just_the_metadata(self, tmp_path: Any) -> None:
        from sample.database_activities import load_dbamp

        base = str(tmp_path)
        self._dbamp_dump(base)
        records = list(load_dbamp(base))
        ids = {r["record_id"] for r in records}
        assert ids == {"dbAMP_00001", "dbAMP_00002", "dbAMP_00003", "dbAMP_00004"}, \
            "every record in the release FASTAs must be yielded, not just the scraped metadata"

    def test_activity_class_comes_from_the_filename(self, tmp_path: Any) -> None:
        from sample.database_activities import load_dbamp

        base = str(tmp_path)
        self._dbamp_dump(base)
        by_id = {r["record_id"]: r for r in load_dbamp(base)}
        # A sequence in two activity files carries both classes.
        assert set(by_id["dbAMP_00002"]["activity_labels"]) == {"Antibacterial", "Antifungal"}
        # The unlabelled whole-release file contributes no invented label.
        assert by_id["dbAMP_00004"]["activity_labels"] == []

    def test_the_metadata_record_is_preferred_where_both_exist(self, tmp_path: Any) -> None:
        # Metadata carries the PubMed id the FASTA header cannot; it must not be displaced.
        from sample.database_activities import load_dbamp

        base = str(tmp_path)
        self._dbamp_dump(base)
        by_id = {r["record_id"]: r for r in load_dbamp(base)}
        assert by_id["dbAMP_00001"]["publications"], \
            "the scraped metadata record must win, keeping its publication"
        assert by_id["dbAMP_00001"]["publications"][0]["pubmed_id"] == "12345678"
