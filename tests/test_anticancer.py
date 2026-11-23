"""
Tests for :mod:`sample.anticancer` -- importing anticancer peptide datasets.

The behaviours worth defending: the plain-text parser keeps only real sequences (dropping headers,
blanks and junk rather than passing them through), positives and negatives get the right labels, and
de-duplication is *positive-preferring* so a sequence that is anticancer in one file is never
demoted to a hard negative because it also appears in a negative file. Fixtures stand in for the
downloaded files so the tests never touch the network.
"""
import os
from typing import Any

import pytest

from sample import anticancer


class TestParseSequencePerLine:
    def test_keeps_valid_sequences_only(self, tmp_path: Any) -> None:
        path = tmp_path / "seqs.txt"
        path.write_text("FAKKFAKK\n\n>header\nICLRLPGC\nnot a sequence 123\nKLAKLAK\n")
        assert anticancer.parse_sequence_per_line(str(path)) == ["FAKKFAKK", "ICLRLPGC", "KLAKLAK"]

    def test_uppercases_and_dedupes_preserving_order(self, tmp_path: Any) -> None:
        path = tmp_path / "seqs.txt"
        path.write_text("fakk\nFAKK\nklak\n")
        assert anticancer.parse_sequence_per_line(str(path)) == ["FAKK", "KLAK"]

    def test_min_length_filter(self, tmp_path: Any) -> None:
        path = tmp_path / "seqs.txt"
        path.write_text("A\nAC\nACD\n")
        assert anticancer.parse_sequence_per_line(str(path), min_length=3) == ["ACD"]


def _write_anticp2_fixture(root: str) -> str:
    """Writes a minimal AntiCP2-shaped directory (a couple of sequences per file)."""
    data_dir = os.path.join(root, "anticp2")
    os.makedirs(data_dir, exist_ok=True)
    contents = {
        "pos_train_main": "FAKKFAKK\nICLRLPGC\n",
        "neg_train_main": "AALRGALR\nSHAREDSEQ\n",   # SHAREDSEQ also appears as a positive below
        "pos_test_main": "FLWWLFKW\nSHAREDSEQ\n",
        "neg_test_main": "ACHAHCQS\n",
        "pos_train_alternate": "GLFDIVKK\n",
        "neg_train_alternate": "AALRGCWT\n",
        "pos_test_alternate": "WKKIPKFL\n",
        "neg_test_alternate": "ACQCPDAI\n",
    }
    for name, text in contents.items():
        with open(os.path.join(data_dir, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    return data_dir


class TestLoadAnticp2:
    def test_labels_positives_and_negatives(self, tmp_path: Any) -> None:
        _write_anticp2_fixture(str(tmp_path))
        frame = anticancer.load_anticp2(os.path.join(str(tmp_path), "anticp2"))
        pos = frame[frame["label"] == 1]
        neg = frame[frame["label"] == 0]
        assert set(pos["activity"]) == {"anticancer"}
        assert set(neg["activity"]) == {"non-anticancer-amp"}
        assert set(frame["source_db"]) == {"AntiCP2"}
        assert set(frame["dataset"]) == {"main", "alternate"}

    def test_missing_files_raise_when_nothing_parses(self, tmp_path: Any) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            anticancer.load_anticp2(str(empty))


class TestBuildAnticancerTable:
    def test_positive_preferring_dedup(self, tmp_path: Any) -> None:
        _write_anticp2_fixture(str(tmp_path))
        out = tmp_path / "anticancer.csv"
        summary = anticancer.build_anticancer_table(
            data_dir=str(tmp_path), output_csv=str(out))

        import pandas as pd
        table = pd.read_csv(str(out))
        # SHAREDSEQ appears as both a positive (pos_test_main) and a negative (neg_train_main);
        # it must survive exactly once, as the positive.
        shared = table[table["sequence"] == "SHAREDSEQ"]
        assert len(shared) == 1
        assert shared.iloc[0]["label"] == 1
        assert summary["positives"] >= 1
        assert summary["by_source"] == {"AntiCP2": summary["rows"]}

    def test_summary_counts_match_file(self, tmp_path: Any) -> None:
        _write_anticp2_fixture(str(tmp_path))
        out = tmp_path / "anticancer.csv"
        summary = anticancer.build_anticancer_table(
            data_dir=str(tmp_path), output_csv=str(out))
        import pandas as pd
        assert summary["rows"] == len(pd.read_csv(str(out)))


class TestSourceRegistry:
    def test_verified_sources_are_importable(self) -> None:
        # Three now have a verified URL and parser; ApInAPDB's host does not respond at all, so it
        # stays explicitly not importable rather than silently omitted.
        assert anticancer.SOURCES["anticp2"]["importable"] is True
        assert anticancer.SOURCES["dctpep"]["importable"] is True
        assert anticancer.SOURCES["cancerppd"]["importable"] is True
        assert not anticancer.SOURCES["apinapdb"]["importable"]

    def test_every_source_records_why_or_how(self) -> None:
        for name, meta in anticancer.SOURCES.items():
            assert meta["note"], f"{name} has no note"


class TestCancerppdParsing:
    def test_bulk_file_skips_header_and_trailing_blanks(self, tmp_path: Any) -> None:
        # The real files carry ~1M trailing blank lines; rows are validated, never counted.
        path = tmp_path / "l_natural.txt"
        path.write_text("id\tSequence\n1001\tFLPLLAGL\n1002\tKWKLFKKI\n" + "\n" * 5000)
        records = anticancer.parse_cancerppd_file(str(path))
        assert [r["entry_id"] for r in records] == ["1001", "1002"]

    def test_bulk_file_drops_non_residue_sequences(self, tmp_path: Any) -> None:
        path = tmp_path / "l_natural.txt"
        path.write_text("id\tSequence\n1001\tFLPLLAGL\n1002\tW-R-W(Dox)\n")
        records = anticancer.parse_cancerppd_file(str(path))
        assert [r["entry_id"] for r in records] == ["1001"]

    def test_detail_parser_reads_labelled_cells(self) -> None:
        # Mirrors the real page's shape: label cell then value cell.
        html = ("<td>PMID</td><td>14499271</td>"
                "<td>SEQUENCE</td><td>FLPLLAGLAANFLPTIICKISYKC</td>"
                "<td>NAME</td><td>GGN6</td>"
                "<td>CELL LINE</td><td>A-549</td>"
                "<td>Tissue</td><td>Lung</td>")
        fields = anticancer.parse_cancerppd_detail(html)
        assert fields["pmid"] == "14499271"
        assert fields["name"] == "GGN6"
        assert fields["cell_line"] == "A-549"
        assert fields["tissue"] == "Lung"

    def test_detail_parser_rejoins_the_split_activity_cells(self) -> None:
        # "IC50=6.49 µg/ml" renders as three cells because the subscript is its own tag.
        html = "<td>ACTIVITY</td><td>IC</td><sub>50</sub><td>=6.49 µg/ml</td>"
        fields = anticancer.parse_cancerppd_detail(html)
        assert "6.49" in fields["activity_raw"] and "µg/ml" in fields["activity_raw"]

    def test_detail_parser_ignores_placeholder_values(self) -> None:
        html = "<td>Patent ID</td><td>None</td><td>NAME</td><td>NA</td>"
        assert "name" not in anticancer.parse_cancerppd_detail(html)

    def test_unknown_labels_are_ignored_not_fatal(self) -> None:
        html = "<td>SOME NEW FIELD</td><td>x</td><td>NAME</td><td>GGN6</td>"
        assert anticancer.parse_cancerppd_detail(html)["name"] == "GGN6"


class TestDctpepParsing:
    def test_null_spellings_become_empty(self) -> None:
        for spelling in ("Not available", "NA", "-", "none", "", "nan"):
            assert anticancer._dctpep_value(spelling) == ""

    def test_real_values_survive(self) -> None:
        assert anticancer._dctpep_value("  A-549 ") == "A-549"

    def test_nan_float_is_not_the_string_nan(self) -> None:
        # str(float('nan')) is the truthy "nan" -- the same trap fixed in sample.cycpeptmpdb.
        assert anticancer._dctpep_value(float("nan")) == ""


class TestPipelineIntegration:
    """The pieces that make an imported anticancer peptide actually reach train_all -- as an
    anticancer label (via the activity table) and as a feature row (via the consensus)."""

    def test_positive_sequences_reads_only_positives(self, tmp_path: Any) -> None:
        _write_anticp2_fixture(str(tmp_path))
        out = tmp_path / "acp.csv"
        anticancer.build_anticancer_table(data_dir=str(tmp_path), output_csv=str(out))
        seqs = anticancer.positive_sequences(output_csv=str(out))
        # Every returned sequence is a positive in the table; SHAREDSEQ (also a negative) is present.
        assert "SHAREDSEQ" in seqs
        assert "AALRGALR" not in seqs  # a pure negative

    def test_positive_sequences_empty_when_nothing_imported(self, tmp_path: Any) -> None:
        # No table and no raw files -> empty, not an error (safe for a fresh checkout).
        assert anticancer.positive_sequences(
            data_dir=str(tmp_path / "nope"), output_csv=str(tmp_path / "nope.csv")) == []

    def test_activity_records_carry_the_anticancer_group(self, monkeypatch: Any) -> None:
        # Exercise the real iter_activity_records; stub only its sequence source.
        monkeypatch.setattr(anticancer, "positive_sequences", lambda *a, **k: ["KLAKLAKKLAK"])
        records = list(anticancer.iter_activity_records())
        assert len(records) == 1
        record = records[0]
        assert record["database"] == "AntiCP2"
        assert record["sequence"] == "KLAKLAKKLAK"
        assert len(record["activities"]) == 1
        act = record["activities"][0]
        assert act["target_group"] == "anticancer"
        # activity_type is empty so no spurious per-assay task is created.
        assert act["activity_type"] == ""
        assert act["value_lower"] is None

    def test_activity_records_empty_without_import(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(anticancer, "positive_sequences", lambda *a, **k: [])
        assert list(anticancer.iter_activity_records()) == []

    def test_consensus_frame_shape(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(anticancer, "positive_sequences",
                            lambda *a, **k: ["KLAKLAKKLAK", "FAKKFAKKFAKK"])
        frame = anticancer.consensus_frame()
        assert list(frame.columns) == ["internal_id", "source", "sequence",
                                       "reference", "biological_activity"]
        assert set(frame["source"]) == {"AntiCP2"}
        assert len(frame) == 2

    def test_consensus_frame_empty_has_the_columns(self, monkeypatch: Any) -> None:
        # An empty frame must still carry the columns merge_databases concatenates on.
        monkeypatch.setattr(anticancer, "positive_sequences", lambda *a, **k: [])
        frame = anticancer.consensus_frame()
        assert len(frame) == 0
        assert list(frame.columns) == ["internal_id", "source", "sequence",
                                       "reference", "biological_activity"]

    def test_loader_is_registered_in_database_activities(self) -> None:
        from sample import database_activities as da
        assert "ANTICANCER" in da._LOADERS
        assert da._LOADERS["ANTICANCER"] is da.load_anticancer

    def test_loader_yields_the_activity_records(self, monkeypatch: Any) -> None:
        from sample import database_activities as da
        monkeypatch.setattr(anticancer, "positive_sequences", lambda *a, **k: ["KLAKLAKKLAK"])
        records = list(da.load_anticancer("unused_base_dir"))
        assert len(records) == 1
        assert records[0]["activities"][0]["target_group"] == "anticancer"


class TestFlashtextIntegration:
    """The imported anticancer peptides must be recognisable by the extraction pipeline's
    known-sequence automata, in both one-letter and three-letter spelling."""

    def test_one_letter_automaton_recognises_an_anticancer_peptide(self, monkeypatch: Any) -> None:
        from sample import nlp
        monkeypatch.setattr(nlp, "_imported_anticancer_sequences", lambda: ["KLAKLAKKLAKLAK"])
        nlp.get_known_seq_1l_kp.cache_clear()
        try:
            kp = nlp.get_known_seq_1l_kp()
            hits = kp.extract_keywords("the peptide KLAKLAKKLAKLAK showed activity")
            assert any(h == "KLAKLAKKLAKLAK" for h in hits)
        finally:
            nlp.get_known_seq_1l_kp.cache_clear()

    def test_three_letter_automaton_recognises_an_anticancer_peptide(self, monkeypatch: Any) -> None:
        from sample import nlp
        from sample.sequence import sequence_to_3_letter
        seq = "KLAKLAKKLAKLAK"
        monkeypatch.setattr(nlp, "_imported_anticancer_sequences", lambda: [seq])
        nlp.get_known_seq_3l_kp.cache_clear()
        try:
            three = sequence_to_3_letter(seq)
            kp = nlp.get_known_seq_3l_kp()
            hits = kp.extract_keywords(f"the sequence {three} was tested")
            # case_sensitive=False lowercases matches, so compare case-insensitively.
            assert any(h.lower() == three.lower() for h in hits)
        finally:
            nlp.get_known_seq_3l_kp.cache_clear()

    def test_automaton_still_builds_when_nothing_imported(self, monkeypatch: Any) -> None:
        from sample import nlp
        monkeypatch.setattr(nlp, "_imported_anticancer_sequences", lambda: [])
        nlp.get_known_seq_1l_kp.cache_clear()
        try:
            assert nlp.get_known_seq_1l_kp() is not None  # resource-file automaton, no crash
        finally:
            nlp.get_known_seq_1l_kp.cache_clear()

    def test_source_helper_is_graceful_on_failure(self, monkeypatch: Any) -> None:
        # If the anticancer import is unreadable, the helper returns [] rather than breaking the
        # automaton build (which the whole extraction pipeline depends on).
        from sample import nlp

        def _boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("unreadable")

        monkeypatch.setattr(anticancer, "positive_sequences", _boom)
        assert nlp._imported_anticancer_sequences() == []
