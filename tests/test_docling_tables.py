"""
Tests for :mod:`sample.docling_tables` — table-to-record conversion.

These run against synthetic cell grids in the shape ``sample/_docling_worker.py`` emits, so none of
them need docling, a PDF, or the ``sample-docling`` environment.

The emphasis is on the ways a table record can be WRONG rather than missing, because those survive
review. A sequence that reads ``HFLPLIGRVLSGIL`` instead of ``FLPLIGRVLSGIL`` is a different
peptide; a micromolar MIC stamped ``ug/mL`` is off by roughly a factor of 1000; a dose recorded as
a target organism attributes a measurement to an entity that was never assayed. Each of those was a
real defect found while building this module, and each has a test here.
"""
from typing import Any, Dict, List

import pytest

from sample import docling_tables as D


def cell(r0: int, r1: int, c0: int, c1: int, text: str, header: bool = False) -> Dict[str, Any]:
    return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "text": text, "header": header}


def table(cells: List[Dict[str, Any]], page_no: int = 1) -> Dict[str, Any]:
    return {"page_no": page_no, "num_rows": None, "num_cols": None, "cells": cells}


#: A conventional layout: one peptide per row, with its sequence and an assay column.
STANDARD = table([
    cell(0, 1, 0, 1, "peptide", header=True),
    cell(0, 1, 1, 2, "sequence", header=True),
    cell(0, 1, 2, 3, "MIC (uM) E. coli", header=True),
    cell(1, 2, 0, 1, "TA"),
    cell(1, 2, 1, 2, "GIGKFLHSAKKFGKAFVGEIMNS"),
    cell(1, 2, 2, 3, "12.5"),
])

#: An assay table turned on its side: peptides across the columns, organisms down the rows, under a
#: single spanning assay header. Common in this literature and the case the pdfplumber path reads
#: backwards.
TRANSPOSED_ORGANISMS = table([
    cell(0, 1, 1, 3, "MIC ( µ M)", header=True),
    cell(1, 2, 1, 2, "TL", header=True),
    cell(1, 2, 2, 3, "TA", header=True),
    cell(2, 3, 0, 1, "Escherichia coli D21"),
    cell(2, 3, 1, 2, "12"),
    cell(2, 3, 2, 3, "> 24"),
], page_no=6)

#: The same shape, but the rows are DOSES rather than organisms -- the row-label column declares a
#: concentration unit of its own.
TRANSPOSED_DOSES = table([
    cell(0, 1, 0, 1, "peptide concn ( µ M)", header=True),
    cell(0, 1, 1, 3, "hemolysis", header=True),
    cell(1, 2, 1, 2, "TL", header=True),
    cell(1, 2, 2, 3, "TA", header=True),
    cell(2, 3, 0, 1, "3"),
    cell(2, 3, 1, 2, "13"),
    cell(2, 3, 2, 3, "3"),
], page_no=6)


class TestSequenceFromCell:
    @pytest.mark.parametrize("raw,expected", [
        ("GIGKFLHSAKKFGKAFVGEIMNS", "GIGKFLHSAKKFGKAFVGEIMNS"),
        ("ILPWKWPWWPWRR-NH2", "ILPWKWPWWPWRR"),
    ])
    def test_reads_one_letter_sequences(self, raw: str, expected: str) -> None:
        assert D.sequence_from_cell(raw) == expected

    def test_keeps_a_low_complexity_designed_peptide(self) -> None:
        # Two distinct residues. Any "must use N different residues" rule rejects this, and it is a
        # real designed AMP, not noise.
        assert D.sequence_from_cell("KLLKLLKKLLKLLK") == "KLLKLLKKLLKLLK"

    def test_reads_three_letter_code_with_positions_and_termini(self) -> None:
        raw = ("H-Phe 1 -Leu 2 -Pro 3 -Leu 4 -Ile 5 -Gly 6 -Arg 7 -Val 8 - "
               "Leu 9 -Ser 10 -Gly 11 -Ile 12 -Leu 13 -NH 2")
        assert D.sequence_from_cell(raw) == "FLPLIGRVLSGIL"

    def test_the_n_terminal_h_is_not_a_histidine(self) -> None:
        # Handed the raw cell, the shared converter returns "HFLPLIGRVLSGIL": the "H-" terminus
        # read as histidine. That is a WRONG sequence rather than an absent one -- it would pass
        # every downstream validity check while naming a different peptide.
        raw = ("H-Phe 1 -Leu 2 -Pro 3 -Leu 4 -Ile 5 -Gly 6 -Arg 7 -Val 8 - "
               "Leu 9 -Ser 10 -Gly 11 -Ile 12 -Leu 13 -NH 2")
        assert D.sequence_from_cell(raw) is not None
        assert not D.sequence_from_cell(raw).startswith("H")

    @pytest.mark.parametrize("primer", [
        "GCGAAGGCATCAACAT", "TCAGGTGCTGTTGCTT", "AGTGCGCCACGCTCTT",
        "ACATTCGGCAACCTCT", "AAGAAGGGCTCCAAGA", "CCTGCTAAACATCATG",
    ])
    def test_a_pcr_primer_is_not_a_peptide(self, primer: str) -> None:
        # A, C, G and T are all one-letter residue codes (Ala/Cys/Gly/Thr), so a primer clears the
        # residue alphabet, the length floor, the dictionary and the prose rule. Methods sections
        # print primer tables routinely: of the ten records from the first two documents that
        # yielded anything, seven were primers -- and they were the ones that SURVIVED the merge,
        # so they would have been written to the database as peptide sequences.
        #
        # nlp.is_valid_sequence carries a composition log-likelihood test for exactly this, which
        # is why it is required as a second condition rather than reimplemented here.
        assert D.sequence_from_cell(primer) is None

    def test_real_peptides_still_pass_the_shared_test(self) -> None:
        # The DNA guard must not cost recall: these are the sequences the table path exists to find.
        for sequence in ("GIGKFLHSAKKFGKAFVGEIMNS", "KLLKLLKKLLKLLK", "FLPLIGRVLSGIL",
                         "GLLSVLGSVAKHVLPHVVPVIAEHL"):
            assert D.sequence_from_cell(sequence) == sequence

    def test_a_hyphenated_clade_heading_is_not_a_sequence(self) -> None:
        # "Gram-negative" defeats every other rule here: all its letters are residue codes, without
        # the hyphen it clears the length floor, and GRAMNEGATIVE is not a dictionary word. It was
        # accepted off a real table's heading row. "Gram-positive" escapes only because "o" is not
        # a residue code, which is not a margin to depend on.
        assert D.sequence_from_cell("Gram-negative") is None
        assert D.sequence_from_cell("Gram-positive") is None
        assert D.sequence_from_cell("Colony-forming units") is None

    @pytest.mark.parametrize("raw", [
        "Escherichia coli", "Pseudomonas aeruginosa",  # residue-alphabet organism names
        "Magainin", "MAGAININ", "ALANINE",             # names spelled only in residue letters
        ">128", "12.5", "n.d.", "ND", "-", "TL", "Pro 3 -TL", "",
    ])
    def test_rejects_cells_that_are_not_sequences(self, raw: str) -> None:
        # nlp.is_valid_sequence returns True for several of these; this module scans every cell of
        # a row, so anything it accepts too readily lands in the database as a peptide.
        assert D.sequence_from_cell(raw) is None


class TestUnitFromHeader:
    @pytest.mark.parametrize("header,expected", [
        ("MIC ( µ M) a.TL", "uM"),
        ("MIC (ug/mL)", "ug/mL"),
        ("peptide concn ( µ M)", "uM"),
        ("MIC (mg/mL)", "mg/mL"),
    ])
    def test_reads_the_declared_unit(self, header: str, expected: str) -> None:
        assert D.unit_from_header(header) == expected

    @pytest.mark.parametrize("header", [
        "Escherichia coli", "Bacterial strain.", "hemolysis.TL", "Characteristic", "",
    ])
    def test_returns_none_when_no_unit_is_declared(self, header: str) -> None:
        # The alternative -- what the pdfplumber path does -- is to stamp "ug/mL" on everything,
        # which reports micromolar measurements as micrograms per millilitre.
        assert D.unit_from_header(header) is None

    def test_does_not_invent_a_unit_from_a_bare_abbreviation(self) -> None:
        # canonicalize_unit maps the string "MM" to "uM" (an OCR repair). Applied to a whole header
        # rather than a parenthesised token, that turns a column named MM into a micromolar unit.
        assert D.unit_from_header("MM") is None


class TestActivityTypeFromHeader:
    @pytest.mark.parametrize("header,expected", [
        ("MIC ( µ M)", "MIC"),
        ("IC50 (uM)", "IC50"),
        ("hemolysis.TL", "Hemolysis"),
        ("MBC (ug/mL)", "MBC"),
        ("zone of inhibition", "Growth Inhibition"),
    ])
    def test_names_the_assay(self, header: str, expected: str) -> None:
        assert D.activity_type_from_header(header) == expected

    def test_returns_none_rather_than_defaulting_to_mic(self, ) -> None:
        # Defaulting to MIC for any numeric cell labels hemolysis percentages and peptide
        # concentrations as minimum inhibitory concentrations.
        assert D.activity_type_from_header("Patients (N 5 156)") is None


class TestOrientation:
    def test_a_conventional_table_is_not_transposed(self) -> None:
        header, body, _ = D._grid(STANDARD)
        assert D.is_transposed(header, body) is False

    def test_peptides_across_the_columns_is_transposed(self) -> None:
        header, body, depth = D._grid(TRANSPOSED_ORGANISMS)
        assert depth == 2
        assert D.is_transposed(header, body) is True

    def test_a_spanning_header_names_every_column_it_covers(self) -> None:
        header, _, _ = D._grid(TRANSPOSED_ORGANISMS)
        labels = D._column_labels(header, 3)
        assert "MIC" in labels[1] and "MIC" in labels[2]
        assert labels[1].endswith("TL") and labels[2].endswith("TA")


class TestRecordsFromTable:
    def test_reads_a_conventional_row(self) -> None:
        records = D.records_from_table(STANDARD, page_text="TA GIGKFLHSAKKFGKAFVGEIMNS 12.5")
        assert len(records) == 1
        record = records[0]
        assert record["peptide_name"] == "TA"
        assert record["peptide_sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"
        assert record["source_page"] == 1
        activity = record["biological_activities"][0]
        assert activity["activity_type"] == "MIC"
        assert activity["activity_unit"] == "uM"
        assert activity["value_lower"] == 12.5

    def test_transposed_table_attributes_values_to_the_right_peptide(self) -> None:
        records = D.records_from_table(TRANSPOSED_ORGANISMS)
        by_name = {r["peptide_name"]: r for r in records}
        assert set(by_name) == {"TL", "TA"}
        tl = by_name["TL"]["biological_activities"][0]
        # Resolved to the binomial, not the raw row label. No strain designation resolves against
        # the NCBI taxonomy -- "Escherichia coli D21" is rejected outright -- and title-casing it
        # ("Escherichia Coli") breaks the binomial as well, so both are handled before the record
        # is built rather than after it is rejected downstream.
        assert tl["target_organism"] == "Escherichia coli"
        assert tl["value_lower"] == 12.0
        assert tl["activity_unit"] == "uM"
        # The censoring must survive: "> 24" is an upper bound, not a point estimate.
        ta = by_name["TA"]["biological_activities"][0]
        assert ta["operator"] == ">"

    def test_a_self_determining_assay_keeps_its_own_class(self) -> None:
        # Hemolysis implies target_group "other". In a dose-response table the row label is a
        # concentration ("3"), which names no assay, so inferring the class from the label produced
        # `antibacterial` on a hemolysis measurement. The assay must be known BEFORE the group is
        # derived -- assigning activity_type afterwards leaves the group derived from the default.
        records = D.records_from_table(TRANSPOSED_DOSES)
        assert records
        activity = records[0]["biological_activities"][0]
        assert activity["activity_type"] == "Hemolysis"
        assert activity["target_group"] == "other", "hemolysis is not an antibacterial result"

    def test_dose_rows_are_not_recorded_as_organisms(self) -> None:
        records = D.records_from_table(TRANSPOSED_DOSES)
        assert records
        activity = records[0]["biological_activities"][0]
        assert activity["activity_type"] == "Hemolysis"
        assert activity["target_organism"] is None, "a concentration is not a target organism"
        assert activity["dose_raw"] == "3"
        assert activity["dose_unit"] == "uM"

    def test_a_column_with_no_declared_unit_yields_no_activity(self) -> None:
        # Not a style preference. strict_schema.normalize_unit(None) returns "", and "" is absent
        # from units_for_activity() for every assay, so a unit-less activity is discarded by
        # enforce_activity several stages later. Refusing to build it here makes the loss visible
        # at the point where the reason is known.
        grid = table([
            cell(0, 1, 0, 1, "peptide", header=True),
            cell(0, 1, 1, 2, "activity", header=True),   # no unit anywhere in the header
            cell(1, 2, 0, 1, "TA"),
            cell(1, 2, 1, 2, "12.5"),
        ])
        assert D.records_from_table(grid) == []

    def test_records_carry_both_group_fields(self) -> None:
        # target_group drives the extraction-time contradiction checks; activity_group is the
        # persisted column the search API filters on. Setting only one disables one of them.
        records = D.records_from_table(STANDARD, page_text="TA")
        activity = records[0]["biological_activities"][0]
        assert activity["target_group"] == activity["activity_group"] == "antibacterial"

    def test_an_organism_is_resolved_to_taxonomy_not_title_cased(self) -> None:
        records = D.records_from_table(TRANSPOSED_ORGANISMS)
        organisms = {a["target_organism"]
                     for r in records for a in r["biological_activities"]}
        assert organisms == {"Escherichia coli"}

    def test_a_clade_heading_row_is_not_recorded_as_an_organism(self) -> None:
        # Assay tables group their rows under headings ("Gram-positive bacteria", "Yeasts") that
        # carry no values. Walking rows naively turns each into a species.
        grid = table([
            cell(0, 1, 1, 3, "MIC ( µ M)", header=True),
            cell(1, 2, 1, 2, "TL", header=True),
            cell(1, 2, 2, 3, "TA", header=True),
            cell(2, 3, 0, 1, "Gram-positive bacteria"),
            cell(3, 4, 0, 1, "Staphylococcus aureus ATCC 25923"),
            cell(3, 4, 1, 2, "12"),
            cell(3, 4, 2, 3, "24"),
        ], page_no=6)
        organisms = {a["target_organism"]
                     for r in D.records_from_table(grid) for a in r["biological_activities"]}
        assert "Gram-positive bacteria" not in organisms
        assert organisms == {"Staphylococcus aureus"}, "strain must reduce to the binomial"

    def test_non_measurements_are_not_recorded(self) -> None:
        grid = table([
            cell(0, 1, 0, 1, "peptide", header=True),
            cell(0, 1, 1, 2, "MIC (uM)", header=True),
            cell(1, 2, 0, 1, "TA"),
            cell(1, 2, 1, 2, "n.d."),
        ])
        assert D.records_from_table(grid) == []


class TestMergeDocumentRecords:
    def test_joins_a_sequence_onto_records_that_only_had_a_name(self) -> None:
        # A paper states its sequences once and then refers to the peptides by name in every assay
        # table. Without this join the measurements never acquire a sequence.
        sequence_only = {"peptide_name": "TL", "peptide_sequence": "FVQWFSKFLGRIL",
                         "biological_activities": [], "context": "", "source_page": 1}
        assay = {"peptide_name": "TL", "peptide_sequence": None,
                 "biological_activities": [{"activity_type": "MIC", "value_lower": 12.0}],
                 "context": "ctx", "source_page": 6}
        merged = D.merge_document_records([sequence_only, assay])
        assert len(merged) == 1
        assert merged[0]["peptide_sequence"] == "FVQWFSKFLGRIL"
        assert merged[0]["biological_activities"]

    def test_folds_the_same_peptide_from_two_tables_into_one_record(self) -> None:
        mic = {"peptide_name": "TL", "peptide_sequence": "FVQWFSKFLGRIL",
               "biological_activities": [{"activity_type": "MIC"}], "context": "a",
               "source_page": 6}
        hemolysis = {"peptide_name": "TL", "peptide_sequence": "FVQWFSKFLGRIL",
                     "biological_activities": [{"activity_type": "Hemolysis"}], "context": "b",
                     "source_page": 6}
        merged = D.merge_document_records([mic, hemolysis])
        assert len(merged) == 1
        assert len(merged[0]["biological_activities"]) == 2

    def test_context_is_dropped_when_the_joined_sequence_is_not_in_it(self) -> None:
        # The extraction pipeline's third gate deletes any peptide whose sequence is absent from its
        # own context window. An assay record has no sequence until the join supplies one, so a
        # window kept at construction time can end up contradicting the sequence it later receives.
        # Checked only at construction, this passed while every real record was being deleted.
        sequence_only = {"peptide_name": "TL", "peptide_sequence": "FVQWFSKFLGRIL",
                         "biological_activities": [], "context": "", "source_page": 1}
        assay = {"peptide_name": "TL", "peptide_sequence": None,
                 "biological_activities": [{"activity_type": "MIC", "value_lower": 12.0}],
                 "context": "MIC values against E. coli were determined in triplicate.",
                 "source_page": 6}
        merged = D.merge_document_records([sequence_only, assay])
        assert len(merged) == 1
        assert merged[0]["peptide_sequence"] == "FVQWFSKFLGRIL"
        assert merged[0]["context"] == "", "a window not supporting the sequence must not be kept"

    def test_context_is_kept_when_it_does_support_the_sequence(self) -> None:
        record = {"peptide_name": "TL", "peptide_sequence": "FVQWFSKFLGRIL",
                  "biological_activities": [{"activity_type": "MIC", "value_lower": 12.0}],
                  "context": "The peptide FVQWFSKFLGRIL was assayed against E. coli.",
                  "source_page": 6}
        merged = D.merge_document_records([record])
        assert merged[0]["context"].startswith("The peptide")

    def test_provenance_follows_the_measurements_not_the_sequence(self) -> None:
        # The sequence table sits on an early page and the assay tables pages later; the first
        # record seen wins the merge, so without care a record cites page 1 for values read from
        # page 6.
        sequence_only = {"peptide_name": "TL", "peptide_sequence": "FVQWFSKFLGRIL",
                         "biological_activities": [], "context": "", "source_page": 1}
        assay = {"peptide_name": "TL", "peptide_sequence": None,
                 "biological_activities": [{"activity_type": "MIC", "value_lower": 12.0}],
                 "context": "", "source_page": 6}
        merged = D.merge_document_records([sequence_only, assay])
        assert merged[0]["source_page"] == 6

    def test_a_sequence_with_no_measurement_is_not_emitted(self) -> None:
        sequence_only = {"peptide_name": "TA", "peptide_sequence": "FLPLIGRVLSGIL",
                         "biological_activities": [], "context": "", "source_page": 1}
        assert D.merge_document_records([sequence_only]) == []


class TestResolution:
    def test_tools_are_resolved_by_absolute_path_never_bare(self) -> None:
        # KNOWN_ISSUES.md records `conda run -n <env>` misrouting on this host, and tesseract is not
        # on PATH here -- a bare command name imports fine and then fails at the first call.
        for resolved in (D.docling_python(), ):
            if resolved:
                assert resolved[0].startswith("/"), "docling interpreter must be an absolute path"
        binary = D.tesseract_command()
        if binary:
            assert binary.startswith("/") and binary.endswith("tesseract")
