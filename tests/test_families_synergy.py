"""
Tests for the layers added after the serving foundation: families, synergy, assay-archive links,
and the contract rules that were declared but not enforced.

The contract tests carry the most weight. ``validation_rules`` publishes what each assay may be
filed under and what it may be measured against, and four of those rules were never applied -- a
MIC filed as ``anticancer``, a MIC against a cell line, an anticancer measurement against a
bacterium, and a range running 64 to 2 all passed through untouched. That is the same class of
defect that put bacteria in the anticancer bucket downstream, so each rule gets a test that fails
if the enforcement is ever removed.

The family tests care about one distinction above all: a Pfam assignment is curated and a cluster
is derived here, and they must stay distinguishable. Collapsing them would present a clustering
artefact with the authority of a curated family.
"""
import os
import re
import sqlite3
from typing import Any

import pytest


def _step_running(subcommand: str) -> int:
    """
    The Makefile step number whose recipe invokes ``sample-cli <subcommand>``.

    :param subcommand: A ``sample-cli`` subcommand, e.g. ``"assembly"``.
    :returns: The step number that runs it.
    :raises AssertionError: If no step runs it, which means either the pipeline dropped the
        command or the test is asking about the wrong one -- both worth failing on.
    """
    makefile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Makefile")
    with open(makefile, encoding="utf-8") as handle:
        source = handle.read()
    for block in re.split(r"^(?=step_\d+_)", source, flags=re.M):
        head = re.match(r"step_(\d+)_", block)
        if head and re.search(rf"sample-cli {re.escape(subcommand)}\b", block[:3000]):
            return int(head.group(1))
    raise AssertionError(f"No Makefile step runs `sample-cli {subcommand}`")


@pytest.fixture
def db_with_extras(tmp_path: Any) -> str:
    """
    Builds a release database carrying the family, bibliography and bioactivity tables.

    :param tmp_path: pytest temporary directory.
    :returns: Path to the database.
    """
    path = str(tmp_path / "sample.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE peptides (entry_id TEXT PRIMARY KEY, sequence TEXT NOT NULL, SMILES TEXT,
            source TEXT, is_decoy INTEGER DEFAULT 0, LITERATURE TEXT, PATENT TEXT,
            Reference TEXT, Biological_Activity TEXT);
        CREATE TABLE descriptors (entry_id TEXT PRIMARY KEY, molecular_weight REAL, logp REAL,
            hydrogen_bond_donors INTEGER, hydrogen_bond_acceptors INTEGER, charge REAL,
            sequence_length INTEGER);
        CREATE TABLE activities (id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT,
            activity_group TEXT, value REAL, unit TEXT, value_mM REAL, value_ug_ml REAL,
            potency REAL, target_organism TEXT, target_taxid INTEGER, activity_type TEXT,
            mechanism_of_action TEXT, source_accession TEXT, source_reference_index TEXT);
        CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT,
            source_id TEXT, UNIQUE(source_type, source_id));
        CREATE TABLE peptide_references (peptide_id TEXT, paper_id INTEGER,
            UNIQUE(peptide_id, paper_id));
        CREATE TABLE peptide_families (entry_id TEXT PRIMARY KEY, family_id TEXT,
            family_name TEXT, source TEXT, evalue REAL, all_domains TEXT);
        CREATE TABLE bibliography (id INTEGER PRIMARY KEY AUTOINCREMENT, ref_key TEXT UNIQUE,
            pmid TEXT, doi TEXT, scopus_eid TEXT, title TEXT, authors TEXT, journal TEXT,
            year INTEGER, volume TEXT, issue TEXT, pages TEXT, apa TEXT, url TEXT,
            source_db TEXT);
        CREATE TABLE peptide_bibliography (peptide_id TEXT, reference_id INTEGER,
            UNIQUE(peptide_id, reference_id));
        CREATE TABLE activity_references (activity_id INTEGER, reference_id INTEGER,
            UNIQUE(activity_id, reference_id));
        CREATE TABLE bioactivity (archive TEXT, identifier TEXT, n_records INTEGER,
            payload TEXT, PRIMARY KEY (archive, identifier));
        CREATE TABLE peptide_bioactivity (entry_id TEXT, archive TEXT, identifier TEXT,
            UNIQUE(entry_id, archive, identifier));
        CREATE TABLE peptide_names (entry_id TEXT, name TEXT, name_kind TEXT, source_db TEXT,
            UNIQUE(entry_id, name));
        CREATE VIRTUAL TABLE peptides_fts USING fts5(entry_id UNINDEXED, sequence,
            literature, patent, tokenize="unicode61");
    """)
    conn.executemany(
        "INSERT INTO peptides (entry_id, sequence, is_decoy) VALUES (?, ?, ?)",
        [("SAMPLE0000000001", "GIGKFLHSAKKFGKAFVGEIMNS", 0),
         ("SAMPLE0000000002", "GIGKFLHSAKKFGKAFVGEIMKS", 0),
         ("SAMPLE0000000003", "ILPWKWPWWPWRR", 0),
         ("SAMPLE_DECOY_00000001", "MFGLGGAAPQISSQQKLQAAEAELDMV", 1)])
    conn.executemany(
        "INSERT INTO peptide_families (entry_id, family_id, family_name, source, evalue) "
        "VALUES (?, ?, ?, ?, ?)",
        [("SAMPLE0000000001", "PF00711", "Defensin_beta", "pfam", 1e-20),
         ("SAMPLE0000000002", "PF00711", "Defensin_beta", "pfam", 1e-18),
         ("SAMPLE0000000003", "SAMPLE_FAM_000007", None, "cluster", None),
         # A decoy carrying a family would be a bug upstream; present here so the member query is
         # shown to exclude it rather than merely never meeting one.
         ("SAMPLE_DECOY_00000001", "PF00711", "Defensin_beta", "pfam", 1e-9)])
    conn.executemany(
        "INSERT INTO peptide_names (entry_id, name, name_kind, source_db) VALUES (?,?,?,?)",
        [("SAMPLE0000000001", "Magainin 2", "primary", "ADP"),
         ("SAMPLE0000000001", "Magainin II", "synonym", "ADP"),
         ("SAMPLE0000000001", "Mag-2", "synonym", "DBAASP"),
         ("SAMPLE0000000002", "Magainin-2 analogue", "primary", "DBAASP"),
         ("SAMPLE0000000003", "Indolicidin", "primary", "CAMP")])
    conn.execute(
        "INSERT INTO activities (id, entry_id, activity_group, value, unit, target_organism, "
        "target_taxid, activity_type) VALUES (1, 'SAMPLE0000000001', 'antibacterial', 16.0, "
        "'ug/mL', 'Escherichia coli', 562, 'MIC')")
    conn.execute(
        "INSERT INTO bibliography (id, ref_key, pmid, doi, apa, url) VALUES "
        "(1, 'pmid:1', '1', '10.1/x', 'Author, A. (2020). Title. Journal.', 'https://doi.org/10.1/x')")
    conn.execute("INSERT INTO peptide_bibliography VALUES ('SAMPLE0000000001', 1)")
    conn.execute("INSERT INTO activity_references VALUES (1, 1)")
    conn.execute("INSERT INTO bioactivity VALUES ('chembl', 'CHEMBL123', 42, '[]')")
    conn.execute(
        "INSERT INTO peptide_bioactivity VALUES ('SAMPLE0000000001', 'chembl', 'CHEMBL123')")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def wired(db_with_extras: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Points :mod:`sample.db` at the fixture database."""
    from sample import db

    monkeypatch.setenv("SAMPLE_DB", db_with_extras)
    monkeypatch.setattr(db, "default_db_path", lambda: db_with_extras)
    db.get_stats.cache_clear()
    return db_with_extras


class TestFamilies:
    def test_lists_families_largest_first(self, wired: str) -> None:
        from sample import db

        rows, total = db.list_families()
        assert total == 2
        assert rows[0]["family_id"] == "PF00711"
        assert rows[0]["n_members"] == 3  # includes the decoy row; membership is not a decoy gate

    def test_pfam_and_cluster_stay_distinguishable(self, wired: str) -> None:
        from sample import db

        by_id = {r["family_id"]: r for r in db.list_families()[0]}
        assert by_id["PF00711"]["source"] == "pfam"
        assert by_id["SAMPLE_FAM_000007"]["source"] == "cluster"

    def test_only_pfam_families_link_out(self, wired: str) -> None:
        # A SAMPLE cluster id resolves to nothing at Pfam; emitting a link would point a reader at
        # an unrelated entry.
        from sample import db

        by_id = {r["family_id"]: r for r in db.list_families()[0]}
        assert "interpro" in by_id["PF00711"]["url"]
        assert by_id["SAMPLE_FAM_000007"]["url"] is None

    @pytest.mark.parametrize("source,expected", [("pfam", 1), ("cluster", 1)])
    def test_source_filter(self, wired: str, source: str, expected: int) -> None:
        from sample import db

        assert db.list_families(source=source)[1] == expected

    def test_members_exclude_decoys(self, wired: str) -> None:
        from sample import db

        rows, _ = db.family_members("PF00711")
        assert [r["entry_id"] for r in rows] == ["SAMPLE0000000001", "SAMPLE0000000002"]

    def test_peptide_carries_its_family(self, wired: str) -> None:
        from sample import db

        family = db.get_peptide("SAMPLE0000000001")["family"]
        assert family["family_id"] == "PF00711"
        assert family["source"] == "pfam"

    def test_missing_table_is_not_an_error(self, tmp_path: Any,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        # A database built before step 35 has no family table, and must still serve peptides.
        from sample import db

        path = str(tmp_path / "bare.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE peptides (entry_id TEXT PRIMARY KEY, sequence TEXT, "
                     "SMILES TEXT, source TEXT, is_decoy INTEGER, LITERATURE TEXT, PATENT TEXT)")
        conn.commit()
        conn.close()
        monkeypatch.setattr(db, "default_db_path", lambda: path)
        assert db.list_families() == ([], 0)
        assert db.family_members("PF00711") == ([], 0)


class TestPipelineOrdering:
    """
    Family assignment must not depend on a step that runs after it.

    ``build_families`` originally read a FASTA derived from ``sample.db``, which step 44 builds --
    nine steps after family assignment. It worked only because that file happened to exist from a
    manual run; on a clean checkout the step had no input at all. It now derives the FASTA from
    ``final_dataset.csv``, which step 29 produces.
    """

    def test_fasta_comes_from_the_assembled_dataset(self, tmp_path: Any) -> None:
        import pandas as pd

        from sample.families import export_peptides_fasta, read_fasta

        dataset = tmp_path / "final_dataset.csv"
        pd.DataFrame([
            {"entry_id": "P1", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "is_decoy": 0},
            {"entry_id": "P2", "sequence": "ILPWKWPWWPWRR", "is_decoy": 0},
            {"entry_id": "D1", "sequence": "MFGLGGAAPQISSQQKLQAAEA", "is_decoy": 1},
            {"entry_id": "P3", "sequence": "GIG", "is_decoy": 0},          # too short
            {"entry_id": "P4", "sequence": "GIGKF-LHSA", "is_decoy": 0},   # non-standard
        ]).to_csv(dataset, index=False)

        out = tmp_path / "peptides.fasta"
        export_peptides_fasta(str(dataset), str(out))
        written = read_fasta(str(out))

        assert set(written) == {"P1", "P2"}
        # A decoy is a shuffled sequence with no provenance; assigning it a family would be a
        # statement about a fabrication.
        assert "D1" not in written

    def test_missing_dataset_names_the_step_to_run(self, tmp_path: Any) -> None:
        # The expected number is read from the Makefile rather than written here. Hardcoding it
        # made this test drift twice: it kept asserting "step 27" through two renumbers, so it
        # passed while the error message sent users to a step that had become something else
        # entirely. Resolved from the target that actually runs `sample-cli assembly`, the
        # assertion now fails if the message and the pipeline disagree -- which is the only thing
        # worth testing about a message whose whole job is to name the right step.
        from sample.families import export_peptides_fasta

        step = _step_running("assembly")
        with pytest.raises(FileNotFoundError, match=f"step {step}"):
            export_peptides_fasta(str(tmp_path / "absent.csv"), str(tmp_path / "x.fasta"))


class TestSimilarityEdgesWorkDirCaching:
    """
    Regression test for a real bug found in production: PepMatch's Preprocessor caches a k-mer
    index next to the target FASTA and reuses it on a later call without checking whether the
    FASTA's content has since changed. ``work_dir`` defaults to a long-lived, shared directory
    (``FAMILY_DIR``), so a stale index from an earlier, differently-sized candidate set silently
    wins and the search runs against sequences that no longer exist -- with no error, just every
    peptide coming back as its own singleton. Measured directly: a real 38,064-peptide clustering
    pass reused an 8-day-old index from a smaller set and found zero real matches.
    """

    def test_stale_target_cache_does_not_mask_a_real_match(self, tmp_path: Any) -> None:
        from sample.families import similarity_edges

        work_dir = str(tmp_path)
        # First call populates a target index in work_dir for a set with no genuine near-identical
        # pair, and in the same length bucket (15-24, k=3) as the second call's pair below.
        similarity_edges({"A1": "GIGKFLHSAKKFGKA", "A2": "WWWWWWWWWWWWWWW"}, work_dir=work_dir)

        # Second call, same work_dir: a genuinely near-identical pair (one mismatch) the stale
        # index left behind by the first call has no knowledge of.
        edges = similarity_edges(
            {"B1": "GIGKFLHSAKKFGKA", "B2": "GIGKFLHSAKKFGKQ"}, work_dir=work_dir)

        pairs = {frozenset((a, b)) for a, b, _ in edges}
        assert frozenset(("B1", "B2")) in pairs


class TestClusterCdhit:
    """
    ``sample.ml_models.GROUPING_IDENTITY`` was lowered to 0.4, below what PepMatch's mismatch-cap
    route can deliver (:data:`sample.families.MAX_MISMATCH_CAP` is 6, so a 50-residue peptide is
    clustered at 88% identity no matter what is requested). cd-hit has no such ceiling, so it is
    now the preferred backend wherever it is installed. Runs the real binary rather than mocking
    subprocess -- these tests are meaningless if they only check that *some* command was invoked.
    """

    def test_skipped_if_cdhit_is_not_on_path(self) -> None:
        from sample.families import cdhit_available

        if not cdhit_available():
            pytest.skip("cd-hit is not installed in this environment")

    def test_near_identical_peptides_share_a_cluster(self, tmp_path: Any) -> None:
        from sample.families import cdhit_available, cluster_cdhit

        if not cdhit_available():
            pytest.skip("cd-hit is not installed in this environment")

        sequences = {
            "A1": "GIGKFLHSAKKFGKAFVGEIMNS",
            "A2": "GIGKFLHSAKKFGKAFVGEIMNT",  # one substitution -- 96% identical to A1
            "B1": "WWWWWWWWWWWWWWWWWWWWWWW",  # unrelated
        }
        assignment = cluster_cdhit(sequences, identity=0.4, work_dir=str(tmp_path))
        assert set(assignment) == set(sequences)
        assert assignment["A1"] == assignment["A2"]
        assert assignment["B1"] != assignment["A1"]

    def test_identity_below_the_floor_is_raised_not_rejected(self, tmp_path: Any) -> None:
        from sample.families import cdhit_available, cluster_cdhit

        if not cdhit_available():
            pytest.skip("cd-hit is not installed in this environment")

        sequences = {"A1": "GIGKFLHSAKKFGKAFVGEIMNS", "A2": "WWWWWWWWWWWWWWWWWWWWWWW"}
        # Must not raise for a value below cd-hit's 40% floor -- silently clamped instead, since a
        # caller sweeping a range of identities should not have to special-case the low end.
        assignment = cluster_cdhit(sequences, identity=0.1, work_dir=str(tmp_path))
        assert set(assignment) == set(sequences)

    def test_a_sequence_below_the_minimum_length_becomes_its_own_singleton(
            self, tmp_path: Any) -> None:
        from sample.families import cdhit_available, cluster_cdhit

        if not cdhit_available():
            pytest.skip("cd-hit is not installed in this environment")

        # cd-hit's own default minimum length is 10; a 3-mer would otherwise be silently dropped
        # from the output entirely, leaving the caller's split to raise a KeyError on it later.
        sequences = {"A1": "GIGKFLHSAKKFGKAFVGEIMNS", "A2": "GIG"}
        assignment = cluster_cdhit(sequences, identity=0.4, work_dir=str(tmp_path))
        assert set(assignment) == {"A1", "A2"}
        assert assignment["A2"] != assignment["A1"]

    def test_not_installed_raises_a_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sample.families import cluster_cdhit

        monkeypatch.setattr("sample.families.cdhit_available", lambda: False)
        with pytest.raises(RuntimeError, match="cd-hit"):
            cluster_cdhit({"A1": "GIGK", "A2": "GIGK"}, identity=0.4)


class TestBibliographyAndBioactivity:
    def test_peptide_carries_formatted_citations(self, wired: str) -> None:
        from sample import db

        bibliography = db.get_peptide("SAMPLE0000000001")["bibliography"]
        assert bibliography[0]["apa"].startswith("Author, A.")

    def test_activity_citations_are_measurement_level(self, wired: str) -> None:
        from sample import db

        activity = db.get_peptide("SAMPLE0000000001")["activities"][0]
        assert [r["pmid"] for r in activity["references"]] == ["1"]

    def test_bioactivity_link_resolves(self, wired: str) -> None:
        from sample import db

        hit = db.get_peptide("SAMPLE0000000001")["bioactivity"][0]
        assert hit["archive"] == "chembl"
        assert hit["n_records"] == 42
        assert "chembl" in hit["url"].lower()

    def test_absent_bioactivity_is_empty_not_missing(self, wired: str) -> None:
        # Empty means "never resolved to an archive identifier", not "no assays exist".
        from sample import db

        assert db.get_peptide("SAMPLE0000000003")["bioactivity"] == []


class TestRegexSearch:
    def test_matches_a_sequence_motif(self, wired: str) -> None:
        from sample import db

        rows, total = db.regex_search("^GIG.{2}[FL]")
        assert total == 2
        assert all(r["entry_id"].startswith("SAMPLE00000000") for r in rows)

    def test_excludes_decoys(self, wired: str) -> None:
        from sample import db

        rows, _ = db.regex_search(".", limit=100)
        assert not any(r["entry_id"].startswith("SAMPLE_DECOY") for r in rows)

    def test_invalid_pattern_is_reported_not_swallowed(self, wired: str) -> None:
        from sample import db

        with pytest.raises(db.InvalidPattern):
            db.regex_search("GIG(")

    def test_unknown_field_is_rejected(self, wired: str) -> None:
        from sample import db

        with pytest.raises(db.InvalidPattern):
            db.regex_search("GIG", field="sequence; DROP TABLE peptides")

    def test_time_budget_is_enforced(self, wired: str) -> None:
        from sample import db

        with pytest.raises(db.SearchTimeout):
            db.regex_search("^(A+)+$", time_budget=0.0)

    def test_overlong_pattern_is_rejected(self, wired: str) -> None:
        from sample import db

        with pytest.raises(db.InvalidPattern):
            db.regex_search("A" * (db.REGEX_MAX_PATTERN + 1))


class TestActivityRulesAreEnforced:
    """
    The four rules ``validation_rules`` publishes and ``enforce_activity`` did not apply.

    Each was found by the error-injection suite at 60 trials apiece, and each is the kind of fault
    that survives into the release looking like data: the record stays well formed, only its
    meaning is wrong.
    """

    @staticmethod
    def _activity(**overrides: Any) -> dict:
        base = {"activity_type": "MIC", "activity_unit": "uM", "value_lower": 4.0,
                "target_organism": "Escherichia coli", "target_cell_line": "",
                "target_group": "antibacterial", "value_operator": ""}
        base.update(overrides)
        return base

    def test_mic_cannot_be_filed_as_anticancer(self) -> None:
        from sample.strict_schema import enforce_activity

        clean, violations = enforce_activity(self._activity(target_group="anticancer"))
        assert violations, "MIC does not permit the anticancer class"
        # Re-derived from the organism rather than dumped into the catch-all.
        assert clean["target_group"] == "antibacterial"

    def test_mic_cannot_be_measured_against_a_cell_line(self) -> None:
        from sample.strict_schema import enforce_activity

        clean, violations = enforce_activity(
            self._activity(target_organism="", target_cell_line="MCF-7"))
        assert violations
        assert clean["target_cell_line"] == ""

    def test_anticancer_requires_an_animal_target(self) -> None:
        from sample.strict_schema import enforce_activity

        clean, violations = enforce_activity(self._activity(
            activity_type="IC50", target_group="anticancer",
            target_organism="Escherichia coli", target_cell_line=""))
        assert violations
        assert clean["target_organism"] == ""

    def test_transposed_range_is_repaired(self) -> None:
        from sample.strict_schema import enforce_activity

        clean, violations = enforce_activity(self._activity(value_lower=64.0, value_upper=2.0))
        assert violations
        # Both numbers are real; only their order was wrong, so the range is fixed rather than
        # dropped.
        assert (clean["value_lower"], clean["value_upper"]) == (2.0, 64.0)

    def test_a_correct_range_is_left_alone(self) -> None:
        from sample.strict_schema import enforce_activity

        clean, violations = enforce_activity(self._activity(value_lower=2.0, value_upper=64.0))
        assert not violations
        assert (clean["value_lower"], clean["value_upper"]) == (2.0, 64.0)

    @pytest.mark.parametrize("group", ["antibacterial", "antifungal", "antiparasitic", "other"])
    def test_permitted_groups_pass_unchanged(self, group: str) -> None:
        # The enforcement must not narrow MIC beyond what the rules actually say.
        from sample.strict_schema import enforce_activity

        organism = {"antibacterial": "Escherichia coli", "antifungal": "Candida albicans",
                    "antiparasitic": "Plasmodium falciparum", "other": ""}[group]
        clean, _ = enforce_activity(self._activity(target_group=group, target_organism=organism))
        assert clean["target_group"] == group


class TestDomainRules:
    """
    ``organism_contradicts_group`` is now derived from the published rules rather than a private
    dict that disagreed with them -- it had no ``anticancer`` entry at all, and mapped
    ``antiparasitic`` to protozoa alone.
    """

    @pytest.mark.parametrize("organism,group,contradicts", [
        ("Escherichia coli", "anticancer", True),
        ("Homo sapiens", "anticancer", False),
        # Helminths are Metazoa, so an animal-domain target is correct for antiparasitic.
        ("Caenorhabditis elegans", "antiparasitic", False),
        ("Plasmodium falciparum", "antiparasitic", False),
        # Archaea are permitted for antibacterial.
        ("Methanocaldococcus jannaschii", "antibacterial", False),
        ("Candida albicans", "antibacterial", True),
        ("Escherichia coli", "antifungal", True),
        # `other` places no restriction.
        ("Escherichia coli", "other", False),
    ])
    def test_domain_pairings(self, organism: str, group: str, contradicts: bool) -> None:
        from sample.nlp import organism_contradicts_group

        assert organism_contradicts_group(organism, group) is contradicts

    def test_unknown_organism_never_contradicts(self) -> None:
        # Conservative by construction: only a positively identified mismatch is acted on.
        from sample.nlp import organism_contradicts_group

        assert organism_contradicts_group("Nonexistent organism xyz", "antibacterial") is False


class TestRuleAccessors:
    def test_groups_for_activity(self) -> None:
        from sample.validation_rules import groups_for_activity

        assert "anticancer" not in groups_for_activity("MIC")
        assert "antibacterial" in groups_for_activity("MIC")

    def test_unknown_assay_is_unconstrained(self) -> None:
        # An empty set means "do not constrain", so an unrecognized assay is not silently narrowed.
        from sample.validation_rules import groups_for_activity

        assert groups_for_activity("NotAnAssay") == set()

    def test_target_kind(self) -> None:
        from sample.validation_rules import target_kind_for_activity

        assert target_kind_for_activity("MIC") == "organism"
        assert target_kind_for_activity("NotAnAssay") is None


class TestErrorInjectionSuite:
    def test_every_injection_declares_a_tier_and_a_rule(self) -> None:
        from sample.validation_extras import INJECTIONS

        for injection in INJECTIONS:
            assert injection.tier in ("structural", "semantic", "blind"), injection.name
            assert injection.probes.strip(), injection.name

    def test_blind_tier_is_excluded_from_the_headline(self) -> None:
        # Scoring the blind tier would make the number fall the more honestly the suite is written,
        # and would let anyone raise it by deleting the hard cases.
        from sample.validation_extras import SCORED_TIERS

        assert "blind" not in SCORED_TIERS

    def test_injections_actually_change_the_record(self) -> None:
        from sample.validation_extras import INJECTIONS, inject_error

        seed = {"peptide_name": "magainin-2", "peptide_sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
                "biological_activities": [{"activity_type": "MIC", "activity_unit": "uM",
                                           "value_lower": 4.0,
                                           "target_organism": "Escherichia coli",
                                           "target_cell_line": "",
                                           "target_group": "antibacterial"}]}
        for injection in INJECTIONS:
            mutated, _ = inject_error(seed, injection.name)
            assert mutated != seed, f"{injection.name} left the record unchanged"

    def test_seed_is_never_mutated(self) -> None:
        from sample.validation_extras import inject_error

        seed = {"peptide_sequence": "GIGKFLHSA", "biological_activities": []}
        before = dict(seed)
        inject_error(seed, "dna_as_peptide")
        assert seed == before


class TestPeptideNames:
    """
    Names and synonyms. The release stored none at all, so "magainin" matched nothing while
    magainin's sequence sat in the table.
    """

    @pytest.mark.parametrize("raw,primary,candidates", [
        ("Magainin-2", "Magainin-2", []),
        ("Ruminococcin A (RumA; Bacteriocin)", "Ruminococcin A", ["RumA", "Bacteriocin"]),
        ("Dermaseptin-B2 (DRS-B2, DS bII; natural AMPs)",
         "Dermaseptin-B2", ["DRS-B2", "DS bII", "natural AMPs"]),
        ("", "", []),
    ])
    def test_split_name(self, raw: str, primary: str, candidates: list) -> None:
        from sample.names import split_name

        assert split_name(raw) == (primary, candidates)

    @pytest.mark.parametrize("name,usable", [
        ("Magainin 2", True), ("DRS-B2", True),
        ("", False), ("Not available", False), ("unnamed", False), ("12345", False),
        ("x" * 200, False),
    ])
    def test_is_usable_name(self, name: str, usable: bool) -> None:
        from sample.names import is_usable_name

        assert is_usable_name(name) is usable

    def test_frequency_separates_synonyms_from_vocabulary(self, tmp_path: Any) -> None:
        # A synonym belongs to one peptide; a class label is shared by many. This is what keeps
        # "natural AMPs" from becoming a name for thousands of peptides -- no stoplist could
        # anticipate every vocabulary term the sources use.
        from sample.names import harvest_names

        adp = tmp_path / "antimicrobial_databases" / "adp"
        adp.mkdir(parents=True)
        rows = ["0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"]
        # Six peptides all tagged "shared class"; each with its own unique synonym.
        for i in range(6):
            rows.append(f"AP0000{i},Pep{i} (Uniq{i}; shared class),src,SEQ{i}"
                        + ",x" * 13)
        (adp / "adb.csv").write_text("\n".join(rows), encoding="utf-8")

        records = harvest_names(data_dir=str(tmp_path), vocabulary_threshold=5)
        names = {r.name for r in records}
        assert "Uniq0" in names, "a one-peptide token is a synonym"
        assert "shared class" not in names, "a token shared by 6 peptides is vocabulary"

    def test_organisms_are_not_peptide_names(self) -> None:
        # dbAMP files the source organism in the name column, so Xenopus laevis and the common
        # name Mouse-ear cress arrived as peptide names.
        from sample.names import is_organism_name

        assert is_organism_name("Xenopus laevis") is True
        assert is_organism_name("Magainin 2") is False

    def test_names_are_searchable(self, wired: str) -> None:
        from sample import db

        rows, total = db.search_peptides(query_str="magainin")
        assert total >= 1
        assert rows[0]["entry_id"] == "SAMPLE0000000001"

    def test_synonym_finds_the_peptide(self, wired: str) -> None:
        from sample import db

        rows, _ = db.search_peptides(query_str="mag-2")
        assert [r["entry_id"] for r in rows] == ["SAMPLE0000000001"]

    def test_one_row_per_peptide_despite_many_names(self, wired: str) -> None:
        # Matched through a subquery, not a join: a peptide with several names must not multiply.
        from sample import db

        rows, total = db.search_peptides(query_str="magainin")
        assert len(rows) == len({r["entry_id"] for r in rows})

    def test_peptide_carries_its_names(self, wired: str) -> None:
        from sample import db

        names = db.get_peptide("SAMPLE0000000001")["names"]
        assert names[0]["name_kind"] == "primary"
        assert {n["name"] for n in names} >= {"Magainin 2", "Mag-2"}

    def test_search_survives_a_database_without_the_table(self, tmp_path: Any,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        # Search is too central to fail wholesale over an optional table added by a later step.
        from sample import db

        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE peptides (entry_id TEXT PRIMARY KEY, sequence TEXT, SMILES TEXT,
                source TEXT, is_decoy INTEGER, LITERATURE TEXT, PATENT TEXT);
            CREATE TABLE activities (id INTEGER PRIMARY KEY, entry_id TEXT, activity_group TEXT,
                value REAL, unit TEXT, target_organism TEXT, activity_type TEXT,
                mechanism_of_action TEXT);
            CREATE TABLE papers (id INTEGER PRIMARY KEY, source_type TEXT, source_id TEXT);
            CREATE TABLE peptide_references (peptide_id TEXT, paper_id INTEGER);
            INSERT INTO peptides VALUES ('S1', 'GIGKFLH', NULL, NULL, 0, NULL, NULL);
        """)
        conn.commit()
        conn.close()
        monkeypatch.setattr(db, "default_db_path", lambda: path)
        rows, total = db.search_peptides(query_str="GIGK")
        assert total == 1


class TestStructureGeneration:
    """
    3D structure generation. The SMILES column is unvalidated upstream, and the previous version
    counted every molecule as a 3D success -- including ones that never embedded.
    """

    def test_optimization_reports_convergence_honestly(self) -> None:
        from rdkit import Chem, RDLogger

        from sample.enrichment import _embed_3d, _optimize_3d

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"))
        assert _embed_3d(mol) is True
        converged, field = _optimize_3d(mol)
        assert converged is True
        assert field in ("MMFF94", "UFF")
        assert mol.GetNumConformers() == 1

    def test_written_structure_is_genuinely_three_dimensional(self, tmp_path: Any) -> None:
        from rdkit import RDLogger

        from sample.enrichment import _generate_structure_for_row

        RDLogger.DisableLog("rdApp.*")
        # Built from the SEQUENCE. Step 27 (structures) takes no SMILES input at all: that column
        # is unvalidated, and every tool that consumed it turned a corrupt cell into a missing
        # descriptor or a failed structure.
        wrote_2d, converged = _generate_structure_for_row(("T1", "GIGKF", str(tmp_path)))
        assert wrote_2d and converged

        text = (tmp_path / "T1.sdf").read_text()
        assert "<Dimensionality>" in text and "3D" in text
        # A flat molecule would have every z at zero.
        zs = [float(line.split()[2]) for line in text.splitlines()[4:40]
              if len(line.split()) > 3 and line.split()[3].isalpha()]
        assert zs and any(abs(z) > 1e-6 for z in zs)

    def test_unbuildable_sequence_is_reported_not_counted(self, tmp_path: Any) -> None:
        from sample.enrichment import _generate_structure_for_row

        # A non-standard residue is a property of the peptide, not a data fault; it yields no
        # structure rather than a wrong one.
        assert _generate_structure_for_row(("T2", "GIGKFXZB", str(tmp_path))) == (False, False)
        assert _generate_structure_for_row(("T3", "", str(tmp_path))) == (False, False)


class TestLiteratureCitations:
    """
    SAMPLE-extracted measurements must be traceable to the paper they were read from.

    They were not: the extraction document's metadata carries only a ``source_file`` path, so
    ``pubmed_id`` and ``doi`` were empty for all 4,333 SAMPLE rows while the third-party rows
    beside them carried a PMID 87 % of the time. The identifiers were one join away the whole
    time -- the extraction filename is the ``internal_id`` the download step recorded.
    """

    def test_extraction_key_from_filename(self) -> None:
        from sample.merge import _extraction_key

        assert _extraction_key("/x/y/abc-123.json", {}) == "abc-123"

    def test_extraction_key_falls_back_to_source_file(self) -> None:
        from sample.merge import _extraction_key

        # A document renamed on the way through still carries the id in its metadata.
        assert _extraction_key("", {"source_file": "/t/def-456.txt"}) == "def-456"

    def test_index_maps_internal_id_to_publication(self, tmp_path: Any) -> None:
        from sample.merge import _literature_index

        (tmp_path / "pubmed_downloaded.csv").write_text(
            "source,title,doi,pmid,link,internal_id,success,pdf_path\n"
            "PubMed,A Paper,10.1/abc,123456,http://x,uuid-1,True,/p.pdf\n",
            encoding="utf-8")
        index = _literature_index(str(tmp_path))
        assert index["uuid-1"]["doi"] == "10.1/abc"
        assert index["uuid-1"]["title"] == "A Paper"

    def test_index_tolerates_missing_and_malformed_files(self, tmp_path: Any) -> None:
        from sample.merge import _literature_index

        # A directory with nothing usable must yield an empty index, not an exception: the merge
        # has to run on a checkout where the literature step has not been executed.
        (tmp_path / "pubmed_downloaded.csv").write_text("no,useful,columns\n1,2,3\n",
                                                        encoding="utf-8")
        assert _literature_index(str(tmp_path)) == {}

    def test_blank_values_do_not_overwrite_real_ones(self, tmp_path: Any) -> None:
        from sample.merge import _literature_index

        # The metadata sweep and the download log both describe the same document; whichever
        # carries a value should win over whichever carries a blank.
        (tmp_path / "pubmed_metadata.csv").write_text(
            "source,title,doi,pmid,link,internal_id\n"
            "PubMed,Real Title,,,http://x,uuid-1\n", encoding="utf-8")
        (tmp_path / "pubmed_downloaded.csv").write_text(
            "source,title,doi,pmid,link,internal_id,success,pdf_path\n"
            "PubMed,,10.1/abc,,http://x,uuid-1,True,/p.pdf\n", encoding="utf-8")
        index = _literature_index(str(tmp_path))
        assert index["uuid-1"]["title"] == "Real Title"
        assert index["uuid-1"]["doi"] == "10.1/abc"


class TestFolding:
    """
    ColabFold structure prediction and the SDF/SMILES derived from it.

    The prediction itself needs a GPU and AlphaFold2 weights, so what is tested here is everything
    around it: which sequences are eligible, how confidence is read back, and the PDB -> SDF ->
    SMILES conversion, which is what replaces the unvalidated SMILES column.
    """

    @pytest.mark.parametrize("sequence,foldable", [
        ("GIGKFLHSAKKFGKAFVGEIMNS", True),
        ("ACDEFGHIKLMNPQRSTVWY", True),
        ("", False),
        ("A", False),
        ("GIGKFLHSAXKFGKA", False),          # X has no AlphaFold2 representation
        ("GIGKFLHSA-KKFGKA", False),         # modification separators
        ("A" * 401, False),                  # past the memory ceiling on a 12 GB card
        ("A" * 15, False),                   # short peptides are distance geometry's job
        ("A" * 16, True),
    ])
    def test_is_foldable(self, sequence: str, foldable: bool) -> None:
        from sample.folding import is_foldable

        assert is_foldable(sequence) is foldable

    def test_structure_routing_is_exhaustive_and_ordered(self) -> None:
        """
        ColabFold by default, RDKit where it is applicable, skip otherwise.

        The boundary is 15 residues: below it a peptide has no tertiary structure to predict, and
        AlphaFold2 in single-sequence mode returns low-confidence output regardless -- a 2-residue
        trial produced residues 30 A apart. Routing short peptides straight to distance geometry
        avoids spending GPU on predictions that geometric validation would discard.
        """
        from sample.enrichment import molecule_from_sequence
        from sample.folding import SHORT_PEPTIDE_LENGTH, is_foldable

        def route(sequence: str) -> str:
            if is_foldable(sequence):
                return "colabfold"
            if (len(sequence) <= SHORT_PEPTIDE_LENGTH
                    and molecule_from_sequence(sequence) is not None):
                return "rdkit"
            return "skip"

        assert route("GIGKFLHSAKKFGKAFVGEIMNS") == "colabfold"   # 23 residues
        assert route("GIGKFLHSAKKFGK") == "rdkit"                # 14 residues
        assert route("GIGKF") == "rdkit"                         # very short
        assert route("A" * 500) == "skip"                        # past the memory ceiling
        assert route("GIGKFLHSAXKFGKAFVGEIMNS") == "skip"        # non-standard, too long for RDKit
        assert route("GIGKFXZ") == "skip"                        # non-standard, unbuildable

    def test_plddt_is_read_from_the_b_factor_column(self, tmp_path: Any) -> None:
        # AlphaFold2 writes per-residue confidence into the B-factor column.
        from sample.folding import _plddt_from_pdb

        pdb = tmp_path / "x.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 80.00           N\n"
            "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00 90.00           C\n"
            "ATOM      3  CA  GLY A   2      12.339   7.071  -4.147  1.00 70.00           C\n"
            "END\n", encoding="utf-8")
        assert _plddt_from_pdb(str(pdb)) == pytest.approx(80.0)

    def test_plddt_missing_file(self) -> None:
        from sample.folding import _plddt_from_pdb

        assert _plddt_from_pdb("/nonexistent/x.pdb") is None

    def test_derives_sdf_and_smiles_from_a_structure(self, tmp_path: Any) -> None:
        # A real (if tiny) glycine PDB: the point is that the SMILES comes from the coordinates,
        # not from the unvalidated SMILES column.
        from sample.folding import derive_sdf_and_smiles

        pdb = tmp_path / "P1.pdb"
        pdb.write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 85.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 85.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00 85.00           C\n"
            "ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00 85.00           O\n"
            "END\n", encoding="utf-8")

        record = derive_sdf_and_smiles("P1", "G", str(tmp_path))
        assert record is not None
        assert record["smiles"]
        assert record["plddt"] == pytest.approx(85.0)
        assert os.path.exists(record["sdf"])

        text = open(record["sdf"], encoding="utf-8").read()
        assert "<pLDDT>" in text
        assert "ColabFold" in text

    def test_smiles_is_connected_not_a_pile_of_atoms(self, tmp_path: Any) -> None:
        # Regression: reading a predicted PDB with proximityBonding disabled parses fine and
        # returns every atom isolated, yielding "C.C.C.C..." -- syntactically valid, chemically
        # meaningless, and silent. A peptide SMILES must be one connected fragment.
        from rdkit import Chem, RDLogger

        from sample.folding import derive_sdf_and_smiles

        RDLogger.DisableLog("rdApp.*")
        pdb = tmp_path / "P2.pdb"
        pdb.write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 85.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 85.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00 85.00           C\n"
            "ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00 85.00           O\n"
            "END\n", encoding="utf-8")

        record = derive_sdf_and_smiles("P2", "G", str(tmp_path))
        assert record is not None
        assert "." not in record["smiles"], "disconnected fragments mean bonds were never perceived"
        assert Chem.MolFromSmiles(record["smiles"]) is not None

    def test_sequence_is_authoritative_over_structure_perception(self, tmp_path: Any) -> None:
        # Bond perception over a predicted structure is geometric and occasionally puts a third
        # bond on a terminal oxygen. The sequence builds the same peptide with correct valences,
        # so it wins wherever it is usable.
        from rdkit import Chem

        from sample.folding import _smiles_for

        expected = Chem.MolToSmiles(Chem.MolFromSequence("GIGKF"))
        assert _smiles_for(None, "GIGKF") == expected

    def test_non_standard_sequence_falls_back_to_the_structure(self, tmp_path: Any) -> None:
        from rdkit import Chem

        from sample.folding import _smiles_for

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        assert _smiles_for(mol, "XZB") == "CCO"

    def test_absent_structure_yields_nothing(self, tmp_path: Any) -> None:
        from sample.folding import derive_sdf_and_smiles

        assert derive_sdf_and_smiles("missing", "G", str(tmp_path)) is None

    def test_folded_entries_are_detected(self, tmp_path: Any) -> None:
        # The distance-geometry fallback consults this so it never overwrites a prediction.
        from sample.folding import folded_entry_ids

        (tmp_path / "A1.pdb").write_text("END\n", encoding="utf-8")
        (tmp_path / "A2.sdf").write_text("", encoding="utf-8")
        assert folded_entry_ids(str(tmp_path)) == {"A1"}

    def test_folded_entries_on_missing_directory(self) -> None:
        from sample.folding import folded_entry_ids

        assert folded_entry_ids("/nonexistent/dir") == set()

    def test_fold_refuses_without_colabfold(self, tmp_path: Any,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        # Refusing beats silently producing distance-geometry conformers under a name that
        # promises a predicted fold.
        from sample import folding

        monkeypatch.setattr(folding, "colabfold_command", lambda: None)
        with pytest.raises(RuntimeError, match="colabfold_batch"):
            folding.fold_sequences([("A1", "GIGKFLHSA")], structure_dir=str(tmp_path))

    def test_rdkit_fallback_skips_predicted_entries(self, tmp_path: Any) -> None:
        import pandas as pd

        from sample.enrichment import enrich_with_2D_and_3D_structures

        frame = pd.DataFrame([
            {"entry_id": "KEEP", "sequence": "GIGKF"},
            {"entry_id": "SKIP", "sequence": "GIGKF"},
        ])
        enrich_with_2D_and_3D_structures(frame, str(tmp_path), skip_entry_ids={"SKIP"})
        assert os.path.exists(tmp_path / "KEEP.sdf")
        assert not os.path.exists(tmp_path / "SKIP.sdf")


class TestSequenceOnlyDescriptors:
    """
    The enrichment pipeline (steps 21-27) takes the sequence and nothing else.

    Every tool in the step used to consume the stored SMILES, which is unvalidated: a corrupt cell
    became a missing descriptor, a failed structure and several lines of RDKit output. Measured
    over the release, the sequence yields descriptors for **100 %** of rows against 98.7 % for the
    SMILES column -- so dropping it costs nothing and removes the whole failure class.
    """

    def test_descriptors_ignore_a_junk_smiles_column(self) -> None:
        import pandas as pd

        from sample.enrichment import enrich_with_descriptors

        frame = pd.DataFrame([
            {"entry_id": "A", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "SMILES": "not"},
            {"entry_id": "B", "sequence": "KWKLFKKIGAVLKVL",
             "SMILES": "ACYCRIPACIAGERRYGTCIYQGRLWAICC"},
        ])
        out = enrich_with_descriptors(frame)
        # Magainin-2's molecular weight, from the sequence, despite the SMILES being the word
        # "not".
        assert out.loc[0, "molecular_weight"] == pytest.approx(2466.9, abs=0.5)
        assert out.loc[1, "molecular_weight"] > 0

    def test_peptide_descriptors_are_present(self) -> None:
        from sample.enrichment import PEPTIDE_DESCRIPTORS, peptide_descriptors

        values = peptide_descriptors("GIGKFLHSAKKFGKAFVGEIMNS")
        for column in PEPTIDE_DESCRIPTORS:
            assert column in values, column
        # Magainin-2 is strongly cationic and amphipathic; these are the properties that
        # characterise an antimicrobial peptide, and they are defined on the sequence.
        assert values["isoelectric_point"] > 9
        assert values["net_charge_ph7"] > 2
        assert values["hydrophobic_moment"] > 0.3

    def test_non_standard_residues_still_yield_peptide_descriptors(self) -> None:
        # peptides.py works from residue composition, so it answers where RDKit cannot build a
        # molecule at all -- which is why molecular weight is taken from it.
        from sample.enrichment import molecule_from_sequence, peptide_descriptors

        assert molecule_from_sequence("GIGKFLHSAXKFGKA") is None
        assert peptide_descriptors("GIGKFLHSAXKFGKA")["molecular_weight"] > 0

    def test_empty_sequence_yields_nothing(self) -> None:
        from sample.enrichment import peptide_descriptors

        assert peptide_descriptors("") == {}
        assert peptide_descriptors(None) == {}


class TestSynergy:
    def test_fici_bands(self) -> None:
        from sample.synergy import classify_fici

        assert classify_fici(0.3) == "synergy"
        assert classify_fici(0.5) == "synergy"       # the cutoff is inclusive
        assert classify_fici(0.75) == "additive"
        assert classify_fici(2.0) == "indifferent"
        assert classify_fici(8.0) == "antagonism"
        assert classify_fici(None) is None

    @pytest.mark.parametrize("raw,expected", [
        ("0.28", 0.28), ("<0.5", 0.5), (">128", 128.0), ("1,5", 1.5),
        ("", None), ("NA", None), (None, None), ("not a number", None),
    ])
    def test_measurement_parsing(self, raw: Any, expected: Any) -> None:
        from sample.synergy import parse_measurement

        assert parse_measurement(raw) == expected

    def test_strain_labels_collapse_to_species(self) -> None:
        # Without this no pathogen has enough measurements to train on: 514 strain labels are
        # really a few dozen species.
        from sample.synergy import normalise_pathogen

        assert normalise_pathogen("Escherichia coli ATCC 25922") == "Escherichia coli"
        assert normalise_pathogen("Pseudomonas aeruginosa Ps4") == "Pseudomonas aeruginosa"
        assert normalise_pathogen("  Staphylococcus   aureus  ") == "Staphylococcus aureus"
        assert normalise_pathogen("") == ""

    def test_recomputed_fici_matches_its_components(self) -> None:
        from sample.synergy import recompute_fici

        record = {"peptide_alone": 8.0, "peptide_combined": 2.0,
                  "antibiotic_alone": 32.0, "antibiotic_combined": 1.0}
        assert recompute_fici(record) == pytest.approx(0.28125)

    @pytest.mark.parametrize("record", [
        {"peptide_alone": None, "peptide_combined": 2.0, "antibiotic_alone": 32.0,
         "antibiotic_combined": 1.0},
        {"peptide_alone": 0.0, "peptide_combined": 2.0, "antibiotic_alone": 32.0,
         "antibiotic_combined": 1.0},
    ])
    def test_recompute_declines_on_missing_or_zero(self, record: dict) -> None:
        from sample.synergy import recompute_fici

        assert recompute_fici(record) is None

    def test_no_models_raises_rather_than_guessing(self, tmp_path: Any) -> None:
        from sample.synergy import predict_synergy

        with pytest.raises(FileNotFoundError):
            predict_synergy(["GIGKFLHSA"], "Rifampicin", model_dir=str(tmp_path))

    def test_available_models_on_empty_dir(self, tmp_path: Any) -> None:
        from sample.synergy import available_models

        assert available_models(model_dir=str(tmp_path)) == []


class TestTrainSynergyModelsRigor:
    """
    ``train_synergy_models`` used to fit a RandomForest with ``cross_val_predict`` over the whole
    dataset (no fixed holdout) and write every model unconditionally. It now goes through
    :func:`sample.ml_models.fit_and_validate` -- the same validated-fit core steps 45-46's own models
    use -- so it should carry the same holdout/CV/bootstrap/y-randomization/quality-gate record
    shape, and a model with no real signal should be rejected rather than written.

    ``build_dataframe`` and ``featurise`` are both monkeypatched: the former to avoid depending on
    real DBAASP records, the latter to avoid a PubChem network dependency for antibiotic
    fingerprints (blocked by the autouse ``no_network`` fixture anyway) and to substitute a feature
    matrix this test fully controls, so the learnable-vs-unlearnable cases are unambiguous.
    """

    def _synthetic_frame(self, rng: Any, signal_strength: float) -> Any:
        import numpy as np
        import pandas as pd

        aminos = list("ACDEFGHIKLMNPQRSTVWY")
        n_peptides = 60
        seqs = ["".join(rng.choice(aminos, 15)) for _ in range(n_peptides)]
        rows = []
        for seq in seqs:
            label = int(rng.integers(0, 2))
            for antibiotic in ("Gentamicin", "Rifampicin", "Ciprofloxacin"):
                rows.append({
                    "sequence": seq, "antibiotic": antibiotic,
                    "pathogen": "Escherichia coli", "target_organism": "Escherichia coli",
                    "is_synergistic": label,
                    "fici": 0.2 if label else 2.0,
                    "_signal": label * signal_strength,
                })
        return pd.DataFrame(rows)

    def _patch_featurise(self, monkeypatch: Any, rng: Any) -> None:
        import numpy as np
        import pandas as pd

        import sample.synergy as S

        def fake_featurise(subset: Any, antibiotics: Any = None, use_network: bool = True) -> Any:
            n = len(subset)
            noise = rng.normal(size=(n, 3))
            informative = subset["_signal"].to_numpy().reshape(-1, 1)
            X = pd.DataFrame(np.hstack([informative, noise]),
                             columns=["signal", "n0", "n1", "n2"], index=subset.index)
            return X, list(X.columns)

        monkeypatch.setattr(S, "featurise", fake_featurise)

    def test_learnable_signal_passes_gate_and_carries_the_full_validation_record(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        import numpy as np

        import sample.synergy as S

        rng = np.random.default_rng(1)
        frame = self._synthetic_frame(rng, signal_strength=5.0)
        monkeypatch.setattr(S, "build_dataframe", lambda: frame)
        self._patch_featurise(monkeypatch, rng)

        report = S.train_synergy_models(
            model_dir=str(tmp_path / "models"), report_path=str(tmp_path / "report.json"),
            min_rows=50)

        assert report, "expected at least one trained/evaluated entry"
        pooled = report.get("synergy/general/all_pathogens/fici/classification")
        assert pooled is not None
        for key in ("holdout", "holdout_ci95", "y_randomization", "null_comparison",
                   "quality_passed", "quality_failures"):
            assert key in pooled, f"missing {key!r} in {pooled.keys()}"
        # The informative feature is proportional to the label itself, so this should clear the
        # quality gate and land on disk.
        assert pooled["quality_passed"] is True
        assert os.path.exists(os.path.join(
            str(tmp_path / "models"), "synergy", "general", "all_pathogens",
            "fici", "classification", "model.pkl"))

        pooled_reg = report.get("synergy/general/all_pathogens/fici/regression")
        assert pooled_reg is not None
        assert "classification_at_cutoff" in pooled_reg

    def test_no_signal_is_rejected_and_not_written_to_disk(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        import numpy as np

        import sample.synergy as S

        rng = np.random.default_rng(2)
        frame = self._synthetic_frame(rng, signal_strength=5.0)
        monkeypatch.setattr(S, "build_dataframe", lambda: frame)

        # Feature matrix carries no relationship to the label at all -- pure noise.
        import pandas as pd

        def unlearnable_featurise(subset: Any, antibiotics: Any = None,
                                  use_network: bool = True) -> Any:
            n = len(subset)
            X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["n0", "n1", "n2", "n3"],
                             index=subset.index)
            return X, list(X.columns)

        monkeypatch.setattr(S, "featurise", unlearnable_featurise)

        model_dir = tmp_path / "models"
        report = S.train_synergy_models(
            model_dir=str(model_dir), report_path=str(tmp_path / "report.json"), min_rows=50)

        pooled = report.get("synergy/general/all_pathogens/fici/classification")
        assert pooled is not None
        assert pooled["quality_passed"] is False
        assert pooled["quality_failures"]
        assert not os.path.exists(os.path.join(
            str(model_dir), "synergy", "general", "all_pathogens", "fici", "classification",
            "model.pkl"))

    def test_grouping_uses_near_identity_clusters_not_exact_sequence(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # A thin spy on ml_models._cluster_groups rather than re-deriving its behaviour here:
        # this test's job is to confirm the grouping call site actually changed, not to
        # re-validate near-identity clustering itself (sample.families already has its own tests).
        import numpy as np

        import sample.ml_models as M
        import sample.synergy as S

        rng = np.random.default_rng(3)
        frame = self._synthetic_frame(rng, signal_strength=5.0)
        monkeypatch.setattr(S, "build_dataframe", lambda: frame)
        self._patch_featurise(monkeypatch, rng)

        calls = []
        real_cluster_groups = M._cluster_groups

        def spy(df: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs.get("work_dir"))
            return real_cluster_groups(df, *args, **kwargs)

        monkeypatch.setattr(M, "_cluster_groups", spy)

        S.train_synergy_models(model_dir=str(tmp_path / "models"),
                               report_path=str(tmp_path / "report.json"), min_rows=50)

        assert calls, "expected sample.ml_models._cluster_groups to be called"
        assert all(w and "synergy_family_scratch" in w for w in calls), (
            "expected an isolated work_dir distinct from step 35's/ml_models.py's own default")


class TestBibliographyScope:
    """
    `linked_only` decides how much network a bibliography rebuild costs.

    The corpus harvester contributes 154,386 of the 218,192 harvest rows and keys them by DOI, so
    a naive "has an accession" test marks every one of the 94,564 references as linked and saves
    nothing -- which is exactly the bug these tests pin. Only the four database harvesters emit an
    accession that reaches a peptide.
    """

    def test_corpus_harvester_is_excluded_from_the_linked_set(self) -> None:
        from sample.references import CORPUS_HARVESTERS, HARVESTERS

        assert "SAMPLE" in CORPUS_HARVESTERS
        assert CORPUS_HARVESTERS < set(HARVESTERS), "corpus harvesters must be a strict subset"
        assert {"DBAASP", "DRAMP", "ADP", "CAMP"}.isdisjoint(CORPUS_HARVESTERS), (
            "the database harvesters are the ones whose accessions reach a peptide")

    def test_enrich_passes_unlinked_references_through_untouched(self) -> None:
        """Restricting the network must never drop a reference from the bibliography."""
        from sample.crossref import enrich
        from sample.references import Reference

        refs = {"linked": Reference(pmid="1", title="Linked"),
                "corpus": Reference(doi="10.1/corpus", title="Corpus")}
        out = enrich(refs, resolve_titles=False, keys=set())

        assert set(out) == set(refs), "every key must survive"
        assert out["corpus"].doi == "10.1/corpus"
        assert out["corpus"].title == "Corpus"

    def test_backfill_pmids_honours_the_key_filter(self) -> None:
        from sample.crossref import backfill_pmids
        from sample.references import Reference

        refs = {"corpus": Reference(doi="10.1/corpus")}
        out = backfill_pmids(refs, keys=set())

        assert set(out) == {"corpus"}
        assert out["corpus"].doi == "10.1/corpus"


class TestPredictorEnrichmentWiring:
    """
    `enrich_with_predictors` was written and tested but had no caller, so its ~135
    `pep_*`/`stab_*`/`hemopi2_*` columns reached no released file. Wired into the monolithic
    `sample-cli enrich` (real peptides; split into steps 21-27 today) -- `sample-cli assembly`
    (decoys) deliberately does NOT call it (removed 2026-08-26): decoys are ML-training negatives
    never shown as release records, and no feature block in `sample.ml_models` reads a predictor
    column, so running it there bought nothing.
    """

    def test_columns_are_added_without_external_tools(self) -> None:
        """The always-available half: peptides.py + BioPython, no shelling out."""
        import pandas as pd

        from sample.peptide_predictors import enrich_with_predictors

        df = pd.DataFrame({"entry_id": ["A", "B"],
                           "sequence": ["GIGKFLHSAKKFGKAFVGEIMNS", "MAGAININ"]})
        out = enrich_with_predictors(df, run_external=False)

        assert len(out.columns) > len(df.columns)
        assert any(c.startswith("pep_") for c in out.columns)
        assert any(c.startswith("stab_") for c in out.columns)
        # Original columns and row count are untouched, only appended to.
        assert len(out) == len(df)
        assert set(df.columns) <= set(out.columns)

    def test_missing_sequence_column_degrades_rather_than_raising(self) -> None:
        import pandas as pd

        from sample.peptide_predictors import enrich_with_predictors

        df = pd.DataFrame({"entry_id": ["A"]})
        out = enrich_with_predictors(df, run_external=False)

        assert list(out.columns) == ["entry_id"], "no sequence column -> nothing to add"

    def test_cli_exposes_skip_flags_on_enrich_only(self) -> None:
        """
        A typo or removal here would silently stop the monolithic `enrich` command from skipping.

        `assembly` deliberately does NOT have these flags (removed 2026-08-26): decoys are
        synthetic ML-training negatives, never shown as release records, and
        `sample.ml_models.feature_blocks` never reads a predictor column as a training feature --
        so running the external predictors over the whole decoy set bought nothing. Only `enrich`
        (and its split-out per-predictor steps) still needs this control.

        `cli.main()` builds its `ArgumentParser` internally rather than exposing one, so the
        flags are checked by grepping the module source -- which is exactly what breaks if either
        argument is renamed or dropped.
        """
        import inspect

        from sample import cli

        source = inspect.getsource(cli)
        assert source.count('"--skip-predictors"') == 1, (
            "expected exactly one registration, on the enrich subparser")
        assert source.count('"--skip-external-predictors"') == 1



class TestCrossReferenceIdentifiers:
    """
    UniProt/PDB accessions were resolved into `activities.csv`'s `cross_references` column during
    step 19, but step 20 onward reads a different lineage (`merged_from_database.csv`) that never
    carries the column, so a successfully-resolved accession reached no released file. See
    `sample/identifiers.py`.
    """

    def test_harvest_restricts_to_identity_bearing_rows(self, tmp_path: Any) -> None:
        import pandas as pd

        from sample.identifiers import harvest_cross_references

        path = str(tmp_path / "activities.csv")
        pd.DataFrame([
            {"sequence": "AAA", "cross_references": "uniprot:P81941"},
            {"sequence": "AAA", "cross_references": "uniprot:P81941"},  # duplicate row, same seq
            {"sequence": "BBB", "cross_references": "pdb:1MAG"},
            {"sequence": "CCC", "cross_references": "pubchem:12345"},   # no uniprot/pdb
            {"sequence": "DDD", "cross_references": ""},
            {"sequence": "EEE", "cross_references": None},
        ]).to_csv(path, index=False)

        refs = harvest_cross_references(path)

        assert set(refs) == {"AAA", "BBB"}, "only sequences carrying uniprot or pdb are kept"
        assert refs["AAA"] == "uniprot:P81941"

    def test_page_link_excludes_assay_archive_namespaces(self) -> None:
        """PubChem/ChEMBL belong to the Assay Archives panel, not this one, to avoid double
        presenting the same accession as two independent confirmations."""
        from sample.api.pages import _identity_links

        links = _identity_links("uniprot:P81941 | pdb:1MAG | pubchem:12345 | chembl:CHEMBL1")

        namespaces = {link["source_type"] for link in links}
        assert namespaces == {"uniprot", "pdb"}

    def test_no_cross_references_yields_no_links(self) -> None:
        from sample.api.pages import _identity_links

        assert _identity_links(None) == []
        assert _identity_links("") == []

    def test_loader_updates_only_matching_sequences(self, tmp_path: Any) -> None:
        import json
        import sqlite3

        from sample.serialize import _load_cross_references

        refs_path = str(tmp_path / "cross_references.json")
        with open(refs_path, "w", encoding="utf-8") as fh:
            json.dump({"cross_references": {"AAA": "uniprot:P81941", "ZZZ": "pdb:9XYZ"}}, fh)

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE peptides (entry_id TEXT PRIMARY KEY, sequence TEXT, "
                     "cross_references TEXT)")
        conn.execute("INSERT INTO peptides VALUES ('E1', 'AAA', NULL)")
        conn.commit()

        n = _load_cross_references(conn.cursor(), {"AAA": "E1"}, refs_json=refs_path)

        assert n == 1, "ZZZ has no matching entry_id and must not be counted"
        assert conn.execute(
            "SELECT cross_references FROM peptides WHERE entry_id='E1'").fetchone()[0] == \
            "uniprot:P81941"




class TestSecondaryStructurePanel:
    """
    Replaces the RDKit 2D chemical-graph depiction: a bond diagram of a 30-residue peptide is
    unreadable at any size and answers nothing this page's readers are asking. See
    `sample/secondary_structure.py`.
    """

    def test_classic_helix_former_classifies_as_helix(self) -> None:
        from sample.secondary_structure import residue_states

        assert set(residue_states("EAEALAKALEALAK")) == {"H"}

    def test_classic_sheet_former_classifies_as_sheet(self) -> None:
        from sample.secondary_structure import residue_states

        assert set(residue_states("VIFVTIFVTVIFVT")) == {"E"}

    def test_summary_fractions_sum_to_one(self) -> None:
        from sample.secondary_structure import summarize

        s = summarize("GIGKFLHSAKKFGKAFVGEIMNS")
        assert abs(sum(s.values()) - 1.0) < 1e-9

    def test_empty_sequence_degrades_rather_than_raising(self) -> None:
        from sample.secondary_structure import residue_states, summarize

        assert residue_states("") == []
        assert summarize("") == {"helix": 0.0, "sheet": 0.0, "coil": 0.0}

    def test_unknown_residue_does_not_crash_and_stays_neutral(self) -> None:
        """A modified/non-standard residue letter must not raise a KeyError."""
        from sample.secondary_structure import residue_states

        states = residue_states("AAAXAAA")
        assert len(states) == 7

    def test_page_helper_pairs_letters_with_states(self) -> None:
        from sample.api.pages import _secondary_structure

        result = _secondary_structure("EAEALAKALEALAK")
        assert result is not None
        assert result["residues"][0] == ("E", "H")
        assert len(result["residues"]) == 14

    def test_page_helper_handles_no_sequence(self) -> None:
        from sample.api.pages import _secondary_structure

        assert _secondary_structure(None) is None
        assert _secondary_structure("") is None

    def test_rdkit_2d_is_fully_removed(self) -> None:
        """No caller of the RDKit chemical-graph renderer should remain anywhere in the app."""
        import inspect

        from sample.api import common, pages

        assert "smiles_to_svg" not in inspect.getsource(common)
        assert "smiles_to_svg" not in inspect.getsource(pages)




class TestGeneOntology:
    """
    CAMP scrapes UniProt's GO annotation into a flattened, undelimited text cell; nothing reads
    it. See `sample/gene_ontology.py` for why a regex is the right tool here (the category and
    evidence-code vocabularies are both small and closed, which is what makes the boundary
    between one term and the next findable in unstructured text) and why Pfam/InterPro are
    deliberately not harvested the same way (already covered, more reliably, by the curated
    HMMER family assignment).
    """

    def test_parses_a_real_camp_style_blob(self) -> None:
        from sample.gene_ontology import parse_go_field

        blob = ("GO ID  Ontology  Definition  Evidence  "
                "GO:0005576  Cellular component  Extracellular region  IDA  "
                "GO:0050829  Biological process  Defense response to Gram-negative bacterium  IDA")
        terms = parse_go_field(blob)

        assert terms == [
            {"go_id": "GO:0005576", "category": "Cellular component",
             "term": "Extracellular region", "evidence": "IDA"},
            {"go_id": "GO:0050829", "category": "Biological process",
             "term": "Defense response to Gram-negative bacterium", "evidence": "IDA"},
        ]

    def test_parses_a_term_with_no_category(self) -> None:
        """~29 % of observed CAMP rows omit the category word entirely."""
        from sample.gene_ontology import parse_go_field

        terms = parse_go_field("GO:0005737  Cytoplasm  ISS")
        assert terms == [{"go_id": "GO:0005737", "category": None,
                          "term": "Cytoplasm", "evidence": "ISS"}]

    def test_header_only_blob_yields_nothing(self) -> None:
        from sample.gene_ontology import parse_go_field

        assert parse_go_field("GO ID  Ontology  Definition  Evidence") == []

    def test_empty_and_none_are_handled(self) -> None:
        from sample.gene_ontology import parse_go_field

        assert parse_go_field("") == []
        assert parse_go_field(None) == []

    def test_harvest_keys_by_sequence_and_deduplicates(self, tmp_path: Any) -> None:
        from sample.gene_ontology import harvest_gene_ontology

        camp_dir = tmp_path / "camp"
        camp_dir.mkdir()
        (camp_dir / "CAMPSQ1.csv").write_text(
            "key,value\n"
            "Sequence,GIGKFLK\n"
            "Gene Ontology,GO ID  Ontology  Definition  Evidence  "
            "GO:0005576  Cellular component  Extracellular region  IDA\n",
            encoding="utf-8")
        # Same sequence republished under a second accession, with an overlapping term (must not
        # duplicate) and a new one (must be added to the same bucket).
        (camp_dir / "CAMPSQ2.csv").write_text(
            "key,value\n"
            "Sequence,gigkflk\n"
            "Gene Ontology,GO ID  Ontology  Definition  Evidence  "
            "GO:0005576  Cellular component  Extracellular region  IDA  "
            "GO:0050829  Biological process  Defense response  IDA\n",
            encoding="utf-8")

        result = harvest_gene_ontology(str(camp_dir))

        assert list(result.keys()) == ["GIGKFLK"]
        assert len(result["GIGKFLK"]) == 2, "the repeated term must not be duplicated"

    def test_record_with_no_sequence_is_skipped(self, tmp_path: Any) -> None:
        from sample.gene_ontology import harvest_gene_ontology

        camp_dir = tmp_path / "camp"
        camp_dir.mkdir()
        (camp_dir / "CAMPSQ1.csv").write_text(
            "key,value\n"
            "Gene Ontology,GO ID  Ontology  Definition  Evidence  "
            "GO:0005576  Cellular component  Extracellular region  IDA\n",
            encoding="utf-8")

        assert harvest_gene_ontology(str(camp_dir)) == {}


class TestGeneOntologyPageGrouping:
    def test_groups_by_category_in_a_fixed_order(self) -> None:
        from sample.api.pages import _go_by_category

        terms = [
            {"go_id": "GO:1", "category": "Molecular function", "term": "A", "evidence": "IEA"},
            {"go_id": "GO:2", "category": "Biological process", "term": "B", "evidence": "IEA"},
            {"go_id": "GO:3", "category": None, "term": "C", "evidence": "IEA"},
        ]
        groups = _go_by_category(terms)

        assert [g["label"] for g in groups] == \
            ["Biological Process", "Molecular Function", "Uncategorised"]

    def test_caps_each_category_and_reports_the_remainder(self) -> None:
        from sample.api.pages import _go_by_category, GO_TERMS_PER_CATEGORY

        terms = [{"go_id": f"GO:{i}", "category": "Biological process",
                 "term": f"term-{i}", "evidence": "IEA"} for i in range(GO_TERMS_PER_CATEGORY + 3)]
        groups = _go_by_category(terms)

        assert len(groups) == 1
        assert groups[0]["shown"] == GO_TERMS_PER_CATEGORY
        assert groups[0]["total"] == GO_TERMS_PER_CATEGORY + 3
        assert len(groups[0]["terms"]) == GO_TERMS_PER_CATEGORY

    def test_no_terms_yields_no_groups(self) -> None:
        from sample.api.pages import _go_by_category

        assert _go_by_category([]) == []





class TestSmilesValidation:
    """
    Restores the garbage-detection logic removed as dead code when step 27 became sequence-only
    -- it is needed again, but for a different job: cleaning the *published* SMILES column rather
    than prefiltering step 27's input. See `sample/smiles_validation.py`.
    """

    @pytest.mark.parametrize("smiles,plausible", [
        ("CCO", True),
        ("N[C@@H](C)C(=O)O", True),                 # (C) is a methyl branch, not a residue leak
        ("N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", True),
        ("not", False), ("Not available", False), ("unknown", False), ("", False),
        ("C", False),                                # methane: parses, but is not a peptide
        ("C(=O)C(G)N", False),                       # residue letter left in place of a structure
        ("CCO<channel|>[{", False),                  # model special tokens
        ("CC(C)CC(C=O)NC(=O)...", False),            # truncated generation
        ("ACYCRIPACIAGERRYGTCIYQGRLWAICC", False),   # the peptide's own sequence pasted in
        ("CCCC", True), ("NCCO", True), ("c1ccccc1", True),
    ])
    def test_prefilter(self, smiles: str, plausible: bool) -> None:
        from sample.smiles_validation import is_plausible_smiles

        assert is_plausible_smiles(smiles) is plausible

    def test_derived_structure_wins_over_a_plausible_stored_value(self, tmp_path: Any) -> None:
        """Sequence-derived beats even a stored value that would pass the prefilter -- it is
        guaranteed round-trippable and connected; the stored one is not guaranteed either."""
        import json

        import pandas as pd

        from sample.smiles_validation import validate_smiles_column

        structures_path = str(tmp_path / "predicted_structures.json")
        with open(structures_path, "w", encoding="utf-8") as fh:
            json.dump({"E1": {"smiles": "CC(N)C(=O)O"}}, fh)

        df = pd.DataFrame({"entry_id": ["E1"], "SMILES": ["CCO"]})
        summary = validate_smiles_column(df, predicted_structures_json=structures_path)

        assert df["SMILES"].iloc[0] == "CC(N)C(=O)O"
        assert summary["replaced_from_structure"] == 1

    def test_garbage_is_nulled_not_replaced_with_a_guess(self, tmp_path: Any) -> None:
        import pandas as pd

        from sample.smiles_validation import validate_smiles_column

        df = pd.DataFrame({"entry_id": ["E1"], "SMILES": ["Not available"]})
        summary = validate_smiles_column(
            df, predicted_structures_json=str(tmp_path / "missing.json"))

        assert df["SMILES"].iloc[0] == ""
        assert summary["rejected"] == 1

    def test_plausible_value_survives_with_no_derived_structure(self, tmp_path: Any) -> None:
        import pandas as pd

        from sample.smiles_validation import validate_smiles_column

        df = pd.DataFrame({"entry_id": ["E1"], "SMILES": ["CCO"]})
        summary = validate_smiles_column(
            df, predicted_structures_json=str(tmp_path / "missing.json"))

        assert df["SMILES"].iloc[0] == "CCO"
        assert summary["kept"] == 1

    def test_already_empty_is_counted_separately_from_rejected(self, tmp_path: Any) -> None:
        """The row was never publishing a lie -- it should not be reported as a catch."""
        import pandas as pd

        from sample.smiles_validation import validate_smiles_column

        df = pd.DataFrame({"entry_id": ["E1"], "SMILES": [""]})
        summary = validate_smiles_column(
            df, predicted_structures_json=str(tmp_path / "missing.json"))

        assert summary["already_empty"] == 1
        assert summary["rejected"] == 0

    def test_missing_predicted_structures_file_degrades_to_validation_only(
            self, tmp_path: Any) -> None:
        from sample.smiles_validation import load_derived_smiles

        assert load_derived_smiles(str(tmp_path / "does_not_exist.json")) == {}





class TestFamilyHMMs:
    """
    Profile HMMs for SAMPLE's own derived family clusters (not Pfam-curated ones, which already
    have a real HMM from Pfam-A itself). See `sample/family_hmms.py`.
    """

    def test_clustered_families_excludes_pfam_and_small_clusters(self, tmp_path: Any) -> None:
        import json

        from sample.family_hmms import clustered_families

        path = str(tmp_path / "families.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "P1": {"family_id": "PF00001", "source": "pfam"},
                "C1": {"family_id": "CLUST1", "source": "cluster"},
                "C2": {"family_id": "CLUST1", "source": "cluster"},
                "C3": {"family_id": "CLUST1", "source": "cluster"},
                "C4": {"family_id": "CLUST1", "source": "cluster"},
                "C5": {"family_id": "CLUST1", "source": "cluster"},
                "C6": {"family_id": "TOOSMALL", "source": "cluster"},
            }, fh)

        result = clustered_families(path, min_size=5)

        assert list(result.keys()) == ["CLUST1"]
        assert len(result["CLUST1"]) == 5

    def test_build_one_hmm_end_to_end(self, tmp_path: Any) -> None:
        """A real MAFFT + hmmbuild run over a handful of related short peptides."""
        from sample.family_hmms import build_one_hmm

        sequences = {
            "A": "GLLDIVKKVVGAFGSL", "B": "GLLDIVKKVAGAFGSL",
            "C": "GLLDIVKKVVGSFGSL", "D": "GLLDIVKKVVGAFGSI",
            "E": "GLLDIVKKVVGAFGSL",
        }
        out_dir = str(tmp_path / "hmms")
        hmm_path = build_one_hmm("TEST_FAM", sequences, out_dir)

        assert hmm_path is not None
        assert os.path.exists(hmm_path)
        with open(hmm_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "NAME  TEST_FAM" in content
        assert "NSEQ  5" in content

    def test_build_one_hmm_finds_a_real_member_with_hmmsearch(self, tmp_path: Any) -> None:
        """The built profile must actually score a genuine member above threshold -- this is
        what makes it useful for classification, not just a file that exists."""
        import subprocess

        from sample.family_hmms import build_one_hmm

        sequences = {f"seq{i}": "KWCFRVCYRGICYRRCR" for i in range(6)}
        out_dir = str(tmp_path / "hmms")
        hmm_path = build_one_hmm("TIGHT_FAM", sequences, out_dir)
        assert hmm_path is not None

        query = str(tmp_path / "query.fasta")
        with open(query, "w", encoding="utf-8") as fh:
            fh.write(">query\nKWCFRVCYRGICYRRCR\n")

        result = subprocess.run(["hmmsearch", hmm_path, query], capture_output=True, text=True,
                                timeout=30)
        assert "No hits detected" not in result.stdout

    def test_build_family_hmms_is_resumable(self, tmp_path: Any) -> None:
        import json

        import pandas as pd

        from sample.family_hmms import build_family_hmms

        families_path = str(tmp_path / "families.json")
        with open(families_path, "w", encoding="utf-8") as fh:
            json.dump({f"E{i}": {"family_id": "FAM1", "source": "cluster"} for i in range(5)}, fh)

        dataset_path = str(tmp_path / "dataset.csv")
        pd.DataFrame({"entry_id": [f"E{i}" for i in range(5)],
                     "sequence": ["GLLDIVKKVVGAFGSL"] * 5}).to_csv(dataset_path, index=False)

        out_dir = str(tmp_path / "hmms")
        first = build_family_hmms(families_json=families_path, dataset_csv=dataset_path,
                                  out_dir=out_dir)
        assert first == {"built": 1, "skipped_existing": 0, "failed": 0, "total_clusters": 1}

        second = build_family_hmms(families_json=families_path, dataset_csv=dataset_path,
                                   out_dir=out_dir)
        assert second["skipped_existing"] == 1
        assert second["built"] == 0

    def test_missing_families_file_degrades_to_empty(self, tmp_path: Any) -> None:
        from sample.family_hmms import clustered_families

        assert clustered_families(str(tmp_path / "missing.json")) == {}



class TestAnalyticsCharts:
    """
    The analytics page previously rendered no charts at all -- checked against the live release
    and found four of five already existed (amino-acid distribution among them; the TODO claim
    was stale). Fixed the one real gap (`umap_esm2.png` was stale-named `umap_protbert.png` after
    ProtBERT's removal) and added the genuinely missing sixth chart, target-organism distribution.
    """

    def test_page_serves_all_six_charts(self) -> None:
        from fastapi.testclient import TestClient

        from sample.api.app import app

        body = TestClient(app).get("/analytics").text
        for name in ("sequence_length_dist", "net_charge_dist", "residue_composition",
                    "umap_morgan", "umap_esm2", "target_organism_dist"):
            assert f"{name}.png" in body, f"{name}.png missing from the rendered page"

    def test_stale_protbert_image_is_gone(self) -> None:
        import os

        assert not os.path.exists("sample/static/img/eda/umap_protbert.png"), (
            "ProtBERT was removed this session; a lingering image under its name is dead output")

    def test_target_organism_distribution_reads_activities_not_final_dataset(self) -> None:
        """target_organism is an activity-level field (one peptide, many organisms); it must not
        be assumed present on the peptide-level dataset."""
        import inspect

        from sample import analysis

        source = inspect.getsource(analysis.run_exploratory_data_analysis)
        assert "activities_csv" in source
        assert "target_organism" in source

    def test_missing_activities_csv_skips_only_that_plot(self, tmp_path: Any) -> None:
        import pandas as pd

        from sample.analysis import run_exploratory_data_analysis

        dataset = str(tmp_path / "dataset.csv")
        pd.DataFrame({"entry_id": ["E1", "E2"], "sequence": ["GIGKFLK", "MAGAININ"],
                     "is_decoy": [0, 0]}).to_csv(dataset, index=False)
        out_dir = str(tmp_path / "eda")

        ok = run_exploratory_data_analysis(
            dataset, out_dir, sample_size=2,
            activities_csv=str(tmp_path / "does_not_exist.csv"))

        assert ok is True
        assert os.path.exists(os.path.join(out_dir, "sequence_length_dist.png"))
        assert not os.path.exists(os.path.join(out_dir, "target_organism_dist.png"))





class TestHmmerSearch:
    """
    HMMER as a search mode over the 519 SAMPLE-derived family profiles built earlier this
    session. Distinct from Pfam assignment, which runs the other direction (hmmsearch at
    ingestion time) and is already exposed through GET /families.
    """

    def test_run_hmm_search_finds_the_right_family(self) -> None:
        from sample.api.common import family_hmm_database_path, run_hmm_search

        db_path = family_hmm_database_path()
        if not db_path:
            import pytest
            pytest.skip("family HMM database not built")

        hits = run_hmm_search("KWCFRVCYRGICYRRCR", db_path)
        assert hits
        assert hits[0]["family_id"] == "SAMPLE_FAM_000005"
        assert hits == sorted(hits, key=lambda h: h["evalue"]), "must be best-evalue first"

    def test_unreliable_cluster_is_never_reported(self) -> None:
        """SAMPLE_FAM_000001's own profile does not reliably score its own members -- see
        KNOWN_ISSUES.md. A hit against it must never reach the caller."""
        from sample.api.common import _UNRELIABLE_FAMILY_HMMS, family_hmm_database_path, \
            run_hmm_search

        assert "SAMPLE_FAM_000001" in _UNRELIABLE_FAMILY_HMMS

        db_path = family_hmm_database_path()
        if not db_path:
            import pytest
            pytest.skip("family HMM database not built")

        hits = run_hmm_search("KWCFRVCYRGICYRRCR", db_path)
        assert not any(h["family_id"] == "SAMPLE_FAM_000001" for h in hits)

    def test_empty_query_returns_empty_without_calling_hmmscan(self) -> None:
        from sample.api.common import run_hmm_search

        assert run_hmm_search("", "/nonexistent/db.hmm") == []

    def test_missing_database_degrades_to_empty_path(self, tmp_path: Any) -> None:
        from sample.api.common import family_hmm_database_path

        # No .h3p pressed index at this path -> not considered available.
        import sample.api.common as common
        old = common.DATA_DIR
        try:
            common.DATA_DIR = str(tmp_path)
            assert family_hmm_database_path() == ""
        finally:
            common.DATA_DIR = old

    def test_rest_endpoint_is_documented_and_returns_typed_hits(self) -> None:
        from fastapi.testclient import TestClient

        from sample.api.app import app

        client = TestClient(app)
        openapi = client.get("/api/v1/openapi.json").json()
        assert "/api/v1/search/hmmer" in openapi["paths"]

        response = client.get("/api/v1/search/hmmer",
                             params={"sequence": "KWCFRVCYRGICYRRCR"})
        if response.status_code == 503:
            return  # database not built in this environment; the 503 contract is what matters
        assert response.status_code == 200
        assert response.json()[0]["family_id"] == "SAMPLE_FAM_000005"

    def test_web_search_mode_is_offered_and_links_to_family_page(self) -> None:
        from fastapi.testclient import TestClient

        from sample.api.app import app

        client = TestClient(app)
        body = client.get("/", params={"q": "KWCFRVCYRGICYRRCR", "type": "hmmer"}).text
        assert 'value="hmmer"' in body
        assert "/search?family=SAMPLE_FAM_000005" in body

    def test_web_search_reports_unavailable_database_rather_than_500(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from sample.api import pages
        from sample.api.app import app

        monkeypatch.setattr(pages, "family_hmm_database_path", lambda: "")
        client = TestClient(app)
        response = client.get("/", params={"q": "SOME_SEQUENCE", "type": "hmmer"})
        assert response.status_code == 200
        assert "unavailable" in response.text


