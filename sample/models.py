"""
The released database as entity classes.

Until now the schema existed only as ``CREATE TABLE`` strings scattered through
:mod:`sample.serialize`, and every read was a hand-written SQL string in :mod:`sample.db`. Nothing
declared what a peptide *is*, so a column added in one place and read in another could disagree
silently, and the schema could only ever be SQLite -- ``AUTOINCREMENT``, ``INSERT OR IGNORE`` and
the FTS5 virtual table are all SQLite spellings.

These classes are the single declaration of the schema. SQLAlchemy renders them into whichever
dialect the engine speaks, so the same code builds SQLite for a laptop and PostgreSQL or MySQL for
a server.

**What portability actually requires, beyond an ORM.** Three things do not translate themselves,
and are handled explicitly rather than pretended away:

* **Full-text search.** FTS5 is a SQLite virtual table with no equivalent elsewhere: PostgreSQL
  uses ``tsvector`` with a GIN index, MySQL uses ``FULLTEXT``. There is no portable spelling, so
  the search layer asks the dialect what it supports -- see :mod:`sample.engine`.
* **Regular expressions.** SQLite has no ``REGEXP`` operator and needs a Python function
  registered per connection; PostgreSQL has ``~`` and MySQL has ``REGEXP`` natively.
* **Upserts.** ``INSERT OR IGNORE`` is SQLite; PostgreSQL wants ``ON CONFLICT DO NOTHING`` and
  MySQL ``INSERT IGNORE``. SQLAlchemy exposes each as a dialect-specific construct.

**The decoy invariant still lives in one place.** Modelling the tables does not weaken it: decoys
share the ``peptides`` table with real entries, so every query still goes through
:mod:`sample.db`, which applies ``is_decoy = 0``. An ORM makes it easier, not harder, to bypass
that by accident -- ``session.query(Peptide).all()`` would happily return fabricated sequences --
so the repository layer remains the only sanctioned entry point.
"""
from typing import List, Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base carrying the shared :class:`~sqlalchemy.MetaData`."""


class Peptide(Base):
    """
    A peptide in the release.

    ``is_decoy`` marks a generated training negative -- a shuffled sequence or a UniProt fragment
    with no provenance. Decoys share this table with real entries and are distinguishable only by
    this flag, which is why no route may query it directly.
    """

    __tablename__ = "peptides"

    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sequence: Mapped[str] = mapped_column(Text, nullable=False)
    # `SMILES` keeps its historical capitalisation: it is part of the published column contract,
    # and renaming it would break every consumer of the CSV export.
    SMILES: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(String(128))
    is_decoy: Mapped[int] = mapped_column(Integer, default=0, index=True)
    LITERATURE: Mapped[Optional[str]] = mapped_column(Text)
    PATENT: Mapped[Optional[str]] = mapped_column(Text)
    Reference: Mapped[Optional[str]] = mapped_column(Text)
    Biological_Activity: Mapped[Optional[str]] = mapped_column(Text)
    #: The organism that produces this peptide, or ``Synthetic`` for a designed sequence.
    #: ``None`` where neither is recorded -- see :mod:`sample.provenance`.
    source_organism: Mapped[Optional[str]] = mapped_column(String(255))

    descriptors: Mapped[Optional["Descriptor"]] = relationship(
        back_populates="peptide", uselist=False, cascade="all, delete-orphan")
    activities: Mapped[List["Activity"]] = relationship(
        back_populates="peptide", cascade="all, delete-orphan")
    names: Mapped[List["PeptideName"]] = relationship(
        back_populates="peptide", cascade="all, delete-orphan")
    family: Mapped[Optional["PeptideFamily"]] = relationship(
        back_populates="peptide", uselist=False, cascade="all, delete-orphan")
    topology: Mapped[Optional["PeptideTopology"]] = relationship(
        back_populates="peptide", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        # MySQL cannot index a TEXT column without a prefix length -- "BLOB/TEXT column used in
        # key specification without a key length" -- and the other dialects ignore the hint. 255
        # characters is far past the point where peptide sequences stop sharing prefixes.
        Index("idx_peptides_sequence", "sequence", mysql_length=255),
        Index("idx_peptides_source", "source"),
    )


class Descriptor(Base):
    """
    Computed properties for one peptide.

    Derived from the sequence, never from the stored SMILES -- that column is unvalidated. The
    peptide-specific columns come from ``peptides.py`` and are corrected for topology: a
    head-to-tail cycle is lighter than the linear chain by a water, and carries no titratable
    termini.
    """

    __tablename__ = "descriptors"

    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), primary_key=True)
    molecular_weight: Mapped[Optional[float]] = mapped_column(Float)
    logp: Mapped[Optional[float]] = mapped_column(Float)
    hydrogen_bond_donors: Mapped[Optional[int]] = mapped_column(Integer)
    hydrogen_bond_acceptors: Mapped[Optional[int]] = mapped_column(Integer)
    charge: Mapped[Optional[float]] = mapped_column(Float)
    sequence_length: Mapped[Optional[int]] = mapped_column(Integer)

    peptide: Mapped["Peptide"] = relationship(back_populates="descriptors")


class Activity(Base):
    """
    One measured biological activity.

    ``source_accession`` and ``source_reference_index`` record which source record the measurement
    came from and that record's own pointer to the paper behind it. Both are needed to cite a
    measurement rather than a peptide: without them a paper gets credited with every measurement
    of every peptide it touched.
    """

    __tablename__ = "activities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), index=True)
    activity_group: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    value: Mapped[Optional[float]] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(32))
    target_organism: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    #: NCBI tax id for the organism. Every lookup already resolves through it, and storing it makes
    #: the row joinable across sources that spell a species differently.
    target_taxid: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    activity_type: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    mechanism_of_action: Mapped[Optional[str]] = mapped_column(Text)
    source_accession: Mapped[Optional[str]] = mapped_column(String(64))
    source_reference_index: Mapped[Optional[str]] = mapped_column(String(16))

    peptide: Mapped[Optional["Peptide"]] = relationship(back_populates="activities")

    __table_args__ = (
        Index("idx_activities_source_ref", "source_accession", "source_reference_index"),
    )


class Paper(Base):
    """
    A source-database record a peptide was copied from.

    Not a bibliography: ``source_type`` holds a database name (``CAMP``, ``DBAASP``) and
    ``source_id`` that database's accession. Citations live in :class:`BibliographyEntry`.
    """

    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(64))
    source_id: Mapped[Optional[str]] = mapped_column(String(128))

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_papers_source"),
        Index("idx_papers_source", "source_type", "source_id"),
    )


class PeptideReference(Base):
    """Association between a peptide and the source record it came from."""

    __tablename__ = "peptide_references"

    peptide_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)


class BibliographyEntry(Base):
    """
    A publication, resolved to identifiers and a formatted citation.

    ``apa`` is preformatted so the CSV export, the JSON API and the HTML page all show the same
    string rather than three renderings that drift apart.
    """

    __tablename__ = "bibliography"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ref_key: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    pmid: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    doi: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    scopus_eid: Mapped[Optional[str]] = mapped_column(String(64))
    title: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[Optional[str]] = mapped_column(Text)
    journal: Mapped[Optional[str]] = mapped_column(String(255))
    year: Mapped[Optional[int]] = mapped_column(Integer)
    volume: Mapped[Optional[str]] = mapped_column(String(32))
    issue: Mapped[Optional[str]] = mapped_column(String(32))
    pages: Mapped[Optional[str]] = mapped_column(String(64))
    apa: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(Text)
    source_db: Mapped[Optional[str]] = mapped_column(String(64))


class PeptideBibliography(Base):
    """Peptide-level citation: this paper concerns this peptide."""

    __tablename__ = "peptide_bibliography"

    peptide_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), primary_key=True)
    reference_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("bibliography.id", ondelete="CASCADE"), primary_key=True)


class ActivityReference(Base):
    """
    Measurement-level citation: this paper reported this specific measurement.

    Distinct from :class:`PeptideBibliography` on purpose. Only DBAASP states which article backs
    which measurement; presenting a peptide-level citation as measurement-level would claim
    provenance the source never gave.
    """

    __tablename__ = "activity_references"

    activity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("activities.id", ondelete="CASCADE"), primary_key=True)
    reference_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("bibliography.id", ondelete="CASCADE"), primary_key=True)


class PeptideName(Base):
    """
    A name or synonym for a peptide.

    Synonyms are what make the corpus searchable across the literature: the same peptide is
    ``Dermaseptin-B2``, ``DRS-B2`` and ``ADENOREGULIN`` in different papers.
    """

    __tablename__ = "peptide_names"

    # Composite natural key rather than a surrogate id: a peptide has a name once, and
    # (entry_id, name) is what the loader already enforces as unique. Declaring an `id` here
    # instead produced a model that could not read the shipped database at all -- the column does
    # not exist in it.
    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"),
        primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    #: ``primary`` or ``synonym``.
    name_kind: Mapped[Optional[str]] = mapped_column(String(16))
    source_db: Mapped[Optional[str]] = mapped_column(String(64))

    peptide: Mapped["Peptide"] = relationship(back_populates="names")


class PeptideFamily(Base):
    """
    A peptide's family.

    ``source`` separates a curated Pfam assignment from a cluster derived here. They are not
    equally strong claims and must stay distinguishable.
    """

    __tablename__ = "peptide_families"

    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), primary_key=True)
    family_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    family_name: Mapped[Optional[str]] = mapped_column(String(255))
    #: ``pfam`` (curated) or ``cluster`` (derived here).
    source: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    evalue: Mapped[Optional[float]] = mapped_column(Float)
    all_domains: Mapped[Optional[str]] = mapped_column(Text)

    peptide: Mapped["Peptide"] = relationship(back_populates="family")


class PeptideTopology(Base):
    """
    A peptide's shape: whether the backbone is closed and how it is braced.

    ``mass_correction`` is what a linear-chain calculation overstates the mass by -- a water for a
    head-to-tail cycle plus two hydrogens per disulfide. On a cyclotide that is 22 Da, enough to
    break an identity match against PubChem or ChEMBL.
    """

    __tablename__ = "peptide_topology"

    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), primary_key=True)
    #: ``linear``, ``head_to_tail``, ``disulfide``, ``cyclic_unspecified`` or ``branched``.
    kind: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    is_cyclic: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    n_disulfides: Mapped[Optional[int]] = mapped_column(Integer)
    mass_correction: Mapped[Optional[float]] = mapped_column(Float)
    #: The source's bond list as JSON: positions, bond chemistry and cycle type.
    bonds: Mapped[Optional[str]] = mapped_column(Text)
    source_db: Mapped[Optional[str]] = mapped_column(String(64))

    peptide: Mapped["Peptide"] = relationship(back_populates="topology")


class Bioactivity(Base):
    """Records an assay archive holds for one external identifier."""

    __tablename__ = "bioactivity"

    archive: Mapped[str] = mapped_column(String(32), primary_key=True, index=True)
    identifier: Mapped[str] = mapped_column(String(64), primary_key=True)
    n_records: Mapped[Optional[int]] = mapped_column(Integer)
    #: The archive's own records, as JSON.
    payload: Mapped[Optional[str]] = mapped_column(Text)


class PeptideBioactivity(Base):
    """
    Bridge from a peptide to an assay-archive identifier.

    Kept separate from :class:`Bioactivity` because the archive records are exact while this
    bridge is a name match; merging them would let the second's uncertainty read as the first's.
    """

    __tablename__ = "peptide_bioactivity"

    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), primary_key=True)
    archive: Mapped[str] = mapped_column(String(32), primary_key=True)
    identifier: Mapped[str] = mapped_column(String(64), primary_key=True)


class PeptideModification(Base):
    """
    One chemical modification: a capped terminus, or a non-canonical residue at a position.

    Separate from :class:`PeptideTopology` because they answer different questions. Topology is a
    property of the whole molecule -- is the backbone closed, how many disulfides brace it.
    A modification is local: *this* terminus, or *that* residue. A peptide commonly has both.

    ``kind`` carries the chemistry rather than the source's code, because the codes are not
    comparable across sources and the chemistry is: ``AMD``, ``NH2`` and ``AMIDE`` are one fact.
    """

    __tablename__ = "peptide_modifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), index=True)
    #: ``N`` or ``C`` for a terminal modification, ``None`` for a residue one.
    terminus: Mapped[Optional[str]] = mapped_column(String(1))
    #: 1-based residue position for a non-canonical residue, ``None`` for a terminal modification.
    position: Mapped[Optional[int]] = mapped_column(Integer)
    #: The source's short code, e.g. ``AMD``, ``ACT``, ``C16``, ``DHF``.
    code: Mapped[str] = mapped_column(String(64))
    description: Mapped[Optional[str]] = mapped_column(Text)
    #: ``amidation``, ``acylation``, ``unusual_residue`` or ``other``.
    kind: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    source_db: Mapped[Optional[str]] = mapped_column(String(64))
    source_accession: Mapped[Optional[str]] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("entry_id", "terminus", "position", "code",
                         name="uq_peptide_modification"),
    )


class PeptideGOTerm(Base):
    """
    One Gene Ontology annotation for a peptide, harvested from CAMP's own scraped GO field.

    ``category`` is nullable because the source scrape omits it for roughly 29 % of terms; the
    term and evidence code are still usable without it, so the row is kept rather than discarded.
    """

    __tablename__ = "peptide_go_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("peptides.entry_id", ondelete="CASCADE"), index=True)
    go_id: Mapped[str] = mapped_column(String(16), index=True)
    #: ``Molecular function``, ``Biological process``, ``Cellular component``, or ``None``.
    category: Mapped[Optional[str]] = mapped_column(String(32))
    term: Mapped[str] = mapped_column(Text)
    #: GO evidence code, e.g. ``IEA``, ``IDA``.
    evidence: Mapped[Optional[str]] = mapped_column(String(8))
    source_db: Mapped[Optional[str]] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("entry_id", "go_id", "evidence", name="uq_peptide_go_term"),
    )


class Admin(Base):
    """
    A person who may sign in to the admin interface.

    The only table here that is **not** derived from the pipeline. Everything else is rebuilt from
    the CSVs by step 44, so `DROP TABLE` and repopulate is the normal way those tables change;
    doing that to accounts would delete every administrator on each release. :func:`create_schema`
    creates this table if it is missing and otherwise leaves it alone.

    Passwords are stored as a scrypt digest and never in the clear -- see
    :func:`sample.admin.hash_password` for the encoding and the reason scrypt was chosen over the
    usual bcrypt/argon2 dependency.
    """

    __tablename__ = "admins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: Case-insensitively unique; :mod:`sample.admin` lowercases before storing and comparing, so
    #: "Alice" and "alice" cannot both exist and either spelling signs in.
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Bumped whenever the password changes, which invalidates that account's outstanding session
    #: cookies: the token carries the value it was signed with. Without it, a password reset after
    #: a cookie was stolen would not end the intruder's session, which is the main reason a person
    #: resets a password in the first place.
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    #: Soft disable, so an account can be suspended without losing the audit trail of who made an
    #: edit. A disabled admin cannot sign in.
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class PeptideOverride(Base):
    """
    A manual correction to one `peptides` field, made through the admin interface, held durably.

    Every editable fact about a peptide otherwise lives in a table step 44 drops and rebuilds
    from the CSVs (see :class:`Peptide`), so a hand-corrected value written directly to `peptides`
    used to be silently gone at the next rebuild -- the operator saw the change take effect, then
    found it reverted weeks later with nothing in the log to explain why. This table has the
    opposite lifecycle, like :class:`Admin`: :func:`sample.serialize.load_sql_database` reapplies
    every row here onto the freshly loaded `peptides` table before returning, so a correction
    survives as many rebuilds as it takes for the underlying source data to fix itself.

    One row per ``(entry_id, field)`` -- the *current* correction, not a history of edits. Who made
    the latest one and when is kept for the audit trail; the log line written at edit time
    (`sample/api/admin.py`) additionally records what the value changed from.
    """

    __tablename__ = "peptide_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: Not a foreign key: this table's whole purpose is to outlive `peptides` being dropped and
    #: recreated, and a peptide legitimately disappearing between releases should not cascade-erase
    #: the correction anyone made for it. `sample.serialize._apply_peptide_overrides` handles that
    #: case at read time instead -- an override whose `entry_id` no longer exists in the rebuilt
    #: `peptides` table is skipped and counted, not applied.
    entry_id: Mapped[str] = mapped_column(String(64), index=True)
    #: Column name in `peptides`; restricted to `sample.config.EDITABLE_PEPTIDE_FIELDS`.
    field: Mapped[str] = mapped_column(String(64))
    value: Mapped[Optional[str]] = mapped_column(Text)
    admin_username: Mapped[str] = mapped_column(String(64))
    changed_at: Mapped[str] = mapped_column(String(32))

    __table_args__ = (
        UniqueConstraint("entry_id", "field", name="uq_peptide_override"),
    )


#: Every entity, in dependency order -- useful for creating or dropping the schema wholesale.
ALL_MODELS = (
    Peptide, Descriptor, Activity, Paper, PeptideReference, BibliographyEntry,
    PeptideBibliography, ActivityReference, PeptideName, PeptideFamily, PeptideTopology,
    Bioactivity, PeptideBioactivity, PeptideModification, PeptideGOTerm, Admin, PeptideOverride,
)
