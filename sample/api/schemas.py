"""
Pydantic models for the public API.

These are the first formal statement of the released entity shape. Until now the shape was
whatever ``peptide_detail_json`` happened to return, which meant a column added to the SQLite
schema appeared in the API without anyone deciding it should. Declaring the fields makes the
public surface an explicit choice, and gives FastAPI the material for the OpenAPI document.

``is_decoy`` is deliberately absent from every response model. Decoys are never served, so a field
that could only ever read ``false`` would imply the API might sometimes return one.
"""
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """A page of results, with enough context for a client to walk the collection."""

    items: List[T] = Field(description="The rows on this page.")
    total: int = Field(description="Total rows matching the query, ignoring pagination.")
    limit: int = Field(description="Maximum rows requested.")
    offset: int = Field(description="Rows skipped before this page.")


class Descriptors(BaseModel):
    """Computed molecular descriptors for one peptide."""

    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    hydrogen_bond_donors: Optional[int] = None
    hydrogen_bond_acceptors: Optional[int] = None
    charge: Optional[float] = None
    sequence_length: Optional[int] = None


class Activity(BaseModel):
    """One measured biological activity."""

    id: Optional[int] = None
    entry_id: Optional[str] = None
    activity_group: Optional[str] = Field(
        None, description="Activity class, e.g. antibacterial, anticancer.")
    activity_type: Optional[str] = Field(None, description="Assay, e.g. MIC, IC50.")
    value: Optional[float] = None
    unit: Optional[str] = None
    target_organism: Optional[str] = Field(
        None, description="Canonical NCBI scientific name, or a cell line name.")
    target_taxid: Optional[int] = Field(
        None, description="NCBI Taxonomy id for the target organism. Stable across sources that "
                          "spell the same species differently, so it is the column to join on.")
    target_taxonomy_url: Optional[str] = Field(
        None, description="NCBI Taxonomy browser link for target_taxid.")
    target_atcc_url: Optional[str] = Field(
        None, description="ATCC product page, when target_organism carries a catalog number "
                          "(e.g. 'Staphylococcus aureus ATCC 29213'). Read from the organism "
                          "string itself, not a separate stored accession.")
    mechanism_of_action: Optional[str] = None
    references: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Papers this specific measurement cites. Populated only where the source "
                    "records it per measurement (DBAASP does); otherwise empty, and the "
                    "peptide-level bibliography applies. An empty list means the provenance is "
                    "peptide-level, not that the measurement is unsourced.")


class Paper(BaseModel):
    """A literature, patent or source-database reference, resolved to an external URL."""

    source_type: Optional[str] = Field(
        None, description="pubmed, doi, patent, or a source database name.")
    source_id: Optional[str] = Field(None, description="The accession in that namespace.")
    label: Optional[str] = Field(None, description="Human-readable resource name.")
    url: Optional[str] = Field(
        None, description="Resolved link; null when the accession could not be resolved.")
    verified: bool = Field(
        False,
        description="Whether the URL pattern was checked against data in this repository. "
                    "False means it follows the resource's documented scheme but is unconfirmed.")


class PeptideSummary(BaseModel):
    """A peptide as it appears in a collection listing."""

    entry_id: str
    sequence: str
    SMILES: Optional[str] = None
    source: Optional[str] = None
    LITERATURE: Optional[str] = None
    PATENT: Optional[str] = None
    Reference: Optional[str] = Field(
        None, description="LITERATURE and PATENT joined, for display.")


class Peptide(PeptideSummary):
    """A peptide with everything attached to it."""

    descriptors: Descriptors = Field(default_factory=Descriptors)
    activities: List[Activity] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)
    family: Optional["Family"] = Field(
        None, description="Family assignment, when one exists.")
    bioactivity: List["BioactivityLink"] = Field(
        default_factory=list,
        description="Assay-archive records for this peptide. Empty means it was never resolved "
                    "to an archive identifier, not that no assays exist -- the bridge runs "
                    "through peptide names, which resolved for only a small fraction.")
    bibliography: List["Reference"] = Field(
        default_factory=list,
        description="Formatted citations for this peptide. Activity-level citations, where the "
                    "source states them, are on the activity instead.")


class Reference(BaseModel):
    """One bibliographic reference, formatted for citation."""

    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = Field(None, description="Semicolon-separated, in source order.")
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    apa: Optional[str] = Field(None, description="Preformatted APA 7 citation.")
    url: Optional[str] = Field(None, description="DOI resolver, else PubMed.")


class ModelInfo(BaseModel):
    """One trained model in the registry."""

    model_id: str = Field(description="Path-shaped identifier, used to request a prediction.")
    label: str
    peptide_type: Optional[str] = None
    scope: Optional[str] = None
    target: Optional[str] = None
    assay_type: Optional[str] = None
    learning_task: Optional[str] = None
    feature_block: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # `model_id` collides with Pydantic v2's protected `model_` namespace; the field is part of
    # the published contract, so the namespace protection is what gives way.
    model_config = {"protected_namespaces": ()}


class ModelDataset(BaseModel):
    """
    Full validation report and train/test peptide split for one model.

    Nested fields (``holdout``, ``holdout_ci95``, ``null_comparison``, ...) are kept as open dicts
    rather than fully typed sub-schemas: their key sets vary by learning task (classification
    reports ``roc_auc``/``mcc``/...; regression reports ``r2``/``rmse``/...) and are themselves
    documented at the source in :mod:`sample.ml_models`, which is what actually produces them.
    """

    model_id: str = Field(description="Path-shaped identifier, matching ModelInfo.model_id.")
    namespace: str
    feature_block: str
    learning_task: str
    n_samples: int
    n_positives: int
    n_negatives: int
    n_train: int = Field(description="Peptides in the training portion.")
    n_test: int = Field(description="Peptides in the holdout set -- the final word.")
    holdout: Dict[str, float] = Field(default_factory=dict, description="Final-word metrics.")
    holdout_ci95: Dict[str, List[float]] = Field(
        default_factory=dict, description="Bootstrap 95% CI per metric, on the holdout.")
    y_randomization: Dict[str, float] = Field(default_factory=dict)
    null_comparison: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per metric: t-test/Kruskal-Wallis results, each distribution's Shapiro-Wilk "
                    "normality check, and which test that check selected as primary.")
    applicability_domain: Dict[str, Any] = Field(default_factory=dict)
    quality_passed: bool
    quality_failures: List[str] = Field(default_factory=list)
    train_entry_ids: List[str] = Field(default_factory=list)
    test_entry_ids: List[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}


class PredictionRequest(BaseModel):
    """A request to score peptides with one model."""

    model_id: str = Field(description="From /api/v1/models.")
    sequences: Optional[str] = Field(
        None, description="FASTA, or one sequence per line. Alternative to `peptides`.")
    peptides: Optional[List[str]] = Field(
        None, description="Sequences as a list. Alternative to `sequences`.")

    model_config = {"protected_namespaces": ()}


class JobAccepted(BaseModel):
    """The response to a queued job submission."""

    job_id: str = Field(description="UUID identifying the job.")
    status: str
    n_peptides: int
    status_url: str
    result_url: str


class JobStatus(BaseModel):
    """The state of a background job."""

    job_id: str
    status: str = Field(description="PENDING, RUNNING, SUCCESS or FAILURE.")
    model_id: Optional[str] = None
    n_peptides: int = 0
    n_completed: int = 0
    error: Optional[str] = None
    done: bool = Field(description="True once the job has reached a terminal state.")

    model_config = {"protected_namespaces": ()}


class PredictionRow(BaseModel):
    """One scored peptide."""

    position: int = Field(description="Index in the submitted list, so rows trace back to input.")
    peptide_name: Optional[str] = None
    sequence: str
    probability: Optional[float] = Field(
        None, description="Classifier output. The label is one reading of it at a 0.5 threshold.")
    predicted_label: Optional[str] = None
    value: Optional[float] = Field(None, description="Regressor output, when the model regresses.")
    in_domain: Optional[bool] = Field(
        None, description="Whether this peptide falls within the model's applicability domain "
                          "(standardized k-NN distance to the training set). The prediction is "
                          "still returned when false -- this is a warning, not a refusal. Null "
                          "when the model predates the applicability domain (a legacy model, or "
                          "an older release) and so was never checked.")
    error: Optional[str] = Field(
        None, description="Why this row could not be scored. Set per row, not per job.")


class JobResults(BaseModel):
    """A finished job and its rows."""

    job: JobStatus
    results: List[PredictionRow] = Field(default_factory=list)


class BlastHit(BaseModel):
    """One blastp alignment."""

    subject_id: str
    identity: float
    align_len: int
    mismatches: int
    gaps: int
    q_start: int
    q_end: int
    s_start: int
    s_end: int
    evalue: float
    bitscore: float


class FamilyHmmHit(BaseModel):
    """
    One `hmmscan` hit against a SAMPLE-derived family profile.

    Distinct from Pfam family assignment (`Family.evalue`, computed the other direction --
    `hmmsearch` at Pfam's own curated threshold): this searches the 519 profiles built from this
    project's own near-identity clusters, the ones with no existing way to test a new sequence
    against them.
    """

    family_id: str = Field(description="SAMPLE cluster id, e.g. SAMPLE_FAM_000005.")
    evalue: float = Field(description="Full-sequence E-value, ascending = better.")
    score: float
    bias: float


class EmbeddingHit(BaseModel):
    """
    One nearest-neighbour hit from the ESM-2 embedding index.

    Distinct from BLAST (exact-alignment identity) and HMMER (profile membership in one of the 519
    SAMPLE clusters): this finds peptides the protein language model places nearby in
    representation space, which can catch a functional similarity too weak to align on raw
    sequence at all.
    """

    entry_id: str
    similarity: float = Field(
        description="Cosine similarity to the query, 1.0 = identical direction, descending order.")
    distance: float = Field(description="Cosine distance (1 - similarity); ascending = better.")


class Family(BaseModel):
    """A peptide family, either a curated Pfam entry or a cluster derived here."""

    family_id: str = Field(description="Pfam accession (PF00711) or SAMPLE cluster id.")
    family_name: Optional[str] = Field(None, description="Pfam family name; null for clusters.")
    source: str = Field(
        description="'pfam' for a curated assignment at Pfam's own gathering threshold, "
                    "'cluster' for one derived by near-identity clustering here. These are not "
                    "equally strong claims and are kept distinguishable on purpose.")
    n_members: int = Field(0, description="Peptides in this family.")
    evalue: Optional[float] = Field(None, description="hmmsearch full-sequence E-value; Pfam only.")
    url: Optional[str] = Field(None, description="InterPro/Pfam link; null for clusters, which "
                                                 "have no external identity.")


class BioactivityLink(BaseModel):
    """One link from a peptide into an assay archive."""

    archive: str = Field(description="chembl, pubchem or bindingdb.")
    identifier: str = Field(
        description="ChEMBL molecule id, PubChem CID, or the UniProt accession BindingDB files "
                    "its affinities under.")
    n_records: int = Field(0, description="Records the archive holds for this identifier.")
    url: Optional[str] = None


class SimilarityHit(BaseModel):
    """One near-identity match from the PepMatch index."""

    entry_id: str
    matched_sequence: Optional[str] = None
    mismatches: int = Field(
        description="Substitutions between query and hit. Exact, not an alignment score.")
    identity: float = Field(description="Percent identity over the query length.")
    query_length: int


class AnnotationRequest(BaseModel):
    """A request to annotate one sequence."""

    sequence: str = Field(description="Query sequence, plain or FASTA.")


class AnnotationJobAccepted(BaseModel):
    """The response to a queued annotation submission."""

    job_id: str = Field(description="UUID identifying the job.")
    status: str
    cached: bool = Field(
        description="True when this sequence was already annotated within the last week and the "
                    "result was returned directly rather than queuing a new job -- status is "
                    "SUCCESS immediately in that case, and result_url is already populated.")
    status_url: str
    result_url: str


class PfamHit(BaseModel):
    """One Pfam family match from the annotator's request-time scan."""

    family: str = Field(description="Pfam family name, e.g. DD_K.")
    accession: str = Field(description="Pfam accession, e.g. PF12121.")
    evalue: float = Field(description="Full-sequence E-value, ascending = better.")


class AnnotationSampleHit(BaseModel):
    """One near-identical peptide already in the release, found while annotating a new sequence."""

    entry_id: str
    identity: float = Field(description="Percent identity over the query length.")
    mismatches: int
    has_literature: bool = Field(description="Whether this peptide carries a literature citation.")
    has_patent: bool = Field(description="Whether this peptide carries a patent citation.")


class AnnotationResult(BaseModel):
    """The annotator battery's output for one sequence."""

    sequence: str = Field(description="The submitted sequence, cleaned.")
    hemopi2: Optional[Dict[str, Any]] = Field(
        None, description="Hemolytic-activity prediction (hemopi2_score, hemopi2_is_hemolytic), "
                          "or null when the tool was unavailable -- distinct from a real "
                          "negative call. See predictors_available.")
    pfam_hits: List[PfamHit] = Field(
        default_factory=list, description="Pfam families matched, best E-value first. Empty is a "
                                          "real result for most AMPs, which have no curated "
                                          "family -- not evidence the scan failed.")
    sample_hits: List[AnnotationSampleHit] = Field(
        default_factory=list, description="Near-identical peptides already in this release.")
    predictors_available: Dict[str, bool] = Field(
        default_factory=dict, description="Every named external predictor's availability, so a "
                                          "consumer can show what did not run and why.")


class AnnotationJobStatus(BaseModel):
    """The state of a background annotation job."""

    job_id: str
    status: str = Field(description="PENDING, RUNNING, SUCCESS or FAILURE.")
    sequence: Optional[str] = None
    error: Optional[str] = None
    done: bool = Field(description="True once the job has reached a terminal state.")


class AnnotationJobResult(BaseModel):
    """A finished annotation job and its result."""

    job: AnnotationJobStatus
    result: Optional[AnnotationResult] = None


class Stats(BaseModel):
    """Dataset counts."""

    consensus_count: int = Field(description="Real peptides. Excludes decoys.")
    decoys_count: int = Field(description="Training negatives. Never served as peptides.")
    final_count: int = Field(description="Rows in the assembled dataset, decoys included.")
    activities_count: int = 0
    papers_count: int = 0


class ErrorResponse(BaseModel):
    """An error."""

    detail: str


# `Peptide` refers to `Family` and `Reference` by name because they are declared after it. Pydantic
# resolves those forward references here, once every model in the module exists.
Peptide.model_rebuild()
