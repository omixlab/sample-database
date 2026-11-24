"""
The HTML dashboard, ported from Flask to FastAPI.

Every URL path is preserved exactly. The templates already hard-code these paths
(``models.html`` posts to ``/predict``, ``prediction_job.html`` polls ``/predict/status/<id>``)
and the compose healthcheck hits ``/``, so the paths are the contract, not an implementation
detail free to change during a framework swap.

Data access goes through :mod:`sample.db`, which is what keeps decoys out of the pages -- the
previous inline SQL had no ``is_decoy`` filter and would return a generated decoy as a search hit.
"""
import json
import logging
import os
import uuid
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from sample import db
from sample.api.common import (DATA_DIR, DATASET_LABELS, DATASET_PATHS, SOURCE_DATABASES,
                               embedding_index_path, extraction_dir, family_hmm_database_path,
                               init_blast_database, run_blast_search, run_embedding_search,
                               run_hmm_search, templates)

logger = logging.getLogger("sample.api.pages")

router = APIRouter(tags=["pages"], include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def index(request: Request, q: str = "", type: str = "text", activity_group: str = "",
          target: str = "", mechanism: str = "", metric: str = "", unit: str = "") -> Any:
    """
    The Explore dashboard: corpus counters plus the search form and its results.

    :param request: Incoming request, required by the template engine.
    :param q: Free-text or sequence query.
    :param type: ``"text"``, ``"fulltext"``, ``"regex"``, ``"similar"``, ``"blast"``, ``"hmmer"``
        or ``"embedding"``.
    :param activity_group: Activity-class filter.
    :param target: Target-organism filter.
    :param mechanism: Mechanism-of-action filter.
    :param metric: Assay-type filter.
    :param unit: Unit filter.
    :returns: The rendered ``index`` template.
    """
    query = q.strip()
    filters = (activity_group.strip(), target.strip(), mechanism.strip(),
               metric.strip(), unit.strip())

    results: List[Any] = []
    search_error: Optional[str] = None
    if query or any(filters):
        if type == "blast" and query:
            blast_db = init_blast_database()
            results = run_blast_search(query, blast_db) if blast_db else []
        elif type == "hmmer" and query:
            hmm_db = family_hmm_database_path()
            if hmm_db:
                results = run_hmm_search(query, hmm_db)
            else:
                search_error = ("Family HMM database unavailable; run "
                               "`sample-cli build-family-hmms --press`.")
        elif type == "embedding" and query:
            index_path = embedding_index_path()
            if index_path:
                results = run_embedding_search(query, index_path)
            else:
                search_error = ("Embedding index unavailable; run "
                               "`sample-cli build-embedding-index`.")
        elif type == "similar" and query:
            from sample import similarity
            results = similarity.search(query, limit=100)
        elif type == "regex" and query:
            # A regex is written deliberately, so its errors are reported rather than swallowed --
            # unlike the full-text box, where a stray parenthesis is usually an accident.
            try:
                results, _ = db.regex_search(query, limit=100)
            except db.InvalidPattern as e:
                search_error = str(e)
            except db.SearchTimeout as e:
                search_error = str(e)
        elif type == "fulltext" and query:
            results, _ = db.full_text_search(query, limit=100)
        else:
            results, _ = db.search_peptides(
                query_str=query, activity_group=filters[0], target=filters[1],
                mechanism=filters[2], metric=filters[3], unit=filters[4], limit=100)

    model_path = os.path.join(DATA_DIR, "sample_final_release", "model.pkl")
    return templates.TemplateResponse(request, "index.html", {
        "active_tab": "explore", "title": "Explore Peptides",
        "stats": db.get_stats(),
        "model_status": "Available" if os.path.exists(model_path) else "Not Trained",
        "query": query, "search_type": type, "search_error": search_error,
        "activity_group": filters[0], "target": filters[1],
        "mechanism": filters[2], "metric": filters[3], "unit": filters[4], "results": results,
    })


#: Family members per page, matching the extraction browser.
FAMILY_MEMBERS_PER_PAGE = 100

#: Extraction directories that are working copies rather than pipeline output. A dataset built
#: before the merge applied `config.iter_extraction_files` carries tokens naming these, and they
#: must not be offered as source documents: the files are duplicates of the live ones, and the
#: link would point at a directory the extraction browser does not serve.
_STALE_SOURCE_DIRS = ("pubmed_old", "pubmed.bak", "scopus_old", "scopus.bak")


def _real_references(reference: Optional[str]) -> List[str]:
    """
    Keeps only reference entries that name a paper.

    ``peptides.Reference`` was built as ``File: <uuid>.json`` -- the name of the extraction file,
    rendered on the page as though it were a citation, for all 12,395 populated rows. The merge
    now resolves each extraction to its title, DOI and PMID, but a dataset built before that still
    holds the filenames, so they are dropped here rather than displayed.

    :param reference: The pipe-delimited cell, or ``None``.
    :returns: The entries worth showing, in order.
    """
    if not reference:
        return []
    return [part.strip() for part in str(reference).split(" | ")
            if part.strip() and not part.strip().startswith("File: ")]


def _charge_correction(modifications: List[dict]) -> float:
    """
    How far the displayed net charge is from the modified molecule's.

    :param modifications: The peptide's modification rows.
    :returns: The signed correction, or ``0.0`` when nothing charged was modified. Shown beside
        the descriptor rather than folded into it -- see :mod:`sample.modifications`.
    """
    try:
        from sample.modifications import charge_correction

        return charge_correction(modifications)
    except Exception:
        return 0.0


def _wikipedia_query(names: List[dict]) -> Optional[str]:
    """
    Picks the name most likely to find the right Wikipedia article.

    Some sources file several synonyms in one cell -- DBAASP records this peptide's primary name
    as ``Adenoregulin, Dermaseptin BII, Dermaseptin B2`` -- and searching for the joined string
    finds nothing. A single name is preferred over a list, and a shorter one over a longer.

    :param names: The peptide's names, primaries first.
    :returns: The chosen query, or ``None`` when there is no usable name.
    """
    candidates = [str(n.get("name") or "").strip() for n in names]
    single = [c for c in candidates if c and "," not in c]
    if single:
        return min(single, key=len)
    return candidates[0].split(",")[0].strip() if candidates else None


#: Terms shown per category before collapsing the rest into a "+N more" count. Some peptides
#: carry 100+ GO terms (CAMP's own UniProt-derived annotation, not this project's judgement of
#: relevance), and showing all of them would bury the rest of the page.
GO_TERMS_PER_CATEGORY = 8

#: Display order and label for each GO namespace, plus the CAMP-scrape rows missing a category.
_GO_CATEGORY_ORDER = (
    ("Biological process", "Biological Process"),
    ("Molecular function", "Molecular Function"),
    ("Cellular component", "Cellular Component"),
    (None, "Uncategorised"),
)


def _go_by_category(go_terms: List[dict]) -> List[dict]:
    """
    Groups GO terms by category for display, capping each group.

    :param go_terms: Rows from :func:`sample.db._go_terms_for`, in category order already.
    :returns: ``[{"label", "terms", "shown", "total"}, ...]`` for non-empty categories only.
    """
    by_category: Dict[Optional[str], List[dict]] = {}
    for term in go_terms:
        by_category.setdefault(term.get("category"), []).append(term)

    groups = []
    for key, label in _GO_CATEGORY_ORDER:
        terms = by_category.get(key)
        if not terms:
            continue
        groups.append({
            "label": label, "terms": terms[:GO_TERMS_PER_CATEGORY],
            "shown": min(len(terms), GO_TERMS_PER_CATEGORY), "total": len(terms),
        })
    return groups


def _link_admet_hemolysis_gap(admet: Optional[dict], annotation: Optional[dict]) -> None:
    """
    Cross-references the ADMET panel's ``hemolysis`` gap against this peptide's pre-computed
    annotation, in place.

    ``sample.admet`` is sequence-only and has no database access, so it cannot know whether
    HemoPI2/macrel have actually scored this peptide -- it can only describe *that* they would.
    This page has both pieces (the ADMET profile and the annotation) already, so it is the right
    place to turn "run the annotator for this peptide" into the peptide's real hemolysis numbers
    once `build-annotations` has produced them, rather than leaving a static pointer forever.

    :param admet: The peptide's ``admet_profile()`` result, mutated in place; a no-op if ``None``.
    :param annotation: The peptide's pre-computed annotation (``sample.peptide_annotations.get``),
        or ``None`` if the batch has not reached this peptide yet.
    """
    if not admet or not annotation:
        return
    predictors = annotation.get("predictors") or {}
    hemolysis_fields = [
        predictors[f] for f in ("hemopi2_score", "macrel_hemolytic_probability") if f in predictors
    ]
    if not hemolysis_fields:
        return
    for gap in admet.get("gaps") or []:
        if gap.get("property") != "hemolysis":
            continue
        values = ", ".join(f"{f['label']} = {f['value']:.3f}" for f in hemolysis_fields)
        gap["status"] = "computed"
        gap["note"] = f"Computed by the annotator: {values}. See the Annotation panel below."


def _secondary_structure(sequence: Optional[str]) -> Optional[dict]:
    """
    Per-residue secondary-structure propensity, for the sequence-strip panel.

    Replaces the RDKit 2D chemical-graph rendering: a full bond diagram of a 30-residue peptide is
    unreadable at any reasonable size and answers a question this page's readers do not have. This
    is what they do reach for -- is a stretch of this sequence likely helical, is it likely to
    form a strand -- and it needs nothing but the sequence, so it covers every peptide rather than
    only the ones ColabFold has folded.

    :param sequence: The peptide's one-letter sequence.
    :returns: ``{"residues": [(letter, state), ...], "summary": {...}}``, or ``None`` when there
        is no sequence to classify.
    """
    if not sequence:
        return None

    from sample.secondary_structure import residue_states, summarize

    states = residue_states(sequence)
    return {
        "residues": list(zip(str(sequence).upper().strip(), states)),
        "summary": summarize(sequence),
    }


def _identity_links(cross_references: Optional[str]) -> List[dict]:
    """
    UniProt and PDB links for one peptide.

    :param cross_references: The raw ``peptides.cross_references`` cell.
    :returns: Renderable link records, restricted to the identity namespaces -- PubChem and
        ChEMBL tokens in the same cell are the "Assay Archives" panel's concern, sourced there
        independently through step 37, and are excluded here so the same accession is not shown
        twice under two different sections.
    """
    if not cross_references:
        return []

    from sample.external_links import cross_reference_links
    from sample.identifiers import IDENTITY_NAMESPACES

    return [link for link in cross_reference_links(cross_references)
            if link.get("source_type") in IDENTITY_NAMESPACES]


_LIT_METADATA_INDEX: Optional[Dict[str, Dict[str, Optional[str]]]] = None
_LIT_METADATA_LOCK = threading.Lock()

_PATENT_METADATA_INDEX: Optional[Dict[str, str]] = None
_PATENT_METADATA_LOCK = threading.Lock()

_PATENT_CLEAN_RE = re.compile(r"\b([A-Z]{2}\d{4,}[A-Z]?\d*)\b")


def _literature_metadata_index() -> Dict[str, Dict[str, Optional[str]]]:
    """
    Cached mapping of internal document stem/UUID -> {doi, title}.
    Populated from pubmed_downloaded.csv and scopus_downloaded.csv.
    """
    global _LIT_METADATA_INDEX
    if _LIT_METADATA_INDEX is not None:
        return _LIT_METADATA_INDEX
    with _LIT_METADATA_LOCK:
        if _LIT_METADATA_INDEX is not None:
            return _LIT_METADATA_INDEX
        index: Dict[str, Dict[str, Optional[str]]] = {}
        sources = [
            os.path.join(DATA_DIR, "literature_data", "pubmed_downloaded.csv"),
            os.path.join(DATA_DIR, "literature_data", "scopus_downloaded.csv"),
        ]
        for path in sources:
            if not os.path.exists(path):
                continue
            try:
                frame = pd.read_csv(path, usecols=lambda c: c in ("internal_id", "doi", "title"), dtype=str)
            except Exception as e:
                logger.debug(f"Could not load literature metadata from {path}: {e}")
                continue
            for _, row in frame.iterrows():
                iid = row.get("internal_id")
                if isinstance(iid, str) and iid.strip():
                    iid = iid.strip()
                    if iid not in index:
                        doi = row.get("doi")
                        title = row.get("title")
                        index[iid] = {
                            "doi": doi.strip() if isinstance(doi, str) and doi.strip() else None,
                            "title": title.strip() if isinstance(title, str) and title.strip() else None,
                        }
        _LIT_METADATA_INDEX = index
        return _LIT_METADATA_INDEX


def _patent_metadata_index() -> Dict[str, str]:
    """
    Cached mapping of normalized patent number -> patent title.
    Populated from patent_data/google_patents.csv.
    """
    global _PATENT_METADATA_INDEX
    if _PATENT_METADATA_INDEX is not None:
        return _PATENT_METADATA_INDEX
    with _PATENT_METADATA_LOCK:
        if _PATENT_METADATA_INDEX is not None:
            return _PATENT_METADATA_INDEX
        index: Dict[str, str] = {}
        path = os.path.join(DATA_DIR, "patent_data", "google_patents.csv")
        if os.path.exists(path):
            try:
                frame = pd.read_csv(path, skiprows=1, usecols=lambda c: c in ("id", "title"), dtype=str)
                for _, row in frame.iterrows():
                    raw_id = row.get("id")
                    title = row.get("title")
                    if isinstance(raw_id, str) and raw_id.strip():
                        clean = re.sub(r"[^A-Za-z0-9]", "", raw_id).upper()
                        if clean and isinstance(title, str) and title.strip():
                            index[clean] = title.strip()
            except Exception as e:
                logger.debug(f"Could not load patent metadata from {path}: {e}")
        _PATENT_METADATA_INDEX = index
        return _PATENT_METADATA_INDEX


def _source_documents(literature: Optional[str], patent: Optional[str]) -> List[dict]:
    """
    Resolves the extraction tokens to public DOI / Google Patents links, skipping working copies.

    Rather than exposing internal extraction UUIDs (e.g. `pubmed:0002a4e1...`), resolves each
    literature item to its DOI (linking to https://doi.org/<doi>, which redirects to the journal
    page) and each patent to its patent number (linking to Google Patents).

    :param literature: The ``LITERATURE`` cell.
    :param patent: The ``PATENT`` cell.
    :returns: ``{kind, source, file_name, label, url, doi, patent_number, title, extraction_url}`` per usable token.
    """
    documents: List[dict] = []
    seen_urls = set()
    lit_meta = _literature_metadata_index()
    pat_titles = _patent_metadata_index()

    for cell, kind in ((literature, "literature"), (patent, "patent")):
        if not cell:
            continue
        # Historically written with either separator, depending on which merge produced the row.
        tokens = str(cell).replace(" | ", ";").split(";")
        for token in tokens:
            source, _, name = token.strip().partition(":")
            if not source or not name or source in _STALE_SOURCE_DIRS:
                continue

            stem = name[:-5] if name.endswith(".json") else name
            extraction_url = f"/extractions/view/{kind}/{source}/{name}"
            doi: Optional[str] = None
            patent_number: Optional[str] = None
            title: Optional[str] = None

            if kind == "patent":
                match = _PATENT_CLEAN_RE.search(stem)
                clean_patent = match.group(1) if match else stem
                patent_number = clean_patent
                url = f"https://patents.google.com/patent/{clean_patent}"
                label = f"Patent: {clean_patent}"
                clean_key = re.sub(r"[^A-Za-z0-9]", "", clean_patent).upper()
                title = pat_titles.get(clean_key) or f"Google Patents: {clean_patent}"
            else:
                meta = lit_meta.get(stem)
                if meta and meta.get("doi"):
                    doi = meta["doi"]
                    raw_title = meta.get("title")
                    title = raw_title.replace('"', '&quot;').replace('\n', ' ').strip() if raw_title else f"DOI: {doi}"
                    url = f"https://doi.org/{doi}"
                    label = f"DOI: {doi}"
                else:
                    # Fallback when DOI is not in metadata (e.g. in synthetic tests)
                    url = extraction_url
                    label = stem
                    title = f"{source}/{name}"

            if url in seen_urls:
                continue
            seen_urls.add(url)

            documents.append({
                "kind": kind,
                "source": source,
                "file_name": name,
                "label": label,
                "url": url,
                "doi": doi,
                "patent_number": patent_number,
                "title": title,
                "extraction_url": extraction_url,
            })
    return documents


@router.get("/search", response_class=HTMLResponse)
def search(request: Request, family: str = "", page: int = 1) -> Any:
    """
    Browses the members of one peptide family.

    The family panel on the peptide page has always linked here (``/search?family=PF12121``) and
    the route did not exist, so every "N members" link answered 404. Only ``/search/peptide/...``
    was defined, and a prefix match does not make ``/search`` itself routable.

    :param request: Incoming request.
    :param family: Pfam accession or SAMPLE cluster id. Empty falls back to the dashboard, since
        there is nothing to list.
    :param page: 1-based page number, clamped into range.
    :returns: The rendered ``family_members`` template.
    """
    family = family.strip()
    if not family:
        return RedirectResponse(url="/", status_code=303)

    total = db.family_members(family, limit=1, offset=0)[1]
    pages = max(1, (total + FAMILY_MEMBERS_PER_PAGE - 1) // FAMILY_MEMBERS_PER_PAGE)
    page = max(1, min(int(page or 1), pages))
    rows, _ = db.family_members(family, limit=FAMILY_MEMBERS_PER_PAGE,
                                offset=(page - 1) * FAMILY_MEMBERS_PER_PAGE)

    return templates.TemplateResponse(request, "family_members.html", {
        "active_tab": "dashboard", "title": f"Family {family}",
        "family_id": family, "members": rows, "total": total,
        "page": page, "pages": pages,
        # Pfam accessions are curated and resolve externally; a SAMPLE cluster id would collide
        # with an unrelated Pfam entry, so only the former is linked out.
        "pfam_url": (f"https://www.ebi.ac.uk/interpro/entry/pfam/{family}/"
                     if family.upper().startswith("PF") else None),
    })


@router.get("/search/peptide/{entry_id}", response_class=HTMLResponse)
def peptide_detail(request: Request, entry_id: str) -> Any:
    """
    Detail page for one peptide.

    :param request: Incoming request.
    :param entry_id: Release identifier.
    :returns: The rendered ``peptide_detail`` template.
    :raises HTTPException: 404 when no such real peptide exists.
    """
    peptide = db.get_peptide(entry_id)
    if peptide is None:
        raise HTTPException(status_code=404, detail=f"No peptide with id {entry_id}.")

    structures_dir = os.path.join(DATA_DIR, "sample_enriched_release", "structures")
    sdf_path = os.path.join(structures_dir, f"{entry_id}.sdf")
    sdf_content: Optional[str] = None
    if os.path.exists(sdf_path):
        try:
            with open(sdf_path, "r", encoding="utf-8") as fh:
                sdf_content = fh.read()
        except OSError as e:
            logger.debug(f"Could not read {sdf_path}: {e}")

    # Structure-derived analysis from the ColabFold PDB, when one exists: DSSP secondary structure,
    # Ramachandran phi/psi and pLDDT-based disorder. None for a peptide that was never folded.
    structure_analysis = None
    try:
        from sample.structure_analysis import analyze_structure
        structure_analysis = analyze_structure(os.path.join(structures_dir, f"{entry_id}.pdb"))
    except Exception as e:
        logger.debug(f"Structure analysis unavailable for {entry_id}: {e}")

    # Pre-computed comprehensive annotation: external predictors (macrel/hemopi2/amplify/anticp2/
    # toxinpred3), near-identity hits and family-HMM hits, with a circular caveat where it applies.
    # Populated by `build-annotations`; None until that batch has run for this peptide.
    annotation = None
    try:
        from sample import peptide_annotations
        annotation = peptide_annotations.get(entry_id)
    except Exception as e:
        logger.debug(f"Annotation unavailable for {entry_id}: {e}")

    # ADMET-lite: a sequence-only pharmacokinetics/toxicity summary. Computed from the sequence
    # (physicochemistry + rule flags), with the endpoints it will not fake -- permeability above all
    # -- listed in `gaps` rather than invented. None when there is no usable sequence.
    admet = None
    try:
        from sample.admet import admet_profile
        admet = admet_profile(peptide.get("sequence") or "") or None
        _link_admet_hemolysis_gap(admet, annotation)
    except Exception as e:
        logger.debug(f"ADMET profile unavailable for {entry_id}: {e}")

    # The template reads descriptors off the peptide object (`peptide.molecular_weight`), while
    # the old route passed them as a separate variable -- so the "no descriptors" branch always
    # won even when they existed. Merge them in, and keep passing `descriptors` for the table
    # that iterates it.
    merged = dict(peptide)
    merged.update(peptide.get("descriptors") or {})

    return templates.TemplateResponse(request, "peptide_detail.html", {
        "active_tab": "dashboard", "title": f"Peptide {entry_id}",
        "real_refs": _real_references(peptide.get("Reference")),
        "wiki_name": _wikipedia_query(peptide.get("names") or []),
        "charge_correction": _charge_correction(peptide.get("modifications") or []),
        "source_documents": _source_documents(peptide.get("LITERATURE"),
                                              peptide.get("PATENT")),
        "identity_links": _identity_links(peptide.get("cross_references")),
        "go_terms": _go_by_category(peptide.get("go_terms") or []),
        "peptide": merged, "descriptors": peptide.get("descriptors") or {},
        "activities": [
            act for act in (peptide.get("activities") or [])
            if act.get("target_organism") and str(act.get("target_organism")).strip()
            and str(act.get("target_organism")).strip().lower() not in ("none", "nan", "n/a", "-")
        ],
        "bibliography": peptide.get("bibliography") or [],
        "family": peptide.get("family"),
        "bioactivity": peptide.get("bioactivity") or [],
        "sdf_content": sdf_content,
        "secondary_structure": _secondary_structure(peptide.get("sequence")),
        "structure_analysis": structure_analysis,
        # Whether a folded structure exists at all, so the page offers a PDB download only when
        # there is really one to serve rather than linking to a 404.
        "has_pdb": os.path.exists(os.path.join(structures_dir, f"{entry_id}.pdb")),
        "admet": admet,
        "annotation": annotation,
    })


@router.get("/search/peptide/{entry_id}/json")
def peptide_detail_json(entry_id: str) -> Any:
    """
    The peptide record as JSON.

    :param entry_id: Release identifier.
    :returns: Peptide with descriptors, activities and papers.
    :raises HTTPException: 404 when no such real peptide exists.
    """
    peptide = db.get_peptide(entry_id)
    if peptide is None:
        raise HTTPException(status_code=404, detail=f"No peptide with id {entry_id}.")
    return peptide


@router.get("/search/peptide/{entry_id}/pdb")
def peptide_pdb(entry_id: str) -> Response:
    """
    The ColabFold-predicted structure for one peptide, as a PDB download.

    Distinct from the ``/sdf`` route below, which serves the RDKit-embedded small-molecule
    conformer the 3D viewer draws: this is the *folded* structure (the one
    :func:`sample.structure_analysis.analyze_structure` reads for DSSP, Ramachandran and pLDDT),
    and it only exists for peptides the corpus fold has actually reached. Not generated on
    demand -- folding is GPU-minutes, not a request-time operation.

    :param entry_id: Release identifier.
    :returns: A PDB file response.
    :raises HTTPException: 404 when the peptide is unknown or has never been folded.
    """
    from fastapi.responses import FileResponse

    pdb_path = os.path.join(DATA_DIR, "sample_enriched_release", "structures", f"{entry_id}.pdb")
    if os.path.exists(pdb_path):
        return FileResponse(pdb_path, media_type="chemical/x-pdb",
                            filename=f"{entry_id}.pdb")

    if db.get_peptide(entry_id) is None:
        raise HTTPException(status_code=404, detail=f"No peptide with id {entry_id}.")
    raise HTTPException(
        status_code=404,
        detail=(f"No predicted structure for {entry_id}. Structure prediction (ColabFold) has not "
                f"reached this peptide; the SDF conformer at /search/peptide/{entry_id}/sdf is "
                f"still available."))


@router.get("/search/peptide/{entry_id}/sdf")
def peptide_sdf(entry_id: str) -> Response:
    """
    The 3D structure for one peptide, pre-computed when available and embedded on demand if not.

    :param entry_id: Release identifier.
    :returns: An SDF response.
    :raises HTTPException: 404 when the peptide is unknown, 400 when no structure can be built.
    """
    from fastapi.responses import FileResponse

    sdf_path = os.path.join(DATA_DIR, "sample_enriched_release", "structures", f"{entry_id}.sdf")
    if os.path.exists(sdf_path):
        return FileResponse(sdf_path, media_type="chemical/x-mdl-sdfile")

    peptide = db.get_peptide(entry_id)
    if peptide is None:
        raise HTTPException(status_code=404, detail=f"No peptide with id {entry_id}.")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from sample.sequence import sequence_to_1_letter

        mol = None
        if peptide.get("SMILES"):
            mol = Chem.MolFromSmiles(str(peptide["SMILES"]))
        if mol is None and peptide.get("sequence"):
            mol = Chem.MolFromSequence(sequence_to_1_letter(str(peptide["sequence"])))
        if mol is None:
            raise HTTPException(status_code=400,
                                detail="Could not build a molecule from SMILES or sequence.")

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) < 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return Response(content=Chem.MolToMolBlock(mol),
                        media_type="chemical/x-mdl-sdfile")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SDF generation failed for {entry_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate the structure.")


@router.get("/datasets", response_class=HTMLResponse)
def datasets(request: Request) -> Any:
    """
    The downloadable release artifacts.

    :param request: Incoming request.
    :returns: The rendered ``datasets`` template.
    """
    listing = []
    for key, label in DATASET_LABELS.items():
        path = DATASET_PATHS[key]
        exists = os.path.exists(path)
        listing.append({
            "name": label,
            "file": os.path.basename(path) if exists else "N/A",
            "size": f"{os.path.getsize(path) / (1024 * 1024):.2f} MB" if exists else "0.00 MB",
            "status": "Available" if exists else "Not Generated",
            "key": key,
        })
    return templates.TemplateResponse(request, "datasets.html", {
        "active_tab": "datasets", "title": "Datasets",
        "datasets_list": listing,
    })


@router.get("/datasets/download/{key}")
def download_dataset(key: str) -> Response:
    """
    Serves one release artifact.

    :param key: Symbolic artifact name, resolved against :data:`DATASET_PATHS` so an arbitrary
        path cannot be requested.
    :returns: A file download.
    :raises HTTPException: 404 when the key is unknown or the file has not been generated.
    """
    from fastapi.responses import FileResponse

    path = DATASET_PATHS.get(key)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found or not generated yet.")
    return FileResponse(path, filename=os.path.basename(path),
                        media_type="application/octet-stream")


#: Records per page. The listing is one row of static markup each, so this is about what the
#: browser renders comfortably rather than what the server can produce.
EXTRACTIONS_PER_PAGE = 100

#: Cached ``(signature, items)`` for the extraction index. Rebuilding it means listing tens of
#: thousands of directory entries, which is far too slow to repeat on every page of a paged view.
_extraction_index: Optional[Tuple[Tuple[float, ...], List[dict]]] = None


def _extraction_dirs() -> List[Tuple[str, str, str]]:
    """:returns: ``(directory, source label, record type)`` for every extraction source."""
    dirs: List[Tuple[str, str, str]] = []
    literature = extraction_dir("literature")
    if os.path.isdir(literature):
        for source in sorted(os.listdir(literature)):
            path = os.path.join(literature, source)
            if os.path.isdir(path):
                dirs.append((path, source.upper(), "Literature"))
    patents = os.path.join(DATA_DIR, "patent_data_json", "google_patents")
    if os.path.isdir(patents):
        dirs.append((patents, "GOOGLE PATENTS", "Patent"))
    return dirs


def extraction_index() -> List[dict]:
    """
    Every extraction record, listed once and cached.

    The listing is ~46,000 files across the literature and patent directories. Reading it per
    request made the page slow before it was paged and would make it slower after, since each page
    would repeat the same full scan to slice 100 rows out of it.

    Invalidated on the directories' modification times, so a pipeline run that writes new
    extractions is picked up without a restart, while ordinary paging costs nothing.

    :returns: One dict per record, ordered by source then filename.
    """
    global _extraction_index

    dirs = _extraction_dirs()
    signature = tuple(os.path.getmtime(d) for d, _, _ in dirs)
    if _extraction_index is not None and _extraction_index[0] == signature:
        return _extraction_index[1]

    items: List[dict] = []
    for path, source, kind in dirs:
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                items.append({"id": name[:-5], "source": source,
                              "file_name": name, "type": kind})

    _extraction_index = (signature, items)
    logger.info(f"Indexed {len(items)} extraction records for the browser.")
    return items


@router.get("/extractions", response_class=HTMLResponse)
def extractions(request: Request, page: int = 1, q: str = "") -> Any:
    """
    Paged browser over the per-document extraction records.

    Paged because the unpaged version rendered every record into one table -- around 46,000 rows,
    several tens of megabytes of HTML. Browsers do parse that, eventually; the page took seconds
    to become interactive and the theme could not be applied until it had.

    :param request: Incoming request.
    :param page: 1-based page number, clamped into range so a hand-edited URL cannot produce an
        empty page or an exception.
    :param q: Optional case-insensitive filter over the record id and filename.
    :returns: The rendered ``extractions`` template.
    """
    items = extraction_index()

    needle = q.strip().lower()
    if needle:
        items = [i for i in items
                 if needle in i["id"].lower() or needle in i["file_name"].lower()]

    total = len(items)
    pages = max(1, (total + EXTRACTIONS_PER_PAGE - 1) // EXTRACTIONS_PER_PAGE)
    page = max(1, min(int(page or 1), pages))
    start = (page - 1) * EXTRACTIONS_PER_PAGE

    return templates.TemplateResponse(request, "extractions.html", {
        "active_tab": "extractions", "title": "LLM Extractions",
        "extracted_items": items[start:start + EXTRACTIONS_PER_PAGE],
        "page": page, "pages": pages, "total": total, "query": q,
        "start_index": start + 1 if total else 0,
        "end_index": min(start + EXTRACTIONS_PER_PAGE, total),
    })


@router.get("/extractions/view/{item_type}/{source}/{filename}", response_class=HTMLResponse)
def view_extraction_file(request: Request, item_type: str, source: str, filename: str) -> Any:
    """
    Shows one extraction record.

    :param request: Incoming request.
    :param item_type: ``"patent"`` or ``"literature"``.
    :param source: Source subdirectory, e.g. ``"pubmed"``.
    :param filename: Basename of the JSON record.
    :returns: The rendered ``view_extraction`` template.
    :raises HTTPException: 404 when the record does not exist, 400 for a traversing filename.
    """
    if os.path.basename(filename) != filename or os.path.basename(source) != source:
        raise HTTPException(status_code=400, detail="Invalid path component.")

    if item_type == "patent":
        path = os.path.join(extraction_dir("patent"), "google_patents", filename)
    else:
        path = os.path.join(extraction_dir("literature"), source, filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Extraction file not found.")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return templates.TemplateResponse(request, "view_extraction.html", {
        "active_tab": "extractions", "title": "View Extraction",
        "filename": filename, "json_data": data,
        "json_pretty": json.dumps(data, indent=4, ensure_ascii=False),
        "item_type": item_type, "source": source,
    })


@router.get("/models", response_class=HTMLResponse)
def models(request: Request) -> Any:
    """
    The model chooser, prediction form and recent jobs.

    :param request: Incoming request.
    :returns: The rendered ``models`` template.
    """
    from sample import model_registry, prediction_store
    from sample.inference import MAX_PEPTIDES_PER_JOB

    report_path = os.path.join(DATA_DIR, "sample_final_release", "validation_report.json")
    metrics = None
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)

    # Step 47's report, read as-is: the webserver only ever displays the comparison already
    # computed on disk, never triggers benchmark_all()/train_all() itself from a page load.
    benchmark_path = os.path.join(DATA_DIR, "sample_final_release", "benchmark_report.json")
    benchmarks = None
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as fh:
            raw_benchmarks = json.load(fh)
        benchmarks = [
            {"model_id": model_id, **result}
            for model_id, result in raw_benchmarks.items() if "error" not in result
        ]

    try:
        available = model_registry.list_models()
    except Exception as e:
        logger.error(f"Could not enumerate models: {e}")
        available = []

    try:
        prediction_store.init_db()
        session_id = request.cookies.get("sample_session_id") or getattr(request.state, "session_id", None)
        jobs = prediction_store.recent_jobs(limit=10, session_id=session_id) if session_id else []
    except Exception as e:
        logger.error(f"Could not read prediction jobs: {e}")
        jobs = []

    return templates.TemplateResponse(request, "models.html", {
        "active_tab": "models", "title": "ML Models",
        "metrics": metrics, "models": available, "jobs": jobs,
        "max_peptides": MAX_PEPTIDES_PER_JOB,
        "benchmarks": benchmarks,
    })


@router.get("/models/{model_id:path}/dataset", response_class=HTMLResponse)
def model_dataset_page(request: Request, model_id: str) -> Any:
    """
    Full validation report and train/test peptide split for one model.

    :param request: Incoming request.
    :param model_id: Id from the models page, e.g. ``antibacterial/general/any/classification/esm2``.
    :returns: The rendered ``model_dataset`` template.
    :raises HTTPException: 404 when the model id is not in the registry.
    """
    from sample import model_registry

    entry = model_registry.get_model_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_id!r}.")
    dataset = model_registry.get_model_dataset(model_id)
    # y-randomization only stores a null distribution for the primary metric (see
    # sample.ml_models._y_randomization) -- AUC for a classifier, R^2 for a regressor. The template
    # needs to know which row of the metrics table that null actually corresponds to.
    is_regression = bool(dataset) and str(dataset.get("learning_task")) == "regression"

    return templates.TemplateResponse(request, "model_dataset.html", {
        "active_tab": "models", "title": f"Dataset — {entry.get('label', model_id)}",
        "entry": entry, "dataset": dataset, "model_id": model_id,
        "primary_metric": "r2" if is_regression else "roc_auc",
    })


@router.get("/models/{model_id:path}/dataset/download")
def download_model_dataset(model_id: str) -> Response:
    """
    Serves the exact train/test peptide split as a CSV, for a model whose page is open.

    :param model_id: Id from the models page.
    :returns: A ``text/csv`` download with one ``entry_id,split`` row per peptide.
    :raises HTTPException: 404 when the model is unknown or has no dataset record.
    """
    from sample import model_registry

    record = model_registry.get_model_dataset(model_id)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"No dataset record for model {model_id!r}.")

    lines = ["entry_id,split"]
    lines += [f"{entry_id},train" for entry_id in record.get("train_entry_ids", [])]
    lines += [f"{entry_id},test" for entry_id in record.get("test_entry_ids", [])]
    filename = model_id.replace("/", "_") + "_dataset.csv"
    return Response(content="\n".join(lines) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/models/{model_id:path}/dataset/holdout.fasta")
def download_model_holdout_fasta(model_id: str) -> Response:
    """
    Serves a model's saved holdout as FASTA -- the same peptides step 47's benchmark scores the
    model and its reference tool against (see :func:`sample.benchmark.benchmark_holdout`), ready to
    hand to a tool this project has not wrapped (a web upload form, an unpackaged CLI) for an
    independent, apples-to-apples comparison.

    Prefers the ``<block>.holdout.fasta`` file train-ml writes beside every kept model's ``.pkl``
    (see :func:`sample.ml_models.train_all`) -- a plain file read, no recomputation. Falls back to
    computing it on demand (:func:`sample.benchmark.holdout_fasta`) for a release built before that
    existed, which still has ``test_entry_ids`` in its ``dataset.json`` but no sibling FASTA file.

    :param model_id: Id from the models page.
    :returns: A FASTA download.
    :raises HTTPException: 404 when the model is unknown or has no stored holdout.
    """
    from fastapi.responses import FileResponse

    from sample import model_registry
    from sample.benchmark import holdout_fasta

    entry = model_registry.get_model_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_id!r}.")
    filename = model_id.replace("/", "_") + "_holdout.fasta"

    fasta_path = os.path.splitext(entry["path"])[0] + ".holdout.fasta"
    if os.path.exists(fasta_path):
        return FileResponse(fasta_path, filename=filename, media_type="text/x-fasta")

    try:
        fasta = holdout_fasta(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(content=fasta, media_type="text/x-fasta",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/databases", response_class=HTMLResponse)
def databases(request: Request) -> Any:
    """
    The source-database descriptions.

    :param request: Incoming request.
    :returns: The rendered ``databases`` template.
    """
    return templates.TemplateResponse(request, "databases.html", {
        "active_tab": "databases", "title": "External Databases",
        "dbs": SOURCE_DATABASES,
    })


@router.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request) -> Any:
    """
    Exploratory figures and the anomaly report.

    :param request: Incoming request.
    :returns: The rendered ``analytics`` template.
    """
    anomaly_path = os.path.join(DATA_DIR, "sample_final_release", "anomaly_report.json")
    report = None
    if os.path.exists(anomaly_path):
        with open(anomaly_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)

    eda_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "static", "img", "eda")
    figures = ("sequence_length_dist", "net_charge_dist", "residue_composition",
               "umap_morgan", "umap_esm2", "target_organism_dist")
    keys = ("length_dist", "charge_dist", "residue_composition", "umap_morgan", "umap_esm2",
           "target_organism_dist")

    return templates.TemplateResponse(request, "analytics.html", {
        "active_tab": "analytics", "title": "Analytics & Anomalies",
        "anomaly_report": report,
        "eda_images": {key: os.path.exists(os.path.join(eda_dir, f"{name}.png"))
                       for key, name in zip(keys, figures)},
    })


# ==========================================
# BACKGROUND PREDICTION JOBS
# ==========================================
# Paths are unchanged from the Flask implementation because models.html and prediction_job.html
# already reference them.

@router.post("/predict")
async def submit_prediction(request: Request) -> JSONResponse:
    """
    Accepts a prediction request and queues it.

    Takes a JSON body (``model_id`` plus ``peptides`` or ``sequences``) or the HTML form's
    ``model_id`` and ``peptides`` fields, so one endpoint serves the page and scripted callers.

    :param request: Incoming request, read as JSON or form depending on content type.
    :returns: 202 with the job id and the URLs to poll, or a 4xx/503 error object.
    """
    from sample import model_registry, prediction_store
    from sample.inference import MAX_PEPTIDES_PER_JOB, parse_peptide_input, validate_sequence
    from sample.tasks import broker_available, predict_peptides

    payload: dict = {}
    try:
        payload = await request.json()
    except Exception:
        form = await request.form()
        payload = {"model_id": form.get("model_id"), "sequences": form.get("peptides")}

    model_id = str(payload.get("model_id") or "").strip()

    raw_peptides = payload.get("peptides")
    if isinstance(raw_peptides, list):
        peptides = [
            {"peptide_name": str(p.get("peptide_name") or f"peptide_{i + 1}"),
             "sequence": str(p.get("sequence", ""))}
            if isinstance(p, dict) else
            {"peptide_name": f"peptide_{i + 1}", "sequence": str(p)}
            for i, p in enumerate(raw_peptides)
        ]
    else:
        peptides = parse_peptide_input(str(payload.get("sequences") or ""))

    if not model_id:
        return JSONResponse({"error": "No model selected."}, status_code=400)
    if model_registry.get_model_entry(model_id) is None:
        return JSONResponse({"error": f"Unknown model id {model_id!r}."}, status_code=404)
    if not peptides:
        return JSONResponse({"error": "No peptide sequences supplied."}, status_code=400)
    if len(peptides) > MAX_PEPTIDES_PER_JOB:
        return JSONResponse(
            {"error": f"{len(peptides)} sequences submitted; the limit is "
                      f"{MAX_PEPTIDES_PER_JOB} per job."}, status_code=400)

    # A submission where nothing is scorable is rejected outright; a partially valid one is
    # queued, and the task records per-row errors for the bad entries.
    if all(validate_sequence(p["sequence"]) for p in peptides):
        return JSONResponse(
            {"error": f"No usable sequences ({validate_sequence(peptides[0]['sequence'])})."},
            status_code=400)

    if not broker_available():
        return JSONResponse(
            {"error": "The prediction queue is unavailable. The Redis broker is not reachable, "
                      "so no job was created."}, status_code=503)

    job_id = str(uuid.uuid4())
    session_id = (request.cookies.get("sample_session_id") or
                  getattr(request.state, "session_id", None) or
                  payload.get("session_id"))
    prediction_store.init_db()
    prediction_store.create_job(
        job_id, model_id, len(peptides),
        meta={"submitted_from": request.client.host if request.client else "",
              "n_peptides": len(peptides),
              "peptides": peptides,
              "session_id": session_id},
        session_id=session_id)

    try:
        async_result = predict_peptides.delay(job_id, model_id, peptides)
        prediction_store.attach_task_id(job_id, async_result.id)
    except Exception as e:
        logger.exception("Failed to dispatch prediction task")
        prediction_store.mark_finished(job_id, prediction_store.STATUS_FAILURE,
                                       error=f"Could not queue the job: {e}")
        return JSONResponse({"error": f"Could not queue the job: {e}", "job_id": job_id},
                            status_code=503)

    logger.info(f"Queued prediction job {job_id}: {len(peptides)} peptide(s), model {model_id}")
    return JSONResponse({
        "job_id": job_id, "status": prediction_store.STATUS_PENDING,
        "n_peptides": len(peptides),
        "status_url": f"/predict/status/{job_id}", "result_url": f"/predict/{job_id}",
    }, status_code=202)


@router.get("/predict/status/{job_id}")
def prediction_status(job_id: str) -> Any:
    """
    Reports a job's state. Polled by the results page while the spinner shows.

    :param job_id: Job identifier.
    :returns: The job row plus a ``done`` flag.
    :raises HTTPException: 404 when the id is unknown.
    """
    from sample import prediction_store

    job = prediction_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return {
        "job_id": job["job_id"], "status": job["status"], "model_id": job["model_id"],
        "n_peptides": job["n_peptides"], "n_completed": job["n_completed"],
        "error": job["error"],
        "done": job["status"] in prediction_store.TERMINAL_STATUSES,
    }


@router.get("/predict/{job_id}/json")
def prediction_results_json(job_id: str) -> Any:
    """
    :param job_id: Job identifier.
    :returns: The job row and its result rows.
    :raises HTTPException: 404 when the id is unknown.
    """
    from sample import prediction_store

    job = prediction_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return {"job": job, "results": prediction_store.get_predictions(job_id)}


@router.get("/predict/{job_id}", response_class=HTMLResponse)
def prediction_results(request: Request, job_id: str) -> Any:
    """
    The results page for one job: spinner while it runs, table when it finishes.

    :param request: Incoming request.
    :param job_id: Job identifier.
    :returns: The rendered ``prediction_job`` template, with 404 status for an unknown id.
    """
    from sample import model_registry, prediction_store

    job = prediction_store.get_job(job_id)
    if job is None:
        return templates.TemplateResponse(request, "prediction_job.html", {
            "active_tab": "models", "title": "Prediction job",
            "job": None, "job_id": job_id, "results": [], "model": None,
        }, status_code=404)

    return templates.TemplateResponse(request, "prediction_job.html", {
        "active_tab": "models", "title": f"Prediction {job_id[:8]}",
        "job": job, "job_id": job_id,
        "model": model_registry.get_model_entry(job["model_id"]),
        "results": prediction_store.get_predictions(job_id),
    })
