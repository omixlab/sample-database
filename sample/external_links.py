"""
Resolves stored identifiers to external resource URLs.

The database records *what* each reference is (``papers.source_type`` = pubmed / doi / patent /
DBAASP / CAMP / ADP / DRAMP / DBAMP, plus ``papers.source_id``) but never recorded where to find
it, so the detail page showed accessions as inert text. This module is the single place that maps
an identifier to a link.

**Confidence differs by source, and is recorded per entry.** ``verified`` means the pattern was
checked by resolving a known accession and confirming the right record came back -- not merely
that it follows a documented scheme. A caller that would rather show nothing than a broken link
can filter on it.

Every source-database link was wrong before this was checked, each differently, and all four had
been shipping dead links:

* **APD** pointed at ``/AP/database/query_output.php``, a path retired with APD2, returning 404
  for every accession. APD6 exposes no addressable entry URL at all, so it is reached by
  submitting :data:`APD_FORM_DEFAULTS` as a POST -- see :func:`form_target`.
* **CAMP** used a host that only 301-redirects.
* **DRAMP** omitted the ``dataset`` parameter its own links carry, and redirected.
* **dbAMP** had both a stale host (``awi`` -> ``ycclab``) and the wrong query parameter
  (``keyword`` -> ``keyword_value``).

The lesson worth keeping: an HTTP 200 does not mean the link works. Three of these sites answer
200 for a nonexistent accession, and APD6 returns a byte-identical page for every id under its
catch-all route. Each pattern here was confirmed by finding expected record content in the
response.
"""
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger("sample.external_links")


class LinkSpec(NamedTuple):
    """
    How to build a URL for one identifier namespace.

    :param label: Human-readable resource name, for the link text.
    :param template: URL with a single ``{id}`` placeholder. For a ``POST`` spec this is the form
        action and carries no placeholder, because the identifier travels in the body.
    :param verified: Whether the pattern was checked against data in this repository. ``False``
        means it follows the resource's documented scheme but has not been confirmed here.
    :param method: ``"GET"`` for an ordinary link, or ``"POST"`` when the resource exposes no
        addressable entry URL and must be reached by submitting a form.
    :param form_field: For a ``POST`` spec, the field carrying the identifier.
    :param form_defaults: For a ``POST`` spec, the other fields the endpoint requires, with the
        values its own form submits.
    :param id_pattern: Optional regex whose first group extracts the portion of the accession the
        resource expects, when that differs from the accession as stored.
    """
    label: str
    template: str
    verified: bool
    method: str = "GET"
    form_field: Optional[str] = None
    form_defaults: Optional[Dict[str, str]] = None
    id_pattern: Optional[str] = None


#: The fields APD6's search form submits alongside the identifier.
#:
#: The endpoint needs all of them, and needs the select boxes at their real defaults -- ``Any``
#: rather than the empty string. Posting ``ID`` alone, or with those fields blank, returns "No
#: Results Found" while still echoing the id back, which reads like a missing record rather than a
#: malformed query. Determined by parsing the live form and bisecting the payload.
APD_FORM_DEFAULTS: Dict[str, str] = {
    "HydrophobicPer": "Any", "Location": "Any", "Method": "Any", "Netcharge": "Any",
    "Type": "Any", "Sort": "ID", "Length": "", "LocationID": "", "Name": "", "Name2": "",
    "Name3": "", "Sequence": "", "Sequence2": "", "author": "", "info": "", "info2": "",
    "info3": "", "source": "",
}


#: Identifier namespace -> how to link it. Keys match ``papers.source_type`` exactly, lowercased.
LINK_SPECS: Dict[str, LinkSpec] = {
    # Verified: stable public resolvers.
    "pubmed": LinkSpec("PubMed", "https://pubmed.ncbi.nlm.nih.gov/{id}/", True),
    "doi": LinkSpec("DOI", "https://doi.org/{id}", True),
    "patent": LinkSpec("Google Patents", "https://patents.google.com/patent/{id}", True),
    # Verified: this is the form the DBAASP loader stores in activities.source_url.
    "dbaasp": LinkSpec("DBAASP", "https://dbaasp.org/peptide-card?id={id}", True),
    # Verified: stable, long-standing resolvers for the molecular archives.
    "uniprot": LinkSpec("UniProt", "https://www.uniprot.org/uniprotkb/{id}/entry", True),
    "pdb": LinkSpec("PDB", "https://www.rcsb.org/structure/{id}", True),
    "pubchem": LinkSpec("PubChem", "https://pubchem.ncbi.nlm.nih.gov/compound/{id}", True),
    "chembl": LinkSpec("ChEMBL",
                       "https://www.ebi.ac.uk/chembl/compound_report_card/{id}/", True),
    "taxonomy": LinkSpec("NCBI Taxonomy",
                         "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={id}", True),
    "alphafold": LinkSpec("AlphaFold DB", "https://alphafold.ebi.ac.uk/entry/{id}", True),
    "interpro": LinkSpec("InterPro", "https://www.ebi.ac.uk/interpro/entry/InterPro/{id}/", True),
    "pfam": LinkSpec("Pfam", "https://www.ebi.ac.uk/interpro/entry/pfam/{id}/", True),

    # Verified against the live site: APD6 has no addressable entry URL, so an accession is
    # reached by submitting its search form. The previous template pointed at the retired APD2
    # path ``/AP/database/query_output.php``, which now returns 404 for every accession -- so
    # every ADP citation in the release was a dead link.
    "adp": LinkSpec("APD6", "https://aps.unmc.edu/database/result", True,
                    method="POST", form_field="ID", form_defaults=APD_FORM_DEFAULTS,
                    # The form wants the digits without the ``AP`` prefix, padding intact.
                    id_pattern=r"AP(\d+)"),

    # Verified against the live sites. Each of these was previously wrong in a different way, and
    # all three were checked by fetching a known accession and confirming the record came back:
    #   CAMP  -- the ``www.camp3`` host only 301-redirects; this is the address it lands on.
    #            CAMPSQ1 returns Ceratotoxin-D.
    #   DRAMP -- the plain ``?id=`` form now redirects; the site's own links carry ``dataset``.
    #            DRAMP00005 returns "The All Information Of DRAMP00005".
    #   dbAMP -- moved host (``awi`` -> ``ycclab``) and the search parameter is ``keyword_value``,
    #            not ``keyword``; the old URL 404s. dbAMP_00001 returns its sequence.
    "camp": LinkSpec("CAMP", "https://camp.bicnirrh.res.in/seqDisp.php?id={id}", True),
    "dramp": LinkSpec(
        "DRAMP",
        "https://dramp.cpu-bioinfor.org/browse/All_Information.php?id={id}&dataset=General", True),
    "dbamp": LinkSpec(
        "dbAMP", "https://ycclab.cuhk.edu.cn/dbAMP/quicksearch.php?keyword_value={id}", True),

    # Verified against the live site: a bare catalog number resolves to the specific strain's
    # product page (25922 -> E. coli, FDA strain Seattle 1946), and a nonexistent number 404s
    # rather than serving a generic catch-all the way APD6 does for every id under its route --
    # the id itself is extracted from a target-organism string by
    # sample.nlp.extract_atcc_id, not stored as its own accession anywhere upstream.
    "atcc": LinkSpec("ATCC", "https://www.atcc.org/products/{id}", True),
}

#: Home pages, used when an accession cannot be deep-linked. Reaching the right database is more
#: useful than reaching nothing.
HOME_PAGES: Dict[str, str] = {
    "camp": "https://camp.bicnirrh.res.in/",
    "adp": "https://aps.unmc.edu/",
    "dramp": "https://dramp.cpu-bioinfor.org/",
    "dbamp": "https://ycclab.cuhk.edu.cn/dbAMP/",
    "dbaasp": "https://dbaasp.org/",
}

#: A DOI, wherever it appears in a longer string.
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# --------------------------------------------------------------------------------------------
# Cross-reference normalization
#
# The `cross_references` column is an unvalidated passthrough of whatever the source databases
# wrote. Alongside real accessions it carries sentinels ("Not available", "No entry found"),
# prose ("1MAG resolved by NMR.", "PDB ID: 1MAG Go to PDB"), comma-separated lists, and
# stringified Python dicts ({'cid': '46891554', 'cidUrl': ...}). Nothing downstream can join on
# it in that state.
#
# Each namespace is validated against the identifier format its issuer actually uses, so a value
# that survives is one that can be looked up. Anything else is dropped rather than guessed at --
# a wrong accession points at a real but different molecule, which is worse than none.
# --------------------------------------------------------------------------------------------

#: UniProtKB accession, per the official specification.
_UNIPROT_PATTERN = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b")

#: PDB entry id: a digit 1-9 followed by three alphanumerics.
_PDB_PATTERN = re.compile(r"\b([1-9][A-Za-z0-9]{3})\b")

#: PubChem CID, either bare or inside the stringified dict DBAASP stores.
_PUBCHEM_PATTERN = re.compile(r"'cid'\s*:\s*'(\d+)'|\b(\d{3,})\b")

#: ChEMBL molecule id.
_CHEMBL_PATTERN = re.compile(r"\b(CHEMBL\d+)\b", re.IGNORECASE)

#: Values that mean "the source had nothing", written where an accession should be.
_NULL_SENTINELS = frozenset({
    "not available", "no entry found", "none", "n/a", "na", "null", "-", "",
    "not found", "unknown", "not applicable", "reference id: ref",
})

_NAMESPACE_PATTERNS = {
    "uniprot": _UNIPROT_PATTERN,
    "pdb": _PDB_PATTERN,
    "pubchem": _PUBCHEM_PATTERN,
    "chembl": _CHEMBL_PATTERN,
}


def normalize_identifier(namespace: str, raw: Any) -> List[str]:
    """
    Extracts every valid accession of one namespace from a raw source value.

    :param namespace: ``uniprot``, ``pdb``, ``pubchem`` or ``chembl``.
    :param raw: The value as the source database wrote it.
    :returns: Distinct valid accessions in order of appearance; empty when the value carries
        none. A sentinel like "Not available" yields ``[]`` rather than being stored as if it
        were an identifier.
    """
    text = str(raw or "").strip()
    if not text or text.lower() in _NULL_SENTINELS:
        return []

    pattern = _NAMESPACE_PATTERNS.get(str(namespace).strip().lower())
    if pattern is None:
        return [text]

    found: List[str] = []
    for match in pattern.finditer(text):
        value = next((g for g in match.groups() if g), None)
        if value is None:
            continue
        value = value.upper() if namespace.lower() != "pubchem" else value
        if value not in found:
            found.append(value)

    # "PDB ID: 1MAG Go to PDB" filed under uniprot is a PDB id in the wrong column, not a UniProt
    # accession -- the UniProt pattern would not match it anyway, but the guard is explicit.
    return found


def normalize_cross_references(blob: Any) -> Dict[str, List[str]]:
    """
    Parses a pipe-delimited ``cross_references`` string into validated per-namespace accessions.

    :param blob: The stored column value, e.g.
        ``"uniprot:P81941 | pdb:Not available | pubchem:{'cid': '46891554', ...}"``.
    :returns: ``{namespace: [accession, ...]}``, omitting namespaces left with nothing valid.
    """
    out: Dict[str, List[str]] = {}
    if not blob:
        return out
    for token in str(blob).split("|"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        namespace, _, value = token.partition(":")
        namespace = namespace.strip().lower()
        for accession in normalize_identifier(namespace, value):
            bucket = out.setdefault(namespace, [])
            if accession not in bucket:
                bucket.append(accession)
    return out


def cross_reference_links(blob: Any) -> List[Dict[str, Any]]:
    """
    Turns a raw ``cross_references`` value into renderable, resolvable links.

    :param blob: The stored column value.
    :returns: One link record per valid accession found.
    """
    links: List[Dict[str, Any]] = []
    for namespace, accessions in normalize_cross_references(blob).items():
        for accession in accessions:
            links.append(describe(namespace, accession))
    return links

#: A patent number as it appears in the LITERATURE/PATENT filename tokens, e.g.
#: "google_patents:US8394762B2.json".
_PATENT_PATTERN = re.compile(r"\b([A-Z]{2}\d{4,}[A-Z]?\d*)\b")


def strip_float_suffix(identifier: str) -> str:
    """
    Drops the fractional part a numeric accession picks up by travelling through a float.

    pandas types an integer column holding any NaN as float64, so PMID 10075760 is written out as
    ``"10075760.0"`` -- the form all 2,217 PubMed accessions in the shipped release are stored in.
    Both the URL builder and the display path need the same cleanup, and they must agree: a page
    that links one identifier while printing another is worse than either mistake alone.

    :param identifier: An accession as stored.
    :returns: The accession without a trailing ``.<digits>``; unchanged if it has none.
    """
    return re.sub(r"\.\d+$", "", identifier)


def build_url(source_type: Any, source_id: Any) -> Optional[str]:
    """
    Builds the external URL for one identifier.

    :param source_type: Namespace, e.g. ``"pubmed"`` or ``"DBAASP"``. Case-insensitive.
    :param source_id: The accession.
    :returns: A URL, the resource home page when the accession cannot be deep-linked, or ``None``
        when the namespace is unknown. For a ``POST`` namespace this is the form action, which is
        not by itself a link to the record -- use :func:`describe`, which carries the form fields
        needed to reach it.
    """
    if not source_type or not source_id:
        return None
    key = str(source_type).strip().lower()
    identifier = str(source_id).strip()
    if not identifier:
        return None

    spec = LINK_SPECS.get(key)
    if spec is None:
        return HOME_PAGES.get(key)

    if key == "doi":
        match = _DOI_PATTERN.search(identifier)
        identifier = match.group(0) if match else identifier
    elif key == "pubmed":
        # A PMID that has travelled through a float loses its fractional part FIRST. pandas types
        # an integer column holding any NaN as float64, so PMID 10075760 is written out as
        # "10075760.0", and every one of the 2,217 PubMed identifiers in the shipped release is
        # stored that way. Stripping non-digits without this deletes the dot but KEEPS its
        # trailing zero -- 10075760 becomes 100757600, a different (nonexistent) article, linked
        # with nothing anywhere to show that the identifier was mangled.
        identifier = strip_float_suffix(identifier)
        digits = re.sub(r"\D", "", identifier)
        if not digits:
            return None
        identifier = digits

    if spec.method == "POST":
        return spec.template

    return spec.template.format(id=identifier)


def form_target(source_type: Any, source_id: Any) -> Optional[Dict[str, Any]]:
    """
    Builds the form submission that reaches a record in a POST-only resource.

    APD6 is the one such resource here: it retired its addressable entry URLs, so an accession is
    reachable only by submitting its search form. A template renders this as a small form styled
    as a link rather than as an anchor.

    :param source_type: Namespace.
    :param source_id: The accession as stored, e.g. ``AP00001``.
    :returns: ``{"action", "fields"}`` ready to render, or ``None`` when the namespace is not
        POST-based or the accession does not match the expected shape.
    """
    key = str(source_type or "").strip().lower()
    spec = LINK_SPECS.get(key)
    if spec is None or spec.method != "POST" or not spec.form_field:
        return None

    identifier = str(source_id or "").strip()
    if not identifier:
        return None

    if spec.id_pattern:
        match = re.search(spec.id_pattern, identifier, re.IGNORECASE)
        if not match:
            return None
        identifier = match.group(1)

    return {
        "action": spec.template,
        "fields": {**(spec.form_defaults or {}), spec.form_field: identifier},
    }


def describe(source_type: Any, source_id: Any) -> Dict[str, Any]:
    """
    Builds a renderable link record for one identifier.

    :param source_type: Namespace.
    :param source_id: The accession.
    :returns: ``{"source_type", "source_id", "label", "url", "verified", "form"}``. ``url`` is
        ``None`` when the identifier cannot be resolved, which the template renders as inert text.
        ``form`` is set only for POST-only resources such as APD6, and a template that finds it
        must submit the form rather than follow ``url``.
    """
    key = str(source_type or "").strip().lower()
    spec = LINK_SPECS.get(key)
    # The accession is shown as well as linked, and the stored PubMed form carries a float's
    # fractional part. Cleaning only the URL would leave the page printing "PubMed: 10075760.0"
    # beside a link to 10075760.
    shown = strip_float_suffix(str(source_id)) if key == "pubmed" and source_id is not None \
        else source_id
    return {
        "source_type": source_type,
        "source_id": shown,
        "label": spec.label if spec else str(source_type or "").upper(),
        "url": build_url(source_type, source_id),
        "verified": bool(spec.verified) if spec else False,
        "form": form_target(source_type, source_id),
    }


def annotate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds ``label``, ``url`` and ``verified`` to each paper row.

    :param papers: Rows from the ``papers`` table.
    :returns: The same rows, annotated. Input is not mutated.
    """
    return [{**paper, **describe(paper.get("source_type"), paper.get("source_id"))}
            for paper in papers or []]


def links_from_reference_tokens(reference: Any) -> List[Dict[str, Any]]:
    """
    Extracts links from the ``LITERATURE``/``PATENT`` filename tokens.

    Those columns hold ``source:filename.json`` strings such as
    ``google_patents:US8394762B2.json`` and ``pubmed:<uuid>.json``. The patent number is a real,
    linkable identifier; the literature uuid is an internal extraction id and is not, so only
    patents produce a link here.

    :param reference: The joined LITERATURE/PATENT string.
    :returns: Link records, empty when nothing linkable was found.
    """
    if not reference:
        return []
    found: List[Dict[str, Any]] = []
    seen = set()
    for token in re.split(r"[;|]", str(reference)):
        if "google_patents" not in token:
            continue
        match = _PATENT_PATTERN.search(token.split(":", 1)[-1])
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            found.append(describe("patent", match.group(1)))
    return found
