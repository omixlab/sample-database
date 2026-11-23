"""
Cell line identity, via the Cellosaurus API, for grouping activities.csv's `target_cell_line` into
trainable categories.

Real data has 106 distinct `target_cell_line` spellings for far fewer actual cell lines --
`MDA-MB-231`, `MDA-MB231` and `MDAMB-231` are the same breast-cancer line typed three ways by three
sources -- and even the best-populated single line (A549, 22 rows) is an order of magnitude below
`sample.ml_models.MIN_POSITIVES_PER_TASK`. A hand-written name-to-cancer-type table was tried first
and caught a real error before it shipped: BGC-823 and Bel-7402 "look like" gastric and liver cancer
lines from their names, but Cellosaurus's own STR-profiling-backed record shows both are
cross-contaminated HeLa (cervical cancer) derivatives -- a well-documented, published finding about
Chinese cell banks of that era, and not something a name-based guess could catch. Cellosaurus is the
field's own controlled vocabulary for this, maintained by SIB and cross-referenced to NCIt disease
terms and UBERON tissue terms, so it is the source of truth here rather than this project's opinion.

Results are cached to disk (:data:`CACHE_PATH`) because a training run may look up the same handful
of cell lines repeatedly and the public API should not be re-queried every time a pipeline step runs.
"""
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("sample.cell_lines")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Where lookups are cached, so repeated pipeline runs do not re-hit the public API for the same
#: handful of cell line names every time.
CACHE_PATH: str = os.path.join(DATA_DIR, "cell_lines", "cellosaurus_cache.json")

CELLOSAURUS_SEARCH_URL: str = "https://api.cellosaurus.org/search/cell-line"

#: Values seen in `target_cell_line` that are not cell lines at all -- an assay-type code and a
#: receptor gene name, each landing in this column on exactly one row of the real release.
_NOT_A_CELL_LINE = frozenset({"ec50", "ccr5"})

#: Disease-label keyword -> coarse cancer-type category, checked in order (first match wins).
#: Restricted to categories with enough real-data support in this project's activities table to be
#: worth a distinct label; anything else falls to "other_cancer" rather than adding a category with
#: no chance of ever clearing `MIN_POSITIVES_PER_TASK`.
_DISEASE_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("leukemia", "leukemia_lymphoma"), ("lymphoma", "leukemia_lymphoma"),
    ("myeloma", "leukemia_lymphoma"),
    ("lung", "lung_cancer"),
    ("breast", "breast_cancer"), ("mammary", "breast_cancer"),
    ("colorectal", "colorectal_cancer"), ("colon", "colorectal_cancer"),
    ("hepatocellular", "liver_cancer"), ("hepat", "liver_cancer"), ("liver", "liver_cancer"),
    ("glioblastoma", "glioma"), ("glioma", "glioma"), ("astrocytoma", "glioma"),
    ("cervi", "cervical_cancer"),
    ("prostate", "prostate_cancer"),
    ("melanoma", "melanoma"),
    ("gastric", "gastric_cancer"), ("stomach", "gastric_cancer"),
    ("pancrea", "pancreatic_cancer"),
    ("osteosarcoma", "osteosarcoma"),
    ("ovarian", "ovarian_cancer"), ("ovary", "ovarian_cancer"),
    ("renal", "kidney_cancer"), ("kidney", "kidney_cancer"),
    ("sarcoma", "sarcoma"),
    ("carcinoma", "other_carcinoma"), ("adenocarcinoma", "other_carcinoma"),
    ("neoplas", "other_cancer"), ("tumour", "other_cancer"), ("tumor", "other_cancer"),
)


def _normalize(text: str) -> str:
    """:returns: `text` uppercased with everything but letters and digits stripped."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def clean_cell_line_name(raw: str) -> Optional[str]:
    """
    :param raw: A `target_cell_line` value, uncleaned.
    :returns: The trimmed value, or ``None`` for an empty string or a known non-cell-line entry
        (:data:`_NOT_A_CELL_LINE` -- an assay-type code and a gene name that landed in this column
        by data-entry error).
    """
    value = str(raw).strip()
    if not value or value.lower() in _NOT_A_CELL_LINE:
        return None
    return value


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {CACHE_PATH}: {e}")
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)


def _query_cellosaurus(name: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """
    :param name: Cell line name to search for.
    :param timeout: Request timeout, in seconds.
    :returns: Raw candidate cell-line entries from the search API, or ``[]`` on any network or
        parse failure -- a lookup this project cannot reach degrades to "unclassified", never to a
        crashed training run.
    """
    query = urllib.parse.quote(f'id:"{name}"')
    url = f"{CELLOSAURUS_SEARCH_URL}?q={query}&fields=id,ac,derived-from-site,di&rows=20"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        logger.warning(f"Cellosaurus lookup failed for {name!r}: {e}")
        return []
    return data.get("Cellosaurus", {}).get("cell-line-list", [])


def _find_exact(name: str, timeout: float) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    One search attempt: queries Cellosaurus for `name` and returns the shortest exactly-matching
    candidate, or ``None``.

    Preferring the SHORTEST matching name is what keeps this pointed at the parent line rather
    than an engineered derivative (`A549 Cas9`, `A549 HIF1A -/-`, ...) that Cellosaurus's
    relevance ranking often returns ahead of the plain parent entry.
    """
    target = _normalize(name)
    candidates = _query_cellosaurus(name, timeout=timeout)
    exact = []
    for entry in candidates:
        names = [n["value"] for n in entry.get("name-list", [])]
        if any(_normalize(n) == target for n in names):
            exact.append((names[0], entry))
    if not exact:
        return None
    exact.sort(key=lambda pair: len(pair[0]))
    return exact[0]


def lookup_cell_line(name: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Finds the Cellosaurus entry for one cell line name, matching case/punctuation-insensitively --
    Cellosaurus's own canonical spelling is not always the one a source paper used, and not always
    reachable through its search index the same way: `id:"A549"` returns 100 engineered
    derivatives and never the plain parent line, while `id:"A-549"` (Cellosaurus's own canonical,
    hyphenated spelling) finds it immediately. A second search with a hyphen inserted between a
    leading letter run and a trailing digit run is retried before giving up, since this covers the
    exact case observed in this project's real `target_cell_line` values.

    :param name: Cell line name, e.g. from `target_cell_line`.
    :param timeout: Request timeout, in seconds.
    :returns: ``{"accession", "name", "tissue": [...], "disease": [...]}`` for the best match, or
        ``None`` when nothing matches exactly even after the hyphenated retry.
    """
    if not _normalize(name):
        return None

    found = _find_exact(name, timeout)
    if found is None:
        hyphenated = re.sub(r"^([A-Za-z]+)(\d.*)$", r"\1-\2", name.strip())
        if hyphenated != name.strip():
            found = _find_exact(hyphenated, timeout)
    if found is None:
        return None

    best_name, best = found
    return {
        "accession": best.get("accession-list", [{}])[0].get("value", ""),
        "name": best_name,
        "tissue": [t["site"]["value"] for t in best.get("derived-from-site-list", [])
                  if t.get("site", {}).get("value")],
        "disease": [d["label"] for d in best.get("disease-list", []) if d.get("label")],
    }


def _classify_disease(disease_labels: Iterable[str]) -> Optional[str]:
    """
    :param disease_labels: Cellosaurus disease-list labels for one cell line.
    :returns: The first matching category from :data:`_DISEASE_KEYWORDS`, ``"other_cancer"`` when
        a disease is recorded but matches no known keyword, or ``None`` when no disease is
        recorded at all (Cellosaurus's own signal for "not a tumour-derived line").
    """
    text = " ".join(disease_labels).lower()
    if not text.strip():
        return None
    for keyword, category in _DISEASE_KEYWORDS:
        if keyword in text:
            return category
    return "other_cancer"


def _tissue_category(tissues: Iterable[str]) -> Optional[str]:
    """
    :param tissues: Cellosaurus derived-from-site labels for one (non-cancer) cell line.
    :returns: A slugified tissue name for the first recorded site, or ``None`` when none is
        recorded.
    """
    for tissue in tissues:
        slug = re.sub(r"[^a-z0-9]+", "_", tissue.strip().lower()).strip("_")
        if slug and slug != "not_specified":
            return slug
    return None


def classify_cell_line(name: str, cache: Optional[Dict[str, Any]] = None,
                       timeout: float = 10.0) -> Optional[Tuple[str, bool]]:
    """
    Categorizes one cell line by cancer type (if tumour-derived) or source tissue (if not),
    via Cellosaurus, with disk caching.

    :param name: Cell line name.
    :param cache: An already-loaded cache dict to read/write in place (avoids a disk round trip
        per name when classifying many); :func:`_load_cache`/:func:`_save_cache` used directly
        when omitted.
    :param timeout: Request timeout, in seconds, for an uncached name.
    :returns: ``(category, is_cancer)``, or ``None`` when the name is not a real cell line, is not
        found in Cellosaurus, or the lookup could not be reached.
    """
    cleaned = clean_cell_line_name(name)
    if not cleaned:
        return None

    owns_cache = cache is None
    if owns_cache:
        cache = _load_cache()
    key = _normalize(cleaned)

    if key not in cache:
        entry = lookup_cell_line(cleaned, timeout=timeout)
        cache[key] = entry
        if owns_cache:
            _save_cache(cache)

    entry = cache.get(key)
    if not entry:
        return None

    category = _classify_disease(entry.get("disease", []))
    if category:
        return category, True
    tissue = _tissue_category(entry.get("tissue", []))
    if tissue:
        return tissue, False
    return None


def build_cell_line_categories(names: Iterable[str],
                               timeout: float = 10.0) -> Dict[str, Tuple[str, bool]]:
    """
    Classifies many cell line names in one pass, loading and saving the disk cache once rather
    than per name.

    :param names: Cell line names to classify (duplicates are fine and only looked up once).
    :param timeout: Request timeout, in seconds, per uncached name.
    :returns: Cleaned cell line name (:func:`clean_cell_line_name`) -> ``(category, is_cancer)``,
        omitting names that could not be classified.
    """
    cache = _load_cache()
    out: Dict[str, Tuple[str, bool]] = {}
    dirty = False
    for raw in names:
        cleaned = clean_cell_line_name(raw)
        if not cleaned:
            continue
        key = _normalize(cleaned)
        if key not in cache:
            cache[key] = lookup_cell_line(cleaned, timeout=timeout)
            dirty = True
            time.sleep(0.1)  # a light rate limit; this is a shared public service
        entry = cache[key]
        if not entry:
            continue
        category = _classify_disease(entry.get("disease", []))
        if not category:
            category = _tissue_category(entry.get("tissue", []))
            is_cancer = False
        else:
            is_cancer = True
        if category:
            out[cleaned] = (category, is_cancer)
    if dirty:
        _save_cache(cache)
    return out
