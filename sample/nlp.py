import logging
import re
import os
import math
import functools

logger = logging.getLogger("sample.nlp")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def extract_context_window(full_text: str, target_term: str = None, token_window: int = 1000) -> str:
    """
    Extracts a ~1000-token (~4000 character) text window centered on target_term from full_text.
    Independent of extraction technique, for downstream validation steps.
    """
    if not full_text or not isinstance(full_text, str):
        return ""

    char_window = token_window * 4  # Approx 4 characters per token (~1000 tokens)
    if len(full_text) <= char_window:
        return full_text.strip()

    if target_term and isinstance(target_term, str):
        match = re.search(re.escape(target_term), full_text, re.IGNORECASE)
        if match:
            center = (match.start() + match.end()) // 2
            start = max(0, center - char_window // 2)
            end = min(len(full_text), center + char_window // 2)
            return full_text[start:end].strip()

    return full_text[:char_window].strip()

import os
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

_aa3_list = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val", "Asx", "Glx", "Xaa", "Orn", "Dab", "Dap"]
_AA3_UPPER = {aa.upper() for aa in _aa3_list}
_aa3_all = []
for aa in _aa3_list:
    _aa3_all.extend([aa, aa.upper(), aa.lower()])
# Atomic groups ((?>...), Python 3.11+) around each optional stereo-prefix are load-bearing, not
# cosmetic: the D/L prefix character class overlaps with valid amino-acid letters themselves
# (D=Asp, L=Leu), so without atomic grouping the engine can re-partition the same run of letters
# between "prefix" and "residue" in exponentially many ways once backtracking kicks in. Measured
# on a plain run of repeated letters (the kind produced by OCR artifacts, table formatting, or
# gene-name runs in real papers): the non-atomic version took >120s on just 45 characters and
# would take vastly longer on anything realistic; the atomic version matches 5000 characters in
# under a millisecond, with identical results on legitimate sequences.
AA3 = r'\[?(?>(?:[DLdl]-?)?)(?:' + '|'.join(_aa3_all) + r')\]?'
TERM_N = r'(?:NH2-|Ac-|AC-|nh2-|ac-)?'
TERM_C = r'(?:-NH2|-COOH|-cooh|-nh2)?'
SEQ_3LETTER = re.compile(rf'\b{TERM_N}(?:{AA3}[-\s]){{4,}}{AA3}{TERM_C}\b')

AA1 = r'\[?(?>(?:[DLdl]-?)?)[ACDEFGHIKLMNPQRSTVWY]\]?'
SEQ_EXACT = re.compile(rf'\b(?:[Nn][Hh]2-|[Aa]c-)?((?:{AA1}-?){{4,}}{AA1})(?:-[Nn][Hh]2|-[Cc][Oo][Oo][Hh])?\b')

SEQ_ALPHANUM = re.compile(r'\b(?:[A-Z]{2,6}|[a-z][A-Z]{1,5})(?:-?\d{1,4}|-alpha|-beta|-gamma)\b')

import ahocorasick

class AhocorasickWrapper:
    """
    Thin wrapper around pyahocorasick's C-native Automaton. Deliberately does NOT keep a
    parallel Python dict of keywords -- for large keyword sets (e.g. get_organism_kp() loads
    ~2M taxonomy name variants) a plain dict duplicate roughly doubled this object's memory
    footprint for no functional benefit (it was only ever used for an emptiness check).
    """
    def __init__(self, case_sensitive: bool = False) -> None:
        """
        :param case_sensitive: When ``False`` (the default) keywords and search text are
            lower-cased, so a match is case-insensitive.
        """
        self.auto = ahocorasick.Automaton()
        self.case_sensitive = case_sensitive
        self._built = False
        self._num_keywords = 0

    def add_keyword(self, keyword: str, clean_name: Optional[str] = None) -> None:
        """
        Registers one keyword. Cheap; the automaton is compiled lazily on first search.

        :param keyword: Surface form to search for.
        :param clean_name: Value returned when this keyword matches -- the canonical name the
            surface form maps to. Defaults to the keyword itself.
        :returns: ``None``.
        """
        if not self.case_sensitive:
            keyword = keyword.lower()
        if clean_name is None:
            clean_name = keyword
        self.auto.add_word(keyword, (len(keyword), clean_name))
        self._num_keywords += 1
        self._built = False

    def extract_keywords(self, text: str, span_info: bool = False) -> List[Any]:
        """
        Finds every registered keyword in `text`, respecting word boundaries.

        A match whose neighbouring character is alphanumeric is discarded, so "Rana" does not
        match inside "Ranatuerin".

        :param text: Text to search.
        :param span_info: When ``True``, return ``(clean_name, start, end)`` tuples instead of
            bare names, so a caller can pair a match with a nearby value by position.
        :returns: The canonical names of the matches, or span tuples when `span_info` is set.
        """
        if not self._built:
            if self._num_keywords > 0:
                self.auto.make_automaton()
            self._built = True

        search_text = text if self.case_sensitive else text.lower()
        results = []

        if self._num_keywords == 0:
            return results

        for end_index, (length, clean_name) in self.auto.iter(search_text):
            start_index = end_index - length + 1

            # Simple word boundary check
            if start_index > 0 and search_text[start_index - 1].isalnum():
                continue
            if end_index < len(search_text) - 1 and search_text[end_index + 1].isalnum():
                continue

            if span_info:
                results.append((clean_name, start_index, end_index + 1))
            else:
                results.append(clean_name)
        return results

    def iter_raw(self, text: str) -> Iterator[Tuple[Any, int, int]]:
        """
        Same underlying automaton scan as extract_keywords(span_info=True), but WITHOUT this
        class's own word-boundary check against `text` itself. For callers matching against a
        transformed copy of some other reference text (e.g. find_whitespace_broken_sequences'
        whitespace-collapsed copy) -- boundary characters in the transformed copy don't correspond
        to real word boundaries in the reference text, so the check needs to happen against that
        reference text instead, which only the caller has.
        """
        if not self._built:
            if self._num_keywords > 0:
                self.auto.make_automaton()
            self._built = True

        search_text = text if self.case_sensitive else text.lower()
        if self._num_keywords == 0:
            return

        for end_index, (length, clean_name) in self.auto.iter(search_text):
            start_index = end_index - length + 1
            yield (clean_name, start_index, end_index + 1)

# Keep KeywordProcessor name so callers don't break
KeywordProcessor = AhocorasickWrapper
from sample.config import PEPTIDE_NAMES_TXT, UMLS_LEXICON_TXT, NAME_SEQUENCE_MAP_TSV

def _normalize_peptide_key(name: str) -> str:
    """Catalogue lookup key: hyphen/space variation is typographic in peptide nomenclature."""
    return re.sub(r'[\s\-]+', ' ', str(name)).strip().upper()


@functools.lru_cache(maxsize=None)
def get_name_sequence_map() -> Dict[str, str]:
    """
    name.upper() -> canonical sequence, built offline from CAMP/DBAASP/DRAMP/ADP by
    scripts/build_name_sequence_lookup.py. Lets a paper that names a well-known peptide
    ("magainin 2", "LL-37") without ever restating its sequence nearby recover the real sequence
    instead of producing a name-only record -- the gap find_local_aliases (in-document,
    per-paper aliasing only) can't close.
    """
    mapping = {}
    if NAME_SEQUENCE_MAP_TSV.exists():
        with open(NAME_SEQUENCE_MAP_TSV, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2 and parts[0] and parts[1]:
                    mapping[_normalize_peptide_key(parts[0])] = parts[1]
    return mapping

@functools.lru_cache(maxsize=1)
def get_catalogued_name_heads() -> Set[str]:
    """
    Leading words of catalogued peptide names ("GRAMICIDIN" from "Gramicidin S"), for recognizing
    a bare family name as peptide nomenclature. Only multi-token entries contribute a head, so a
    single-word catalogue entry does not license itself through this path -- it already matches
    the exact lookup.

    The remainder after the head must be a bare DESIGNATOR -- a letter, a number, or a short code
    ("Gramicidin S", "Nisin Z", "Temporin B", "Defensin 1"). That is what peptide nomenclature
    looks like: a family name followed by the member's label.

    Without that requirement this set was the single largest source of junk peptide names in the
    output. Any entry beginning with a generic word contributed that word as a head, and a head
    outranks the dictionary gate below, so "Cationic antimicrobial protein CAP37" licensed
    "cationic", "Pyruvate synthase" licensed "pyruvate", "Phenol soluble modulin alpha 4" licensed
    "phenol", and "Heparin cofactor 2" licensed "heparin" -- each then appearing in the database as
    a peptide in its own right, 124 times in the case of "pyruvate". Those all have a following
    WORD rather than a designator, which is precisely what distinguishes a descriptive entry title
    from a family name. Applying it drops the head set from 1,685 to 935 while keeping every real
    family checked (gramicidin, nisin, temporin, magainin, brevinin, melittin, defensin).
    """
    heads = set()
    for name in get_name_sequence_map():
        head, _, rest = name.partition(" ")
        if rest and len(head) >= 5 and head.isalpha() and _NAME_DESIGNATOR.match(rest.strip()):
            heads.add(head)
    return heads


# "S", "B", "Z", "2", "1SPB", "U2", "A7" -- a member label, not another descriptive word.
_NAME_DESIGNATOR = re.compile(r'^[A-Z]{0,3}[-\s]?\d{0,4}[A-Z]?$')


@functools.lru_cache(maxsize=None)
def get_known_peptide_names_kp() -> AhocorasickWrapper:
    """
    Aho-Corasick automaton over every name key in get_name_sequence_map(), for finding one of
    these known peptides by name anywhere in a document's raw text. Hyphen vs space is
    typographic variation in peptide nomenclature ("LL-37" / "LL 37"), not semantic, so each name
    is also registered with the opposite spelling, mapped back to the map's own key (clean_name)
    so the lookup in extractor.py still hits.
    """
    kp = KeywordProcessor(case_sensitive=False)
    for name in get_name_sequence_map():
        kp.add_keyword(name, name)
        if '-' in name:
            kp.add_keyword(name.replace('-', ' '), name)
        elif ' ' in name:
            kp.add_keyword(name.replace(' ', '-'), name)
    return kp

@functools.lru_cache(maxsize=None)
def get_seq_names_kp() -> AhocorasickWrapper:
    """
    Automaton over the curated peptide-name list.

    :returns: A shared, lazily-built automaton mapping each known peptide name to itself.
    """
    kp = KeywordProcessor(case_sensitive=False)
    if PEPTIDE_NAMES_TXT.exists():
        with open(PEPTIDE_NAMES_TXT, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    kp.add_keyword(name)
    
    # Vector 4 Correction: Add macro-classes directly to get_seq_names_kp()
    macro_classes = ["COLICIN", "BACTERIOCIN", "NISIN", "MUREIN", "CASIDIN", "CIDIN", "DEFENSIN", "MAGAININ"]
    for mc in macro_classes:
        kp.add_keyword(mc)
    return kp

#: Shortest catalogued sequence allowed to become an Aho-Corasick SEARCH KEY.
#:
#: Every entry in the catalogued word list is matched verbatim against every document, so a short
#: entry is not a lookup -- it is a substring search that fires on ordinary prose. Measured in the
#: first run under the new gates: "IP" was extracted as a peptide from "prostacyclin (IP)
#: receptors", "FG" from "FGF", "PG" from "protegrin PG-1" and "DPG", "PP" from photodynamic-therapy
#: prose, and "DEVD" from "GDEVDGSGK". All five are genuine entries in the catalogued sources, and
#: `is_valid_sequence` accepts all of them, so nothing downstream could reject them.
#:
#: 5 matches SEQ_EXACT's own floor, so the two discovery paths agree on what is too short to be
#: findable by text search. The word list holds 352 entries below it (1.01% of 34,895) and the
#: imported anticancer set contributes the 2-residue ones.
#:
#: This does NOT delete those peptides from the release -- they remain valid database records, and
#: a paper that states such a sequence alongside a name still yields it through the name path. What
#: it removes is the claim that finding those letters in running text identifies that peptide.
MIN_CATALOGUED_SEQUENCE_LEN: int = 5


def _imported_anticancer_sequences() -> List[str]:
    """
    One-letter sequences of the imported anticancer peptides (AntiCP2, ...), for the known-sequence
    automata below.

    Sourced from the anticancer import (:func:`sample.anticancer.positive_sequences`) rather than the
    static resource files, so re-importing keeps the search current without regenerating a 1.3 MB
    word list. Empty and silent when nothing has been imported, so extraction never depends on it.
    """
    try:
        from sample import anticancer
        return [s for s in anticancer.positive_sequences() if s]
    except Exception:  # noqa: BLE001 -- the automaton must build even if the import is unreadable
        return []


@functools.lru_cache(maxsize=None)
def get_known_seq_1l_kp() -> AhocorasickWrapper:
    """
    Automaton over known peptide sequences in one-letter code.

    Includes the catalogued word list *and* the imported anticancer peptides, so a paper naming an
    AntiCP2 sequence verbatim is recognised the same way a catalogued one is.

    :returns: A shared, lazily-built automaton for spotting a catalogued sequence verbatim.
    """
    kp = KeywordProcessor(case_sensitive=True)
    known_1l_path = os.path.join(os.path.dirname(__file__), "..", "data", "resources", "known_1letter.txt")
    skipped = 0
    if os.path.exists(known_1l_path):
        with open(known_1l_path, 'r', encoding='utf-8') as f:
            for line in f:
                seq = line.strip()
                if not seq:
                    continue
                # See MIN_CATALOGUED_SEQUENCE_LEN: a 2-4 residue key is a substring search over
                # ordinary prose, not a peptide lookup.
                if len(seq) < MIN_CATALOGUED_SEQUENCE_LEN:
                    skipped += 1
                    continue
                kp.add_keyword(seq)
    # The imported anticancer peptides, in one-letter code (add_keyword is idempotent, so a sequence
    # already in the word list is not double-counted).
    for seq in _imported_anticancer_sequences():
        if len(seq) < MIN_CATALOGUED_SEQUENCE_LEN:
            skipped += 1
            continue
        kp.add_keyword(seq)
    if skipped:
        logger.info("Known-sequence automaton: %d catalogued sequence(s) shorter than %d residues "
                    "excluded as search keys (they match ordinary prose, not peptides).",
                    skipped, MIN_CATALOGUED_SEQUENCE_LEN)
    return kp

@functools.lru_cache(maxsize=None)
def get_known_seq_3l_kp() -> AhocorasickWrapper:
    """
    Automaton over known peptide sequences written in three-letter code.

    Includes the catalogued word list *and* the imported anticancer peptides converted to the
    ``Gly-Ile-Gly`` spelling, so both spellings of an anticancer sequence are recognised.

    :returns: A shared, lazily-built automaton for the ``Gly-Ile-Gly`` spelling.
    """
    from sample.sequence import sequence_to_3_letter

    kp = KeywordProcessor(case_sensitive=False)
    known_3l_path = os.path.join(os.path.dirname(__file__), "..", "data", "resources", "known_3letter.txt")
    if os.path.exists(known_3l_path):
        with open(known_3l_path, 'r', encoding='utf-8') as f:
            for line in f:
                three = line.strip()
                if not three:
                    continue
                # Same floor as the one-letter automaton, counted in RESIDUES: a three-letter
                # spelling holds one residue per hyphen-separated token ("Gly-Ile-Gly" is 3).
                if len([t for t in re.split(r'[-\s]+', three) if t]) < MIN_CATALOGUED_SEQUENCE_LEN:
                    continue
                kp.add_keyword(three)
    # The imported anticancer peptides, converted from one-letter to three-letter code.
    for seq in _imported_anticancer_sequences():
        if len(seq) < MIN_CATALOGUED_SEQUENCE_LEN:
            continue
        three = sequence_to_3_letter(seq)
        if three:
            kp.add_keyword(three)
    return kp

# Bare metric acronyms. Every addition here has to be checked for collisions against ordinary
# biomedical prose before it goes in -- "MHC" (major histocompatibility complex), "MLC" (myosin
# light chain) and "Ki" are all real assay acronyms in other contexts AND far more common as
# something else in this literature, so they are recognized only through their spelled-out phrases
# in _METRIC_PHRASE_TYPE_MAP below, never as bare letters.
METRIC_PATTERN_EXACT = re.compile(
    r'\b(MICs?|MIC50|MIC80|MIC90|MIC100|IC50|IC90|EC50|EC90|MBCs?|MBC50|MBC90|MFCs?'
    r'|CC50|CC90|LC50|LD50|LD90|ED50|ED90|ZOI|MBIC|MBEC|HC50|GI50|TC50|TD50|FIC)\b')
METRIC_PATTERN_IGNORE = re.compile(r'\b(minimum inhibitory concentration|zone of inhibition|minimum bactericidal concentration|antimicrobial activity|antibacterial activity|antifungal activity|hemolytic activity|cytotoxicity|hemolysis)\b', re.IGNORECASE)

# Descriptive-phrase fallback for activity_type when no bare acronym (METRIC_PATTERN_EXACT) is
# present in the sentence -- e.g. "the peptide showed strong growth inhibition of E. coli"
# never says "MIC" but is still a valid, typeable experimental metric.
_METRIC_PHRASE_TYPE_MAP = [
    # Plurals are load-bearing, not cosmetic: table captions -- the place a results table names its
    # assay, and the scope every row below inherits (see extractor.build_scope_types) -- are almost
    # always written in the plural ("Table 2. Minimum inhibitory concentrations of ..."). Without
    # `s?` the trailing \b failed against "concentrations" and captions matched nothing at all.
    (re.compile(r'\bminimum inhibitory concentrations?\b', re.IGNORECASE), "MIC"),
    (re.compile(r'\bminimum bactericidal concentrations?\b', re.IGNORECASE), "MBC"),
    (re.compile(r'\bminimum fungicidal concentrations?\b', re.IGNORECASE), "MFC"),
    (re.compile(r'\bzones? of inhibition\b|\binhibition zones?\b'
                r'|\binhibition halos?\b|\bclear(?:ing)? zones?\b'
                r'|\bdiameters? of (?:the )?(?:inhibition|growth[- ]inhibition|clear\w*) zones?\b',
                re.IGNORECASE), "ZOI"),
    (re.compile(r'\bgrowth inhibition\b', re.IGNORECASE), "Growth Inhibition"),
    (re.compile(r'\bh(?:a)?emolytic activity\b|\bh(?:a)?emolysis\b', re.IGNORECASE), "Hemolysis"),
    (re.compile(r'\bcytotoxicity\b|\bcytotoxic activity\b', re.IGNORECASE), "Cytotoxicity"),

    # Additions below were derived from the 16,812 corpus records that carried a value and a target
    # but no resolvable assay type. Each phrase was counted against that population first: these
    # cover ~2,400 of them, and every one names an assay whose reported number is genuinely the
    # peptide's own measurement rather than an experimental parameter.
    (re.compile(r'\bminimum (?:lethal|bacteriostatic) concentrations?\b', re.IGNORECASE), "MLC"),
    (re.compile(r'\bminimum h(?:a)?emolytic concentrations?\b', re.IGNORECASE), "MHC"),
    (re.compile(r'\bfractional inhibitory concentrations?\b', re.IGNORECASE), "FIC"),
    (re.compile(r'\bhalf[- ]maximal inhibitory concentrations?\b', re.IGNORECASE), "IC50"),
    # Biofilm assays report a concentration exactly like an MIC does, and are a distinct endpoint
    # from planktonic MIC -- collapsing them into "MIC" would misreport what was measured.
    (re.compile(r'\bbiofilm (?:inhibition|eradication|formation|biomass|reduction)\b'
                r'|\bantibiofilm activity\b'
                r'|\bminimum biofilm (?:inhibitory|eradication) concentrations?\b',
                re.IGNORECASE), "Biofilm Inhibition"),
    # Viability readouts are cytotoxicity measurements under a different name. Note MTT also
    # appears in extractor._REAGENT_CONTEXT as a reagent; naming the assay here takes precedence,
    # which is correct and matches how a stated metric already overrides that guard.
    (re.compile(r'\bcell viabilit\w*\b|\bMTT assay\b|\bCCK-?8\b|\bresazurin\b|\bAlamar\s*Blue\b',
                re.IGNORECASE), "Cytotoxicity"),
    (re.compile(r'\bantiprolifer\w+\b|\bproliferation (?:inhibition|assay)\b'
                r'|\binhibition of (?:cell )?proliferation\b', re.IGNORECASE), "Growth Inhibition"),
    (re.compile(r'\bbactericidal (?:activity|assay|concentration)\b'
                r'|\btime[- ]kill(?:ing)?\b|\bkilling (?:assay|kinetics|curves?)\b',
                re.IGNORECASE), "Bactericidal"),
    (re.compile(r'\bmembrane permeabili\w+\b|\bpermeabilization assay\b'
                r'|\bmembrane depolari\w+\b|\bcalcein (?:release|leakage)\b'
                r'|\bdye leakage\b', re.IGNORECASE), "Membrane Permeabilization"),
]


def detect_activity_type(sent: str, default: str = "unknown") -> str:
    """
    Identifies the specific experimental metric named in a sentence: MIC, MIC50/90/100, IC50/90,
    EC50/90, MBC, MBC50/90, MFC, CC50, LC50, LD50, ED50, ZOI, or a descriptive-phrase fallback
    (Growth Inhibition, Hemolysis, Cytotoxicity). Falls back to `default` if none is found.
    """
    m = METRIC_PATTERN_EXACT.search(sent)
    if m:
        return m.group(1).rstrip('s')  # "MICs"/"MBCs"/"MFCs" -> "MIC"/"MBC"/"MFC"
    for pattern, label in _METRIC_PHRASE_TYPE_MAP:
        if pattern.search(sent):
            return label
    return default

ORGANISM_PATTERN = re.compile(r'\b(?:[A-Z]\.\s*[a-z]{3,}|(?:Escherichia|Staphylococcus|Pseudomonas|Acinetobacter|Klebsiella|Candida|Enterococcus|Salmonella|Bacillus|Streptococcus|Listeria|Mycobacterium|Vibrio|Helicobacter|Neisseria)\s+[a-z]{3,})\b')

@functools.lru_cache(maxsize=None)
def get_organism_kp() -> AhocorasickWrapper:
    """
    Automaton over every NCBI taxonomy name variant, mapping each to its canonical binomial.

    ~2M entries, built once and shared by all worker threads; this singleton is the main reason
    extraction uses a thread pool rather than a process pool.

    :returns: A shared, lazily-built automaton keyed on name variants.
    """
    kp = KeywordProcessor(case_sensitive=False)
    _org_db_path = os.path.join(os.path.dirname(__file__), "..", "data", "taxonomy.db")
    if os.path.exists(_org_db_path):
        import sqlite3 as _sqlite3
        try:
            _org_conn = _sqlite3.connect(_org_db_path)
            # Each row: (name variant, canonical scientific name)
            for (_name_lower, _sci_name) in _org_conn.execute(
                "SELECT o.name_lower, s.scientific_name "
                "FROM organism_names o "
                "JOIN scientific_names s ON o.tax_id = s.tax_id"
            ):
                # Key = the variant (e.g. "e. coli"), value = canonical ("Escherichia coli")
                kp.add_keyword(_name_lower, _sci_name)
            _org_conn.close()
        except Exception as _e:
            logger.warning(f"Could not load taxonomy.db for get_organism_kp(): {_e}")
            # Graceful fallback to organism_names.txt
            _org_txt = os.path.join(os.path.dirname(__file__), "..", "data", "organism_names.txt")
            if os.path.exists(_org_txt):
                with open(_org_txt, 'r', encoding='utf-8') as _f:
                    for _line in _f:
                        _name = _line.strip()
                        if _name:
                            kp.add_keyword(_name, _name)
    return kp

@functools.lru_cache(maxsize=None)
def get_cellosaurus_kp() -> AhocorasickWrapper:
    """
    Automaton over Cellosaurus cell-line names.

    :returns: A shared, lazily-built automaton mapping cell-line synonyms to a canonical name.
    """
    kp = KeywordProcessor(case_sensitive=False)
    cellosaurus_names_path = os.path.join(os.path.dirname(__file__), "..", "data", "cellosaurus_names.txt")
    if os.path.exists(cellosaurus_names_path):
        with open(cellosaurus_names_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    kp.add_keyword(name, name)
    return kp

# Activity group taxonomy KeywordProcessor
@functools.lru_cache(maxsize=None)
def get_organism_group_kp() -> AhocorasickWrapper:
    """
    Automaton over coarse organism-group labels ("Gram-positive bacteria", "yeast").

    :returns: A shared, lazily-built automaton used when a paper names a clade, not a species.
    """
    kp = KeywordProcessor(case_sensitive=False)
    group_taxonomy = {
        "antibacterial": [
            "escherichia coli", "e. coli", "staphylococcus aureus", "s. aureus", "pseudomonas aeruginosa", "p. aeruginosa",
            "bacillus subtilis", "b. subtilis", "acinetobacter baumannii", "a. baumannii", "klebsiella pneumoniae", "k. pneumoniae",
            "enterococcus faecalis", "e. faecalis", "enterococcus faecium", "salmonella typhimurium", "s. typhimurium", "salmonella enterica",
            "helicobacter pylori", "h. pylori", "listeria monocytogenes", "l. monocytogenes", "mycobacterium tuberculosis", "m. tuberculosis",
            "streptococcus pneumoniae", "s. pneumoniae", "vibrio cholerae", "v. cholerae", "propionibacterium acnes", "p. acnes",
            "gram-negative", "gram-positive", "bacteria", "bacterial"
        ],
        "antifungal": [
            "candida albicans", "c. albicans", "candida glabrata", "c. glabrata", "candida tropicalis", "c. tropicalis",
            "aspergillus niger", "a. niger", "aspergillus fumigatus", "a. fumigatus", "cryptococcus neoformans", "c. neoformans",
            "saccharomyces cerevisiae", "s. cerevisiae", "trichophyton rubrum", "fungi", "fungal", "yeast"
        ],
        "antiviral": [
            "hiv", "hsv", "herpes", "influenza", "sars-cov-2", "covid", "dengue", "hepatitis", "zika", "ebola", "rhinovirus", "adenovirus", "virus", "viral"
        ],
        "antiparasitic": [
            "plasmodium falciparum", "p. falciparum", "leishmania", "trypanosoma", "giardia", "entamoeba", "toxoplasma", "parasite", "parasitic", "malaria"
        ],
        "anticancer": [
            "hela", "mcf-7", "mcf7", "a549", "u87", "ht-29", "ht29", "k562", "mda-mb-231", "hepg2", "pc-3", "hct116", "hct-116",
            "jurkat", "cancer", "tumor", "carcinoma", "leukemia", "melanoma", "cytotoxicity", "cytotoxic"
        ]
    }
    for group_name, keywords in group_taxonomy.items():
        for kw in keywords:
            kp.add_keyword(kw, group_name)
    return kp

def parse_activity_value_fast(val_str: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Parses a raw activity string (e.g. '>100', '10-20', '< 0.5', '12.5 ug/mL')
    into (operator, value_lower, value_upper).
    """
    if val_str is None:
        return None, None, None
    val_str = str(val_str).strip()
    if not val_str or val_str.upper() in ["NULL", "NONE", "N/A", "TBD"]:
        return None, None, None
        
    val_str = val_str.replace('&lt;', '<').replace('&gt;', '>').replace('≤', '<=').replace('≥', '>=')

    # Disambiguate the comma before any number is read out of the string. A comma is a thousands
    # separator when exactly three digits follow it and no digit precedes the group boundary
    # ("1,000" -> 1000), and a European decimal point when one or two digits follow ("3,7" -> 3.7).
    # Without this the regexes below stop at the comma and read "1,000 ng/ml" as 1.0 -- a silent
    # 1000-fold under-reporting, seen in real output on several CAP37/CAP18 records.
    val_str = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', val_str)
    val_str = re.sub(r'(?<=\d),(?=\d{1,2}(?!\d))', '.', val_str)

    # Convert scientific notation: 'x 10^5' or '× 10^-5' to 'e5' or 'e-5'
    val_str = re.sub(r'\s*(?:x|×)\s*10\^([+-]?\d+)', r'e\1', val_str)

    # OCR'd tables sometimes concatenate adjacent cell values into one giant run with no
    # separator (observed in real output: "102103104105"). A contiguous run of more than 6 digits
    # is never a real activity value in this domain -- treat it as a table-parsing artifact.
    if re.search(r'\d{7,}', val_str):
        return None, None, None

    # 0. Standard deviation check e.g. "5.0 ± 1.0" or "5.0 +/- 1.0"
    # We extract the mean as the primary value (lower), and ignore the SD for the basic parsed output.
    m_sd = re.search(r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:±|\+/-)\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', val_str)
    if m_sd:
        try:
            return None, float(m_sd.group(1)), None
        except ValueError:
            pass
            
    # 1. Range check e.g. "5.0 - 10.0" or "5-10"
    m_range = re.search(r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', val_str)
    if m_range:
        try:
            return None, float(m_range.group(1)), float(m_range.group(2))
        except ValueError:
            pass

    # 2. Operator check e.g. "< 0.5", ">= 10", "~ 5", "100"
    m_op = re.search(r'([<>]=?|~|=)?\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', val_str)
    if m_op and m_op.group(2):
        op = m_op.group(1) if m_op.group(1) else None
        try:
            return op, float(m_op.group(2)), None
        except ValueError:
            pass
            
    return None, None, None

_ORGANISM_BLOCKLIST = {
    "CANCER", "DATA", "THIS", "TIME", "SYNTHETIC CONSTRUCT",
    "UNIDENTIFIED", "ALL", "MIN", "MIC", "BETA",
    # Generic top-level taxonomic-rank names (real taxonomy.db entries, e.g. tax_id=10239
    # "Viruses") are too coarse to be a meaningful target_organism when specific organism names
    # are what a real activity record needs -- found via a real corpus example where a sentence
    # naming several specific viruses (HSV-1, Influenza virus, Cytomegalovirus...) still got the
    # generic word "viruses" proximity-paired to a value instead of any of the actual named
    # viruses. Blocking it here forces proximity pairing to fall through to a real organism name,
    # or to no organism at all -- both better than a false, over-generic label.
    "VIRUS", "VIRUSES", "BACTERIA", "BACTERIUM", "FUNGI", "FUNGUS", "ARCHAEA",
    # Manuscript-structure words and lab vocabulary that collide with real NCBI names. Every one
    # of these was observed as a target_organism in real output: "Figs." (a cross-reference) is
    # catalogued as a common name for the fig genus Ficus; "plasmids", "clones" and "vectors" are
    # literal NCBI entries; "Center" arrived from an institutional address. They validate against
    # NCBI perfectly, so only a blocklist can catch them.
    "FIG", "FIGS", "FIGURE", "FIGURES", "TABLE", "TABLES", "PANEL", "LANE", "LANES",
    "PLASMID", "PLASMIDS", "VECTOR", "VECTORS", "CLONE", "CLONES", "LIBRARY",
    "CENTER", "CENTRE", "GROUP", "GROUPS", "CONTROL", "CONTROLS", "SAMPLE", "SAMPLES",
    "MEDIUM", "MEDIA", "BUFFER", "SERUM", "COLUMN", "PEAK", "PEAKS", "PROBE", "PROBES",
    # Place names and eponyms that are also catalogued taxa/cell-line codes. Observed as
    # target_organism / target_cell_line in real output: "California" and "Calif." from supplier
    # addresses, "Luria" from Luria-Bertani broth, and the author surnames "Smith", "Jones" and
    # "Mata" from citations -- Cellosaurus lists short codes that collide with all of them.
    "CALIFORNIA", "CALIF", "LURIA", "SMITH", "JONES", "MATA", "OLIGO", "COMBI",
}

# Surnames reach target_cell_line via Cellosaurus codes that collide with them, and the citation
# context around them is distinctive: "(Smith et al., 1990)", "Jones and Brown reported", a year
# in parentheses. Checked near the match rather than sentence-wide, so a genuine cell line in a
# sentence that also carries a citation is unaffected.
_CITATION_CONTEXT = re.compile(
    r'\bet\s*al\b|\(\s*(?:19|20)\d{2}\s*[a-z]?\s*\)|,\s*(?:19|20)\d{2}\s*[);]'
    r'|\b(?:reported|described|observed|showed|found)\s+(?:by|that)\b|\baccording\s+to\b',
    re.IGNORECASE
)

# Company/supplier parentheticals -- "(Amgen, Thousand Oaks, CA)", "(Chiron, Emeryville, CA)",
# "(2200 Ci/mmol, Phoenix)". Several vendor names are also real genera (Phoenix is the date palm,
# Chiron a real genus), so they pass NCBI validation and can only be recognized by the company
# metadata sitting beside them.
_VENDOR_CONTEXT = re.compile(
    r'\b(?:Inc|Ltd|Co|Corp|GmbH|LLC|AG|BV|SA|Pharmaceuticals?|Biosciences?|Biotech\w*|'
    r'Laboratories|Labs?|Technologies|Scientific|Chemical\s+Co|Separation\s+Group)\b\.?'
    r'|,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*,\s*[A-Z]{2}\b'          # ", Thousand Oaks, CA"
    r'|,\s*[A-Z]{2}\s*(?:\)|,)'                                        # ", CA)"
    # Spelled-out US state abbreviations as used in older papers' supplier addresses --
    # "(4.6 by 250 mm, Vydac; The Separation Group, Hesperia, Calif.)" put the California city
    # Hesperia (also a real genus) into target_organism nine times in a single document.
    r'|,\s*(?:Calif|Mass|Conn|Ill|Ind|Mich|Minn|Miss|Mo|Neb|Nev|Okla|Ore|Pa|Tenn|Tex|Va|Wash|Wis|'
    r'Ala|Ariz|Ark|Colo|Del|Fla|Ga|Kan|Ky|La|Md|Mont|N\.?[JYCDHM]|S\.?[CD]|Vt|W\.?Va|Wyo)\b\.?'
    r'|\b(?:Sigma|Amgen|Merck|Roche|Pfizer|Bachem|Novartis|Genentech|Invitrogen|Gibco|Vydac|'
    r'Pharmacia|Whatman|Millipore|Beckman|Waters|BioRad|Bio-Rad|Costar|Falcon|Nunc)\b',
    re.IGNORECASE
)
_ORGANISM_CORROBORATION = re.compile(
    r'\b(cell\s*line|cells?|ATCC|passage|clone|xenograft|carcinoma|tumor(?:al)?|fibroblasts?)\b',
    re.IGNORECASE
)
_AMBIGUOUS_ORGANISM_LEN = 4

def is_valid_organism_match(candidate: str, sentence: str = "", is_cell_line_source: bool = False) -> bool:
    """
    Guards get_organism_kp()/get_cellosaurus_kp() matches against two systemic false-positive
    sources found in a corpus-wide audit: pure-numeric strings matched as organisms (17.2% of all
    activity records had e.g. "100", "01460", "12.5" as target_organism -- almost certainly OCR
    table digits colliding with a taxonomy/cellosaurus name variant), and short (<=4 char)
    Cellosaurus codes (PD3, TIME, BTB, MIN, ALL, Ca2) that are real cell-line codes but collide
    with common English words/lab shorthand.

    The length+corroboration check below applies ONLY when `is_cell_line_source=True` (the match
    came from get_cellosaurus_kp()) -- every example of ambiguous short-code collision found in
    the audit was a cell-line code, never a genus/species name. Applying it to
    get_organism_kp() matches too was a real regression, caught retroactively: "Rana" (a common,
    unambiguous frog genus behind many published AMPs -- temporins, brevinins) is exactly 4
    characters and was being rejected 257 times across the corpus for lack of a "cell line"/
    "ATCC"-style cue that would never appear near a genuine organism mention in the first place.
    Taxonomy-sourced matches are backed by an authoritative ~2M-entry database; once past the
    numeric/blocklist gates, length alone isn't a real ambiguity signal for them.
    """
    if not candidate:
        return False
    core = candidate.strip()
    if re.fullmatch(r'[\d.]+', core.rstrip('+-')):
        return False
    if core.upper() in _ORGANISM_BLOCKLIST:
        return False
    # 3-letter amino-acid residue codes (Pro, Gly, Val...) get matched as organism-name variants
    # by some taxonomy entries -- found retroactively in a fresh extraction run, where a table of
    # synthetic peptide sequences ("R-Trp-Gly-Lys-Pro-Val-NH2") produced "PRO"/"Gly"/"VAL" as
    # target_organism. These are always a sequence-context artifact here, never a genuine
    # organism reference.
    if core.upper() in _AA3_UPPER:
        return False
    if is_cell_line_source:
        alnum_len = len(re.sub(r'[^A-Za-z0-9]', '', core))
        if alnum_len <= _AMBIGUOUS_ORGANISM_LEN:
            # Established cell-line designations in this literature essentially always carry a
            # digit (MCF-7, P815, A549, HT-29, K562, HEK293, U937). A short PURE-ALPHA Cellosaurus
            # hit is nearly always an unrelated acronym or an English word that happens to collide
            # with an entry -- observed in real output: "MSH" (melanocyte-stimulating hormone),
            # "MOR" (mu-opioid receptor), "RAT", and "MAST" (from "mast cells" -- a cell *type*,
            # not a cell line). A generic nearby "cells" is not enough to separate these, since
            # that word appears next to the true and false cases alike, so a short pure-alpha code
            # additionally requires an explicit "cell line" phrase.
            if not re.search(r'\d', core):
                return bool(re.search(r'\bcell\s*lines?\b', sentence, re.IGNORECASE))
            return bool(_ORGANISM_CORROBORATION.search(sentence))
    return True

_TAXONOMY_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "taxonomy.db")


@functools.lru_cache(maxsize=1)
def _taxonomy_conn() -> Optional[Any]:
    """
    Read-only connection to the NCBI taxonomy DB, shared across worker threads.

    check_same_thread=False is safe here because every query through it is a read (sqlite3
    serializes concurrent reads on one connection internally), and the alternative -- a connection
    per thread -- would multiply the page cache across the pool for no benefit.
    """
    if not os.path.exists(_TAXONOMY_DB_PATH):
        return None
    try:
        import sqlite3
        return sqlite3.connect(_TAXONOMY_DB_PATH, check_same_thread=False)
    except Exception as e:
        logger.warning(f"Could not open taxonomy.db for name validation: {e}")
        return None


_TAX_RESOLVE_SQL = (
    "SELECT s.scientific_name FROM organism_names o "
    "JOIN scientific_names s ON o.tax_id = s.tax_id WHERE o.name_lower = ? LIMIT 1"
)

#: Same lookup, keeping the tax_id it already joins through. Every organism query in this module
#: resolved a name via tax_id and then dropped it, so the released tables carried organism names
#: with no stable identifier -- nothing could be linked to NCBI Taxonomy or joined across sources
#: that spell a species differently.
_TAX_RESOLVE_WITH_ID_SQL = (
    "SELECT s.scientific_name, s.tax_id FROM organism_names o "
    "JOIN scientific_names s ON o.tax_id = s.tax_id WHERE o.name_lower = ? LIMIT 1"
)


@functools.lru_cache(maxsize=100_000)
def resolve_taxon(name: str) -> Tuple[str, Optional[int]]:
    """
    Resolves an organism string to its NCBI scientific name *and* tax id.

    Tolerates a strain designation the way :func:`sample.database_activities.resolve_target_species`
    does -- "Escherichia coli BL21(DE3)" is not an NCBI name, but the binomial it qualifies is --
    by retrying on progressively shorter prefixes.

    :param name: Raw organism string.
    :returns: ``(scientific_name, tax_id)``; ``("", None)`` when nothing resolves.
    """
    if not name:
        return "", None
    conn = _taxonomy_conn()
    if conn is None:
        return str(name), None

    candidates = [" ".join(str(name).split()).lower()]
    tokens = re.sub(r"[^A-Za-z .\-]", " ", str(name)).split()
    for width in (3, 2, 1):
        if len(tokens) >= width:
            candidate = " ".join(tokens[:width]).lower()
            if candidate not in candidates:
                candidates.append(candidate)

    for key in candidates:
        if not key:
            continue
        try:
            row = conn.execute(_TAX_RESOLVE_WITH_ID_SQL, (key,)).fetchone()
        except Exception as e:
            logger.debug(f"Taxonomy lookup failed for {name!r}: {e}")
            return str(name), None
        if row:
            return row[0], int(row[1])
    return "", None


@functools.lru_cache(maxsize=100_000)
def resolve_scientific_name(name: str) -> str:
    """
    Resolves a raw organism string to its NCBI canonical scientific name, or "" if NCBI has no
    such name. Accepts any NCBI-known variant -- common names and abbreviations included, so
    "human" -> "Homo sapiens", "mouse" -> "Mus musculus", "E. coli" -> "Escherichia coli".

    This is the authority for what may appear as `target_organism` in the output: a value that
    does not resolve here is not a real organism name, and downstream consumers of this database
    should be able to join on it directly.
    """
    if not name:
        return ""
    conn = _taxonomy_conn()
    if conn is None:
        return name  # no DB available -- do not silently discard data over a missing resource
    key = " ".join(str(name).split()).lower()
    if not key:
        return ""
    try:
        row = conn.execute(_TAX_RESOLVE_SQL, (key,)).fetchone()
    except Exception as e:
        logger.debug(f"Taxonomy lookup failed for {name!r}: {e}")
        return name
    return row[0] if row else ""


# Template/placeholder text, and the bracket characters that only appear in it. Real peptide
# nomenclature uses letters, digits, hyphens and parentheses -- never square or curly brackets.
_PLACEHOLDER_NAME = re.compile(
    r'[\[\]{}<>]'
    r'|\b(?:insert|placeholder|example|sample|your|the)\s+(?:peptide|name|sequence|value)\b'
    r'|\b(?:TBD|TODO|N/?A|FIXME|XXX+)\b'
    r'|^\s*(?:none|null|unknown|unnamed|undefined|not\s+\w+)\s*$',
    re.IGNORECASE
)

_ABBREVIATED_BINOMIAL = re.compile(r'^([A-Z])\.\s*([a-z][a-z-]{2,})$')


def collect_document_genera(text: str) -> frozenset:
    r"""
    Capitalized words in a document that could be a spelled-out genus.

    Deliberately unfiltered and deliberately not cached: it is keyed on the document text itself,
    so caching would leak one paper's genera into another's, and a non-genus word that slips in
    can do no harm -- resolve_abbreviated_binomial only ever uses these to *build* a candidate
    binomial, which then has to resolve against NCBI on its own merits.

    The leading anchor is (?<![A-Z]) rather than \b because this corpus's OCR routinely drops the
    space between words: the document that motivated this reads "growth-inhibitory activity
    againstMicrococcus luteus", where \b finds no boundary before the capital M and the genus is
    invisible. Requiring only that the capital not follow another capital keeps acronyms
    ("DNAse") from being split while still finding a genus fused to the preceding word.
    """
    return frozenset(m.group(1) for m in re.finditer(r'(?<![A-Z])([A-Z][a-z]{3,})(?![a-z])', text))


def resolve_abbreviated_binomial(surface: str, document_genera: frozenset) -> str:
    """
    Resolves an abbreviated binomial ("M. luteus") using the genera the document actually spells
    out, returning "" when the document does not settle it.

    Why this cannot be left to NCBI alone: NCBI registers "m. luteus" as a name variant of
    Methylobacter luteus (tax_id 415), so the plain lookup returns a real, validly-formed species
    name -- just not the one the paper meant. Sixteen species share the pattern "M... luteus", and
    in this literature it is always Micrococcus luteus, a standard test organism, never a
    methane-oxidizing environmental bacterium. The error is invisible downstream precisely because
    the wrong answer is well-formed and taxonomically valid; 64 records across 28 documents carried
    it, and every one of those documents spells out "Micrococcus" and none ever says
    "Methylobacter".

    Ambiguity is resolved to "" rather than guessed: if two spelled-out genera in the same document
    both yield a valid binomial with this epithet, the abbreviation genuinely is ambiguous and the
    caller keeps NCBI's answer instead of inventing a different one.
    """
    m = _ABBREVIATED_BINOMIAL.match(" ".join(str(surface).split()))
    if not m:
        return ""
    initial, epithet = m.group(1), m.group(2).lower()
    hits = set()
    for genus in document_genera:
        if genus[0] != initial:
            continue
        resolved = resolve_scientific_name(f"{genus} {epithet}")
        if resolved:
            hits.add(resolved)
            if len(hits) > 1:
                return ""
    return hits.pop() if len(hits) == 1 else ""


# FALLBACK ONLY -- the authoritative source is the tax_domain table built by
# scripts/build_taxonomy_domains.py from NCBI's lineage (see classify_organism_domain). These
# lists exist so the check still functions on a taxonomy.db built before that table was added.
# They are not maintained as a complete catalogue and never can be: the corpus names thousands of
# species, which is precisely why the hand-written version was replaced.
_BACTERIAL_GENERA = {
    "ESCHERICHIA", "STAPHYLOCOCCUS", "STREPTOCOCCUS", "PSEUDOMONAS", "SALMONELLA", "KLEBSIELLA",
    "ENTEROCOCCUS", "LISTERIA", "BACILLUS", "ACINETOBACTER", "PROTEUS", "SERRATIA", "SHIGELLA",
    "NEISSERIA", "HAEMOPHILUS", "HELICOBACTER", "CAMPYLOBACTER", "CLOSTRIDIUM", "CLOSTRIDIOIDES",
    "MYCOBACTERIUM", "MICROCOCCUS", "LACTOBACILLUS", "LACTOCOCCUS", "LEUCONOSTOC", "BACTEROIDES",
    "AEROMONAS", "VIBRIO", "ALTEROMONAS", "XANTHOMONAS", "ERWINIA", "AGROBACTERIUM", "RHIZOBIUM",
    "TREPONEMA", "BORRELIA", "CHLAMYDIA", "LEGIONELLA", "YERSINIA", "BORDETELLA", "BRUCELLA",
    "CORYNEBACTERIUM", "PROPIONIBACTERIUM", "CUTIBACTERIUM", "ACTINOMYCES", "NOCARDIA",
    "STREPTOMYCES", "CAPNOCYTOPHAGA", "AGGREGATIBACTER", "PORPHYROMONAS", "PREVOTELLA",
    "FUSOBACTERIUM", "MORGANELLA", "CITROBACTER", "ENTEROBACTER", "PROVIDENCIA", "PASTEURELLA",
    "ALCALIGENES", "BURKHOLDERIA", "STENOTROPHOMONAS", "RALSTONIA", "SPHINGOMONAS", "MORAXELLA",
    "GARDNERELLA", "VEILLONELLA", "PEPTOSTREPTOCOCCUS", "MICROBISPORA", "BIFIDOBACTERIUM",
    "ARTHROBACTER", "BREVIBACTERIUM", "RHODOCOCCUS", "MYCOPLASMA", "UREAPLASMA", "RICKETTSIA",
    "COXIELLA", "FRANCISELLA", "PLESIOMONAS", "EDWARDSIELLA", "PHOTOBACTERIUM", "ACTINOPLANES",
}
_FUNGAL_GENERA = {
    "CANDIDA", "ASPERGILLUS", "CRYPTOCOCCUS", "SACCHAROMYCES", "TRICHOPHYTON", "MICROSPORUM",
    "EPIDERMOPHYTON", "FUSARIUM", "PENICILLIUM", "RHIZOPUS", "MUCOR", "MALASSEZIA", "SPOROTHRIX",
    "BLASTOMYCES", "HISTOPLASMA", "COCCIDIOIDES", "PARACOCCIDIOIDES", "PNEUMOCYSTIS",
    "TRICHOSPORON", "RHODOTORULA", "GEOTRICHUM", "ALTERNARIA", "BOTRYTIS", "COLLETOTRICHUM",
    "VERTICILLIUM", "CLADOSPORIUM", "CURVULARIA", "SCEDOSPORIUM", "EXOPHIALA", "PICHIA",
    "KLUYVEROMYCES", "YARROWIA", "DEBARYOMYCES", "ZYGOSACCHAROMYCES", "NEUROSPORA", "TRICHODERMA",
}
_PROTOZOAN_GENERA = {
    "PLASMODIUM", "LEISHMANIA", "TRYPANOSOMA", "TOXOPLASMA", "ENTAMOEBA", "GIARDIA",
    "ACANTHAMOEBA", "PARAMECIUM", "TETRAHYMENA", "CRYPTOSPORIDIUM", "BABESIA", "TRICHOMONAS",
    "NAEGLERIA", "EIMERIA", "THEILERIA",
}
_ANIMAL_GENERA = {
    "HOMO", "MUS", "RATTUS", "ORYCTOLAGUS", "BOS", "SUS", "OVIS", "CAPRA", "EQUUS", "CANIS",
    "FELIS", "GALLUS", "XENOPUS", "RANA", "LITHOBATES", "DANIO", "DROSOPHILA", "APIS",
    "CAENORHABDITIS", "CAVIA", "MESOCRICETUS", "MACACA", "LIMULUS", "BOMBYX", "GALLERIA",
    "PHYLLOMEDUSA", "LITORIA", "BUFO", "HYLA", "SALMO", "ONCORHYNCHUS", "CIONA", "STYELA",
}
_PLANT_GENERA = {
    "GLYCINE", "THEOBROMA", "ARABIDOPSIS", "ORYZA", "ZEA", "TRITICUM", "NICOTIANA", "SOLANUM",
    "ARMORACIA", "PHOENIX", "FICUS", "HORDEUM", "PISUM", "PHASEOLUS", "VIGNA", "CAPSICUM",
    "ALLIUM", "BRASSICA", "HELIANTHUS", "GOSSYPIUM", "MEDICAGO", "VITIS", "MALUS", "CITRUS",
}
# Suffixes that settle a genus this list does not name. -MYCES is bacterial for the actinomycetes
# (Streptomyces, Actinomyces) and fungal otherwise, so it is resolved by the explicit sets above
# and deliberately left out here.
_BACTERIAL_SUFFIXES = ("BACTER", "BACTERIUM", "COCCUS", "BACILLUS", "MONAS", "SPIRILLUM",
                       "SPIRA", "PLASMA", "VIBRIO", "SPORA", "PHILUS", "ELLA")
_FUNGAL_SUFFIXES = ("MYCETE", "MYCOTA", "ASCUS", "SPORIUM", "DERMA")

@functools.lru_cache(maxsize=1)
def _domains_by_group() -> Dict[str, frozenset]:
    """
    Allowed NCBI domains per activity class, read from the published rules.

    Previously a hand-written dict whose docstring claimed it was inverted from
    ``validation_rules`` "so the two cannot disagree". They disagreed. It omitted ``anticancer``
    entirely, so an anticancer measurement against *Escherichia coli* raised no objection -- the
    exact mis-pairing this function exists to catch. It also mapped ``antiparasitic`` to protozoa
    alone, while the rules allow protozoa *and* animal, so every helminth target (Caenorhabditis,
    Schistosoma, Brugia -- all Metazoa) would have been flagged as a contradiction it is not.

    Deriving it from :data:`~sample.validation_rules.TARGET_GROUP_DOMAINS` makes the claim true and
    supports several domains per class. Classes with no restriction (``other``) are omitted, so
    they never contradict.

    :returns: Activity class -> the domains it may target.
    """
    from sample.validation_rules import TARGET_GROUP_DOMAINS

    return {rule.target_group: frozenset(rule.domains)
            for rule in TARGET_GROUP_DOMAINS if rule.domains}


_TAX_DOMAIN_SQL = (
    "SELECT d.domain FROM organism_names o "
    "JOIN tax_domain d ON o.tax_id = d.tax_id WHERE o.name_lower = ? LIMIT 1"
)


@functools.lru_cache(maxsize=1)
def _has_tax_domain_table() -> bool:
    """
    :returns: ``True`` when the ``tax_domain`` table built by
        ``scripts/build_taxonomy_domains.py`` is present, so organism domains can be resolved from
        NCBI lineage instead of the hand-written genus fallback.
    """
    conn = _taxonomy_conn()
    if conn is None:
        return False
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tax_domain'"
        ).fetchone() is not None
    except Exception:
        return False


@functools.lru_cache(maxsize=50_000)
def classify_organism_domain(name: str) -> str:
    """
    Broad domain of an organism name: 'bacteria', 'archaea', 'fungi', 'virus', 'protozoa',
    'animal', 'plant', or '' when it cannot be determined. '' means unknown, never "none of
    these" -- callers must not treat it as a contradiction.

    Answered from NCBI's own lineage: scripts/build_taxonomy_domains.py walks every tax_id's
    parent links to the root and stores the clade it belongs to, so this is a database lookup
    rather than a judgement about a name's spelling. The genus lists below are a fallback for
    installations whose taxonomy.db predates that table; they cover the common cases and nothing
    more, which is exactly why they are not the primary path.
    """
    if not name:
        return ""
    key = " ".join(str(name).split()).lower()
    if not key:
        return ""

    if _has_tax_domain_table():
        conn = _taxonomy_conn()
        # Full name first, then the genus alone. The shipped organism_names table does not carry
        # every binomial NCBI knows -- "Wickerhamomyces anomalus" is absent while the genus
        # "Wickerhamomyces" is present -- and a genus fixes the domain regardless of species, so
        # this recovers the answer instead of reporting unknown over a missing name variant.
        for candidate in (key, key.split()[0]):
            try:
                row = conn.execute(_TAX_DOMAIN_SQL, (candidate,)).fetchone()
            except Exception as e:
                logger.debug(f"tax_domain lookup failed for {name!r}: {e}")
                break
            if row and row[0]:
                return row[0]
        # A name NCBI does not know cannot be classified from the database; fall through to the
        # heuristics rather than reporting a confident "unknown".

    upper = key.upper()
    if "VIRUS" in upper or "PHAGE" in upper or "VIROID" in upper:
        return "virus"
    genus = upper.split()[0].rstrip(".,")
    for genera, domain in ((_BACTERIAL_GENERA, "bacteria"), (_FUNGAL_GENERA, "fungi"),
                           (_PROTOZOAN_GENERA, "protozoa"), (_ANIMAL_GENERA, "animal"),
                           (_PLANT_GENERA, "plant")):
        if genus in genera:
            return domain
    if genus.endswith(_BACTERIAL_SUFFIXES):
        return "bacteria"
    if genus.endswith(_FUNGAL_SUFFIXES):
        return "fungi"
    return ""


def organism_contradicts_group(target_organism: str, target_group: str) -> bool:
    """
    True when the organism demonstrably cannot be the target of the claimed activity class -- an
    "antibacterial" record whose target is a plant, an animal or a fungus.

    This mismatch is manufactured by infer_activity_group(): when the organism matches none of its
    known groups it falls back to the assay name, and "MIC" alone yields "antibacterial". So
    Theobroma cacao, Homo sapiens and Xenopus laevis all came out labelled antibacterial in real
    output, each of them a mis-paired organism rather than a real target.

    Conservative by construction: an unclassifiable organism (domain "") never contradicts, so
    only a positively identified mismatch is acted on.
    """
    expected = _domains_by_group().get((target_group or "").lower())
    if not expected:
        return False
    domain = classify_organism_domain(target_organism)
    return bool(domain) and domain not in expected


#: NCBI domain -> activity group. Inverted from validation_rules.DOMAINS_BY_GROUP so the two
#: cannot disagree. "animal" is deliberately absent: it maps to both anticancer (a human cell
#: line) and antiparasitic (a helminth), so it is not decidable from the domain alone and falls
#: through to the text rules below.
_GROUP_BY_DOMAIN: Dict[str, str] = {
    "bacteria": "antibacterial",
    "archaea": "antibacterial",
    "fungi": "antifungal",
    "virus": "antiviral",
    "protozoa": "antiparasitic",
}

#: Metric abbreviations, anchored on word boundaries. Unanchored substrings matched inside other
#: words: "mic" inside "Microsporum" made a dermatophyte antibacterial, and "ic50" inside "MIC50"
#: made an Escherichia coli MIC anticancer.
_METRIC_HINT_BACTERIAL = re.compile(r"\b(?:mic|mbc|mbic|mbec|mic\d{2})\b|bacteri", re.IGNORECASE)
_METRIC_HINT_FUNGAL = re.compile(r"\b(?:mfc|mfc\d{2})\b|fung|yeast", re.IGNORECASE)
_METRIC_HINT_CANCER = re.compile(r"\b(?:ic50|gi50|cc50|tc50)\b|cytotox|tumou?r|cancer|"
                                 r"cell[\s\-]?line", re.IGNORECASE)
_WORD_VIRUS = re.compile(r"\bvir(?:us|al)\b", re.IGNORECASE)

#: Non-alphabetic noise in a strain designation, stripped before the binomial fallback.
_STRAIN_NOISE = re.compile(r"[^A-Za-z .\-]")

#: An ATCC catalog number as sources write it beside the species: "ATCC 29213", "ATCC# 6538",
#: "ATCC No. 25922", "(ATCC 4868T)". The trailing letter on the last example marks a type strain
#: in microbiology nomenclature and is not part of the catalog number ATCC's own product URLs use
#: (https://www.atcc.org/products/29213, not .../29213T) -- verified live: a bare digit path
#: resolves to the specific strain's product page, and a nonexistent number 404s rather than
#: falling through to a generic search page the way some of this project's other source-database
#: links used to (see sample.external_links's module docstring on that failure mode).
#: Three ATCC numbering series appear in the data: plain digits ("ATCC 29213"), and two prefixed
#: series -- BAA (Bacteriology and Applied microbiology Accession) and MYA (mycology) -- used for
#: more recently deposited strains, e.g. "ATCC BAA-1605", "ATCC MYA-4646". All three resolve on
#: the live site as the lowercase id in the URL path (.../products/29213, .../products/baa-1605,
#: .../products/mya-4646 -- each confirmed live against the correct, specific strain page). The
#: prefixed series allow shorter numeric suffixes than the plain series -- BAA-42 is real and
#: 2 digits -- so only the plain series keeps a 3-digit floor, chosen to avoid matching incidental
#: short numbers near the word "ATCC" that are not actually part of a catalog number.
#:
#: No trailing `\b` after the digits: a type-strain marker ("4868T") is itself a word character
#: immediately following the last digit, so a boundary never occurs there and the match would
#: silently fail for every type strain. `(?!\d)` gets the same "don't grab a longer number"
#: protection without also rejecting a trailing letter.
_ATCC_ID = re.compile(r"\bATCC[\s#]*(?:No\.?\s*)?((?:BAA|MYA)-\d{1,7}|\d{3,7})(?!\d)", re.IGNORECASE)


def extract_atcc_id(name: Any) -> Optional[str]:
    """
    Pulls the ATCC catalog number out of a target-organism string, when one is present.

    Sources (DBAASP especially -- 59,904 of 174,163 released activities, 34.4%) write the strain
    together with the species rather than as its own field, so this is the only place the number
    is recoverable at all: :func:`sample.database_activities.resolve_target_species` and
    :func:`_domain_with_strain_fallback` both discard it on the way to a binomial the taxonomy can
    resolve, and rightly so for that purpose -- this is a separate, later read of the same raw
    string for a different one.

    :param name: Raw target-organism string, e.g. ``"Staphylococcus aureus ATCC 29213"`` or
        ``"Acinetobacter baumannii ATCC BAA-1605"``.
    :returns: The bare catalog number lowercased, e.g. ``"29213"`` or ``"baa-1605"`` -- the form
        ATCC's own product URLs use -- or ``None`` when the string carries no recognizable ATCC
        designation.
    """
    match = _ATCC_ID.search(str(name or ""))
    return match.group(1).lower() if match else None


def _domain_with_strain_fallback(name: str) -> str:
    """
    Resolves an organism's NCBI domain, tolerating a strain designation.

    Sources write the strain with the species -- "Escherichia coli BL21(DE3)",
    "Staphylococcus aureus ATCC 29213" -- and no strain designation resolves against the taxonomy.
    Falls back to the binomial and then the genus, which is what the strain is a strain of. Mirrors
    :func:`sample.database_activities.resolve_target_species`, kept separate only to avoid a
    circular import.

    :param name: Target organism string, possibly strain-qualified.
    :returns: The NCBI domain, or ``""`` when nothing resolves.
    """
    domain = classify_organism_domain(name)
    if domain:
        return domain
    tokens = _STRAIN_NOISE.sub(" ", name).split()
    for width in (3, 2, 1):
        if len(tokens) >= width:
            domain = classify_organism_domain(" ".join(tokens[:width]))
            if domain:
                return domain
    return ""


def infer_activity_group(target_organism: str = "", activity_type: str = "", is_cell_line: bool = None) -> str:
    """
    Infers the biological activity group enum ('antibacterial', 'antifungal', 'antiviral',
    'antiparasitic', 'anticancer', 'other') from target organism or metric text.

    Precedence, most authoritative first:

    1. **NCBI taxonomy**, when it resolves the organism to a non-animal domain. A confirmed
       bacterium, fungus, virus or protozoan is what it is, whatever else the string looks like.
    2. **Cell-line name**, for anything the taxonomy did not settle.
    3. **Text hints** from the organism and metric.

    The taxonomy comes first because Cellosaurus contains short codes that collide with strain
    designations: ``BL21``, ``K12``, ``D21`` and ``K88`` are all cell-line names, so
    "Escherichia coli K88" matched as a cell line and every one of its MICs was recorded as
    ANTICANCER. "animal" is excluded from step 1 precisely because it is the ambiguous case --
    it covers both a human tumour cell line and a parasitic helminth.

    `is_cell_line`, when provided, is trusted as-is instead of being recomputed here. Callers that
    have already run a candidate through is_valid_organism_match() (sample/extractor.py's sentence
    loop) know the filtered answer; recomputing via an unfiltered get_cellosaurus_kp() call here
    would let exactly the false positives that check exists to catch flow back in (this was
    measured: 59% of all activities got target_group="anticancer" before this fix, traced to this
    short-circuit re-running the unguarded lookup on ambiguous short organism strings). Callers
    that don't have a pre-filtered flag (sample/pdf_tables.py) pass nothing and get the original,
    unfiltered behavior.

    :param target_organism: Target organism or cell-line name.
    :param activity_type: Assay name, used only as a weak hint when the organism is unresolvable.
    :param is_cell_line: Pre-computed, already-filtered cell-line flag; recomputed when ``None``.
    :returns: One of the activity-group enum values.
    """
    combined = f"{target_organism or ''} {activity_type or ''}".lower()

    # 1. Taxonomy. Consulted before anything else, and before the keyword processor, so that a
    #    resolved microbe cannot be overridden by a name collision or a substring.
    if target_organism:
        group = _GROUP_BY_DOMAIN.get(_domain_with_strain_fallback(target_organism))
        if group:
            return group

    matches = get_organism_group_kp().extract_keywords(combined)
    if matches:
        return matches[0]

    # 2. Cell lines, for whatever the taxonomy could not resolve.
    if is_cell_line is None:
        is_cell_line = bool(target_organism) and bool(
            get_cellosaurus_kp().extract_keywords(target_organism))
    if is_cell_line:
        return "anticancer"

    # Text fallbacks, for targets the taxonomy cannot resolve. Metric abbreviations are matched on
    # word boundaries so they cannot fire from inside a longer word.
    if _METRIC_HINT_BACTERIAL.search(combined):
        return "antibacterial"
    if _METRIC_HINT_FUNGAL.search(combined):
        return "antifungal"
    if _METRIC_HINT_CANCER.search(combined):
        return "anticancer"
    if any(k in combined for k in ["hiv", "hsv", "influenza"]) or _WORD_VIRUS.search(combined):
        return "antiviral"
    if any(k in combined for k in ["parasit", "plasmodium", "leishmania"]):
        return "antiparasitic"
    return "other"


@functools.lru_cache(maxsize=None)
def get_dictionary_words() -> Set[str]:
    """
    Uppercase English vocabulary, for rejecting ordinary words matched as peptide names.

    :returns: A shared, lazily-built set of uppercase words.
    """
    words = set()
    if os.path.exists('/usr/share/dict/words'):
        with open('/usr/share/dict/words', 'r', encoding='utf-8') as f:
            words = {w.strip().upper() for w in f if len(w.strip()) >= 3}
    
    try:
        import nltk
        from nltk.corpus import names
        try:
            nltk.data.find('corpora/names')
        except LookupError:
            nltk.download('names', quiet=True)
        name_words = {w.strip().upper() for w in names.words()}
        words.update(name_words)

        # General English vocabulary (~234k words). /usr/share/dict/words alone is too sparse
        # to catch common scientific/lab terms -- it doesn't contain "phenol", "ethyl",
        # "amylase", "biosynthesis", or "inhibitor", all of which are literal entries in
        # data/resources/peptide_names.txt and get treated as "peptide name" candidates
        # (is_valid_peptide_name) without this.
        from nltk.corpus import words as nltk_words
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)
        words.update(w.upper() for w in nltk_words.words())
    except Exception:
        pass
        
    # Load surnames dataset if available
    surnames_path = os.path.join(os.path.dirname(__file__), "..", "data", "surnames.txt")
    if os.path.exists(surnames_path):
        with open(surnames_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                w = line.strip().upper()
                if len(w) >= 3:
                    words.add(w)
                    
    return words

N_TERM_MOD = re.compile(r'^(?i:NH2-|Ac-)')
C_TERM_MOD = re.compile(r'(?i:-NH2|-COOH)$')

@functools.lru_cache(maxsize=None)
def get_spacy_nlp() -> Any:
    """
    :returns: The shared scispaCy pipeline used for sentence segmentation, loaded on first use.
    """
    try:
        import spacy
        return spacy.load("en_core_sci_sm", disable=["tok2vec", "tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
    except ImportError:
        print("Warning: SciSpaCy not installed. Biomedical vocabulary filtering disabled.")
        return None

@functools.lru_cache(maxsize=None)
def get_spacy_nlp_syntax() -> Any:
    """
    :returns: The shared spaCy pipeline with the parser enabled, for clause-level analysis.
    """
    try:
        import spacy
        return spacy.load("en_core_sci_sm", disable=["ner", "lemmatizer"])
    except ImportError:
        return None

@functools.lru_cache(maxsize=None)
def get_biomed_vocab() -> Set[str]:
    """
    :returns: A shared set of biomedical terms, used to keep domain words out of the
        English-dictionary rejection path.
    """
    nlp = get_spacy_nlp()
    if nlp:
        return nlp.vocab
    return []

@functools.lru_cache(maxsize=None)
def get_umls_vocab() -> Set[str]:
    """
    :returns: A shared set of UMLS lexicon terms, used the same way as :func:`get_biomed_vocab`.
    """
    vocab = set()
    if UMLS_LEXICON_TXT.exists():
        with open(UMLS_LEXICON_TXT, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                vocab.add(line.strip().upper())
    return vocab

_ACRONYMS = {
    "IFN", "TNF", "PBS", "BSA", "FCS", "FBS", "LPS", "MIC", "MBC", "FIC", "FITC", "PEG", "WT", "DNA", "RNA", "ATP", "ADP", "AMP", "NAD", "FAD", "HIV", "HSV", "MRSA", "VRE", "ESBL", "ATCC", "CFU", "BHI", "TSB", "TSA", "ROS", "RNS", "CDNA", "MHA", "MHB",
    "EDTA", "EGTA", "CMC", "DAHA", "HEPES", "TRIS", "DAPI", "SDS", "TFA", "DMEM", "MEM", "NADH", "CAMP", "CGMP", "PVDF", "PAGE", "DPBS", "HBSS", "MTT", "MTS", "WST", "DTT", "PMSF", "NTA", "TAE", "TBE",
    "DEAE", "SPDP", "CHAPS", "CAPS", "MOPS", "MES", "TAPS", "PIPES", "SSC", "CTAB", "IPTG", "XGAL", "DEPC", "DMSO", "DMF", "THF", "DCM", "TCA", "HSA", "GTP", "GDP", "GMP", "CTP", "UTP", "TTP",
    "NADP", "NADPH", "FADH", "MRNA", "TRNA", "RRNA", "SIRNA", "MIRNA", "SHRNA", "PCR", "QPCR", "RTPCR", "ELISA", "FACS", "HPLC", "FPLC", "NMR", "MS", "GCMS", "LCMS", "FTICR", "MALDI", "TOF", "ESI",
    "SPR", "ITC", "CD", "UV", "VIS", "IR", "FTIR", "EM", "TEM", "SEM", "AFM", "MFC", "ZOI", "PFU", "MOI", "OD", "GFP", "RFP", "YFP", "CFP", "TRITC", "PI", "EB", "XTT", "LDH", "NO", "IL", "TGF", "EGF",
    "FGF", "VEGF", "PDGF", "NGF", "BDNF", "GDNF", "HCV", "HBV", "HPV", "RSV", "VRSA", "CRE", "KPC", "NDM", "OXA", "IMP", "VIM", "LB", "SOC", "YPD", "YPG", "RPMI", "GASPAK", "GASPAKS",
    "NP-40", "NP40", "RPMI-1640", "RPMI1640", "HC50", "HC1", "CO2", "OD600", "OD", "N2", "O2", "PO4", "SO4", "NH4",
    "NH2", "CH2", "CH3", "MW"
}
# Unbounded junk-token families (open-ended by design, e.g. any CD-number or IC/EC/ED-number) that
# can't be enumerated as a static set the way _ACRONYMS/_FALSE_POSITIVE_SEQS are. Found via a
# corpus-wide audit of extraction output: the unvalidated SEQ_ALPHANUM candidate path (2-6 letters
# + optional digit suffix, meant to catch real alphanumeric peptide names like "LL37") was the
# single largest source of junk peptide_name values -- assay-metric acronyms (IC50, EC50, MIC50),
# immune-cell markers (CD3/CD4/CD8/CD14), pH notation (pH7, pH8), and reversed unit+concentration
# OCR artifacts (mM3, mM10) all match that same shape.
_JUNK_SHAPE_PATTERNS = [
    re.compile(r'CD\d{1,3}$', re.IGNORECASE),
    re.compile(r'IL-?\d{1,2}[A-Z]{0,3}$', re.IGNORECASE),
    re.compile(r'PH\d{1,2}(?:\.\d+)?$', re.IGNORECASE),
    re.compile(r'(?:IC|EC|ED|LD|MIC|MBC|MFC|HC|CC)\d{1,3}$', re.IGNORECASE),
    re.compile(r'[MUNµ]M\d+$', re.IGNORECASE),
]
_FALSE_POSITIVE_SEQS = {
    "PEPTIDE", "PEPTIDES", "ANTIMYCIN", "LYSINE", "LISINE", "LYSIN", "TRYPTIC", 
    "ACTIVE", "ACTIVITY", "ACID", "WATER", "CELL", "CELLS", "TEST", "ASSAY", 
    "DATA", "STRAIN", "STRAINS", "EFFECT", "EFFECTS", "YEAST", "SYNTHESIS", 
    "CYTOTOXIC", "CYTOTOXICITY", "HEMOLYTIC", "HEMOLYSIS", "GHA", "LAHA", 
    "DAP", "IDAP", "CIDAP", "FS", "EF", "GLYCINE", "ALANINE", "VALINE", 
    "SERINE", "CYSTEINE", "PHENYLALANINE", "ASPARTATE", "ASPARAGINE", 
    "ARGININE", "HISTIDINE", "GLYCIN", "ALANIN", "VALIN", "SERIN", "CYSTEIN",
    "ASPARTIC", "ARGININ", "HISTIDIN", "AMINO", "AMINOACID", "PROTEASE", 
    "KINASE", "LIGASE", "PHASE", "MEDIA", "MEDIUM", "MICE", "RATS", "HUMAN",
    "ANIMAL", "ANIMALS", "TISSUE", "TISSUES", "MEMBRANE", "SARCIN", "ARNETT",
    "COLICIN", "BACTERIOCIN", "NISIN", "MUREIN", "CASIDIN", "CIDIN", "DEFENSIN", "MAGAININ",
    "LYSOZYME", "SCHLESSINGER", "GRAMNEGATIVE", "GRAMPOSITIVE", "WAKSMANN", "WINTERSTEINER",
    "CARDWELL", "SIIEMYAKIN", "SHEMYAKIN", "SIIEMVAKIN", "SHANKMAN", "KANAMYCIN",
    "GENTAMICIN", "INTRATHECAL", "CLINDAMYCIN", "PENICILLINASERESISTANTP", "BACITRACIN",
    "CCNSC", "NCCLS", "WAKISAKA", "RATNAM", "TSIALI", "CEPHALEXIN", "ANTIFUNGAL",
    "AUREUS", "COLI", "AERUGINOSA", "BAUMANNII", "SUBTILIS", "PNEUMONIAE", "FAECALIS", 
    "TYPHIMURIUM", "ALBICANS", "NIGER", "FUMIGATUS", "NEOFORMANS", "CEREVISIAE", "FALCIPARUM", 
    "HELA", "INHIBITORY", "CONCENTRATION", "MINIMUM", "MIC", "MBC", "MFC", "IC50", "EC50", 
    "ZOI", "ZONE", "GROWTH", "VALUE", "UNIT", "ORGANISM", "TARGET", "TABLE", "FIGURE",
    "MICROGRAM", "MILLIGRAM", "MICROMOLAR", "MILLIMOLAR", "PERCENTAGE", "UG/ML", "MG/L", "UM", "NM",
    "DIPEPTIDE", "DIPEPTIDES", "TRIPEPTIDE", "TRIPEPTIDES", "TETRAPEPTIDE", "POLYPEPTIDE", "POLYPEPTIDES",
    "PENICILLIN", "PENICILLINS", "ENICILLIN", "ENICILLINS", "ANILIDE", "ANILIDES", "GASPAK", "GASPAKS",
    "NAKAHARA", "NAKAHARAS", "AMPHOTERICIN", "CEPHALOSPORIN", "CEPHALOSPORINS", "CARBAPENEM", "FLOXACIN",
    "QUINOLONE", "SULFONAMIDE", "AMIDE", "AMIDES", "DERIVATIVE", "DERIVATIVES",    "COMPOUND", "COMPOUNDS",
    "ANALOG", "ANALOGS", "FRACTION", "FRACTIONS",
    "PROTEIN", "PROTEINS", "SOLUTION", "SOLUTIONS",
    "CONCENTRATION", "CONCENTRATIONS", "AMINOACIDS", "VITAMIN"
}

# Expression hosts and reference strains of this literature. Standard, stable and small enough to
# enumerate, unlike the open-ended strain codes (CH-112 and friends) left to the LLM triage.
_STRAIN_DESIGNATIONS = {
    "DE3", "BL21", "JM101", "JM109", "DH5A", "DH5ALPHA", "TOP10", "XL1", "XL1BLUE", "MG1655",
    "K12", "K-12", "W3110", "BW25113", "ROSETTA", "ORIGAMI", "SHUFFLE", "C41", "C43", "HB101",
    "PAO1", "PA14", "PAK", "USA300", "NCTC8325", "SG511", "RN4220", "COL", "MU50", "N315",
    "SC5314", "AF293", "H37RV", "H37RA", "ATCC", "NCTC", "NCIMB", "DSM", "CECT", "KCTC",
}

# Peptide antibiotics that are ordinary dictionary words AND absent from the merged catalogue, so
# neither the exact lookup nor the family-head path can rescue them from the dictionary gate.
# Verified against the catalogue: "bacitracin" has no entry under any spelling, while tyrocidine,
# polymyxin, gramicidin and colistin do and need no help here. This is a genuine gap in the source
# databases rather than a heuristic, so it is a short explicit list and should shrink if the
# catalogue is ever rebuilt from a more complete source.
_CATALOGUE_GAP_PEPTIDES = {"BACITRACIN", "SUBTILIN", "SURFACTIN", "VALINOMYCIN", "ALAMETHICIN"}

# Class/category nouns that name a kind of molecule, cell or preparation rather than a specific
# peptide. All were observed as peptide_name in real extraction output.
_BIOLOGICAL_CATEGORY_WORDS = {
    "NEUTROPHIL", "NEUTROPHILS", "LYMPHOCYTE", "LYMPHOCYTES", "MACROPHAGE", "MACROPHAGES",
    "MONOCYTE", "MONOCYTES", "PLATELET", "PLATELETS", "ERYTHROCYTE", "ERYTHROCYTES",
    "RIBOSOME", "RIBOSOMES", "VIRION", "VIRIONS", "PLASMID", "PLASMIDS", "LIPOSOME", "LIPOSOMES",
    "HOMODIMER", "HETERODIMER", "MONOMER", "DIMER", "TRIMER", "TETRAMER", "OLIGOMER",
    "LIGAND", "LIGANDS", "RECEPTOR", "RECEPTORS", "SUBSTRATE", "SUBSTRATES", "INHIBITOR",
    "LYMPHOKINE", "LYMPHOKINES", "CYTOKINE", "CYTOKINES", "CHEMOKINE", "CHEMOKINES",
    "HORMONE", "HORMONES", "ANTIBODY", "ANTIBODIES", "ANTIGEN", "ANTIGENS", "ADJUVANT",
    "HYDROLYSATE", "HYDROLYSATES", "LYSATE", "LYSATES", "EXTRACT", "EXTRACTS", "SUPERNATANT",
    "RECOMBINANT", "SYNTHETIC", "NATIVE", "PURIFIED", "GLYCOSYLATED", "PHOSPHORYLATED",
    "TRANSMEMBRANE", "CYTOPLASMIC", "EXTRACELLULAR", "INTRACELLULAR", "MEMBRANE",
    "LIPOPOLYSACCHARIDE", "PEPTIDOGLYCAN", "GLYCOPROTEIN", "LIPOPROTEIN", "PROTEOGLYCAN",
    "SEROTYPE", "SEROTYPES", "GENOTYPE", "ISOFORM", "ISOFORMS", "PRECURSOR", "PROPEPTIDE",
    "FRAGMENT", "FRAGMENTS", "HOMOLOG", "HOMOLOGUE", "ANALOGUE", "CONJUGATE", "DERIVATIVE",
    # Enzymes and carrier/serum proteins whose names predate the "-ase" convention, so the suffix
    # rule above cannot reach them. They are common assay reagents in this literature and were
    # appearing as peptide_name in real output.
    "TRYPSIN", "CHYMOTRYPSIN", "PEPSIN", "PAPAIN", "THROMBIN", "PLASMIN", "PRONASE",
    "PEPSTATIN", "APROTININ", "GLOBULIN", "ALBUMIN", "GLUTATHIONE", "HEMOGLOBIN",
    # Protein/molecule classes and lab shorthand that survived every other gate in the corpus
    # audit, each because the catalogue contained an entry beginning with the word (see
    # get_catalogued_name_heads) or because it is absent from the dictionary. A cathepsin is a
    # protease family, a lectin is a sugar-binding protein class, and "oligo"/"combi" are clipped
    # lab words ("oligonucleotide", "combination"), none of them the name of a peptide.
    "CATHEPSIN", "CATHEPSINS", "LECTIN", "LECTINS", "OLIGO", "OLIGOS", "COMBI",
    # Structural classes of peptide. A class is not a name: "lipopeptide" appeared 40 times in the
    # corpus as a peptide_name, and no chemical database can resolve it either.
    "LIPOPEPTIDE", "LIPOPEPTIDES", "GLYCOPEPTIDE", "GLYCOPEPTIDES", "CYCLOPEPTIDE",
    "DEPSIPEPTIDE", "LANTIBIOTIC", "LANTIBIOTICS", "PSEUDOPEPTIDE", "PEPTIDOMIMETIC",
    "METABOLITE", "METABOLITES", "SURFACTANT", "DETERGENT", "CHELATOR", "SCAFFOLD",
    "PYRUVATE", "LACTATE", "ACETATE", "CITRATE", "GLUTAMATE", "ASPARTATE",
    "MYOGLOBIN", "FIBRIN", "FIBRINOGEN", "CASEIN", "GELATIN", "COLLAGEN", "KERATIN",
    "SUPEROXIDE", "PEROXIDE", "PEROXIDASE",
}


def is_valid_peptide_name(name: Any) -> bool:
    """
    Filters candidate peptide *names* pulled from get_seq_names_kp() (data/resources/peptide_names.txt).
    Unlike sequence-type candidates, these were never checked against anything before being
    treated as a valid peptide name in sample/extractor.py -- and that reference list turns out
    to contain a fair number of generic scientific/English words alongside real peptide names
    (e.g. "biosynthesis", "antifungal", "phenol", "ethyl", "inhibitor", "subtilis", "peptides"
    are all literal entries in it). With the LLM fallback able to actually reach a model
    (recover_peptide_llm), an unfiltered generic-word "candidate" gets a fabricated activity
    record invented for it instead of being silently dropped, so this check matters more than
    it used to. Rejects known false-positive/acronym terms and plain English dictionary words;
    real peptide names are typically coined terms that won't appear in a standard dictionary.
    """
    if not name:
        return False
    core = str(name).strip()
    upper = core.upper()
    # Placeholder text and template residue. Never a peptide name, and it reaches this function
    # because the checks below are shaped for real words: brackets and spaces make core.isalpha()
    # false, so "[insert peptide name]" skipped the dictionary gate and every pattern, and was
    # accepted. Found by the error-injection suite (sample/validation_extras.py), which is the kind
    # of gap that suite exists to surface.
    if _PLACEHOLDER_NAME.search(core):
        return False
    # These macro-class names are deliberately seeded into get_seq_names_kp() (see
    # macro_classes in that function) and must not be rejected even though some of them also
    # appear in _FALSE_POSITIVE_SEQS -- that list exists to stop e.g. "MAGAININ" from being
    # misread as a literal 8-residue amino-acid sequence, which is a different question from
    # whether "Magainin" is a valid peptide *name* (it is).
    if upper in {"COLICIN", "BACTERIOCIN", "NISIN", "MUREIN", "CASIDIN", "CIDIN", "DEFENSIN", "MAGAININ"}:
        return True
    if upper in _CATALOGUE_GAP_PEPTIDES:
        return True
    if upper in _ACRONYMS or upper in _FALSE_POSITIVE_SEQS:
        return False
    # A trailing digit run stripped off a known acronym is still that acronym, not a peptide --
    # e.g. "UV230" (a UV-detector wavelength setting, "UV230 nm"), "FBS4" (a reagent concentration
    # column smashed together with "10% FBS" in an OCR'd table), "MW13"/"MW2" (molecular-weight
    # labels). Catches the whole trailing-digit-suffix family for every _ACRONYMS entry at once,
    # rather than needing a separate blocklist entry per observed suffix.
    stripped = re.sub(r'-?\d+$', '', upper)
    if stripped != upper and stripped in _ACRONYMS:
        return False
    if any(p.match(upper) for p in _JUNK_SHAPE_PATTERNS):
        return False
    # Enzyme names. The "-ase" suffix is the standard enzyme ending, and an enzyme is not the kind
    # of molecule this database catalogues -- observed as peptide_name in real output: elastase,
    # synthetase, glycosidase, dismutase, hydrolase, collagenase. Deliberately suffix-based rather
    # than a word list, since the family is open-ended. Names shorter than 6 characters are left
    # alone so a real peptide code that happens to end in those letters is not caught.
    # Compared on letters only. Testing core.isalpha() directly meant any multi-word enzyme name
    # skipped this rule entirely, because the space makes it false -- "dismutase" was rejected
    # while "superoxide dismutase" sailed through, as did "alkaline phosphatase".
    # Threshold is 5, not 6, so the two shortest enzyme names in this corpus are covered: RNase and
    # DNase are both exactly 5 letters and were the only "-ase" names still reaching the output
    # ("rnase", 35 records). No real peptide name of 5 letters or more ends in "-ase".
    alpha_only = re.sub(r'[^A-Za-z]', '', core)
    if len(alpha_only) >= 5 and alpha_only.upper().endswith("ASE"):
        return False
    # Adjectives describing an activity or property. They are never a molecule's name, and the
    # suffixes are closed sets, unlike the open-ended families handled by word lists.
    # IONIC covers "cationic"/"anionic" and OTIC covers "apoptotic"/"necrotic" -- charge and
    # cell-fate adjectives that describe a peptide but do not name one.
    if alpha_only.isalpha() and upper.endswith((
        "CIDAL", "STATIC", "GENIC", "LYTIC", "PHILIC", "PHOBIC", "TROPIC", "OGENOUS",
        "IONIC", "OTIC",
    )):
        return False
    # Microbial strain and expression-host designations. These sit next to a peptide in exactly the
    # places a peptide name would ("BL21(DE3) cells", "P. aeruginosa PAO1"), carry no dictionary
    # entry to be caught by, and were among the most frequent junk names in the corpus -- "DE3"
    # 127 records, "PAO1" 42. Unlike the open-ended classes left to the LLM, the standard hosts and
    # reference strains of this literature are a small, stable, enumerable set.
    if upper in _STRAIN_DESIGNATIONS:
        return False
    # Category nouns from cell biology and biochemistry: real words naming a *class* of thing
    # rather than a specific peptide. These are absent from /usr/share/dict/words and from NLTK's
    # general wordlist (verified), so the dictionary gate below cannot reach them, and they cannot
    # be separated from real peptide names by vocabulary alone -- "defensin" and "lactoferrin" are
    # in the same biomedical lexicons. Enumerated from what actually appeared in extraction output.
    if upper in _BIOLOGICAL_CATEGORY_WORDS:
        return False
    # A name catalogued in CAMP/DBAASP/DRAMP/ADP is a peptide by definition, so it outranks the
    # dictionary gate below. That gate was silently discarding exactly the molecules this database
    # exists to collect: bacitracin, gramicidin and tyrocidine are classic peptide antibiotics and
    # also ordinary dictionary entries, so every mention of them was being dropped as an English
    # word. Consulting the catalogue (built by scripts/build_name_sequence_lookup.py) resolves the
    # collision with evidence instead of another hand-maintained exception list.
    if _normalize_peptide_key(core) in get_name_sequence_map():
        return True
    # Bare family names ("gramicidin", "bacitracin") are usually absent from the catalogue, which
    # lists the specific members ("Gramicidin S", "Gramicidin C", "Bacitracin A"). Matching the
    # leading token recognizes them as peptide nomenclature: a word that begins a catalogued
    # peptide name is being used as a peptide name here, whatever the dictionary says about it.
    if core.isalpha() and _normalize_peptide_key(core) in get_catalogued_name_heads():
        return True
    if core.isalpha() and upper in get_dictionary_words():
        return False
    return True


# Average amino-acid composition of UniProtKB/Swiss-Prot (published per-release statistics,
# https://web.expasy.org/docs/relnotes/relstat.html). Four of the twenty one-letter amino-acid
# codes -- A (Ala), C (Cys), G (Gly), T (Thr) -- are also the four DNA bases, so SEQ_EXACT's
# character class ([ACDEFGHIKLMNPQRSTVWY]) happily matches a raw nucleotide read as if it were a
# valid peptide. Confirmed against real extraction output: runs like "TGTTAATAATTT",
# "CAGGTCGCAGTTCAAGTCATCAGG" were coming out as `peptide_sequence`. A composition model is the
# right level to catch this -- every one of A/C/G/T's Swiss-Prot frequencies is well below the
# 25% a uniform-random DNA base would have, so a log-likelihood-ratio test cleanly separates the
# two once a handful of characters are restricted to just these four letters.
_AA_FREQ_SWISSPROT = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0546, 'C': 0.0138,
    'Q': 0.0393, 'E': 0.0672, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
    'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
    'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687,
}
# No basis for a non-uniform base composition at the point this check runs (source organism is
# unknown) -- uniform is the standard neutral null model for this kind of composition test.
_DNA_BASE_FREQ = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
_DNA_ALPHABET = frozenset(_DNA_BASE_FREQ)
_DNA_MIN_LEN = 5  # matches SEQ_EXACT's own minimum candidate length


def dna_log_likelihood_ratio(seq: str) -> float:
    """
    Log-likelihood ratio of `seq` under a Swiss-Prot amino-acid composition model versus a
    uniform-DNA-base model, summed per character: positive favors peptide, negative favors DNA.
    Returns None if `seq` contains any character outside {A, C, G, T} -- such a sequence cannot be
    a plain nucleotide read in the first place, so the DNA hypothesis doesn't apply.
    """
    core = str(seq).upper().strip()
    if not core or not set(core) <= _DNA_ALPHABET:
        return None
    return sum(math.log(_AA_FREQ_SWISSPROT[c] / _DNA_BASE_FREQ[c]) for c in core)


def is_likely_dna_sequence(seq: str) -> bool:
    """
    True if `seq` is restricted to {A, C, G, T} and scores as more likely a DNA read than a
    peptide under the log-likelihood-ratio test above. Every one of the four bases' Swiss-Prot
    amino-acid frequencies is below the 25% a uniform DNA base has, so this triggers on
    essentially any real nucleotide run once it clears `_DNA_MIN_LEN` -- a real AMP composed
    entirely of Ala/Cys/Gly/Thr for that many consecutive residues does not occur in practice.
    """
    core = str(seq).upper().strip()
    if len(core) < _DNA_MIN_LEN:
        return False
    llr = dna_log_likelihood_ratio(core)
    return llr is not None and llr < 0


def is_valid_sequence(seq: Any, allow_dictionary: bool = False) -> bool:
    """
    Whether a candidate string is a genuine peptide sequence.

    Applies, in order: length bounds, a check that the residues are real one-letter codes, a
    log-likelihood test that rejects nucleotide runs spelling valid amino-acid letters
    (see :func:`dna_log_likelihood_ratio`), and a dictionary gate that rejects ordinary English
    words such as "SPLICED".

    :param seq: Candidate sequence, one-letter code.
    :param allow_dictionary: Skip the English-word rejection. Used where surrounding context has
        already established that the token is a sequence.
    :returns: ``True`` if the candidate should be treated as a peptide sequence.
    """
    if not seq:
        return False
    core_seq = str(seq).upper().strip()

    # Catch false positives early before any regex mangling
    flat_seq = core_seq.replace('-', '')
    if flat_seq in _ACRONYMS or flat_seq in _FALSE_POSITIVE_SEQS:
        return False

    # A run restricted to just A/C/G/T is far more likely a raw nucleotide read than a peptide --
    # see dna_log_likelihood_ratio() above. Checked before the more expensive dictionary lookups
    # below since it's a cheap, decisive rejection for this specific false-positive class.
    if is_likely_dna_sequence(flat_seq):
        return False
        
    # Block common antibiotic suffixes, chemical classes, and English word endings
    if (flat_seq.endswith("MYCIN") or flat_seq.endswith("MICIN") or flat_seq.endswith("CYCLINE") or 
        flat_seq.endswith("ZYME") or flat_seq.endswith("SPORIN") or flat_seq.endswith("PEPTIDE") or 
        flat_seq.endswith("PEPTIDES") or flat_seq.endswith("CILLIN") or flat_seq.endswith("CILLINS") or 
        flat_seq.endswith("ANILIDE") or flat_seq.endswith("ANILIDES") or flat_seq.endswith("PAK") or 
        flat_seq.endswith("PAKS") or flat_seq.endswith("SYSTEM") or flat_seq.endswith("MEDIUM") or 
        "RESISTANT" in flat_seq or "NEGATIVE" in flat_seq or "POSITIVE" in flat_seq):
        return False
        
    # Block stereochemistry artifacts (e.g. 'LDLDLD' or 'DDLLDD')
    if set(flat_seq) <= {'L', 'D'} and len(flat_seq) >= 4:
        return False

    # Block masked/degenerate sequences, where X stands for "any or unspecified residue" -- e.g.
    # "XXNXXXX", observed in real output as a peptide_sequence. A string that is mostly placeholder
    # identifies no actual peptide, so it cannot be a database record no matter how it was found.
    if flat_seq.count('X') * 2 >= len(flat_seq):
        return False
        
    # Block extremely low-complexity sequences (e.g., homopolymers 'AAAAA' or repeats 'ABABAB')
    if len(set(flat_seq)) == 1 and len(flat_seq) >= 3:  # e.g., AAA, LLL
        return False
    if len(set(flat_seq)) == 2 and len(flat_seq) >= 6:  # e.g., ABABAB
        # Optional: check if it's purely alternating or just very low entropy
        pass
        
    # If the string is a known peptide name (e.g. 'defensin'), it is a NAME, not a 1-letter sequence!
    # get_seq_names_kp() holds all known names.
    if get_seq_names_kp().extract_keywords(flat_seq):
        # Only reject if the name completely matches the sequence
        # e.g. "defensin" matches "defensin"
        # We don't want to reject if the sequence just happens to contain a substring, 
        # but extract_keywords returns a list. Let's just check if the whole string is in it.
        # Actually, if the entire flat_seq is a known name, reject it.
        found = get_seq_names_kp().extract_keywords(flat_seq, span_info=True)
        for kw, start, end in found:
            if start == 0 and end == len(flat_seq):
                return False
        
    has_term_mod = bool(N_TERM_MOD.match(seq) or C_TERM_MOD.search(seq))
    
    core_seq = N_TERM_MOD.sub('', core_seq)
    core_seq = C_TERM_MOD.sub('', core_seq)
    
    # Only remove stereochemistry markers if they have a hyphen (e.g. D- or [D]-)
    core_seq = re.sub(r'\[?[DLdl]\]?-', '', core_seq)
    
    has_hyphens = '-' in core_seq
    core_seq = core_seq.replace('-', '')
    
    if not allow_dictionary:
        if not (has_hyphens or has_term_mod):
            # Reject if in OS dictionary
            if core_seq in get_dictionary_words():
                return False
            # Reject if in SciSpaCy PubMed vocabulary (check both upper and lowercase)
            biomed = get_biomed_vocab()
            if core_seq in biomed or core_seq.lower() in biomed:
                return False
            # Reject if in UMLS Lexicon
            if core_seq in get_umls_vocab():
                return False
        
    return True

_ALIAS_SHORTCODE = r'[A-Za-z][\w.\-]{1,13}'
_ALIAS_SUFFIX = re.compile(rf'[\s,]{{0,2}}\(\s*({_ALIAS_SHORTCODE})\s*\)')
_ALIAS_PREFIX = re.compile(rf'({_ALIAS_SHORTCODE}),\s*$')
_PURE_AA_LETTERS = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$', re.IGNORECASE)

def find_local_aliases(text: str, sequences_list: list) -> dict:
    """
    Detects in-document shorthand-name <-> sequence bindings, e.g. a paper stating
    "NH2-ELPKLPDDKVLIRSRSNCPKGKVWNGFDCKSPFAFS (SP4080)" once and then discussing that peptide's
    activity using only "SP4080" afterward. Without this, the sequence-bearing candidate never
    gets an activity sentence linked to it (the shorthand is paper-local, never in any global
    known-name list) and is silently dropped by extract_peptides_from_text's final
    "no activities -> discard" filter -- confirmed against real documents in a corpus-wide audit
    (493 records had an empty peptide_sequence despite the real sequence being present in their
    own captured context, exactly this pattern).

    Deliberately NOT @functools.lru_cache'd, unlike every other lookup in this module -- this is
    built from the `text` argument itself, once per document, so caching it would leak one
    document's aliases into another's.
    """
    alias_map = {}
    known_exact_vals = {s["value"] for s in sequences_list if s["type"] in ("exact", "3letter")}
    if not known_exact_vals:
        return alias_map

    def _register(shortcode: str, seq_val: str) -> None:
        """
        Binds a document-local shortcode to a sequence, refusing ambiguous bindings.

        :param shortcode: Paper-local label, e.g. ``"SP4080"``.
        :param seq_val: Sequence the label refers to.
        :returns: ``None``. A shortcode already bound to a different sequence is dropped rather
            than overwritten, since a wrong binding fabricates data.
        """
        if not is_valid_peptide_name(shortcode):
            return
        # Reject a shortcode that is itself composed purely of valid amino-acid letters and
        # independently passes the sequence blocklist -- that's two distinct peptides mentioned
        # adjacently ("GACFSIAHECGA next to CTTHWGFTLC"), not a name/sequence pair.
        if _PURE_AA_LETTERS.match(shortcode) and is_valid_sequence(shortcode):
            return
        # Real paper-local shorthand codes (SP4080, hBD-2, SV-4814) are short and carry a digit,
        # hyphen, or internal capital -- a plain run-together lowercase word never is. Measured
        # against a real document: without this, "...poly(A) addition signal (ATAAA)..." bound
        # the nonsense shortcode "additionsignal" to an unrelated DNA motif that happened to also
        # spell valid amino-acid letters.
        if shortcode.isalpha() and shortcode.islower():
            return
        existing = alias_map.get(shortcode)
        if existing is not None and existing != seq_val:
            alias_map[shortcode] = None  # ambiguous within this document -- drop, don't guess
            return
        if shortcode not in alias_map:
            alias_map[shortcode] = seq_val

    for m in SEQ_EXACT.finditer(text):
        seq_val = m.group(1).replace('-', '')
        if seq_val not in known_exact_vals:
            continue

        tail = _ALIAS_SUFFIX.match(text, m.end())
        if tail:
            _register(tail.group(1), seq_val)

        next_char = text[m.end():m.end() + 1]
        if next_char in (',', '.', ')'):
            head_window = text[max(0, m.start() - 20):m.start()]
            head = _ALIAS_PREFIX.search(head_window)
            if head:
                _register(head.group(1), seq_val)

    return {k: v for k, v in alias_map.items() if v is not None}


def _flatten_whitespace(text: str) -> Tuple[str, List[int]]:
    """
    Returns (flattened_text, index_map): flattened_text has every whitespace/line-break character
    removed, and index_map[i] gives flattened_text[i]'s position in the original text -- lets a
    match found in the flattened text be traced back to its real span in the original.
    """
    chars = []
    index_map = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(ch)
        index_map.append(i)
    return ''.join(chars), index_map


# A genuine PDF line-wrap or table-cell rewrap inserts at most a handful of whitespace characters
# into an otherwise-contiguous known sequence. Without a bound, collapsing ALL whitespace from a
# whole document before scanning risks stitching two unrelated, distant known-sequence fragments
# across a large intervening gap (e.g. across a page break) into a false match.
_MAX_WHITESPACE_BREAK_CHARS = 30


def find_whitespace_broken_sequences(text: str) -> dict:
    """
    Finds known peptide sequences (get_known_seq_1l_kp()/get_known_seq_3l_kp()) present in the
    document but split across a PDF line-wrap or extra internal whitespace -- e.g. a table column
    rendering "GIGKFLHSAK\\nKFGKAFVGEIMNS" instead of the contiguous sequence. Plain Aho-Corasick
    matching (case-folded exact substring search) can't tolerate this: the registered keyword and
    the on-page text differ by the inserted whitespace. Scans a whitespace-collapsed copy of the
    text instead, then maps each hit back to its real (still whitespace-broken) span in the
    original -- that literal, broken substring is what's returned, since it's what the sentence
    loop's plain substring search (extractor.py) needs to actually find later on the page.

    Returns {original_broken_surface_text: (clean_known_sequence_value, seq_type)}, seq_type is
    "exact" (1-letter) or "3letter". Contiguous (unbroken) matches are excluded -- those are
    already found directly by get_known_seq_1l_kp()/get_known_seq_3l_kp() in extractor.py's normal
    candidate discovery.
    """
    flat_text, index_map = _flatten_whitespace(text)
    if not flat_text:
        return {}

    candidates = []
    for kp, seq_type in ((get_known_seq_1l_kp(), "exact"), (get_known_seq_3l_kp(), "3letter")):
        # iter_raw(), not extract_keywords(): the wrapper's own word-boundary check runs against
        # flat_text, where whitespace-adjacent characters have been pulled directly next to each
        # other -- it would (and, before this fix, silently did) reject a genuine line-wrap break
        # like "...peptide GIGKFLHSAK\nKFGKAFVGEIMNS showed..." as "embedded in a longer word",
        # because in flat_text that's exactly what it looks like ("...peptideGIGK...MNSshowed...").
        # The boundary check below runs against the ORIGINAL text instead, at the mapped-back
        # span, which still has the real word-boundary whitespace in place.
        for name_key, fstart, fend in kp.iter_raw(flat_text):
            orig_start = index_map[fstart]
            orig_end = index_map[fend - 1] + 1
            if orig_start > 0 and text[orig_start - 1].isalnum():
                continue
            if orig_end < len(text) and text[orig_end].isalnum():
                continue
            surface = text[orig_start:orig_end]
            if surface == name_key:
                continue  # contiguous in the original -- not a whitespace-break case
            if (orig_end - orig_start) - len(name_key) > _MAX_WHITESPACE_BREAK_CHARS:
                continue
            # A wrapped sequence crosses exactly one line break. A span crossing two or more is
            # more likely several DIFFERENT peptides listed on consecutive lines of a table, which
            # this must not fuse into one fabricated sequence -- so the multi-line case is left to
            # the per-line SEQ_EXACT matches, which already capture each peptide separately.
            # (Intervening non-whitespace text can't reach here at all: flattening removes only
            # whitespace, so any row label between two fragments breaks the keyword match.)
            if text.count("\n", orig_start, orig_end) > 1:
                continue
            # A wrapped sequence and two peptides on consecutive table rows are indistinguishable
            # by line count -- both put exactly one newline between the fragments. What separates
            # them is the catalogue: if EVERY fragment is independently a catalogued peptide, this
            # is a list of distinct peptides that happens to concatenate into another catalogued
            # sequence, and fusing them would fabricate a peptide the document never reported. A
            # genuine line wrap splits a sequence at an arbitrary point, so its fragments are not
            # themselves catalogued entries.
            fragments = [f for f in surface.split() if len(f) >= 5]
            if len(fragments) > 1 and all(
                kp.extract_keywords(f) == [f] for f in fragments
            ):
                continue
            if seq_type == "exact" and not is_valid_sequence(name_key):
                continue
            candidates.append((orig_start, orig_end, surface, name_key, seq_type))

    # The known-sequence resource files catalog both full peptides and their fragments/truncated
    # analogs as separate entries (e.g. "Magainin-2" and "Magainin-2 (1-17)"), so a single broken
    # run in a document can match several different known sequences at once, nested inside each
    # other. Keep only the longest (maximal) span at each position -- a shorter one fully
    # contained within it is the same on-page text, not a separate additional finding.
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    result = {}
    last_end = -1
    for orig_start, orig_end, surface, name_key, seq_type in candidates:
        if orig_start < last_end:
            continue
        result[surface] = (name_key, seq_type)
        last_end = orig_end
    return result





_REF_SECTION_PATTERN = re.compile(
    r'\n(?:\d+\.\s*)?(?:R[\s]*E[\s]*F[\s]*E[\s]*R[\s]*E[\s]*N[\s]*C[\s]*E[\s]*S?|L[\s]*I[\s]*T[\s]*E[\s]*R[\s]*A[\s]*T[\s]*U[\s]*R[\s]*E[\s]+C[\s]*I[\s]*T[\s]*E[\s]*D|B[\s]*I[\s]*B[\s]*L[\s]*I[\s]*O[\s]*G[\s]*R[\s]*A[\s]*P[\s]*H[\s]*Y)[\s\n]*\n',
    re.IGNORECASE
)


def truncate_references(text: str) -> str:
    """
    Cuts off a manuscript's bibliography/references section. Candidate discovery run over the
    full raw text (the norm before this was factored out -- there was no reference truncation
    anywhere in the active extraction pipeline) picks up author surnames, years, and journal
    abbreviations from citations as if they were peptide names, which then get fed to the LLM
    fallback (recover_peptide_llm) and can come back with a fabricated activity record for
    someone's surname. First tries an exact "REFERENCES"/"LITERATURE CITED"/"BIBLIOGRAPHY"
    header match; falls back to a citation-density heuristic (blocks packed with years/"et al."/
    volume:page numbers) for scanned/OCR text where the header itself didn't extract cleanly.
    """
    ref_match = _REF_SECTION_PATTERN.search(text)
    if ref_match:
        return text[:ref_match.start()]

    blocks = re.split(r'\n{2,}', text)
    start_idx = len(blocks) // 2
    for i in range(start_idx, len(blocks)):
        block = blocks[i]
        years = len(re.findall(r'\b(?:19|20)\d{2}\b', block))
        et_als = len(re.findall(r'et\s+al\.', block, re.IGNORECASE))
        vol_pages = len(re.findall(r'\b\d+\s*:\s*\d+-\d+\b', block))
        if (years >= 2 and et_als >= 1) or (years >= 3) or (years >= 1 and vol_pages >= 1):
            return '\n\n'.join(blocks[:i])
    return text



