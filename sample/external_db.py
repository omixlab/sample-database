from sample.config import DATA_DIR
import os
import json
import logging
import concurrent.futures
import requests
import sqlite3
import threading
import time
import urllib.parse
import re
import math
from functools import lru_cache
from typing import Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from sample.rf_filter import filter_sequence
from sample.sequence import sequence_to_1_letter

# --- Shared HTTP session (connection pooling + automatic retry/backoff) ---
_SESSION = None
_SESSION_LOCK = threading.Lock()


def _get_session(pool_size: int = 20) -> requests.Session:
    """
    The shared HTTP session, created on first use.

    One session for the whole process so connections are pooled and the retry policy is decided
    in a single place.

    :param pool_size: Connection-pool size for the mounted adapters.
    :returns: The shared ``requests.Session``.
    """
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                session = requests.Session()
                # One retry, not three. A read timeout here is the endpoint being slow rather than
                # a lost packet, so repeating the same query mostly multiplies the wait: at the
                # previous total=3 with a 10s read timeout, a single unresolvable name could hold a
                # worker for 40s and still return nothing. Persistent failure is handled by
                # _HostBreaker instead, which is the right tool for a degraded host.
                #
                # 429 and 503 are deliberately NOT retried here. They are the two codes that mean
                # "you are sending too much", and a transport-level retry answers them by sending
                # more -- immediately, without consulting the rate limiter. Against PubChem's
                # quota-based throttle that is self-reinforcing: retries raise the request count,
                # the higher count returns more 503s, and the run gets slower the longer it runs,
                # which is exactly the degradation seen in this step's logs. Both are handled in
                # the callers instead, where the throttling headers can be read and obeyed.
                retry = Retry(
                    total=1,
                    backoff_factor=0.5,
                    status_forcelist=[500, 502, 504],
                    allowed_methods=frozenset(["GET"]),
                )
                adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _SESSION = session
    return _SESSION


class _RateLimiter:
    """
    Minimum spacing between requests to one host, shared across threads, that SLOWS DOWN when the
    host starts failing and speeds back up when it recovers.

    The adaptive part is the point. Measured directly, ChEMBL answers a molecule search in
    0.8-1.7s -- it is not a slow endpoint. The wall of 10s read timeouts in this step's logs was
    self-inflicted: EBI throttles per IP, so ten worker threads issuing up to ten requests a second
    got queued server-side until they timed out. A fixed rate cannot cope with that, because the
    rate that is safe depends on how the server feels at the time. Worse, it degrades: the run's
    own ETA drifted from 2h to 27h as sustained load made the throttling harder.

    Backing off on failure and recovering on success keeps the step near whatever rate the server
    is actually willing to serve, instead of guessing one in advance and hammering through it.
    """

    def __init__(self, min_interval: float, max_interval: float = 4.0) -> None:
        """
        :param min_interval: Starting seconds between request starts; also the floor that
            :meth:`relax` decays back toward.
        :param max_interval: Ceiling the backoff may reach.
        """
        self.base_interval = min_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        """
        Blocks until the configured interval has elapsed since the previous request start.

        :returns: ``None``.
        """
        with self._lock:
            interval = self.min_interval
            elapsed = time.monotonic() - self._last_call
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_call = time.monotonic()

    def penalize(self, factor: float = 2.0) -> None:
        """
        Backs off after a timeout or an explicit throttle response.

        :param factor: Multiplier applied to the current interval, capped at ``max_interval``.
        :returns: ``None``.
        """
        with self._lock:
            self.min_interval = min(self.min_interval * factor, self.max_interval)

    def relax(self) -> None:
        """:returns: ``None``. Eases the interval back toward the base rate after a success."""
        with self._lock:
            if self.min_interval > self.base_interval:
                self.min_interval = max(self.base_interval, self.min_interval * 0.9)


# Starting rates are deliberately well below what each service tolerates in a burst. The total work
# is only ~2,700 distinct names (the rest are cache hits), so even 3 requests/sec finishes ChEMBL
# in about fifteen minutes -- there is nothing to gain by crowding the server, and the previous
# 10/sec is what turned a fast endpoint into a wall of timeouts.
_PUBCHEM_LIMITER = _RateLimiter(1.0 / 5)   # PubChem PUG-REST usage policy: ~5 requests/sec
_CHEMBL_LIMITER = _RateLimiter(1.0 / 3)

# Concurrency caps, which the rate limiter alone does not provide: spacing request STARTS says
# nothing about how many are in flight at once, and it is simultaneous connections that EBI
# throttles. Without this, a slow response lets the next request start anyway and in-flight
# requests pile up without bound.
_PUBCHEM_SEMAPHORE = threading.BoundedSemaphore(4)
_CHEMBL_SEMAPHORE = threading.BoundedSemaphore(3)

# Long enough to ride out a queued request, short enough not to tie a thread up indefinitely.
# Raising this instead of reducing pressure would be the wrong trade -- under throttling it just
# holds each thread longer for the same throughput.
_PUBCHEM_TIMEOUT = 15
_CHEMBL_TIMEOUT = 20


class _HostBreaker:
    """
    Stops querying a host that is persistently failing, for a cooldown, then tries again.

    The last line of defence, after the adaptive rate limiter and the concurrency caps. Those two
    handle a host that is merely throttling; this one handles a host that is genuinely down or has
    blocked us, where every request costs its full timeout to learn nothing and worker threads
    would otherwise spend the rest of the run discovering that one at a time.
    """

    def __init__(self, name: str, threshold: int = 8, cooldown: float = 300.0) -> None:
        """
        :param name: Host label used in log messages.
        :param threshold: Consecutive failures required to open the breaker.
        :param cooldown: Seconds to stay open before allowing a retry.
        """
        self._name = name
        self._threshold = threshold
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._consecutive = 0
        self._open_until = 0.0
        self._skipped = 0

    def is_open(self) -> bool:
        """
        :returns: ``True`` while lookups should be skipped. Closes the breaker automatically once
            the cooldown has elapsed, logging how many calls were skipped.
        """
        with self._lock:
            if self._open_until and time.monotonic() >= self._open_until:
                logger.warning(
                    f"{self._name}: retrying after cooldown ({self._skipped} lookup(s) skipped)."
                )
                self._open_until, self._skipped, self._consecutive = 0.0, 0, 0
            if self._open_until:
                self._skipped += 1
                return True
        return False

    def record_success(self) -> None:
        """:returns: ``None``. Resets the consecutive-failure counter."""
        with self._lock:
            self._consecutive = 0

    def record_failure(self) -> None:
        """:returns: ``None``. Counts a failure and opens the breaker at `threshold` in a row."""
        with self._lock:
            self._consecutive += 1
            if self._consecutive >= self._threshold and not self._open_until:
                self._open_until = time.monotonic() + self._cooldown
                logger.warning(
                    f"{self._name}: {self._consecutive} consecutive failures; "
                    f"pausing lookups for {self._cooldown:.0f}s."
                )


_PUBCHEM_BREAKER = _HostBreaker("PubChem")
_CHEMBL_BREAKER = _HostBreaker("ChEMBL")

# PubChem reports its own view of how hard we are pushing it, on every response, and its usage
# policy asks clients to read it and adapt rather than to guess a safe rate. Probing it directly
# while this step was running returned "Request Count status: Black (125%)" on every request --
# the quota was already blown, so every lookup came back 503 having done no work. Guessing a fixed
# rate cannot fix that, because the budget is per-minute and shared with anything else on this IP;
# only the server knows the real number, and it is telling us.
#
# Header format: "Request Count status: Green (12%), Request Time status: Green (2%), Service
# status: Green". Green/Yellow/Red/Black is the escalation, Black meaning requests are refused.
_THROTTLE_COUNT = re.compile(r'Request Count status:\s*(\w+)', re.IGNORECASE)
_THROTTLE_BACKOFF = {"green": None, "yellow": 1.5, "red": 2.5, "black": 4.0}


def _apply_pubchem_throttle(response: Any) -> bool:
    """
    Adjusts the PubChem rate from its throttling header. Returns True if we are being refused.
    """
    status = ""
    match = _THROTTLE_COUNT.search(response.headers.get("X-Throttling-Control", "") or "")
    if match:
        status = match.group(1).lower()
    factor = _THROTTLE_BACKOFF.get(status)
    if factor:
        _PUBCHEM_LIMITER.penalize(factor)
        if status == "black":
            logger.warning(
                "PubChem request quota exceeded (status Black); backing off to "
                f"{_PUBCHEM_LIMITER.min_interval:.1f}s between requests."
            )
    elif status == "green":
        _PUBCHEM_LIMITER.relax()
    return status == "black"

# PubChem and ChEMBL are independent services, so a peptide should not wait for one before asking
# the other. Serially, every uncached peptide costs PubChem's latency PLUS ChEMBL's, and ChEMBL's
# free-text search is by far the slower of the two -- so the fast lookup spends most of its time
# queued behind the slow one for no reason.
#
# Deliberately a separate pool from the file-level workers: submitting into the pool a task is
# already running in would deadlock once every worker is blocked waiting on its own children.
_LOOKUP_POOL = None
_LOOKUP_POOL_LOCK = threading.Lock()


def _get_lookup_pool() -> concurrent.futures.ThreadPoolExecutor:
    """
    :returns: The shared pool used to run the PubChem and ChEMBL lookups for one peptide
        concurrently. Separate from the file-level pool: submitting into the pool a task is
        already running in would deadlock.
    """
    global _LOOKUP_POOL
    if _LOOKUP_POOL is None:
        with _LOOKUP_POOL_LOCK:
            if _LOOKUP_POOL is None:
                # Sized to the two host semaphores (4 + 3) plus one. A larger pool would only
                # park more threads on those semaphores; the caps, not the pool, set the pressure.
                _LOOKUP_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=8, thread_name_prefix="extdb-lookup")
    return _LOOKUP_POOL

# One in-flight request per (source, name). The sqlite cache dedups across time but not across
# threads: with a pool of workers, every thread that reaches the same name before the first
# response lands issues its own identical request. That is visible in the step's own logs, where
# the same query ("peptide C", "Magainin-2") is retried by several threads within the same second.
_NAME_LOCKS = {}
_NAME_LOCKS_GUARD = threading.Lock()


def _name_lock(key: tuple) -> threading.Lock:
    """
    :param key: ``(source, name)`` identifying one lookup.
    :returns: A lock unique to that key, so concurrent threads issue one request per name rather
        than one each.
    """
    with _NAME_LOCKS_GUARD:
        lock = _NAME_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _NAME_LOCKS[key] = lock
        return lock


# Placeholder and shorthand "names" that no chemical database can resolve, but which are shaped
# like a real compound name and so were being queried anyway. Taken from this step's own log and
# cache: "peptide C", "peptide P", "peptide YL", "Peptide D1", "compound 1", "cyclic decapeptide",
# "cyclo", "CFW-1" (a mouse strain), "SRP-1"/"SRP-3"/"OLP-4". Each costs two round trips plus
# retries and returns nothing, every time it appears.
_UNSEARCHABLE_NAME = re.compile(
    r'^(?:peptides?|compounds?|fractions?|analogs?|analogues?|derivatives?|samples?|extracts?)'
    r'[\s\-]*[A-Za-z]{0,3}\d{0,3}$'
    r'|^(?:cyclo|linear|cyclic|native|synthetic|crude|purified|recombinant)\b'
    r'|^[A-Za-z]{1,2}\d{0,2}$',
    re.IGNORECASE
)


def is_searchable_name(name: str) -> bool:
    """
    Whether a peptide name is worth spending two HTTP round trips on.

    Reuses the extraction pipeline's own name validator, so a name this database would not accept
    as a peptide is not looked up either, and adds the placeholder shapes above that are valid
    peptide names in context ("peptide C" is how a paper labels its own compound) but meaningless
    to PubChem or ChEMBL.
    """
    if not name or len(name.strip()) < 3:
        return False
    core = name.strip()
    if _UNSEARCHABLE_NAME.match(core):
        return False
    try:
        from sample.nlp import is_valid_peptide_name
        return is_valid_peptide_name(core)
    except Exception:
        # Never let a missing resource turn into a silent skip of all enrichment.
        return True

# --- Persistent on-disk lookup cache (survives across runs, unlike functools.lru_cache) ---
_CACHE_CONN = None
_CACHE_LOCK = threading.Lock()


def _get_cache_conn() -> sqlite3.Connection:
    """
    :returns: The shared connection to the on-disk lookup cache, created on first use. The cache
        survives across runs, unlike an in-process memo, so a re-run costs no network traffic.
    """
    global _CACHE_CONN
    if _CACHE_CONN is None:
        with _CACHE_LOCK:
            if _CACHE_CONN is None:
                db_path = os.path.join(DATA_DIR, "external_db_cache.sqlite")
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lookup_cache ("
                    "source TEXT NOT NULL, query TEXT NOT NULL, ext_id TEXT, smiles TEXT, "
                    "PRIMARY KEY (source, query))"
                )
                conn.commit()
                _CACHE_CONN = conn
    return _CACHE_CONN


def _cache_get(source: str, query: str) -> Optional[tuple]:
    """
    :param source: Database name, e.g. ``"pubchem"``.
    :param query: The name that was looked up.
    :returns: ``(ext_id, smiles)`` if the lookup was cached -- including a cached miss, stored as
        ``(None, None)`` -- or ``None`` when the name has never been looked up.
    """
    conn = _get_cache_conn()
    with _CACHE_LOCK:
        row = conn.execute(
            "SELECT ext_id, smiles FROM lookup_cache WHERE source=? AND query=?", (source, query)
        ).fetchone()
    return row


def _cache_set(source: str, query: str, ext_id: Optional[str], smiles: Optional[str]) -> None:
    """
    Records a lookup result, including a miss.

    :param source: Database name.
    :param query: The name that was looked up.
    :param ext_id: Identifier found, or ``None`` for a miss.
    :param smiles: Structure found, or ``None``.
    :returns: ``None``.
    """
    conn = _get_cache_conn()
    with _CACHE_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO lookup_cache (source, query, ext_id, smiles) VALUES (?, ?, ?, ?)",
            (source, query, ext_id, smiles),
        )
        conn.commit()

# Amino acid frequencies derived from known peptide databases (UniProt)
PROTEIN_FREQS = {
    'A': 0.070, 'C': 0.011, 'D': 0.054, 'E': 0.062, 'F': 0.038,
    'G': 0.065, 'H': 0.022, 'I': 0.059, 'K': 0.058, 'L': 0.099,
    'M': 0.024, 'N': 0.041, 'P': 0.047, 'Q': 0.039, 'R': 0.057,
    'S': 0.083, 'T': 0.053, 'V': 0.068, 'W': 0.011, 'Y': 0.029
}

# Standard nucleotide frequencies (assuming equal distribution)
DNA_FREQS = {
    'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25, 'U': 0.25
}

def predict_sequence_type(sequence: str) -> str:
    """
    Classifies a string sequence as DNA or PROTEIN using log-likelihood.
    """
    seq = sequence.upper().strip()
    if not seq:
        raise ValueError("Empty sequence provided.")

    log_prob_dna = 0.0
    log_prob_protein = 0.0
    epsilon = 1e-9  # Smoothing factor for characters not in the expected alphabet

    for char in seq:
        p_dna = DNA_FREQS.get(char, epsilon)
        log_prob_dna += math.log(p_dna)

        p_protein = PROTEIN_FREQS.get(char, epsilon)
        log_prob_protein += math.log(p_protein)

    # Return the model with the higher log-likelihood
    return "DNA" if log_prob_dna > log_prob_protein else "PROTEIN"

def check_tanimoto(seq_1l: str, db_smiles: str, threshold: float = 0.8) -> bool:
    """
    Morgan-fingerprint similarity between a sequence and a database structure.

    Retained as a secondary accept path for :func:`check_peptide_identity`; on its own it rejects
    correct peptide matches, because a linear model built from a sequence scores poorly against
    the real cyclized or amidated molecule.

    :param seq_1l: One-letter sequence to build the reference molecule from.
    :param db_smiles: SMILES from the external database.
    :param threshold: Minimum Tanimoto similarity to accept.
    :returns: ``True`` if similarity meets `threshold`; ``False`` on any parse failure.
    """
    if not seq_1l or not db_smiles:
        return False
    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        
        ref_mol = Chem.MolFromSequence(seq_1l)
        if not ref_mol:
            return False
            
        db_mol = Chem.MolFromSmiles(db_smiles)
        if not db_mol:
            return False
            
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = mfpgen.GetFingerprint(ref_mol)
        fp2 = mfpgen.GetFingerprint(db_mol)
        
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return sim >= threshold
    except Exception:
        return False

# Fraction by which a database compound's exact mass may differ from the mass computed from the
# extracted sequence and still be accepted as the same peptide. It is not a similarity score: it
# is a budget for the chemistry a one-letter sequence cannot express. Measured against real
# matches this step had already found and thrown away -- indolicidin (ChEMBL415216) differs by
# 0.1%, the single dalton of C-terminal amidation, and nisin (ChEMBL526744) by 4.2%, the water lost
# forming its lanthionine rings and dehydro residues. Cyclization, disulfides and N-acetylation all
# fall inside the same budget, and an unrelated compound does not: the "compound 1" hit that name
# search returned is a ~300 Da small molecule against a 1,900 Da peptide.
_MASS_TOLERANCE = 0.05


def check_peptide_identity(seq_1l: str, db_smiles: str,
                           tolerance: float = _MASS_TOLERANCE) -> bool:
    """
    Whether a database compound is the peptide described by `seq_1l`, judged by exact mass.

    Replaces a Morgan-fingerprint Tanimoto gate at 0.8, which was rejecting correct matches and so
    made this entire step produce almost nothing: every name lookup that succeeded had its result
    discarded, leaving the network traffic to buy no output at all. The cause is structural rather
    than a bad threshold. Chem.MolFromSequence builds a linear peptide with free termini, while the
    database holds the real molecule -- amidated, cyclized, cross-linked -- and a circular-
    fingerprint comparison between the two scores low however similar they are as molecules:
    indolicidin scored under 0.8 and nisin under 0.4 against their own correct database entries.

    Mass is the right invariant here because the name match is what already establishes identity;
    this only has to catch a name COLLIDING with something unrelated, which mass does decisively.
    A strong fingerprint match is still honoured as an independent accept path, so nothing the old
    gate would have kept is lost.
    """
    if not seq_1l or not db_smiles:
        return False
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import ExactMolWt

        ref_mol = Chem.MolFromSequence(seq_1l)
        db_mol = Chem.MolFromSmiles(db_smiles)
        if ref_mol is None or db_mol is None:
            return False
        ref_mass = ExactMolWt(ref_mol)
        if ref_mass <= 0:
            return False
        if abs(ref_mass - ExactMolWt(db_mol)) / ref_mass <= tolerance:
            return True
    except Exception:
        return False
    # Outside the mass budget, an unambiguous structural match is still a match.
    return check_tanimoto(seq_1l, db_smiles)


logger = logging.getLogger("sample.external_db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_peptide_name(name: str) -> str:
    """
    Strips modifiers that make a name unresolvable to a chemical database.

    :param name: Peptide name as extracted, e.g. ``"magainin-2-NH2 analog"``.
    :returns: The name with terminal-modification and analog suffixes removed.
    """
    # Strip common modifiers that break APIs
    name = re.sub(r'(?i)-?NH2', '', name)
    name = re.sub(r'(?i)\(mutant\)', '', name)
    name = re.sub(r'(?i)\s+analog.*$', '', name)
    return name.strip()

@lru_cache(maxsize=4096)
def fetch_pubchem(peptide_name: str) -> tuple:
    """
    Looks a peptide name up in PubChem, honouring the cache, rate limit and breaker.

    :param peptide_name: Name to search for.
    :returns: ``(cid, smiles)``, both ``None`` when nothing matched or the service refused.
        A refusal is never cached, so a momentary quota breach does not become a permanent miss.
    """
    cached = _cache_get("pubchem", peptide_name)
    if cached is not None:
        return cached[0], cached[1]

    # Serialize identical concurrent lookups; the loser re-reads the cache the winner just wrote.
    with _name_lock(("pubchem", peptide_name)):
        cached = _cache_get("pubchem", peptide_name)
        if cached is not None:
            return cached[0], cached[1]
        if _PUBCHEM_BREAKER.is_open():
            return None, None

        clean_name = urllib.parse.quote(peptide_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{clean_name}/property/Title,IsomericSMILES,CanonicalSMILES/JSON"
        cid, smiles = None, None
        try:
            with _PUBCHEM_SEMAPHORE:
                _PUBCHEM_LIMITER.wait()
                response = _get_session().get(url, timeout=_PUBCHEM_TIMEOUT)
            blocked = _apply_pubchem_throttle(response)
            if blocked or response.status_code in (429, 503):
                # Being refused is not an answer about this compound, so it is neither cached nor
                # counted as a result -- caching it would make a momentary quota breach a permanent
                # "not found" for that name.
                _PUBCHEM_BREAKER.record_failure()
                return None, None
            # A 404 is PubChem's normal "no such compound" and is a real, cacheable answer;
            # only transport-level failures count against the breaker.
            if response.status_code == 200:
                data = response.json()
                if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                    props = data["PropertyTable"]["Properties"][0]
                    cid = props.get("CID")
                    smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
            _PUBCHEM_BREAKER.record_success()
        except Exception:
            _PUBCHEM_LIMITER.penalize()
            _PUBCHEM_BREAKER.record_failure()
            # Not cached: a timeout says nothing about whether the compound exists, and writing a
            # negative result here would make an outage permanent for that name.
            return None, None
        _cache_set("pubchem", peptide_name, cid, smiles)
        return cid, smiles

@lru_cache(maxsize=4096)
def fetch_chembl(peptide_name: str) -> tuple:
    """
    Looks a peptide name up in ChEMBL, honouring the cache, rate limit and breaker.

    :param peptide_name: Name to search for.
    :returns: ``(chembl_id, smiles)``, both ``None`` when nothing matched or the service refused.
    """
    cached = _cache_get("chembl", peptide_name)
    if cached is not None:
        return cached[0], cached[1]

    with _name_lock(("chembl", peptide_name)):
        cached = _cache_get("chembl", peptide_name)
        if cached is not None:
            return cached[0], cached[1]
        if _CHEMBL_BREAKER.is_open():
            return None, None

        clean_name = urllib.parse.quote(peptide_name)
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={clean_name}&format=json"
        chembl_id, smiles = None, None
        try:
            with _CHEMBL_SEMAPHORE:
                _CHEMBL_LIMITER.wait()
                response = _get_session().get(url, timeout=_CHEMBL_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if data.get("molecules"):
                    mol = data["molecules"][0]
                    chembl_id = mol.get("molecule_chembl_id")
                    structs = mol.get("molecule_structures")
                    smiles = structs.get("canonical_smiles") if structs else None
            elif response.status_code in (429, 503):
                _CHEMBL_LIMITER.penalize()
                _CHEMBL_BREAKER.record_failure()
                return None, None
            _CHEMBL_BREAKER.record_success()
            _CHEMBL_LIMITER.relax()
        except Exception:
            _CHEMBL_LIMITER.penalize()
            _CHEMBL_BREAKER.record_failure()
            return None, None
        _cache_set("chembl", peptide_name, chembl_id, smiles)
        return chembl_id, smiles

def fetch_chemspider(peptide_name: str) -> tuple:
    """
    ChemSpider lookup placeholder.

    ChemSpider requires an API key in the request headers; without one every call would be an
    unauthorized error, so this returns empty rather than generating failed traffic.

    :param peptide_name: Name that would be searched.
    :returns: Always ``(None, None)``.
    """
    # Public chemically aware search on ChemSpider requires an API key in headers.
    # Without an API key, we return None to avoid unauthorized errors.
    return None, None

def enrich_single_json(file_path: str) -> tuple:
    """
    Filters and enriches one extraction file. The unit of work for the enrichment pool.

    Applies the local filters first (length bound, RF word filter, DNA check), then looks up only
    the peptides whose result could actually be used -- a name-only peptide has no sequence to
    confirm a match against, so its lookup is skipped rather than discarded afterwards.

    :param file_path: Extraction JSON to process, rewritten in place only if it changed.
    :returns: ``(file_path, smiles_added, peptides_before, peptides_after)``.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        peptides = data.get("peptides", []) if isinstance(data, dict) else data
        if not isinstance(peptides, list): return file_path, 0, 0, 0
        
        updated_count = 0
        initial_count = len(peptides)
        retained_peptides = []
        
        for item in peptides:
            seq_1l = None
            
            # 1. Apply Size and ML Filter (RF Model) to sequence
            seq = item.get("peptide_sequence")
            if seq:
                seq_str = str(seq).upper().strip()
                
                # Length check: Drop if exact amino acid length > 50
                seq_1l = sequence_to_1_letter(seq_str)
                if len(seq_1l) > 50:
                    continue
                
                # RF filter for purely alphabetical strings
                if re.match(r'^[A-Z]+$', seq_str):
                    if not filter_sequence(seq_str):
                        continue # Drop the peptide entirely if it fails the RF filter
                
                # DNA log-likelihood filtering
                try:
                    if predict_sequence_type(seq_str) == "DNA":
                        continue # Drop if it is misidentified as a peptide instead of DNA
                except ValueError:
                    pass
            
            retained_peptides.append(item)
            
            # 2. Proceed with enrichment on retained peptides
            name = item.get("peptide_name") or ""
            name = str(name).strip()
            
            if not name or name.lower() in ["unnamed", "none", "null"]:
                continue
                
            has_smiles = bool(item.get("peptide_sequence_SMILES_format"))

            if not has_smiles:
                # Every lookup below is accepted only if check_tanimoto confirms the fetched
                # structure against this peptide's own sequence, and check_tanimoto returns False
                # whenever there is no sequence to compare against. So for a name-only peptide the
                # result is discarded no matter what comes back -- the two round trips cannot
                # change the output. A third of all peptide records are name-only, which made this
                # a third of the step's entire network traffic, spent to reach a foregone
                # conclusion. Skipping is behaviour-preserving: the same records are written.
                if not seq_1l:
                    continue
                # Placeholders and generic labels no chemical database can resolve.
                if not is_searchable_name(name):
                    continue

                search_name = clean_peptide_name(name)
                pool = _get_lookup_pool()
                f_pubchem = pool.submit(fetch_pubchem, search_name)
                f_chembl = pool.submit(fetch_chembl, search_name)
                cid, p_smiles = f_pubchem.result()
                chembl_id, c_smiles = f_chembl.result()
                cs_id, cs_smiles = fetch_chemspider(search_name)   # local no-op without an API key

                cross_refs = item.get("database_cross_reference", [])
                if not isinstance(cross_refs, list):
                    cross_refs = []
                    
                best_smiles = None
                
                # Each fetched database SMILES is confirmed against the text-extracted sequence by exact
                # mass before its cross-reference is recorded -- see check_peptide_identity.
                if cid and check_peptide_identity(seq_1l, p_smiles):
                    cross_refs.append(f"PubChem:{cid}")
                    best_smiles = best_smiles or p_smiles
                    
                if chembl_id and check_peptide_identity(seq_1l, c_smiles):
                    cross_refs.append(f"ChEMBL:{chembl_id}")
                    best_smiles = best_smiles or c_smiles
                    
                if cs_id and check_peptide_identity(seq_1l, cs_smiles):
                    cross_refs.append(f"ChemSpider:{cs_id}")
                    best_smiles = best_smiles or cs_smiles
                    
                if cross_refs:
                    item["database_cross_reference"] = list(set(cross_refs))
                    
                if best_smiles:
                    item["peptide_sequence_SMILES_format"] = best_smiles
                    updated_count += 1
                    
        # Replace the original list with the filtered list
        if isinstance(data, dict):
            data["peptides"] = retained_peptides
        else:
            data = retained_peptides

        # Only rewrite a file this step actually changed. Most documents gain nothing here -- no
        # peptide was dropped by the filters and no SMILES was found -- and rewriting all 46,189
        # regardless costs a full serialize plus write each, resets every mtime (so nothing
        # downstream can tell what this step touched), and risks corrupting untouched files if the
        # run is interrupted mid-write.
        if updated_count or len(retained_peptides) != initial_count:
            tmp_path = f"{file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, file_path)

        return file_path, updated_count, initial_count, len(retained_peptides)
    except Exception as e:
        logger.error(f"Error enriching {file_path}: {e}")
        return file_path, 0, 0, 0

def run_external_enrichment(sources: Any, workers: int = 10) -> None:
    """
    Step 12: enrich extraction records with PubChem and ChEMBL cross-references.

    :param sources: Sources to process -- ``"all"`` or a list naming ``pubmed``, ``scopus``,
        ``patents`` or a source database.
    :param workers: Concurrent file workers. Network pressure is bounded separately by the
        per-host semaphores and rate limiters, not by this number.
    :returns: ``None``; totals are logged.
    """
    logger.info("Starting external database enrichment (PubChem, ChEMBL, ChemSpider)...")
    
    # Collect all JSONs
    base_dirs = []
    db_papers_json_base = f"{DATA_DIR}/antimicrobial_databases/papers_json"
    
    if "all" in sources or "pubmed" in sources:
        base_dirs.append(f"{DATA_DIR}/literature_data_json/pubmed")
    if "all" in sources or "scopus" in sources:
        base_dirs.append(f"{DATA_DIR}/literature_data_json/scopus")
    if "all" in sources or "patents" in sources:
        base_dirs.append(f"{DATA_DIR}/patent_data_json/google_patents")
        
    if os.path.exists(db_papers_json_base):
        for sub in os.listdir(db_papers_json_base):
            if "all" in sources or "databases" in sources or "antimicrobial_databases" in sources or sub in sources:
                base_dirs.append(os.path.join(db_papers_json_base, sub))
        
    json_files = []
    for d in base_dirs:
        if os.path.exists(d):
            for file in os.listdir(d):
                if file.endswith(".json"):
                    json_files.append(os.path.join(d, file))
                    
    logger.info(f"Found {len(json_files)} JSON files to process.")
    
    total_updated = 0
    total_initial = 0
    total_retained = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(enrich_single_json, f): f for f in json_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Enriching JSONs"):
            f, updated, init_cnt, ret_cnt = future.result()
            total_updated += updated
            total_initial += init_cnt
            total_retained += ret_cnt
            
    dropped = total_initial - total_retained
    logger.info(f"ML Filtering complete: initially {total_initial} peptides -> retained {total_retained} peptides (dropped {dropped} false positives).")
    logger.info(f"External database enrichment complete! Updated SMILES for {total_updated} peptides.")
