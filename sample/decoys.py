import os
import random
import logging
import pandas as pd
from Bio import SeqIO
from sample.sequence import is_valid_peptide

logger = logging.getLogger("sample.decoys")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# pandas' default `read_csv` na_values -- every CSV round-trip in this pipeline (decoys.csv,
# final_dataset.csv, and beyond) reads through pandas with its default settings, so a decoy
# sequence that happens to spell one of these is silently read back as NaN, not the string it was
# written as. Caught live: a shuffled/fallback decoy sequence "NA" (a valid Asn-Ala dipeptide --
# `is_valid_peptide` and RDKit both accepted it) became `peptides.sequence IS NULL` on every load,
# which `NOT NULL constraint failed` then rejected outright, silently dropping that one decoy row.
# Checked case-insensitively since not every candidate string here is guaranteed uppercase.
_PANDAS_NA_SENTINELS = frozenset(s.upper() for s in (
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
))


def _csv_round_trip_safe(sequence: str) -> bool:
    """:returns: ``False`` when ``sequence`` would be misread as missing by a later ``pd.read_csv``."""
    return sequence.upper() not in _PANDAS_NA_SENTINELS

def shuffle_sequence(seq: str) -> str:
    """
    Shuffles a sequence to build a composition-matched decoy.

    Shuffling rather than sampling random residues keeps amino-acid composition, length and net
    charge identical to the positive it came from, so a classifier cannot separate the two on those
    features alone and has to learn something about residue ORDER.

    :param seq: Positive-class sequence to derive a decoy from.
    :returns: A random permutation of the input residues.
    """
    seq_list = list(seq)
    random.shuffle(seq_list)
    return "".join(seq_list)

#: Sequence identity above which a candidate decoy is dropped as too close to a REAL positive to
#: trust as a negative -- a "hidden positive" (Step 3 of a hardened negative-sampling protocol
#: this module was reviewed against; see :func:`_drop_near_positive_decoys`'s own docstring).
#: Deliberately much stricter than :data:`sample.ml_models.GROUPING_IDENTITY` (0.4): that threshold
#: exists to keep ANALOGS of one real peptide on the same side of a train/test split, a different
#: question ("could these be confused for each other during splitting?") from this one ("is this
#: candidate decoy actually a near-duplicate of something known to be active?"). Not imported from
#: sample.ml_models to avoid pulling that module's own import weight into decoy generation, which
#: runs earlier in the pipeline.
DECOY_POSITIVE_IDENTITY: float = 0.7

#: UniProt keyword/text exclusions for the "uniprot" decoy source. Broader than the original
#: keyword-only list: UniProt's own keyword hierarchy nests many antimicrobial-peptide subfamilies
#: (defensin, cathelicidin, bacteriocin, lantibiotic) under "Antimicrobial [KW-0044]" already, but
#: relying on that alone assumes annotation is complete and consistent, which curated databases
#: are not guaranteed to be -- free-text protein-name exclusions catch a name that never got the
#: keyword applied. KW-IDs are UniProt's stable identifiers for the top-level categories this
#: project already excluded; the additional keyword lines use names only (no KW-ID), which UniProt
#: also accepts, for categories this module could not verify a stable KW-ID for offline.
UNIPROT_EXCLUDED_KEYWORDS: tuple = (
    'keyword:"Antimicrobial [KW-0044]"', 'keyword:"Toxin [KW-0800]"',
    'keyword:"Antiviral [KW-1191]"', 'keyword:"Antifungal [KW-1195]"',
    'keyword:"Tumor suppressor [KW-0043]"', 'keyword:Antiparasitic',
    'keyword:Defensin', 'keyword:Bacteriocin',
)
UNIPROT_EXCLUDED_PROTEIN_NAMES: tuple = (
    "antimicrobial", "antibacterial", "bacteriocin", "defensin", "cathelicidin", "lantibiotic",
    "nisin", "venom", "toxin", "host defense", "host-defense",
)

#: Residues counted as cationic, anionic and hydrophobic for :func:`is_amp_like`. Single-letter
#: codes only; histidine is deliberately NOT counted as cationic (mostly neutral at pH 7.4).
_CATIONIC_RESIDUES = frozenset("KR")
_ANIONIC_RESIDUES = frozenset("DE")
_HYDROPHOBIC_RESIDUES = frozenset("AVILMFWYC")

#: The cationic-amphipathic bar a non-AMP fragment has to clear to be worth keeping as an
#: ADVERSARIAL negative ("uniprot_amp_like").
#:
#: A length- and composition-matched decoy rules out the crudest shortcuts, but it does not stop a
#: classifier separating the classes on the one property nearly every membrane-active AMP shares
#: and a random Swiss-Prot fragment usually does not: net positive charge paired with a substantial
#: hydrophobic fraction. A negative set without that property lets "is it cationic and amphipathic"
#: stand in for "is it active", which scores well and generalises badly -- the specific criticism
#: this source exists to answer. These fragments come from proteins carrying none of the
#: AMP/toxin annotations in :data:`UNIPROT_EXCLUDED_KEYWORDS` and are still screened against the
#: real positives by :func:`_drop_near_positive_decoys`, so they are non-AMP by annotation while
#: being AMP-like by physicochemistry: a hard negative rather than an easy one.
#:
#: +2 and 0.30 are the low end of the range quoted for natural cationic AMPs, chosen to admit
#: enough of Swiss-Prot to fill a quota rather than to describe a typical AMP -- a stricter bar
#: starves this source (most real proteins are not cationic), and the point is a harder negative,
#: not a fake positive.
AMP_LIKE_MIN_NET_CHARGE: int = 2
AMP_LIKE_MIN_HYDROPHOBIC_FRACTION: float = 0.30

#: How ``method="combined"`` divides its decoy budget between the three sources. Was a 50/50
#: shuffled/UniProt split with no adversarial component at all; the third source is the one that
#: makes the negative class something other than "strings a generator built" (see
#: :data:`AMP_LIKE_MIN_NET_CHARGE`). Weights, not counts -- normalised against whatever
#: ``decoy_ratio`` asks for.
#:
#: Changing this changes what every model trained afterwards is trained against, so a release whose
#: decoys predate it is NOT comparable to one generated after: regenerate the decoy set (step 28)
#: before reporting any metric alongside one computed under the old mix.
DECOY_METHOD_MIX: dict = {"shuffled": 1.0, "uniprot_fragment": 1.0, "uniprot_amp_like": 1.0}


def net_charge(seq: str) -> int:
    """
    :param seq: A peptide sequence, upper- or lower-case.
    :returns: Net charge at physiological pH, counted as (K + R) - (D + E). A whole-number
        approximation, not a Henderson-Hasselbalch calculation: this decides which fragments are
        worth keeping as hard negatives, a question that does not turn on a fractional charge.
    """
    upper = seq.upper()
    return (sum(1 for c in upper if c in _CATIONIC_RESIDUES)
            - sum(1 for c in upper if c in _ANIONIC_RESIDUES))


def hydrophobic_fraction(seq: str) -> float:
    """
    :param seq: A peptide sequence, upper- or lower-case.
    :returns: Fraction of residues in :data:`_HYDROPHOBIC_RESIDUES`; 0.0 for an empty sequence
        (rather than a ZeroDivisionError, since this is called on generated fragments).
    """
    if not seq:
        return 0.0
    upper = seq.upper()
    return sum(1 for c in upper if c in _HYDROPHOBIC_RESIDUES) / len(upper)


def is_amp_like(seq: str) -> bool:
    """
    :param seq: A candidate decoy sequence.
    :returns: Whether it is cationic AND hydrophobic enough to serve as an adversarial negative
        (:data:`AMP_LIKE_MIN_NET_CHARGE`, :data:`AMP_LIKE_MIN_HYDROPHOBIC_FRACTION`). Says nothing
        about whether the sequence is actually active -- it comes from a protein annotated as
        none of the AMP/toxin categories this module excludes; it says the easy physicochemical
        shortcut is not available for this one.
    """
    return (net_charge(seq) >= AMP_LIKE_MIN_NET_CHARGE
            and hydrophobic_fraction(seq) >= AMP_LIKE_MIN_HYDROPHOBIC_FRACTION)


def _drop_near_positive_decoys(positive_seqs: "set[str]", candidate_decoys: dict,
                               identity: float = DECOY_POSITIVE_IDENTITY,
                               work_dir: "str | None" = None) -> dict:
    """
    Drops any candidate decoy sitting within `identity` sequence identity of a REAL positive.

    A UniProt-sourced fragment survives the keyword exclusion in :func:`generate_decoys` by
    protein-level annotation alone -- a real antimicrobial region inside a protein UniProt never
    tagged as such, or a coincidental near-duplicate of a curated positive from an unrelated
    source, would still be accepted as a "negative" without this check. Uses cd-hit (already a
    pipeline dependency -- see :func:`sample.families.cluster_cdhit`) rather than a fresh
    pairwise-alignment pass: clustering POSITIVES and CANDIDATE DECOYS together in one pass and
    dropping any decoy that lands in a cluster containing a real positive is the same operation
    :func:`sample.families.cluster_cdhit` already does at scale, just read for membership rather
    than for a group id.

    A shuffled decoy is NOT expected to trigger this: a random permutation of a sequence's
    residues destroys the positional alignment cd-hit's identity search depends on, so it clusters
    with its own parent only by coincidence, not by construction -- this check is aimed at the
    UniProt-fragment half, where a real, unshuffled sequence is exactly the kind of thing that CAN
    coincidentally match a positive too closely.

    :param positive_seqs: The real peptide sequences being defended against.
    :param candidate_decoys: ``{sequence: generation_method}`` candidates to filter.
    :param identity: Required fraction of identical positions to count as "too close";
        :data:`DECOY_POSITIVE_IDENTITY` when omitted.
    :param work_dir: Scratch directory for cd-hit's index files; a fresh temp directory when
        omitted, so this does not race step 35's own clustering scratch space.
    :returns: `candidate_decoys` with any near-positive entry removed. Unchanged (with a warning,
        not silently) when cd-hit is not installed -- this check is a defense in depth, not the
        only guard against a hidden positive (the keyword/name exclusion already ran), so its
        absence should not block decoy generation outright.
    """
    from sample.families import cdhit_available, cluster_cdhit

    if not candidate_decoys:
        return candidate_decoys
    if not cdhit_available():
        logger.warning("cd-hit is not installed; skipping the near-positive decoy cross-check. "
                       "Candidate decoys are filtered by keyword/name exclusion alone.")
        return candidate_decoys

    import tempfile

    work_dir = work_dir or tempfile.mkdtemp(prefix="sample_decoy_crosscheck_")
    combined = {f"pos_{i}": s for i, s in enumerate(positive_seqs)}
    combined.update({f"dec_{i}": s for i, s in enumerate(candidate_decoys)})
    dec_key_to_seq = {f"dec_{i}": s for i, s in enumerate(candidate_decoys)}

    assignment = cluster_cdhit(combined, identity=identity, work_dir=work_dir)
    positive_clusters = {assignment[k] for k in combined if k.startswith("pos_")}
    dropped = {dec_key_to_seq[k] for k in dec_key_to_seq if assignment.get(k) in positive_clusters}

    if dropped:
        logger.warning(f"Dropping {len(dropped)} candidate decoy(s) within {identity:.0%} "
                       f"sequence identity of a real positive -- a hidden-positive risk, not a "
                       f"negative this dataset can trust.")
    return {s: m for s, m in candidate_decoys.items() if s not in dropped}


def generate_decoys(positives_csv: str, output_csv: str, decoy_ratio: float = 1.0, method: str = "combined") -> pd.DataFrame:
    """
    Generates negative samples (decoys) for machine learning by:
    - Shuffling positive sequences (paired-analysis) -- ``method="shuffling"``.
    - Random length-matched fragments of UniProt-SwissProt entries that are not AMPs/toxins/etc.
      -- ``method="uniprot"``.
    - ADVERSARIAL fragments from that same annotation-excluded pool, selected for AMP-like
      physicochemistry (cationic and hydrophobic, see :func:`is_amp_like`) so the easy
      charge/hydrophobicity shortcut is not available to a classifier -- ``method="amp_like"``.
    - Combined approach, the default: all three in the proportions of
      :data:`DECOY_METHOD_MIX`. NOTE this was formerly a 50/50 shuffled/UniProt split with no
      adversarial component, so a decoy set generated before that change is not comparable to one
      generated after it.

    Every candidate is checked against real positives for accidental near-duplication
    (:func:`_drop_near_positive_decoys`) before being accepted, and the METHOD that produced it
    (``shuffled`` / ``uniprot_fragment`` / ``fallback_random``) is recorded in the output's
    ``negative_type`` column -- auditable provenance a flat, undifferentiated decoy pool cannot
    offer: this project's own review of decoy quality (comparing shuffled-only, UniProt-only, and
    combined decoys, and measuring how well composition-based features separate the two decoy
    sources from each other) depended on being able to tell which decoy came from which method,
    which the pre-existing output could not do at all.
    """
    logger.info(f"Generating decoy datasets from positive baseline {positives_csv} using method '{method}'...")
    if not os.path.exists(positives_csv):
        raise FileNotFoundError(f"Baseline positive dataset not found at {positives_csv}")

    df_pos = pd.read_csv(positives_csv)
    positive_seqs = set(df_pos["sequence"].dropna().apply(lambda x: str(x).upper().strip()))

    decoy_count = int(len(positive_seqs) * decoy_ratio)
    logger.info(f"Targeting {decoy_count} decoy sequences...")

    # {sequence: generation_method} -- a dict, not a bare set, so every decoy's provenance
    # survives into the output file (see this function's own docstring).
    decoys: dict = {}

    # 1. Determine sizes
    shuffled_target = 0
    uniprot_target = 0
    amp_like_target = 0

    if method == "shuffling":
        shuffled_target = decoy_count
    elif method == "uniprot":
        uniprot_target = decoy_count
    elif method == "amp_like":
        amp_like_target = decoy_count
    else:  # combined -- the three-source mix, see DECOY_METHOD_MIX
        total_weight = sum(DECOY_METHOD_MIX.values()) or 1.0
        shuffled_target = int(decoy_count * DECOY_METHOD_MIX.get("shuffled", 0.0) / total_weight)
        uniprot_target = int(decoy_count * DECOY_METHOD_MIX.get("uniprot_fragment", 0.0)
                             / total_weight)
        # The adversarial source takes the remainder rather than its own separately rounded share,
        # so the three quotas always sum to exactly decoy_count instead of losing a row or two to
        # floor division.
        amp_like_target = decoy_count - shuffled_target - uniprot_target

    # 2. Shuffling (paired-analysis)
    if shuffled_target > 0:
        logger.info(f"Generating {shuffled_target} shuffled decoy sequences...")
        pos_list = list(positive_seqs)
        shuffled_attempts = 0
        n_before = len(decoys)
        while len(decoys) - n_before < shuffled_target and shuffled_attempts < shuffled_target * 5:
            shuffled_attempts += 1
            source_seq = random.choice(pos_list)
            shuf = shuffle_sequence(source_seq)
            if (shuf not in positive_seqs and shuf not in decoys and is_valid_peptide(shuf)
                    and _csv_round_trip_safe(shuf)):
                decoys[shuf] = "shuffled"
        logger.info(f"Generated {len(decoys) - n_before} shuffled decoys.")

    # 3. UniProt Swiss-Prot sampling (Length-matched chopping)
    if uniprot_target > 0 or amp_like_target > 0:
        logger.info(f"Fetching sequences from UniProt Swiss-Prot (non-AMP, non-toxins)...")
        import requests
        from urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

        # We fetch longer proteins and chop them to perfectly match the positive length distribution.
        # The keyword list alone assumes UniProt's annotation is both complete and consistent for
        # every excluded category; it is neither, so free-text protein-name exclusions ride along
        # as a second, independent check (see UNIPROT_EXCLUDED_PROTEIN_NAMES' own comment) -- a
        # protein whose keyword was never applied can still be caught by name.
        keyword_clause = " AND ".join(f"NOT {kw}" for kw in UNIPROT_EXCLUDED_KEYWORDS)
        name_clause = " AND ".join(f'NOT protein_name:"{name}"' for name in UNIPROT_EXCLUDED_PROTEIN_NAMES)
        q = f'reviewed:true AND {keyword_clause} AND {name_clause} AND length:[50 TO 500]'
        url = "https://rest.uniprot.org/uniprotkb/stream"

        uniprot_seqs = []
        try:
            r = requests.get(url, params={'query': q, 'format': 'fasta'}, verify=False, stream=True, timeout=30)
            if r.status_code == 200:
                current_seq = []
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith(">"):
                            if current_seq:
                                seq = "".join(current_seq).upper()
                                uniprot_seqs.append(seq)
                                current_seq = []
                        else:
                            current_seq.append(line_str)
                if current_seq:
                    seq = "".join(current_seq).upper()
                    uniprot_seqs.append(seq)
            else:
                logger.warning(f"UniProt stream returned HTTP {r.status_code}. Falling back to random generation.")
        except Exception as e:
            logger.warning(f"Failed to query UniProt API: {e}. Falling back to random generation.")

        if uniprot_seqs:
            added_plain = 0
            added_amp_like = 0
            lengths = [len(s) for s in positive_seqs]
            random.shuffle(lengths)
            # Both UniProt-derived quotas are filled from ONE pass of cut fragments, each fragment
            # routed to a bucket by its own physicochemistry (is_amp_like), rather than from two
            # separate cutting loops. A cationic, hydrophobic fragment is far rarer in
            # annotation-excluded Swiss-Prot than a plain one, so cutting for the adversarial
            # quota on its own would either starve it or re-cut the same few proteins repeatedly.
            # Attempts are bounded: the AMP-like quota is not guaranteed reachable from a given
            # protein sample, and falling short must log a shortfall rather than loop forever.
            max_attempts = max(len(lengths), (uniprot_target + amp_like_target) * 50)
            attempts = 0
            while ((added_plain < uniprot_target or added_amp_like < amp_like_target)
                   and attempts < max_attempts):
                target_len = lengths[attempts % len(lengths)]
                attempts += 1
                source = random.choice(uniprot_seqs)
                if len(source) <= target_len:
                    continue
                start = random.randint(0, len(source) - target_len)
                fragment = source[start:start + target_len]
                if (fragment in positive_seqs or fragment in decoys
                        or not is_valid_peptide(fragment)
                        or not _csv_round_trip_safe(fragment)):
                    continue
                if is_amp_like(fragment):
                    if added_amp_like < amp_like_target:
                        decoys[fragment] = "uniprot_amp_like"
                        added_amp_like += 1
                elif added_plain < uniprot_target:
                    decoys[fragment] = "uniprot_fragment"
                    added_plain += 1
            logger.info(f"Added {added_plain}/{uniprot_target} UniProt length-matched fragment "
                       f"decoys and {added_amp_like}/{amp_like_target} AMP-like adversarial "
                       f"fragment decoys ({attempts} fragment attempt(s)).")
            if added_amp_like < amp_like_target:
                logger.warning(
                    f"AMP-like adversarial decoy quota short by "
                    f"{amp_like_target - added_amp_like} of {amp_like_target}: cationic, "
                    f"hydrophobic fragments are rare in the annotation-excluded Swiss-Prot "
                    f"sample fetched. The negative set is still usable, but its adversarial "
                    f"share is smaller than requested -- report the realised composition "
                    f"(negative_type) rather than the requested mix.")

    # 4. Fallback if still under target count
    if len(decoys) < decoy_count:
        logger.info(f"Target count not met ({len(decoys)}/{decoy_count}). Generating fallback random amino acid sequences...")
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        lengths = [len(s) for s in positive_seqs]
        fallback_attempts = 0
        while len(decoys) < decoy_count and fallback_attempts < decoy_count * 5:
            fallback_attempts += 1
            target_len = random.choice(lengths)
            rand_seq = "".join(random.choices(amino_acids, k=target_len))
            if (rand_seq not in positive_seqs and rand_seq not in decoys and is_valid_peptide(rand_seq)
                    and _csv_round_trip_safe(rand_seq)):
                decoys[rand_seq] = "fallback_random"

    logger.info(f"Assembled {len(decoys)} candidate decoy sequence(s); cross-checking against "
               f"real positives for accidental near-duplication...")
    decoys = _drop_near_positive_decoys(positive_seqs, decoys)
    logger.info(f"Successfully assembled final decoy dataset with {len(decoys)} unique sequences "
               f"({sum(1 for m in decoys.values() if m == 'shuffled')} shuffled, "
               f"{sum(1 for m in decoys.values() if m == 'uniprot_fragment')} uniprot_fragment, "
               f"{sum(1 for m in decoys.values() if m == 'uniprot_amp_like')} uniprot_amp_like, "
               f"{sum(1 for m in decoys.values() if m == 'fallback_random')} fallback_random).")

    # Build decoy DataFrame
    decoy_rows = []
    ID_PREFIX = "SAMPLE_DECOY_"
    ID_ZFILL = 8

    from rdkit import Chem

    entry_count = 0
    unparseable_decoy_smiles = 0
    for seq, negative_type in decoys.items():
        entry_count += 1
        entry_id = f"{ID_PREFIX}{str(entry_count).zfill(ID_ZFILL)}"

        smiles = None
        try:
            mol = Chem.MolFromSequence(seq)
            if mol:
                smiles = Chem.MolToSmiles(mol)
        except Exception:
            unparseable_decoy_smiles += 1

        decoy_rows.append({
            "entry_id": entry_id,
            "sequence": seq,
            "SMILES": smiles,
            "is_decoy": 1,
            "source": "decoy_generator",
            "negative_type": negative_type,
            "pubmed_ids": None,
            "dois": None,
            "patent_ids": None,
            "ADP": None, "CAMP": None, "DBAMP": None, "DRAMP": None, "DBAASP": None
        })

    if unparseable_decoy_smiles:
        logger.info(f"{unparseable_decoy_smiles} of {entry_count} decoys got no SMILES (RDKit "
                   f"could not build a molecule from the sequence); sequence is still valid.")

    df_decoys = pd.DataFrame(decoy_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_decoys.to_csv(output_csv, index=False)

    # Copy to baseline sample_decoy_databases too
    compat_path = "data/sample_decoy_databases/decoys.csv"
    os.makedirs(os.path.dirname(compat_path), exist_ok=True)
    df_decoys.to_csv(compat_path, index=False)

    logger.info(f"Decoy generation complete. Saved decoy dataset to {output_csv} and {compat_path}.")
    return df_decoys
