"""
Third-party descriptor and activity predictors for the enrichment step.

Adds three families of computed property to each peptide:

* **Descriptors** -- ~100 published scales from ``peptides.py`` (Cruciani, BLOSUM, FASGAI, Kidera,
  MS-WHIM, ProtFP, ST-scales, T-scales, VHSE, Z-scales) plus the named physicochemical properties
  every AMP paper reports: hydrophobic moment, Boman index, aliphatic index, instability index.
* **Stability** -- N-end-rule half-life, instability index, and protease cleavage-site counts. These
  matter for whether a peptide is developable at all: a sequence with excellent MIC and a trypsin
  site every four residues is not a therapeutic candidate.
* **Predicted activities** -- AMP, hemolytic, anticancer and related predictions from external
  tools, run through adapters.

On the external predictors, deliberately: each is an OPTIONAL adapter that reports whether it ran.
Every one (macrel, AMPlify, AntiCP2, HemoPI2, AlgPred2) needs its own isolated conda environment --
each pins a scikit-learn/numpy pair, or a Python version, that conflicts with the others and with
the shared pipeline pin -- so requiring any of them would make the whole enrichment step fail on a
machine that merely lacks one. Every adapter is probed once, its availability recorded in the
output, and a missing tool leaves its columns absent rather than filled with a fabricated default.
A consumer can then tell "predicted non-hemolytic" from "hemolysis was never predicted", which a
0.0 in a column cannot express.
"""
import contextlib
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("sample.peptide_predictors")

#: Half-life of a protein by its N-terminal residue, in hours -- the N-end rule (Bachmair, Finley &
#: Varshavsky 1986), as tabulated by ExPASy ProtParam. Three systems because the same residue is
#: stabilizing in one and destabilizing in another, and a peptide intended for mammalian
#: administration is judged on the first column.
N_END_RULE_HALF_LIFE: Dict[str, Dict[str, float]] = {
    # residue: {mammalian (h), yeast (h), e_coli (h)}
    "A": {"mammalian": 4.4, "yeast": 20.0, "e_coli": 10.0},
    "R": {"mammalian": 1.0, "yeast": 0.033, "e_coli": 0.033},
    "N": {"mammalian": 1.4, "yeast": 0.05, "e_coli": 10.0},
    "D": {"mammalian": 1.1, "yeast": 0.05, "e_coli": 10.0},
    "C": {"mammalian": 1.2, "yeast": 20.0, "e_coli": 10.0},
    "Q": {"mammalian": 0.8, "yeast": 0.167, "e_coli": 10.0},
    "E": {"mammalian": 1.0, "yeast": 0.5, "e_coli": 10.0},
    "G": {"mammalian": 30.0, "yeast": 20.0, "e_coli": 10.0},
    "H": {"mammalian": 3.5, "yeast": 10.0, "e_coli": 10.0},
    "I": {"mammalian": 20.0, "yeast": 30.0, "e_coli": 10.0},
    "L": {"mammalian": 5.5, "yeast": 3.0, "e_coli": 0.033},
    "K": {"mammalian": 1.3, "yeast": 0.033, "e_coli": 0.033},
    "M": {"mammalian": 30.0, "yeast": 20.0, "e_coli": 10.0},
    "F": {"mammalian": 1.1, "yeast": 3.0, "e_coli": 0.033},
    "P": {"mammalian": 20.0, "yeast": 20.0, "e_coli": 10.0},
    "S": {"mammalian": 1.9, "yeast": 20.0, "e_coli": 10.0},
    "T": {"mammalian": 7.2, "yeast": 20.0, "e_coli": 10.0},
    "W": {"mammalian": 2.8, "yeast": 3.0, "e_coli": 0.033},
    "Y": {"mammalian": 2.8, "yeast": 10.0, "e_coli": 2.0},
    "V": {"mammalian": 100.0, "yeast": 20.0, "e_coli": 10.0},
}

#: Cleavage specificity of the proteases a peptide therapeutic actually meets -- in serum, in the
#: gut, and at the site of a bacterial infection. Each entry is (residues cleaved after, residues
#: that block cleavage when they follow). Counting sites is a coarse proxy for susceptibility, but
#: it is a real, checkable one: zero trypsin sites is a meaningful statement about a cationic
#: peptide, which are usually rich in exactly the K and R that trypsin cuts.
PROTEASE_RULES: Dict[str, Dict[str, str]] = {
    "trypsin": {"after": "KR", "not_before": "P"},
    "chymotrypsin": {"after": "FYW", "not_before": "P"},
    "pepsin": {"after": "FL", "not_before": ""},
    "elastase": {"after": "AVSG", "not_before": "P"},
    "thermolysin": {"after": "", "before": "ILVFAM"},
    "proteinase_k": {"after": "AFILVWY", "not_before": ""},
    # Staphylococcal V8, secreted by S. aureus -- directly relevant to whether an antistaphylococcal
    # peptide survives long enough at the infection site to act.
    "glutamyl_endopeptidase": {"after": "DE", "not_before": "P"},
}


def _clean_sequence(sequence: Any) -> str:
    """
    :param sequence: Sequence in any supported notation.
    :returns: Uppercase canonical one-letter sequence containing only standard residues, or ``""``.
    """
    from sample.sequence import sequence_to_1_letter

    seq = (sequence_to_1_letter(str(sequence or "").upper()) or "").strip().upper()
    return "".join(ch for ch in seq if ch in "ACDEFGHIKLMNPQRSTVWY")


def compute_peptides_descriptors(sequence: str,
                                 failure_counts: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """
    Descriptor scales and named physicochemical properties from ``peptides.py``.

    :param sequence: Canonical one-letter sequence.
    :param failure_counts: When given, incremented in place per named property that failed for
        this sequence (e.g. an aliphatic index needing a lookup a non-standard residue defeats).
        Callers processing many sequences pass a shared dict and log one summary afterward; omit
        it to leave the property simply absent from the result, as every caller before this one
        did.
    :returns: ``pep_*``-prefixed descriptors. Empty when the package is unavailable or the sequence
        cannot be parsed, so a caller can distinguish absent from zero.
    """
    try:
        import peptides
    except ImportError:
        return {}
    if not sequence:
        return {}
    try:
        pep = peptides.Peptide(sequence)
        out: Dict[str, float] = {f"pep_{k}": float(v) for k, v in pep.descriptors().items()}
        # The named properties AMP papers report directly, which are not in descriptors().
        for name, fn in (
            ("aliphatic_index", pep.aliphatic_index),
            ("boman", pep.boman),
            ("charge", pep.charge),
            ("hydrophobicity", pep.hydrophobicity),
            ("instability_index", pep.instability_index),
            ("isoelectric_point", pep.isoelectric_point),
            ("molecular_weight", pep.molecular_weight),
            ("hydrophobic_moment", pep.hydrophobic_moment),
        ):
            try:
                value = fn()
            except Exception:
                if failure_counts is not None:
                    failure_counts[name] = failure_counts.get(name, 0) + 1
                continue
            if value is not None:
                out[f"pep_{name}"] = float(value)
        # Amphipathicity as an alpha helix, the conformation most membrane-active AMPs adopt.
        try:
            moment = pep.hydrophobic_moment(angle=100, window=min(len(sequence), 11))
            if moment is not None:
                out["pep_hydrophobic_moment_helix"] = float(moment)
        except Exception:
            if failure_counts is not None:
                failure_counts["hydrophobic_moment_helix"] = (
                    failure_counts.get("hydrophobic_moment_helix", 0) + 1)
        return out
    except Exception as e:
        logger.debug(f"peptides.py failed on {sequence[:20]!r}: {e}")
        return {}


def compute_stability(sequence: str) -> Dict[str, float]:
    """
    Half-life, instability and protease-susceptibility features.

    :param sequence: Canonical one-letter sequence.
    :returns: ``stab_*``-prefixed features: N-end-rule half-lives, instability index, GRAVY,
        aromaticity, secondary-structure fractions, and per-protease cleavage-site counts with a
        length-normalized rate.
    """
    if not sequence:
        return {}
    out: Dict[str, float] = {}

    n_terminal = sequence[0]
    for system, hours in N_END_RULE_HALF_LIFE.get(n_terminal, {}).items():
        out[f"stab_half_life_{system}_h"] = float(hours)

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        analysis = ProteinAnalysis(sequence)
        out["stab_instability_index"] = float(analysis.instability_index())
        # Guruprasad's threshold: above 40 the protein is predicted unstable in vitro.
        out["stab_predicted_unstable"] = float(out["stab_instability_index"] > 40.0)
        out["stab_gravy"] = float(analysis.gravy())
        out["stab_aromaticity"] = float(analysis.aromaticity())
        out["stab_isoelectric_point"] = float(analysis.isoelectric_point())
        helix, turn, sheet = analysis.secondary_structure_fraction()
        out["stab_frac_helix"] = float(helix)
        out["stab_frac_turn"] = float(turn)
        out["stab_frac_sheet"] = float(sheet)
        try:
            out["stab_molar_extinction_reduced"] = float(analysis.molar_extinction_coefficient()[0])
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"ProtParam failed on {sequence[:20]!r}: {e}")

    total_sites = 0
    for protease, rule in PROTEASE_RULES.items():
        sites = 0
        for i, residue in enumerate(sequence[:-1]):
            following = sequence[i + 1]
            if rule.get("after") and residue in rule["after"]:
                if following not in (rule.get("not_before") or ""):
                    sites += 1
            elif rule.get("before") and following in rule["before"]:
                sites += 1
        out[f"stab_cleavage_sites_{protease}"] = float(sites)
        total_sites += sites
    out["stab_cleavage_sites_total"] = float(total_sites)
    out["stab_cleavage_density"] = float(total_sites / len(sequence)) if sequence else 0.0
    # A peptide with no site for any of these is a plausible protease-resistant candidate. Reported
    # as its own flag because it is the question a medicinal chemist asks first.
    out["stab_protease_resistant"] = float(total_sites == 0)
    return out


#: Conda environment holding the third-party predictors. They pin dependency versions that
#: conflict with the pipeline's own -- hemopi2 requires scikit-learn==1.3.1 while umap-learn and
#: the ML step need >=1.6 -- so installing them alongside downgrades scikit-learn under the whole
#: project. Isolating them is the only way both can be right: the predictors keep the exact
#: versions their pickled models were trained with, and the pipeline keeps its own.
#:
#: Create it once with::
#:
#:     conda create -y -n sample-predictors python=3.11
#:     conda run -n sample-predictors pip install hemopi2 toxinpred3 macrel
PREDICTOR_ENV: str = os.environ.get("SAMPLE_PREDICTOR_ENV", "sample-predictors")


def _predictor_command(executable: str, env: Optional[str] = None) -> Optional[List[str]]:
    """
    Builds the command used to invoke a predictor, preferring its isolated environment.

    :param executable: The tool's entry-point name.
    :param env: The conda environment the tool lives in. Each predictor pins its OWN environment
        (macrel, AMPlify and AntiCP2 each need a scikit-learn/numpy pin that would fight the others
        in a shared env), so this is per-tool rather than one global predictors environment. Falls
        back to :data:`PREDICTOR_ENV` when omitted.
    :returns: The argv prefix to run it with, or ``None`` when the tool cannot be found in either
        its environment or on ``PATH``.
    """
    # The executable must be shown to EXIST, not merely be plausible. Returning a `conda run`
    # command just because the environment exists reported every adapter as available -- including
    # tools that were never installed -- so the availability map claimed predictions that would
    # have failed at call time, which is worse than reporting the tool as missing.
    env = env or PREDICTOR_ENV
    for env_root in (os.environ.get("CONDA_ENVS_PATH"),
                     os.path.join(os.path.expanduser("~"), "miniconda3", "envs"),
                     os.path.join(os.path.expanduser("~"), "anaconda3", "envs")):
        if not env_root:
            continue
        candidate = os.path.join(env_root, env, "bin", executable)
        if os.path.exists(candidate):
            return [candidate]
    if shutil.which(executable):
        return [shutil.which(executable)]
    return None


def _log_subprocess_failure(tool_name: str, exc: Exception) -> None:
    """
    Logs a predictor subprocess failure with its actual captured output, not the generic
    ``str(CalledProcessError)`` exit-status message alone.

    That generic message ("Command '[...]' returned non-zero exit status 1") was ALL that ever
    reached the log for hemopi2 and, separately, AMPlify in production -- both call
    ``subprocess.run(..., capture_output=True)``, which already captures the real reason in
    ``e.stderr``/``e.stdout``; it just was never logged. A negative returncode means the process
    was killed by a signal (SIGKILL/-9 most often means the OOM killer), which explains a
    "silent" failure with no Python-level traceback at all.

    :param tool_name: The predictor's name, for the log line.
    :param exc: The exception caught around the ``subprocess.run`` call.
    """
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip() \
            if isinstance(exc.stderr, bytes) else str(exc.stderr or "").strip()
        stdout = (exc.stdout or b"").decode("utf-8", "replace").strip() \
            if isinstance(exc.stdout, bytes) else str(exc.stdout or "").strip()
        detail = stderr[-2000:] or stdout[-2000:] or "<no output captured>"
        if exc.returncode < 0:
            detail = f"killed by signal {-exc.returncode} (likely OOM or external kill) -- {detail}"
        logger.warning(f"{tool_name} failed (exit {exc.returncode}): {detail}")
    elif isinstance(exc, subprocess.TimeoutExpired):
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip() if exc.stderr else ""
        logger.warning(f"{tool_name} timed out after {exc.timeout}s: {stderr[-2000:]}")
    else:
        logger.warning(f"{tool_name} failed: {exc}")


class ExternalPredictor:
    """
    An optional third-party activity predictor, invoked as a command-line tool.

    Availability is probed once. A tool that is not installed contributes nothing and is recorded
    as unavailable, rather than the pipeline inventing a neutral value -- "not predicted" and
    "predicted negative" are different facts and the database must not conflate them.
    """

    def __init__(self, name: str, executable: str, description: str,
                 columns: Sequence[str], env: Optional[str] = None) -> None:
        """
        :param name: Short identifier, used as the column prefix.
        :param executable: Command that must be on ``PATH`` for the tool to be usable.
        :param description: What the tool predicts, for the availability report.
        :param columns: Column names this predictor contributes.
        :param env: The conda environment this tool lives in. Each predictor gets its own so their
            conflicting scikit-learn/numpy pins never meet; defaults to :data:`PREDICTOR_ENV`.
        """
        self.name = name
        self.executable = executable
        self.description = description
        self.columns = list(columns)
        self.env = env or PREDICTOR_ENV
        self._available: Optional[bool] = None
        self._command: Optional[List[str]] = None

    def available(self) -> bool:
        """
        :returns: Whether the tool can be run, probed once and cached.
        """
        if self._available is None:
            self._command = _predictor_command(self.executable, self.env)
            self._available = self._command is not None
            if not self._available:
                logger.info(
                    f"{self.name} not installed ({self.executable} not on PATH) -- "
                    f"skipping {self.description}.")
        return self._available

    def predict(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        Runs the tool over a batch of sequences.

        :param sequences: Canonical one-letter sequences.
        :returns: One result dict per sequence; empty dicts when the tool is unavailable or failed.
        """
        if not self.available():
            return [{} for _ in sequences]
        return self._run(sequences)

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        Tool-specific invocation. The base implementation is a no-op.

        :param sequences: Canonical one-letter sequences.
        :returns: One result dict per sequence.
        """
        return [{} for _ in sequences]



#: Sequences macrel cannot be given, because they crash it rather than being skipped.
#:
#: macrel's own ``normalize_seq`` strips a leading ``M`` and then reads ``seq[-1]``::
#:
#:     if seq[0] == 'M': seq = seq[1:]     # "M" becomes ""
#:     if seq[-1] == '*':                  # IndexError on ""
#:
#: so a sequence that is exactly ``"M"`` raises ``IndexError: string index out of range`` and the
#: whole run exits non-zero. Its caller guards for an empty sequence both before and after that
#: call, so the intent is clearly to skip these -- the guard simply cannot fire from inside the
#: function that crashes first. Found on a real run: one junk single-residue row
#: (``SAMPLE0000015473``, sequence ``M``) cost all 23,368 macrel predictions in step 22.
#:
#: Filtering here rather than waiting for an upstream fix, because the cost asymmetry is extreme:
#: one unusable row should never take a whole batch with it.
def _macrel_safe(sequence: str) -> bool:
    """
    :param sequence: A candidate sequence.
    :returns: Whether macrel can be given this sequence without crashing. Anything it merely
        *skips* (non-standard residues, ordinary short peptides) still returns ``True`` -- macrel
        reports those cleanly and the batch survives.
    """
    seq = str(sequence or "").strip()
    # Mirrors normalize_seq EXACTLY rather than being conservative: it strips at most ONE leading
    # M, so "MM" survives as "M" and is fine, and a trailing "*" is removed only after the index
    # that crashes. Over-filtering here would silently drop real predictions, which is the failure
    # this guard exists to prevent -- so the only inputs excluded are the ones that genuinely crash:
    # the empty string, and exactly "M".
    return bool(seq) and seq != "M"


class MacrelPredictor(ExternalPredictor):
    """
    Macrel -- AMP classification and hemolytic-activity prediction from sequence.

    Chosen as the installable stand-in for the Raghava-lab web servers (HemoPI, AntiCP), which have
    no distributable package. Macrel is pip-installable, publishes its training data, and predicts
    both the AMP call and hemolysis, which is the pair this database most needs.
    """

    def __init__(self) -> None:
        super().__init__(
            name="macrel", executable="macrel",
            description="AMP and hemolysis prediction",
            columns=["macrel_amp_probability", "macrel_is_amp", "macrel_hemolytic_probability"],
            # Its own environment: macrel pins a scikit-learn/numpy pair that would fight hemopi2's
            # in the shared predictors env. See environments/sample-macrel.yml.
            env="sample-macrel",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the AMP and hemolysis calls; empty dicts on failure.
        """
        import csv
        import os
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results
        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.faa")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i, seq in enumerate(sequences):
                    if _macrel_safe(seq):
                        fh.write(f">s{i}\n{seq}\n")
            out_dir = os.path.join(tmp, "out")
            try:
                subprocess.run(
                    # --keep-negatives: without it, macrel drops every peptide it does not call an
                    # AMP from the output entirely, rather than writing a negative row -- verified
                    # against a real run (a D/E-only negative control was simply absent from the
                    # file, no "is_AMP" column at all, whereas a real AMP candidate appeared with
                    # is_AMP=True). This adapter needs a call for every submitted sequence,
                    # including the ones macrel calls negative.
                    #
                    # self._command (not a bare "macrel"): available()'s per-environment resolution
                    # is the whole reason each predictor gets its own isolated environment, and a
                    # hardcoded bare command name bypasses it entirely -- verified as a real bug on
                    # this host, where "macrel" is not on PATH but self._command's resolved absolute
                    # path is.
                    self._command + ["peptides", "--fasta", fasta, "--output", out_dir, "-t", "4",
                                     "--keep-negatives"],
                    check=True, capture_output=True, timeout=1800,
                )
            except Exception as e:
                _log_subprocess_failure("macrel", e)
                return results
            for root, _dirs, files in os.walk(out_dir):
                for fname in files:
                    # The real output file is "<tag>.prediction.gz" -- not ".tsv.gz" -- verified
                    # against a real run; the ".tsv"/".tsv.gz" filter this used to have never
                    # matched it, so this adapter silently returned empty results for every call
                    # regardless of whether macrel itself succeeded.
                    if "prediction" not in fname or not fname.endswith((".gz", ".tsv")):
                        continue
                    try:
                        import gzip
                        opener = gzip.open if fname.endswith(".gz") else open
                        with opener(os.path.join(root, fname), "rt", encoding="utf-8") as fh:
                            rows = [r for r in fh if not r.startswith("#")]
                        reader = csv.DictReader(rows, delimiter="\t")
                        for row in reader:
                            access = str(row.get("Access") or row.get("access") or "")
                            if not access.startswith("s"):
                                continue
                            idx = int(access[1:])
                            if 0 <= idx < len(results):
                                results[idx] = {
                                    "macrel_amp_probability": float(row.get("AMP_probability") or 0.0),
                                    "macrel_is_amp": float(str(row.get("is_AMP", "")).lower() == "true"),
                                    "macrel_hemolytic_probability": float(
                                        row.get("Hemolytic_probability") or 0.0),
                                }
                    except Exception as e:
                        logger.debug(f"Could not parse macrel output {fname}: {e}")
        return results


#: AMPlify's own accepted-input footprint (AMPlify.py's `valid_ix` filter, read directly from its
#: source): 2-200 residues, every character one of these 20 standard amino acids, or standard
#: except for a trailing '*' stop-codon marker.
_AMPLIFY_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _is_amplify_valid(seq: str) -> bool:
    """
    Whether AMPlify's own script would keep `seq` in its valid-sequence list.

    A sequence outside AMPlify's footprint (too short/long, or a non-standard residue code such as
    X/B/Z from a shuffled decoy) is not rejected gracefully by AMPlify.py itself -- when it is the
    ONLY sequence in a call (this project now scores one sequence per call; see
    :func:`sample.benchmark.reference_scores`), AMPlify's valid-sequence list ends up empty and its
    own `ensemble()` crashes with an opaque tensor-shape error deep inside Keras, rather than
    reporting "nothing to score" the way every other unscorable case does. Checked here so that
    specific, known failure mode never reaches the subprocess at all -- confirmed against a real
    12,712-peptide benchmark holdout: exactly 7 sequences fail this check (one single-residue real
    peptide, six decoys carrying an ambiguity code), and each one previously cost a full AMPlify
    subprocess spin-up (TensorFlow/Keras import, five model loads) only to crash.

    :param seq: A canonical one-letter sequence.
    :returns: Whether AMPlify's own filter would accept it.
    """
    if not (2 <= len(seq) <= 200):
        return False
    body = seq[:-1] if seq.endswith("*") else seq
    return set(body) <= _AMPLIFY_STANDARD_AA


class AMPlifyPredictor(ExternalPredictor):
    """
    AMPlify -- deep-learning AMP prediction (bcgsc/AMPlify), installed from bioconda.

    Invoked via the ``sample-amplify`` environment's OWN python running ``AMPlify.py`` directly,
    not the ``AMPlify`` console-script bioconda installs: that script execs ``AMPlify.py``, whose
    own shebang (``#!/usr/bin/env python3``) resolves through the CALLER's ``PATH`` rather than
    pinning to its own environment. Run under any other active environment (or with a different
    environment earlier on ``PATH``), it silently picks up the wrong python and fails with
    ``ModuleNotFoundError: No module named 'keras'`` -- confirmed on this host. Naming the
    interpreter and script explicitly sidesteps that regardless of the caller's active environment.
    ``available()`` still probes for the ``AMPlify`` executable (unchanged from the base class), so
    the per-environment probing this project tests stays intact; only the invocation in ``_run``
    differs.
    """

    def __init__(self) -> None:
        super().__init__(
            name="amplify", executable="AMPlify", description="AMP prediction",
            columns=["amplify_score", "amplify_is_amp"], env="sample-amplify",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the AMP probability score and call; empty on failure.
        """
        import csv
        import glob
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results

        # A sequence AMPlify's own script would reject (see _is_amplify_valid) is left out of the
        # FASTA entirely rather than submitted and left to fail: AMPlify crashes rather than
        # gracefully skipping when its OWN valid-sequence list ends up empty, which is exactly
        # what happens when this is the only sequence in the call.
        valid_positions = [i for i, seq in enumerate(sequences) if seq and _is_amplify_valid(seq)]
        if not valid_positions:
            return results

        env_bin = os.path.dirname(self._command[0])
        python = os.path.join(env_bin, "python")
        script = os.path.join(os.path.dirname(env_bin), "share", "amplify", "src", "AMPlify.py")
        if not (os.path.exists(python) and os.path.exists(script)):
            logger.warning(f"amplify: could not locate AMPlify.py alongside its environment "
                           f"(expected {script}) -- skipping {self.description}.")
            return results

        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.fasta")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i in valid_positions:
                    fh.write(f">s{i}\n{sequences[i]}\n")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir, exist_ok=True)
            try:
                subprocess.run(
                    [python, script, "-s", fasta, "-od", out_dir, "-of", "tsv"],
                    check=True, capture_output=True, timeout=3600,
                )
            except Exception as e:
                _log_subprocess_failure("amplify", e)
                return results
            tsv_files = glob.glob(os.path.join(out_dir, "*.tsv"))
            if not tsv_files:
                return results
            try:
                with open(tsv_files[0], newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh, delimiter="\t"):
                        seq_id = str(row.get("Sequence_ID") or "").strip()
                        if not seq_id.startswith("s"):
                            continue
                        idx = int(seq_id[1:])
                        score = row.get("Probability_score")
                        call = str(row.get("Prediction") or "").strip().upper()
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "amplify_score": float(score) if score not in (None, "") else float("nan"),
                                "amplify_is_amp": float(call == "AMP"),
                            }
            except Exception as e:
                logger.debug(f"Could not parse amplify output: {e}")
        return results


#: The character set AMPidentifier's own feature extractor (modlamp's ``GlobalDescriptor``)
#: recognizes -- read directly from ``modlamp.core.aa_weights()``, the table its molecular-weight
#: step indexes into.
#:
#: A residue outside these 20 is not skipped gracefully: ``calculate_MW`` does a bare
#: ``weights[aa]`` lookup with no ``.get()``/try-except around it, so one non-standard residue (X,
#: B, Z, U, a trailing ``*``, ...) raises ``KeyError`` and kills feature extraction for the WHOLE
#: submitted batch -- confirmed against a real 3-sequence run: one sequence with a trailing 'X' lost
#: every prediction, including the two clean sequences alongside it (exit code 1, no output file at
#: all). Filtered here for the same reason ``_is_amplify_valid`` filters AMPlify's own footprint:
#: one bad sequence should cost nothing beyond itself.
_AMPIDENTIFIER_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _is_ampidentifier_valid(seq: str) -> bool:
    """
    Whether AMPidentifier's own feature extractor would accept `seq` without crashing.

    :param seq: A canonical one-letter sequence.
    :returns: Whether every residue is one of the 20 standard amino acids. No length restriction is
        enforced -- verified directly against a real run that a single-residue sequence ("A") scores
        normally, unlike AMPlify's length-2 floor.
    """
    return bool(seq) and set(seq) <= _AMPIDENTIFIER_STANDARD_AA


class AMPidentifierPredictor(ExternalPredictor):
    """
    AMPidentifier -- a soft-voting ensemble (Random Forest, SVM-RBF, Gradient Boosting, XGBoost,
    LightGBM) over 22 modlamp-derived physicochemical descriptors (madsondeluna/AMPidentifier),
    installed from PyPI.

    Benchmark-only, exactly like :class:`AMPlifyPredictor` beside it: never added to
    :data:`EXTERNAL_PREDICTORS`, only reachable through
    :func:`sample.benchmark._benchmark_only_predictors`. See that function's docstring for why.

    Invoked via its OWN console-script directly (``self._command``, unlike
    :class:`AMPlifyPredictor`, which must derive its environment's python and run ``AMPlify.py``
    explicitly). AMPlify needs that indirection because bioconda's wrapper script has a
    ``#!/usr/bin/env python3`` shebang that resolves through the CALLER's ``PATH``. AMPidentifier's
    ``pip``-installed ``ampidentifier`` entry-point script carries an ABSOLUTE shebang baked in at
    install time (``#!<env-prefix>/bin/python3.11``) -- confirmed by reading the installed script
    directly on this host -- so the environment-probed command already resolves to the right
    interpreter regardless of the caller's active environment, and no indirection is needed.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ampidentifier", executable="ampidentifier", description="AMP prediction",
            columns=["ampidentifier_score", "ampidentifier_is_amp"], env="sample-ampidentifier",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the voting-ensemble's AMP probability and 0/1 call
            (its own default and recommended mode -- see the class docstring); empty dicts on
            failure, or for a sequence AMPidentifier's own feature extractor would reject (see
            :func:`_is_ampidentifier_valid`).
        """
        import csv
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results

        # A sequence outside AMPidentifier's own accepted alphabet is left out of the FASTA
        # entirely rather than submitted and left to fail: one such sequence crashes feature
        # extraction for the WHOLE batch (see _is_ampidentifier_valid), not just itself.
        valid_positions = [i for i, seq in enumerate(sequences)
                           if seq and _is_ampidentifier_valid(seq)]
        if not valid_positions:
            return results

        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.fasta")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i in valid_positions:
                    fh.write(f">s{i}\n{sequences[i]}\n")
            out_dir = os.path.join(tmp, "out")
            try:
                subprocess.run(
                    # No -m/--model flag: "voting" (the soft-voting ensemble of all five models) is
                    # both the tool's own default and its recommended mode.
                    self._command + ["-i", fasta, "-o", out_dir],
                    check=True, capture_output=True, timeout=3600,
                )
            except Exception as e:
                _log_subprocess_failure("ampidentifier", e)
                return results
            # Fixed filename: AMPidentifier writes "predictions_{model}.csv", and the model here is
            # always "voting" (no -m flag passed above).
            pred_path = os.path.join(out_dir, "predictions_voting.csv")
            if not os.path.exists(pred_path):
                return results
            try:
                with open(pred_path, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        seq_id = str(row.get("ID") or "").strip()
                        if not seq_id.startswith("s"):
                            continue
                        idx = int(seq_id[1:])
                        score = row.get("probability_AMP")
                        prediction = row.get("prediction")
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "ampidentifier_score": (
                                    float(score) if score not in (None, "") else float("nan")),
                                "ampidentifier_is_amp": (
                                    float(prediction) if prediction not in (None, "")
                                    else float("nan")),
                            }
            except Exception as e:
                logger.debug(f"Could not parse ampidentifier output: {e}")
        return results


#: amPEPpy's own repository (tlawrence3/amPEPpy), cloned by ``make env-ampeppy`` purely to reach
#: ``pretrained_models/amPEP.model`` -- the tool's published, manuscript-accuracy Random Forest
#: (160 trees over the Distribution descriptor set). ``pip install`` (from the git URL
#: environments/sample-ampeppy.yml uses, or a local clone) installs only the ``amPEPpy`` python
#: package: its ``setup.py`` declares no ``package_data``/``MANIFEST.in``, so the model file is
#: never part of that install -- confirmed by reading setup.py directly. The ``ampep`` command
#: itself still comes from the ``sample-ampeppy`` conda environment; this path is reached only for
#: the ``-m`` model argument.
AMPEPPY_DIR: str = os.environ.get("SAMPLE_AMPEPPY_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ampeppy")
AMPEPPY_MODEL_PATH: str = os.path.join(AMPEPPY_DIR, "pretrained_models", "amPEP.model")


class AmPEPpyPredictor(ExternalPredictor):
    """
    amPEPpy -- a Random Forest classifier over the Distribution descriptor set from the Global
    Protein Sequence Descriptors (tlawrence3/amPEPpy, Bioinformatics 2021), installed from its own
    git repository (not on PyPI or bioconda).

    Benchmark-only, exactly like :class:`AMPlifyPredictor` and :class:`AMPidentifierPredictor`
    beside it: never added to :data:`EXTERNAL_PREDICTORS`, only reachable through
    :func:`sample.benchmark._benchmark_only_predictors`. See that function's docstring for why.

    Uses the tool's OWN bundled pretrained model (:data:`AMPEPPY_MODEL_PATH`) rather than training a
    fresh one -- exactly the model the paper's accuracy figures refer to. That file does not travel
    with ``pip install`` (see :data:`AMPEPPY_DIR`'s comment), so ``make env-ampeppy`` clones
    amPEPpy's repo into ``data/ampeppy/`` to reach it; a missing file here means that step has not
    been run yet, not that the tool is broken.

    Invoked via its OWN console-script directly (``self._command``), like
    :class:`AMPidentifierPredictor` and unlike :class:`AMPlifyPredictor`: amPEPpy's pip-installed
    ``ampep`` entry-point script carries an absolute shebang baked in at install time
    (``#!<env-prefix>/bin/python3.11``) -- confirmed by reading the installed script directly on
    this host -- so no python/script indirection is needed.

    Verified directly against a live run (2026-09-04) through the real ``ampep predict`` CLI and the
    bundled model: magainin-2 (``GIGKFLHSAKKFGKAFVGEIMNS``) scores ``probability_AMP=0.959375``,
    called ``AMP``. Unlike macrel/AMPlify/AMPidentifier, amPEPpy never crashed on ANY degenerate
    input tried directly against its CLI -- a single residue, a non-standard residue (``X``), a
    trailing ``*`` stop marker, or even a fully empty sequence all returned an ordinary scored row;
    its Distribution-descriptor featurizer degrades to an all-zero feature row rather than raising
    on any of them. Empty sequences are still excluded below (see ``valid_positions``) because
    scoring "nothing" is not a prediction worth keeping, not because amPEPpy needs the guard.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ampeppy", executable="ampep",
            description="AMP prediction (Random Forest / Global Protein Sequence Descriptors)",
            columns=["ampeppy_amp_probability", "ampeppy_is_amp"], env="sample-ampeppy",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the Random Forest's AMP probability and call; empty
            dicts on failure, when the sequence was empty, or when the bundled pretrained model
            (see :data:`AMPEPPY_MODEL_PATH`) has not been fetched by ``make env-ampeppy`` yet.
        """
        import csv
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results

        valid_positions = [i for i, seq in enumerate(sequences) if seq]
        if not valid_positions:
            return results

        if not os.path.exists(AMPEPPY_MODEL_PATH):
            logger.warning(
                f"ampeppy: pretrained model not found at {AMPEPPY_MODEL_PATH} -- run "
                f"`make env-ampeppy` to fetch it -- skipping {self.description}.")
            return results

        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.fasta")
            out_tsv = os.path.join(tmp, "out.tsv")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i in valid_positions:
                    fh.write(f">s{i}\n{sequences[i]}\n")
            try:
                subprocess.run(
                    self._command + ["predict", "-m", AMPEPPY_MODEL_PATH, "-i", fasta,
                                     "-o", out_tsv, "-t", "4"],
                    # cwd=tmp: amPEPpy's own feature-scoring step unconditionally writes a
                    # "<today>_scored_features.csv" side-effect file into the CURRENT working
                    # directory -- verified against a real run, not documented anywhere in its
                    # README, and not controlled by -o. Without cwd=tmp that file lands in whatever
                    # directory this process happens to be running from (this project's own root,
                    # in production), growing without bound across repeated benchmark runs.
                    check=True, capture_output=True, timeout=1800, cwd=tmp,
                )
            except Exception as e:
                _log_subprocess_failure("ampeppy", e)
                return results
            if not os.path.exists(out_tsv):
                return results
            try:
                with open(out_tsv, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh, delimiter="\t"):
                        seq_id = str(row.get("seq_id") or "").strip()
                        if not seq_id.startswith("s"):
                            continue
                        idx = int(seq_id[1:])
                        score = row.get("probability_AMP")
                        call = str(row.get("predicted") or "").strip()
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "ampeppy_amp_probability": (
                                    float(score) if score not in (None, "") else float("nan")),
                                "ampeppy_is_amp": float(call == "AMP"),
                            }
            except Exception as e:
                logger.debug(f"Could not parse ampeppy output: {e}")
        return results


class AntiCP2Predictor(ExternalPredictor):
    """
    AntiCP 2.0 -- anticancer-peptide prediction (raghavagps/anticp2), installed from PyPI.

    The package's own metadata declares ``Requires-Python >=3.3,<=3.7`` -- a PEP 440 specifier
    *without* a wildcard, so it is a strict version comparison and even ``python=3.7.12`` fails it
    (``3.7.12 <= 3.7`` is False). ``environments/sample-anticp2.yml`` pins ``python=3.6``, which
    satisfies the bound unambiguously; see that file for the fuller account.
    """

    def __init__(self) -> None:
        super().__init__(
            name="anticp", executable="anticp2", description="anticancer peptide prediction",
            columns=["anticp_score", "anticp_is_anticancer"], env="sample-anticp2",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the anticancer score and call; empty dicts on failure.
        """
        import csv
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results
        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.fasta")
            out_csv = os.path.join(tmp, "out.csv")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i, seq in enumerate(sequences):
                    if seq:
                        fh.write(f">s{i}\n{seq}\n")
            try:
                subprocess.run(
                    self._command + ["-i", fasta, "-o", out_csv, "-d", "2"],
                    check=True, capture_output=True, timeout=3600,
                )
            except Exception as e:
                _log_subprocess_failure("anticp2", e)
                return results
            if not os.path.exists(out_csv):
                return results
            try:
                with open(out_csv, newline="", encoding="utf-8") as fh:
                    for row in csv.reader(fh):
                        if not row or row[0].strip().startswith("#"):
                            continue
                        seq_id = row[0].strip().lstrip(">")
                        if not seq_id.startswith("s") or len(row) < 4:
                            continue
                        idx = int(seq_id[1:])
                        call, score = row[2].strip(), row[3].strip()
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "anticp_score": float(score) if score else float("nan"),
                                "anticp_is_anticancer": float(call == "AntiCP"),
                            }
            except Exception as e:
                logger.debug(f"Could not parse anticp2 output: {e}")
        return results


def _hemopi2_safe(sequence: str) -> bool:
    """
    :param sequence: A candidate sequence.
    :returns: Whether hemopi2 can be given this sequence without crashing.

    hemopi2's own feature pipeline computes CTC (Conjoint Triad, a 3-mer count over a 7-category
    reduced alphabet) for every sequence, and its own ``CTC()`` explicitly checks for a sequence
    too short to contain a single triad (``Max_MM == 0`` in
    ``hemopi2/Model/composition_calculate.py``), prints an error, and returns WITHOUT writing its
    output file. That file's absence then crashes the unrelated concatenation step immediately
    after it (an uncaught ``pd.read_csv`` under ``os.system``, so the failure is swallowed there
    too), which is why the final `hemopi2_classification` call dies with a ``FileNotFoundError``
    on a completely different, downstream file (confirmed: `'EL'`, a real corpus entry, row 2078,
    2026-08-26). Any sequence shorter than 3 residues hits this deterministically -- there is no
    partial/degraded result to salvage, so it is excluded before ever being sent, the same way
    `_macrel_safe` excludes macrel's one genuinely-crashing input.
    """
    return len(str(sequence or "").strip()) >= 3


class HemoPI2Predictor(ExternalPredictor):
    """
    HemoPI 2.0 -- hemolytic-activity prediction (Raghava group).

    Hemolysis is the property that decides whether an otherwise potent AMP is usable, and this
    database extracts far more MIC values than hemolysis values, so a computed prediction fills a
    real gap. Uses the ESM+MERCI hybrid (model 4), the paper's best-performing configuration.

    Runs from the isolated predictor environment: HemoPI 2.0 pins ``scikit-learn==1.3.1`` to match
    the version its models were pickled with, which conflicts with the pipeline's own requirement.
    """

    def __init__(self) -> None:
        super().__init__(
            name="hemopi2", executable="hemopi2_classification",
            description="hemolytic activity prediction",
            columns=["hemopi2_score", "hemopi2_is_hemolytic"],
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the hemolysis score and call; empty dicts on failure.
        """
        import csv
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results
        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.faa")
            out_csv = os.path.join(tmp, "out.csv")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i, seq in enumerate(sequences):
                    if _hemopi2_safe(seq):
                        fh.write(f">s{i}\n{seq}\n")
            # Two quirks of HemoPI2's path handling, both worked around here rather than patched
            # upstream:
            #
            # 1. It writes intermediates into a "tmp" subdirectory of its working directory and
            #    never creates it.
            # 2. It re-joins the working directory onto itself, so an absolute --working of
            #    /x/y produces the path /x/y/tmp/x/y and the run dies with "Cannot save file into
            #    a non-existent directory".
            #
            # Passing "." and setting cwd instead makes that doubling harmless -- it resolves to
            # ./tmp/. -- which is why the tool works when run by hand from its own directory and
            # failed only from the adapter. Paths are therefore relative to `tmp` throughout.
            os.makedirs(os.path.join(tmp, "tmp"), exist_ok=True)
            try:
                subprocess.run(
                    # Model 1 (Random Forest) rather than the ESM hybrid: it is the configuration
                    # that runs without downloading the ESM weights at call time, which matters for
                    # a batch step that must not stall on a network fetch mid-run.
                    self._command + ["-i", os.path.basename(fasta),
                                     "-o", os.path.basename(out_csv), "-wd", ".", "-m", "1"],
                    check=True, capture_output=True, timeout=3600, cwd=tmp,
                )
            except Exception as e:
                _log_subprocess_failure("hemopi2", e)
                return results
            if not os.path.exists(out_csv):
                return results
            try:
                with open(out_csv, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        seq_id = str(row.get("SeqID") or row.get("ID") or "").strip().lstrip(">")
                        if not seq_id.startswith("s"):
                            continue
                        idx = int(seq_id[1:])
                        score = row.get("ML Score") or row.get("Score")
                        call = str(row.get("Prediction") or "").strip().lower()
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "hemopi2_score": float(score) if score not in (None, "") else float("nan"),
                                # "Non-Hemolytic" contains "hemolytic", so the negative
                                # prefix has to be tested first or every row reads as positive.
                                "hemopi2_is_hemolytic": float(call == "hemolytic"),
                            }
            except Exception as e:
                logger.debug(f"Could not parse hemopi2 output: {e}")
        return results


class AlgPred2Predictor(ExternalPredictor):
    """
    AlgPred 2.0 -- allergenicity prediction (raghavagps/algpred2), installed from PyPI.

    Its shipped RandomForest pickle (``algpred2/model/rf_model``) was trained with scikit-learn
    0.19.2 -- old enough that any current scikit-learn fails the same way toxinpred3's pickle does
    ("node array from the pickle has an incompatible dtype": the pickle predates the
    ``missing_go_to_left`` Tree field 1.3 introduced). Unlike toxinpred3, this one IS fixable:
    ``scikit-learn==1.2.2`` loads it, but only once ``numpy<2`` is pinned alongside it (1.2.2's
    compiled extensions were built against numpy 1.x's C ABI). See
    ``environments/sample-algpred2.yml`` for the fuller account and the version search that found
    this pin.
    """

    def __init__(self) -> None:
        super().__init__(
            name="algpred2", executable="algpred2", description="allergenicity prediction",
            columns=["algpred2_score", "algpred2_is_allergen"], env="sample-algpred2",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the allergenicity score and call; empty dicts on
            failure.
        """
        import csv
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results
        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.fasta")
            out_csv = os.path.join(tmp, "out.csv")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i, seq in enumerate(sequences):
                    if seq:
                        fh.write(f">s{i}\n{seq}\n")
            try:
                subprocess.run(
                    self._command + ["-i", fasta, "-o", out_csv, "-d", "2"],
                    check=True, capture_output=True, timeout=3600,
                )
            except Exception as e:
                _log_subprocess_failure("algpred2", e)
                return results
            if not os.path.exists(out_csv):
                return results
            try:
                with open(out_csv, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        seq_id = (row.get("ID") or "").strip().lstrip(">")
                        if not seq_id.startswith("s"):
                            continue
                        idx = int(seq_id[1:])
                        score = (row.get("ML_Score") or "").strip()
                        prediction = (row.get("Prediction") or "").strip()
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "algpred2_score": float(score) if score else float("nan"),
                                "algpred2_is_allergen": float(prediction == "Allergen"),
                            }
            except Exception as e:
                logger.debug(f"Could not parse algpred2 output: {e}")
        return results


class ToxinPred2Predictor(ExternalPredictor):
    """
    ToxinPred 2.0 -- general peptide/protein toxicity prediction (raghavagps/toxinpred2), installed
    from PyPI. Benchmark-only (see ``sample.benchmark._benchmark_only_predictors``), not part of
    :data:`EXTERNAL_PREDICTORS`: it is NOT a hemolysis-specific predictor -- it scores general
    toxicity -- so it is wired in as the closest adjacent comparator for the hemolytic-activity
    benchmark, not as a replacement for a hemolysis-specific tool the way HemoPI2 is used live. See
    ``sample.benchmark.REFERENCE_TOOLS``' "hemolytic" entry for that scope caveat spelled out for
    readers of the comparison.

    Chosen after this project's own ToxinPred 3.0 integration attempt turned out to be a confirmed,
    unfixable dead end (its shipped scikit-learn pickle is incompatible with every scikit-learn
    version tried -- see ``environments/sample-toxinpred3.yml`` and ``KNOWN_ISSUES.md``). ToxinPred2
    does not have that problem: its shipped model is an ONNX export
    (``onnxruntime.InferenceSession`` over ``RF_model.onnx``), so there is no scikit-learn pickle
    version to fight, unlike hemopi2/algpred2/toxinpred3.

    Uses Model 1 (AAC-composition-based Random Forest, the tool's own default), NOT Model 2 (the
    hybrid RF+BLAST+MERCI ensemble the paper reports as best-performing): Model 2 has a real,
    reproducible upstream bug, confirmed 2026-09-04 against the real installed tool. Its own
    ``BLAST_processor`` (``toxinpred2/python_scripts/toxinpred2.py``) references an undefined name
    ``seqid`` instead of its own parameter ``name1`` in the branch taken whenever BLAST finds ZERO
    hits across the whole submitted batch, crashing the entire run with ``NameError: name 'seqid'
    is not defined`` instead of reporting "no BLAST support" gracefully for those sequences. That
    branch is not a rare corner case -- it is the common case for peptides with no close match in
    ToxinPred2's own small bundled BLAST database, exactly what this project submits. Model 1 needs
    no BLAST/MERCI/perl step at all and was verified clean end to end on real peptides.
    """

    def __init__(self) -> None:
        super().__init__(
            name="toxinpred2", executable="toxinpred2", description="general toxicity prediction",
            columns=["toxinpred2_score", "toxinpred2_is_toxic"], env="sample-toxinpred2",
        )

    def _run(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """
        :param sequences: Canonical one-letter sequences.
        :returns: One dict per sequence with the toxicity score and call; empty dicts on failure.
        """
        import csv
        import tempfile

        results: List[Dict[str, float]] = [{} for _ in sequences]
        if not self._command:
            return results

        valid_positions = [i for i, seq in enumerate(sequences) if seq]
        if not valid_positions:
            return results

        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "in.fasta")
            out_csv = os.path.join(tmp, "out.csv")
            with open(fasta, "w", encoding="utf-8") as fh:
                for i in valid_positions:
                    fh.write(f">s{i}\n{sequences[i]}\n")
                # ToxinPred2's own feature step (aac_comp's output read back with np.loadtxt)
                # writes exactly one row per submitted sequence, and np.loadtxt collapses a file
                # with only ONE row to a 1-D array rather than 2-D -- onnxruntime then rejects it
                # ("Invalid rank for input: Got 1 Expected 2") and the whole call crashes.
                # Confirmed 2026-09-04: a batch of exactly one sequence fails this way regardless
                # of the sequence itself (a real 23-residue AMP and a bare 2-residue string both
                # crashed identically; 2+ sequences never did). A second, throwaway sequence forces
                # the 2-D shape the tool's own code assumes; its row is dropped below by ID, never
                # returned to the caller.
                if len(valid_positions) == 1:
                    fh.write(">__toxinpred2_pad__\nAAAA\n")
            try:
                subprocess.run(
                    # -m 1: AAC-based RF, not the hybrid (-m 2) -- see the class docstring for the
                    # confirmed upstream crash in the hybrid's BLAST-processing path.
                    # -d 2: report every submitted sequence, not only the ones called "Toxin" --
                    # -d 1, the tool's own default, silently drops every negative call from the
                    # output entirely, the same trap anticp2/algpred2 need -d 2 for.
                    self._command + ["-i", fasta, "-o", out_csv, "-m", "1", "-d", "2"],
                    check=True, capture_output=True, timeout=3600, cwd=tmp,
                )
            except Exception as e:
                _log_subprocess_failure("toxinpred2", e)
                return results
            if not os.path.exists(out_csv):
                return results
            try:
                with open(out_csv, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        seq_id = (row.get("ID") or "").strip().lstrip(">")
                        if not seq_id.startswith("s"):
                            continue
                        idx = int(seq_id[1:])
                        score = (row.get("ML_Score") or "").strip()
                        prediction = (row.get("Prediction") or "").strip()
                        if 0 <= idx < len(results):
                            results[idx] = {
                                "toxinpred2_score": float(score) if score else float("nan"),
                                "toxinpred2_is_toxic": float(prediction == "Toxin"),
                            }
            except Exception as e:
                logger.debug(f"Could not parse toxinpred2 output: {e}")
        return results


#: How many sequences go to an external tool in one call.
#:
#: These tools were being handed the WHOLE corpus in a single subprocess call, which does not
#: scale: AMPlify timed out after its full 3600s on a 23,368-peptide run (2027-8-18),
#: losing every AMPlify prediction for the entire corpus. Batching bounds each call's runtime and,
#: more importantly, its blast radius -- a tool that fails or times out now costs one batch rather
#: than the whole dataset, and the rest of the corpus still gets scored.
#:
#: A size-1 default (one subprocess call per peptide) was tried after hemopi2 failed on every
#: batch of a 53,614-peptide run at size 2000 with no informative error: it pins a failure to the
#: exact peptide that caused it, but it turned out to cost far more than intended. Most of these
#: tools' per-call time is fixed model-load overhead, not per-sequence work -- AMPlify alone ran at
#: ~31s/peptide (real run, 2027-8-21/24), which is that fixed cost paid 53,614 times
#: instead of amortised, and would have taken this one tool over two weeks to finish the corpus.
#:
#: The fix is `enrich_with_predictors`' chunk-with-fallback below, not a smaller default: run
#: batches of 100 (amortising the fixed overhead over 100 peptides), and ONLY when a whole batch
#: comes back empty, retry that batch one peptide at a time to isolate exactly which sequence
#: broke it. A tool that is behaving normally never pays the per-peptide cost at all; one that
#: hits a bad peptide loses at most the model-load overhead of a size-100 retry sequence, not the
#: whole corpus.
PREDICTOR_BATCH_SIZE: int = int(os.environ.get("SAMPLE_PREDICTOR_BATCH_SIZE", "100"))

#: How many batches of a single predictor run concurrently. Default 1 (fully sequential, identical
#: to the original behaviour) because this is a real resource tradeoff a caller should opt into
#: deliberately, not a change that silently alters memory/CPU usage under everyone.
#:
#: Each call is a fresh `subprocess.run` (`_predict_chunk_with_fallback`), which blocks on the
#: child process and releases the GIL, so a `ThreadPoolExecutor` gets genuine OS-level parallelism
#: -- that part is sound. What is NOT safe on its own: measured on a real 24-core host
#: (2026-08-25), a SINGLE AMPlify (TensorFlow) subprocess claimed only ~1.5 cores when nothing
#: else was contending for them, but running just TWO concurrent AMPlify calls pushed load average
#: past 50 -- each process's own math library (OpenMP/BLAS/TensorFlow's intra-op pool) scales its
#: thread count up to use whatever cores it can see, unaware that N-1 siblings are about to do the
#: same thing. Naive concurrency here does not parallelize the work; it oversubscribes the machine
#: and can end up SLOWER than sequential. `_run_predictor` therefore caps every common
#: threading-library env var (`OMP_NUM_THREADS` and friends, see `_capped_thread_env`) to
#: `cpu_count // PREDICTOR_WORKERS` for the duration of a concurrent run, so N processes together
#: stay within the machine's real core budget. Also mind memory: workers * per-instance RSS
#: (~500MB-1GB for AMPlify) must fit in what is actually available, not just "free".
PREDICTOR_WORKERS: int = int(os.environ.get("SAMPLE_PREDICTOR_WORKERS", "1"))


#: The external predictors attempted, in order. Add an adapter here to extend the enrichment; the
#: pipeline picks it up without further wiring and reports whether it was available.
#:
#: AMPlify (`AMPlifyPredictor`, still defined below) is deliberately NOT in this list, and has no
#: CLI command or Makefile step -- fully retired from the pipeline. Across multiple real
#: production runs of the full 53,614-peptide corpus (2026-08-24/26) it crashed on a sustained
#: 7-12%+ of batches with a genuine upstream bug -- `ValueError: expected Input to have 3
#: dimensions, but got array with shape (0, 1)` inside its own `ensemble()`/Keras `predict()` call
#: -- not a resource, batching, or environment issue on this project's side. The class is kept,
#: undeleted, as a working reference for anyone who wants to pick the bug apart later or re-enable
#: it once fixed: add it back here, restore its `enrich-amplify` CLI command and Makefile step
#: (see git history around 2026-08-26 for exactly what to restore), done.
EXTERNAL_PREDICTORS: List[ExternalPredictor] = [
    MacrelPredictor(),
    AntiCP2Predictor(),
    HemoPI2Predictor(),
    AlgPred2Predictor(),
]


def predictor_availability() -> Dict[str, bool]:
    """
    :returns: Predictor name -> whether it is installed. Recorded alongside the enriched data so a
        consumer knows which predictions were even attempted for a given release.
    """
    return {p.name: p.available() for p in EXTERNAL_PREDICTORS}


def _predict_chunk_with_fallback(predictor: "ExternalPredictor",
                                 chunk: List[str]) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Runs one predictor call over a chunk of sequences, falling back to one-at-a-time only when the
    whole chunk comes back empty.

    Batching amortises a tool's fixed model-load overhead over many sequences -- the entire reason
    it is faster than one call per peptide. That overhead is paid exactly once here in the common
    case (the chunk succeeds). Only on a chunk that fails entirely does this pay it again, per
    peptide, to isolate exactly which sequence(s) actually broke it -- a single bad peptide then
    costs one retry sequence, not the whole chunk and not the whole corpus.

    :param predictor: The predictor to run.
    :param chunk: Sequences to predict (a batch of ``PREDICTOR_BATCH_SIZE``, or fewer for the last
        one).
    :returns: ``(predictions, failed_indices)`` -- one prediction dict per input sequence (empty
        for any with no result), and the indices within ``chunk`` that ended up empty.
    """
    predictions = predictor.predict(chunk)
    if any(predictions) or len(chunk) <= 1:
        return predictions, [i for i, p in enumerate(predictions) if not p]

    retried: List[Dict[str, float]] = []
    failed_indices: List[int] = []
    for i, seq in enumerate(chunk):
        single = predictor.predict([seq])
        prediction = single[0] if single else {}
        if not prediction:
            failed_indices.append(i)
        retried.append(prediction)
    return retried, failed_indices


@contextlib.contextmanager
def _capped_thread_env(workers: int):
    """
    Caps common math/threading-library env vars so N concurrent predictor subprocesses do not each
    try to claim every CPU core.

    The real incident this exists for: two concurrent AMPlify (TensorFlow) subprocess calls, on a
    24-core host where a single call alone used ~1.5 cores, pushed load average past 50 (measured
    2026-08-25). Each process's own math library scales its own thread pool up to the visible core
    count, with no awareness that siblings are about to do the same -- naive concurrency does not
    parallelize the work, it oversubscribes the machine. Every subprocess a predictor's ``_run``
    spawns inherits ``os.environ`` (none of them pass an explicit ``env=``), so setting these here,
    once, before dispatch is enough to reach all of them regardless of which predictor class it is.

    :param workers: The concurrency level about to be used. A no-op (yields immediately) at 1,
        so the sequential default path is completely unaffected by this existing.
    """
    if workers <= 1:
        yield
        return
    threads = str(max(1, (os.cpu_count() or workers) // workers))
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = threads
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_predictor(predictor: "ExternalPredictor", sequences: List[str],
                   rows: List[Dict[str, float]]) -> None:
    """
    Runs one predictor over `sequences` in ``PREDICTOR_BATCH_SIZE`` chunks (with the per-peptide
    fallback on a failed chunk), updating `rows` **in place**, one dict per sequence.

    Shared by :func:`enrich_with_predictors` (all tools, over the whole corpus) and
    :func:`enrich_one_predictor` (a single tool, independently rerunnable) so the batching,
    fallback and logging behavior can never quietly drift apart between the two entry points.

    :param predictor: The (already ``available()``-checked) predictor to run.
    :param sequences: Sequences to predict, aligned index-for-index with ``rows``.
    :param rows: Output accumulator, one dict per sequence; updated in place.
    """
    workers = max(1, PREDICTOR_WORKERS)
    logger.info(f"Running {predictor.name} ({predictor.description}) over "
                f"{len(sequences)} sequences in batches of {PREDICTOR_BATCH_SIZE}"
                f"{f', {workers} concurrent' if workers > 1 else ''}...")
    chunks_with_failures = 0
    failed_peptides = 0
    logged_failures = 0
    chunk_starts = list(range(0, len(sequences), PREDICTOR_BATCH_SIZE))

    def handle_result(start: int, chunk: List[str],
                      predictions: List[Dict[str, float]], failed_in_chunk: List[int],
                      bar: "tqdm") -> None:
        nonlocal chunks_with_failures, failed_peptides, logged_failures
        if failed_in_chunk:
            chunks_with_failures += 1
            failed_peptides += len(failed_in_chunk)
            # The one-at-a-time fallback (chunk > 1) or the direct call (chunk == 1) both pin
            # the exact peptide a tool failed on, rather than only a count.
            for i in failed_in_chunk:
                if logged_failures >= 20:
                    break
                seq_preview = chunk[i][:40] + ("..." if len(chunk[i]) > 40 else "")
                # logger.warning, not bar.write: this pipeline runs headless
                # (setsid ... > out.log 2> err.log), where every diagnostic needs to land
                # through the same logger everything else uses, not a side channel that a
                # log-only run would never capture.
                logger.warning(f"{predictor.name}: no prediction for '{seq_preview}' "
                               f"(row {start + i})")
                logged_failures += 1
            bar.set_postfix(failed=failed_peptides, refresh=False)
        for record, prediction in zip(rows[start:start + len(chunk)], predictions):
            record.update(prediction)
        bar.update(len(chunk))

    # One bar per tool, not one for the whole predictor stack: each tool has its own runtime per
    # peptide and its own failure rate, so a single shared bar would hide exactly which tool is
    # slow or failing -- the thing this bar exists to show.
    with tqdm(total=len(sequences), desc=predictor.name, unit="pep") as bar:
        if workers == 1:
            # No executor at all in the default case: same call stack as before this existed,
            # so nothing about the common path changes by this feature merely existing.
            for start in chunk_starts:
                chunk = sequences[start:start + PREDICTOR_BATCH_SIZE]
                predictions, failed_in_chunk = _predict_chunk_with_fallback(predictor, chunk)
                handle_result(start, chunk, predictions, failed_in_chunk, bar)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Each chunk call is `subprocess.run` underneath (a fresh external process) -- it
            # blocks on the child and releases the GIL, so a thread pool gets genuine multi-core
            # concurrency across the real bottleneck (the tool's own CPU-bound work), not
            # something fighting the GIL. Keyed by start offset so results land in the right rows
            # regardless of which chunk's subprocess happens to finish first.
            #
            # _capped_thread_env wraps the whole dispatch: without it, each concurrent subprocess's
            # own math library (BLAS/OpenMP/TensorFlow) tries to claim every visible core, and N
            # processes doing that together oversubscribes the machine instead of parallelizing it
            # (measured real incident in that function's docstring).
            with _capped_thread_env(workers), ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_predict_chunk_with_fallback, predictor,
                                    sequences[start:start + PREDICTOR_BATCH_SIZE]): start
                    for start in chunk_starts
                }
                for future in as_completed(futures):
                    start = futures[future]
                    chunk = sequences[start:start + PREDICTOR_BATCH_SIZE]
                    predictions, failed_in_chunk = future.result()
                    handle_result(start, chunk, predictions, failed_in_chunk, bar)
    if failed_peptides:
        # Named explicitly: a partly-populated column is a different fact from a missing one,
        # and silently shipping it would misrepresent which peptides were scored.
        logger.warning(
            f"{predictor.name}: {failed_peptides} of {len(sequences)} peptides got no "
            f"prediction ({chunks_with_failures} of "
            f"{-(-len(sequences) // PREDICTOR_BATCH_SIZE)} chunk(s) needed a per-peptide "
            f"fallback).")


def enrich_one_predictor(df: pd.DataFrame, predictor_name: str, sequence_column: str = "sequence",
                         only_missing: bool = True) -> pd.DataFrame:
    """
    Runs exactly ONE external predictor over a peptide table that already carries every other
    column (descriptors, the other predictors, embeddings, ...) -- so re-running one tool after a
    partial failure costs only that tool's time, not the whole enrichment pipeline's.

    This is what the real 2026-08-24 production incident needed: AMPlify failed on ~7% of a
    53,614-peptide corpus partway through a run that also computes descriptors and four other
    predictors, and the only way to fix just AMPlify's gap was to redo everything from scratch.
    Each predictor now has its own CLI command and Makefile target built on this function.

    :param df: A peptide table with at least a sequence column. Every other existing column
        (including other predictors' results) is carried through completely unchanged.
    :param predictor_name: One of :data:`EXTERNAL_PREDICTORS`' names (e.g. ``"amplify"``).
    :param sequence_column: Name of the sequence column.
    :param only_missing: Skip rows that already carry a non-null value in this predictor's first
        column -- the common case: filling in gaps left by a prior partial failure. ``False``
        recomputes every row (e.g. after a model update).
    :returns: A copy of `df` with that predictor's columns added or updated. Rows outside the
        recomputed subset keep whatever they already had for this predictor's columns.
    :raises ValueError: When `predictor_name` does not match any known predictor.
    """
    predictor = next((p for p in EXTERNAL_PREDICTORS if p.name == predictor_name), None)
    if predictor is None:
        known = ", ".join(p.name for p in EXTERNAL_PREDICTORS)
        raise ValueError(f"Unknown predictor '{predictor_name}'. Known predictors: {known}.")

    out = df.copy()
    if sequence_column not in out.columns:
        logger.error(f"No '{sequence_column}' column; nothing to enrich.")
        return out
    for col in predictor.columns:
        if col not in out.columns:
            out[col] = float("nan")
    if not predictor.available():
        logger.warning(f"{predictor.name} is not available in this environment; leaving its "
                       f"columns as they were.")
        return out

    sentinel_col = predictor.columns[0]
    todo_index = out.index[out[sentinel_col].isna()] if only_missing else out.index
    if len(todo_index) == 0:
        logger.info(f"{predictor.name}: every peptide already has a result; nothing to do "
                    f"(pass only_missing=False to force a full recompute).")
        return out
    if only_missing and len(todo_index) < len(out):
        logger.info(f"{predictor.name}: {len(todo_index)} of {len(out)} peptides are missing a "
                    f"result; recomputing only those.")

    sequences = [_clean_sequence(s) for s in out.loc[todo_index, sequence_column]]
    rows: List[Dict[str, float]] = [{} for _ in todo_index]
    _run_predictor(predictor, sequences, rows)

    for idx, record in zip(todo_index, rows):
        for key, value in record.items():
            out.at[idx, key] = value
    return out


def compute_descriptors_and_stability(df: pd.DataFrame,
                                      sequence_column: str = "sequence") -> pd.DataFrame:
    """
    Adds ``peptides.py`` descriptor scales, N-end-rule half-life, instability index and protease
    cleavage-site counts.

    Split out from :func:`enrich_with_predictors` as its own step: this half is sequence-only,
    dependency-free (no external tool, no subprocess, no failure mode beyond an unparseable
    sequence), unlike the five external predictors below it -- bundling them into one pipeline step
    meant a slow or failing predictor blocked re-running this fast, reliable half too.

    :param df: Table with a sequence column.
    :param sequence_column: Name of that column.
    :returns: A copy of `df` with the new columns appended. Rows whose sequence cannot be parsed
        keep the original columns and get no new ones rather than zeros.
    """
    if sequence_column not in df.columns:
        logger.error(f"No '{sequence_column}' column; nothing to enrich.")
        return df

    sequences = [_clean_sequence(s) for s in df[sequence_column]]
    logger.info(f"Computing descriptors and stability features for {len(sequences)} peptides...")

    rows: List[Dict[str, float]] = []
    descriptor_failures: Dict[str, int] = {}
    for seq in sequences:
        record: Dict[str, float] = {}
        record.update(compute_peptides_descriptors(seq, descriptor_failures))
        record.update(compute_stability(seq))
        rows.append(record)
    if descriptor_failures:
        logger.info(f"peptides.py named-property computation: {descriptor_failures} (property -> "
                   f"failure count) across {len(sequences)} sequences; that property is simply "
                   f"absent from the row, one failure does not block the others.")

    # Built in one concat rather than column-by-column assignment: with ~120 new columns the
    # incremental form fragments the frame and is markedly slower.
    additions = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, additions], axis=1)


def enrich_with_predictors(df: pd.DataFrame, sequence_column: str = "sequence",
                           run_external: bool = True) -> pd.DataFrame:
    """
    Adds descriptor, stability and predicted-activity columns to a peptide table.

    :param df: Table with a sequence column.
    :param sequence_column: Name of that column.
    :param run_external: Whether to invoke the external command-line predictors. Set ``False`` for
        a fast, dependency-free run using only ``peptides.py`` and BioPython.
    :returns: A copy of `df` with the new columns appended. Rows whose sequence cannot be parsed
        keep the original columns and get no new ones rather than zeros.
    """
    if sequence_column not in df.columns:
        logger.error(f"No '{sequence_column}' column; nothing to enrich.")
        return df

    out = compute_descriptors_and_stability(df, sequence_column)

    if run_external:
        sequences = [_clean_sequence(s) for s in df[sequence_column]]
        rows: List[Dict[str, float]] = [{} for _ in sequences]
        for predictor in EXTERNAL_PREDICTORS:
            if not predictor.available():
                continue
            _run_predictor(predictor, sequences, rows)
        additions = pd.DataFrame(rows, index=df.index)
        out = pd.concat([out, additions], axis=1)
        logger.info(
            f"Predictor enrichment complete: {len(additions.columns)} new columns "
            f"({', '.join(sorted(predictor_availability())) or 'no external tools'} probed).")
    return out
