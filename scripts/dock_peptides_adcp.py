"""
Docks peptides into a prepared AutoGridFR target with AutoDock CrankPep (ADCP).

Reads the peptide set and the target/box definition produced alongside it (see
``data/docking/docking_config.json``, which records WHICH PDB structure was chosen, why, and how
the search box was derived), runs one ADCP job per peptide, and collects the per-peptide best
scores into a single CSV.

Run with the ADFRsuite environment on PATH:

    export PATH=data/docking/tools/ADFRsuite/bin:$PATH
    python scripts/dock_peptides_adcp.py --peptides data/docking/pilot_peptides.fasta \\
        --target data/docking/msba_g907site.trg --out-dir data/docking/results

Runs ``--workers`` peptides at a time, each ADCP job using ``--cores`` cores internally. The two
multiply: keep ``workers * cores`` at or below the core count, or the jobs fight each other (three
concurrent all-core jobs drove load average past 45 on 24 cores earlier in this project).

Why parallelise across peptides at all, when ADCP already parallelises its own replicas: it
parallelises them in WAVES of `cores` replicas, so the wall-clock per peptide is
``ceil(replicas / cores) * time_per_replica`` -- a step function. At the default 50 replicas, 20
cores buys 3 waves where 17 would do, and the last wave runs 10 replicas on 20 cores with half the
machine idle. Spreading the same cores over several peptides converts that quantisation waste into
throughput: measured at 50 replicas, 1 worker x 20 cores is ~46 h for 100 peptides while 6 workers
x 4 cores is ~33 h on the same 24-core machine.

Each job gets its OWN working directory, which is not optional: ADCP unzips the target's maps and
writes ``transpoints``/``constrains`` into the *current working directory* under fixed names, so
concurrent jobs sharing one directory silently overwrite each other's target files and score
against whichever maps landed last.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_fasta(path):
    """
    :param path: FASTA file; headers may carry ``|``-separated metadata (``>id|MIC_uM=..|len=..``).
    :returns: ``[(identifier, sequence)]`` in file order.
    """
    records, name, chunks = [], None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    records.append((name, "".join(chunks)))
                name, chunks = line[1:].split("|")[0], []
            else:
                chunks.append(line.upper())
    if name:
        records.append((name, "".join(chunks)))
    return records


def parse_adcp_scores(log_text):
    """
    Pulls the ranked cluster table out of an ADCP run's stdout.

    ADCP prints a table of solution modes with affinity in kcal/mol; the first data row is the
    best-scoring cluster. Parsed defensively -- a peptide whose run produced no table (a crash, a
    timeout, a sequence ADCP refused) yields ``None`` rather than a fabricated score.

    :param log_text: Captured ADCP stdout.
    :returns: ``(best_affinity, n_modes)``, or ``(None, 0)`` when no score table was produced.
    """
    rows = []
    for line in log_text.splitlines():
        # e.g. "     1     -12.3     ..." -- rank then affinity, both numeric
        m = re.match(r"\s*(\d+)\s+(-?\d+\.\d+)\s", line)
        if m:
            rows.append(float(m.group(2)))
    if not rows:
        return None, 0
    return min(rows), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peptides", required=True, help="FASTA of peptides to dock")
    ap.add_argument("--target", required=True, help="AGFR .trg target file")
    ap.add_argument("--out-dir", required=True, help="Directory for per-peptide ADCP output")
    ap.add_argument("--replicas", type=int, default=50,
                    help="ADCP replicas per peptide (-N). 50 is ADCP's own recommended production "
                         "value; lower trades pose quality for wall-clock time.")
    ap.add_argument("--steps", type=int, default=2500000,
                    help="Monte-Carlo steps per replica (-n). 2.5M is ADCP's recommended value.")
    ap.add_argument("--cores", type=int, default=4, help="Cores per ADCP job (-c)")
    ap.add_argument("--workers", type=int, default=6,
                    help="Peptides docked concurrently. workers * cores should not exceed the "
                         "machine's core count.")
    ap.add_argument("--timeout", type=int, default=21600,
                    help="Per-peptide wall-clock cap in seconds; a peptide exceeding it is "
                         "recorded as timed out rather than stalling the whole batch.")
    ap.add_argument("--helix", action="store_true",
                    help="Start from a helical conformation (UPPER-case sequence) instead of coil. "
                         "Coil is the default and the right choice for proline-rich AMPs, which do "
                         "not form helices.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target_abs = os.path.abspath(args.target)
    target_name = os.path.basename(args.target)

    peptides = read_fasta(args.peptides)
    n_cpu = os.cpu_count() or 1
    if args.workers * args.cores > n_cpu:
        print(f"  WARNING: workers*cores = {args.workers * args.cores} exceeds {n_cpu} cores; "
              f"the jobs will contend and each one will run slower than it would alone.",
              flush=True)
    waves = -(-args.replicas // args.cores)  # ceil
    print(f"{len(peptides)} peptide(s) against {args.target}: {args.workers} worker(s) x "
          f"{args.cores} core(s), {args.replicas} replicas in {waves} wave(s) per peptide",
          flush=True)

    summary_path = os.path.join(args.out_dir, "docking_summary.csv")
    out = open(summary_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(out)
    writer.writerow(["entry_id", "sequence", "length", "best_affinity_kcal_mol",
                     "n_modes", "seconds", "status"])
    out.flush()
    write_lock = threading.Lock()
    done = [0]

    def dock_one(item):
        """Docks one peptide in its own working directory. :returns: the summary row."""
        name, seq = item
        # Each job needs a private cwd: ADCP unzips the maps and writes transpoints/constrains
        # there under FIXED names, so a shared directory means concurrent jobs overwrite each
        # other's target and score against whichever maps landed last.
        work = os.path.join(args.out_dir, "work", name)
        os.makedirs(work, exist_ok=True)
        # ADCP builds os.path.join('./tmp_<job>/', target[:-4], ...) from the target argument, so
        # -t must be a BARE FILENAME in the job's cwd: '../x.trg' yields './tmp_job/../x/...' and
        # an absolute path makes os.path.join drop the './tmp_job/' prefix. Both fail with an
        # opaque IOError on translationPoints.npy. Hence a per-job symlink named exactly that.
        link = os.path.join(work, target_name)
        if not os.path.lexists(link):
            os.symlink(target_abs, link)

        # ADCP reads case as secondary-structure intent: lower = coil, UPPER = helix.
        seq_arg = seq.upper() if args.helix else seq.lower()
        # -o is a job NAME, not a path: ADCP writes <name>_<replica>.pdb into its cwd.
        cmd = ["adcp", "-t", target_name, "-s", seq_arg,
               "-N", str(args.replicas), "-n", str(args.steps),
               "-c", str(args.cores), "-o", name]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                                  cwd=work)
            log = (proc.stdout or "") + (proc.stderr or "")
            status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
        except subprocess.TimeoutExpired:
            log, status = "", "timeout"
        except Exception as e:  # one peptide must not take the batch down
            log, status = str(e), "error"
        elapsed = time.time() - t0

        with open(os.path.join(args.out_dir, f"{name}.log"), "w", encoding="utf-8") as fh:
            fh.write(log)
        best, n_modes = parse_adcp_scores(log)
        row = [name, seq, len(seq), f"{best:.3f}" if best is not None else "", n_modes,
               f"{elapsed:.0f}", status]

        # Written as each peptide lands, not at the end: a multi-day batch has to stay readable
        # while it runs, and has to survive being interrupted with its finished work intact.
        with write_lock:
            writer.writerow(row)
            out.flush()
            done[0] += 1
            print(f"[{done[0]}/{len(peptides)}] {name} len={len(seq)} {status}, "
                  f"best={best if best is not None else 'n/a'} kcal/mol, {elapsed / 60:.1f} min",
                  flush=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(dock_one, p) for p in peptides]
            for future in as_completed(futures):
                future.result()
    finally:
        out.close()

    print(f"\nwrote {summary_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
