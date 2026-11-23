"""
One-time offline build step: adds a `tax_domain` table to data/taxonomy.db mapping every NCBI
tax_id to its broad domain (bacteria / archaea / virus / fungi / plant / animal / protozoa).

Why this exists: target_group and target_organism have to agree -- an "antibacterial" record whose
target is a plant or an animal is a mis-pairing, not a finding. Deciding that requires knowing what
kind of organism a name refers to, and the shipped taxonomy.db carries only name -> tax_id, with no
ranks and no lineage. The first version of the check used a hand-written list of bacterial and
fungal genera, which cannot cover a corpus that names thousands of species and silently treats
everything it does not recognize as unknown.

This resolves each tax_id by walking NCBI's own parent links to the root and recording the first
recognized clade it falls under, so the answer comes from the taxonomy rather than from a list
someone maintained by hand.

Run: python scripts/build_taxonomy_domains.py [--taxdump PATH]
Re-run when taxonomy.db is rebuilt or NCBI's taxdump is refreshed.
"""
import argparse
import os
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample.config import DATA_DIR

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
TAXONOMY_DB = DATA_DIR / "taxonomy.db"

# Clades whose members define a domain, most specific first. Walking upward from a node and taking
# the first match gives the most specific answer: Plasmodium reaches Apicomplexa before it reaches
# Eukaryota, so it resolves to protozoa rather than being lost among the eukaryotes.
#
# "protozoa" is not a clade in modern taxonomy -- it is a practical grouping of the eukaryotic
# parasites this literature tests peptides against, so the relevant lineages are listed explicitly.
DOMAIN_ROOTS = {
    2: "bacteria",
    2157: "archaea",
    10239: "virus",
    4751: "fungi",
    33090: "plant",        # Viridiplantae
    33208: "animal",       # Metazoa
    5794: "protozoa",      # Apicomplexa  (Plasmodium, Toxoplasma, Cryptosporidium, Eimeria)
    33682: "protozoa",     # Euglenozoa   (Trypanosoma, Leishmania)
    554915: "protozoa",    # Amoebozoa    (Entamoeba, Acanthamoeba)
    207245: "protozoa",    # Fornicata    (Giardia)
    5878: "protozoa",      # Ciliophora   (Paramecium, Tetrahymena)
    5719: "protozoa",      # Parabasalia  (Trichomonas)
}


def fetch_nodes(taxdump_path=None):
    """Yields (tax_id, parent_tax_id) from nodes.dmp, downloading the taxdump if not supplied."""
    tmpdir = None
    if not taxdump_path:
        tmpdir = tempfile.mkdtemp()
        taxdump_path = os.path.join(tmpdir, "taxdump.tar.gz")
        print(f"Downloading {TAXDUMP_URL} ...")
        urllib.request.urlretrieve(TAXDUMP_URL, taxdump_path)
        print(f"  {os.path.getsize(taxdump_path) / 1e6:.0f} MB")

    print("Reading nodes.dmp ...")
    with tarfile.open(taxdump_path, "r:gz") as tar:
        member = tar.extractfile("nodes.dmp")
        for line in member:
            # nodes.dmp is "tax_id\t|\tparent tax_id\t|\trank\t|\t..."
            parts = line.decode("utf-8", "ignore").split("\t|\t", 2)
            if len(parts) >= 2:
                try:
                    yield int(parts[0]), int(parts[1])
                except ValueError:
                    continue


def resolve_domains(parent_of):
    """
    Domain for every tax_id, by walking parent links to the root.

    Ancestors are memoised as the walk proceeds, so each node is resolved once no matter how many
    descendants share its path -- without that this is quadratic over ~2.5M nodes.
    """
    domain_of = {}
    for tax_id in parent_of:
        if tax_id in domain_of:
            continue
        path, node = [], tax_id
        while True:
            if node in domain_of:
                answer = domain_of[node]
                break
            if node in DOMAIN_ROOTS:
                answer = DOMAIN_ROOTS[node]
                break
            path.append(node)
            parent = parent_of.get(node)
            if parent is None or parent == node:      # reached root, or a self-parented node
                answer = ""
                break
            node = parent
        for seen in path:
            domain_of[seen] = answer
        domain_of.setdefault(tax_id, answer)
    return domain_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxdump", help="Path to an already-downloaded taxdump.tar.gz")
    args = ap.parse_args()

    if not TAXONOMY_DB.exists():
        print(f"ERROR: {TAXONOMY_DB} not found; nothing to annotate.")
        return 1

    parent_of = dict(fetch_nodes(args.taxdump))
    print(f"  {len(parent_of)} taxonomy nodes")

    print("Resolving domains by lineage ...")
    domain_of = resolve_domains(parent_of)

    conn = sqlite3.connect(str(TAXONOMY_DB))
    conn.execute("DROP TABLE IF EXISTS tax_domain")
    conn.execute("CREATE TABLE tax_domain (tax_id INTEGER PRIMARY KEY, domain TEXT NOT NULL)")
    conn.executemany(
        "INSERT OR REPLACE INTO tax_domain (tax_id, domain) VALUES (?, ?)",
        ((t, d) for t, d in domain_of.items() if d),
    )
    conn.commit()

    counts = dict(conn.execute("SELECT domain, COUNT(*) FROM tax_domain GROUP BY domain"))
    total = conn.execute("SELECT COUNT(*) FROM tax_domain").fetchone()[0]
    conn.close()

    print(f"Wrote {total} tax_id -> domain rows into {TAXONOMY_DB}")
    for domain, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {domain:10} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
