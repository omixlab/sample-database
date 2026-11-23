"""
Renders the manuscript to PDF (step 48, after :mod:`paper.generate_dataset_report` has refreshed
the figures).

Run under the docs environment, which is where pandoc and tectonic live:

    conda run -n sample python paper/build_pdf.py

The manuscript (``paper/README.md``) embeds its figures inline with ``![caption](figures/NN_*.png)``
next to the prose that cites them, so this is a straight conversion -- no assembly step. Image
sizing lives in ``paper/pdf-header.tex`` rather than in per-image pandoc attributes, so the
manuscript stays readable as Markdown on the repository page.

PDF engine is tectonic (XeTeX): pdflatex cannot set this document at all, since the manuscript
uses real Unicode mathematical characters. ``paper/pdf-header.tex`` maps the ones Latin Modern has
no glyph for -- without it they are DROPPED SILENTLY rather than erroring, which turns
"roc_auc >= 0.75" into "roc_auc  0.75".
"""
import os
import re
import subprocess
import sys
import tempfile

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PAPER_DIR, "figures")
HEADER = os.path.join(PAPER_DIR, "pdf-header.tex")


def check_figures(manuscript_path):
    """
    Warns about figure tags pointing at files that are not there.

    A missing image is a hard tectonic error rather than a silent gap, but the message it prints
    names a LaTeX temp file, not the manuscript -- so the miss is reported here, against the
    manuscript, before pandoc is invoked at all.
    """
    with open(manuscript_path, encoding="utf-8") as fh:
        body = fh.read()
    referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)
    missing = [p for p in referenced if not os.path.exists(os.path.join(PAPER_DIR, p))]
    for path in missing:
        print(f"  WARNING: {manuscript_path} references {path}, which does not exist -- run "
              f"paper/generate_dataset_report.py first.", flush=True)
    print(f"  {len(referenced)} figure reference(s), {len(missing)} missing", flush=True)
    return body


def render(markdown_text, out_pdf, title):
    """Runs pandoc over `markdown_text`, writing `out_pdf`. :returns: True on success."""
    # Written inside PAPER_DIR (not a system temp dir) so the relative figure paths above resolve.
    fd, tmp_md = tempfile.mkstemp(suffix=".md", dir=PAPER_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(markdown_text)
        cmd = ["pandoc", tmp_md, "-o", out_pdf,
               "--pdf-engine=tectonic",
               "-H", HEADER,
               "--toc", "--toc-depth=2",
               "-V", "geometry:margin=2.2cm",
               "-V", "fontsize=10pt",
               "-V", "linkcolor=blue",
               "-M", f"title={title}",
               "--resource-path", PAPER_DIR]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PAPER_DIR)
        if proc.returncode != 0:
            # tectonic is verbose on success too, so only the failing tail is worth printing.
            sys.stderr.write((proc.stderr or proc.stdout or "")[-3000:])
            print(f"  FAILED to build {out_pdf}", flush=True)
            return False
        # A dropped glyph is only a warning, but it silently corrupts the text -- surface it.
        for line in (proc.stderr or "").splitlines():
            if "could not represent character" in line or "Missing character" in line:
                print(f"  WARNING from TeX: {line.strip()}", flush=True)
        print(f"  wrote {out_pdf} ({os.path.getsize(out_pdf) / 1024:.0f} KB)", flush=True)
        return True
    finally:
        if os.path.exists(tmp_md):
            os.remove(tmp_md)


def main():
    print("Building the paper PDF ...", flush=True)
    ok = True

    manuscript = os.path.join(PAPER_DIR, "README.md")
    if os.path.exists(manuscript):
        ok &= render(check_figures(manuscript), os.path.join(PAPER_DIR, "sAMPle.pdf"),
                     "sAMPle: A Continuously Updatable Knowledgebase for Automated Mining, "
                     "Curation, and Machine Learning of Bioactive Peptides")
    else:
        print(f"  {manuscript} not found; nothing to build.", flush=True)
        return 1

    # The supplement is a separate document and is built as one; it cites no figures of its own.
    supplementary = os.path.join(PAPER_DIR, "SUPPLEMENTARY.md")
    if os.path.exists(supplementary):
        with open(supplementary, encoding="utf-8") as fh:
            ok &= render(fh.read(), os.path.join(PAPER_DIR, "sAMPle_supplementary.pdf"),
                         "sAMPle: Supplementary Material")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
