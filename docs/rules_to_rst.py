"""
Renders the validation rule tables into reStructuredText for the Sphinx build.

``sample/validation_rules.py`` is the file a reviewer edits, but autodoc only shows the module
docstring and the names of the tables -- not the rows, which are the part worth reading. A rule
nobody can see in the documentation is a rule nobody checks.

Generated from the live tables on every build rather than transcribed, so the published rules are
by construction the rules the pipeline enforces.
"""
from typing import Any, List, Sequence


def _grid(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """
    Renders a list-table directive, which handles long cells better than a fixed-width grid.

    :param headers: Column headings.
    :param rows: Row cells, already stringified.
    :returns: RST lines.
    """
    out = [".. list-table::", "   :header-rows: 1", "   :widths: auto", ""]
    for i, row in enumerate([list(headers)] + [list(r) for r in rows]):
        for j, cell in enumerate(row):
            text = str(cell) if str(cell).strip() else "—"
            out.append(f"   {'*' if j == 0 else ' '} - {text}")
    out.append("")
    return out


def _section(title: str, char: str = "-") -> List[str]:
    """
    :param title: Section heading text.
    :param char: Underline character; ``=`` for the page title, ``-`` for sections.
    :returns: The title and an underline generated to match its length, so a reworded heading
        cannot leave a short underline behind.
    """
    return [title, char * len(title), ""]


def _fmt_tuple(value: Any) -> str:
    """
    :param value: A tuple of allowed values from a rule row.
    :returns: Comma-separated literals, or "any" when the tuple is empty (meaning unrestricted).
    """
    if not value:
        return "*any*"
    return ", ".join(f"``{v}``" for v in value)


def render() -> str:
    """
    :returns: The full RST page documenting every validation rule table.
    """
    from sample import validation_rules as R

    out: List[str] = [
        *_section("Validation rules", "="),
        ".. NOTE: generated from sample/validation_rules.py by docs/rules_to_rst.py during the",
        "   Sphinx build. Edit the rule tables in that module, not this file.",
        "",
        "Every constraint the pipeline enforces on a measurement is listed here. These are",
        "scientific claims rather than implementation details, so they are published in full: a",
        "reviewer should be able to check them, and disagree.",
        "",
        "A record must satisfy all of them to be published. Where two fields are each valid but",
        "contradict one another, the conflict is referred to an LLM together with the paragraph the",
        "record came from -- and whatever the model proposes is re-checked against these same tables",
        "before it is accepted. See :func:`sample.strict_schema.check_activity_coherence` and",
        ":func:`sample.llm_validator.apply_coherence_review`.",
        "",
        *_section("Units and plausible ranges", "-"),
        "Bounds exist to catch OCR and parsing damage, not to express what a good peptide looks",
        "like. A real but unusual result must survive them.",
        "",
    ]
    out += _grid(
        ("Unit", "Quantity", "Minimum", "Maximum"),
        [(f"``{r.unit}``" if r.unit else "*(dimensionless)*", r.quantity,
          f"{r.min_value:g}", f"{r.max_value:g}") for r in R.UNIT_RANGES],
    )

    out += [
        *_section("Activity classes and their valid target domains", "-"),
        "Domains are resolved from NCBI lineage (``scripts/build_taxonomy_domains.py``), so these",
        "are checked against the taxonomy rather than a hand-written list of genera. An organism",
        "whose domain cannot be resolved is never treated as a contradiction.",
        "",
    ]
    out += _grid(
        ("Activity class", "Valid target domains", "Rationale"),
        [(f"``{r.target_group}``", _fmt_tuple(r.domains), r.note) for r in R.TARGET_GROUP_DOMAINS],
    )

    out += [
        *_section("Assays", "-"),
        "``Target kind`` distinguishes assays run against a whole organism from those run on",
        "cultured cells -- a MIC is not measured on a cell line, and hemolysis is not measured on a",
        "bacterium. An empty ``Activity classes`` means the assay is not restricted to one class.",
        "",
    ]
    out += _grid(
        ("Assay", "Activity classes", "Unit quantities", "Target kind", "Meaning"),
        [(f"``{r.activity_type}``", _fmt_tuple(r.target_groups), _fmt_tuple(r.quantities),
          f"``{r.target_kind}``", r.note) for r in R.ACTIVITY_RULES],
    )

    out += [
        *_section("Unit equivalences", "-"),
        "Applied before validation so that one physical quantity has one spelling. A factor of 1.0",
        "means the two strings denote the same thing and only the label changes.",
        "",
    ]
    out += _grid(
        ("Reported as", "Stored as", "Factor", "Note"),
        [(f"``{r.source}``", f"``{r.target}``", f"{r.factor:g}", r.note)
         for r in R.UNIT_EQUIVALENCES],
    )
    return "\n".join(out) + "\n"


def write_rules_rst(output_path: str) -> str:
    """
    Renders the rule tables and writes them.

    :param output_path: Destination ``.rst`` file.
    :returns: The path written.
    """
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(render())
    return output_path
