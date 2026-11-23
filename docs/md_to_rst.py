"""
Converts README.md to reStructuredText for the Sphinx build.

Sphinx cannot read Markdown without myst-parser, which this project does not depend on, so the
README is translated to RST instead. It is regenerated from conf.py on every build rather than
checked in as a hand-converted copy: a second, manually-maintained rendering of the README is a
copy that silently goes stale, and the whole point of putting it in the docs is that the two agree.

Deliberately narrow. It handles exactly the constructs README.md uses -- verified by inventorying
the file -- and does not attempt to be a general Markdown implementation. If the README grows a
table or a hyperlink, this needs a case for it; the build will show the raw syntax rather than fail
silently, which is the intended failure mode.
"""
import re

# RST underline characters, in the order Sphinx assigns heading levels within one document.
_UNDERLINE = {1: "=", 2: "-", 3: "~", 4: "^", 5: '"', 6: "'"}

# Markdown fence languages -> Pygments lexers. "env" is not a lexer; those blocks are dotenv
# fragments, which highlight acceptably as plain text and, unlike an unknown lexer name, do not
# make the build emit a warning per block.
_LEXERS = {"bash": "bash", "sh": "bash", "env": "text", "": "text"}


def _escape_inline(text):
    """
    Escapes the RST inline markup that plain Markdown prose can trip over.

    The one that actually bites here is the trailing underscore: RST reads `foo_` as a reference to
    a target named `foo`, so a bare pipeline step name like `step_08_screening_extract_peptide_data`
    outside backticks would become a broken reference. Text already inside a literal is never
    passed to this function.
    """
    # Protect inline code spans first -- their contents must not be escaped or reinterpreted.
    spans = []

    def stash(match):
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r'`([^`\n]+)`', stash, text)

    text = text.replace("\\", "\\\\")
    # A word-final underscore is a reference in RST; escape it.
    text = re.sub(r'(?<=\w)_(?=\s|$|[.,;:)])', r'\_', text)
    # A bare "|" would start a substitution reference.
    text = text.replace("|", "\\|")

    # Restore code spans as RST inline literals.
    def restore(match):
        return f"``{spans[int(match.group(1))]}``"

    return re.sub(r'\x00(\d+)\x00', restore, text)


def convert(markdown, asset_prefix="../"):
    """
    Returns the RST rendering of `markdown`.

    `asset_prefix` is prepended to relative image paths, because the README resolves them from the
    repository root while the generated page is built from docs/.
    """
    out = []
    lines = markdown.split("\n")
    i = 0
    in_fence = False
    fence_lang = "text"
    list_indent = None      # indent of the list item most recently emitted, or None outside a list

    while i < len(lines):
        line = lines[i]

        # Fenced code block -> literal block with a lexer.
        fence = re.match(r'^```(\w*)\s*$', line)
        if fence and not in_fence:
            in_fence, fence_lang = True, _LEXERS.get(fence.group(1), "text")
            out += ["", f".. code-block:: {fence_lang}", ""]
            i += 1
            continue
        if in_fence:
            if line.strip() == "```":
                in_fence = False
                out.append("")
            else:
                out.append("   " + line)     # body is indented, never escaped
            i += 1
            continue

        # Images. A standalone ![alt](path) becomes a figure; the path is rebased because the
        # README resolves relative paths from the repository root and this page is built from docs/.
        image = re.match(r'^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$', line)
        if image:
            alt, path = image.group(1).strip(), image.group(2).strip()
            if not re.match(r'^(?:https?:)?//', path) and not path.startswith("/"):
                path = asset_prefix + path
            out += ["", f".. image:: {path}"]
            if alt:
                out.append(f"   :alt: {alt}")
            out.append("")
            list_indent = None
            i += 1
            continue

        # Headings.
        heading = re.match(r'^(#{1,6})\s+(.*)$', line)
        if heading:
            level, title = len(heading.group(1)), _escape_inline(heading.group(2).strip())
            out += ["", title, _UNDERLINE[level] * max(len(title), 3), ""]
            i += 1
            continue

        # Horizontal rules. RST transitions cannot sit directly before a section title, which is
        # exactly where this README puts them, so they are dropped -- the headings already provide
        # the visual break.
        if line.strip() in ("---", "***", "___"):
            i += 1
            continue

        # Blockquote -> indented block.
        if line.startswith("> "):
            block = []
            while i < len(lines) and (lines[i].startswith("> ") or lines[i].strip() == ">"):
                block.append(_escape_inline(lines[i][2:].strip()))
                i += 1
            out += ["", "   " + " ".join(block).strip(), ""]
            continue

        # Bullets. RST needs a blank line before a list and consistent markers.
        bullet = re.match(r'^(\s*)[*-]\s+(.*)$', line)
        if bullet:
            indent, body = bullet.group(1), _escape_inline(bullet.group(2))
            # RST needs a blank line before a list starts AND before its nesting level changes.
            # Without the second case a sub-list under "7. Interactive Dashboard:" was absorbed into
            # the parent item and rendered as literal "* " text instead of a nested list.
            if out and out[-1].strip() and (list_indent is None or len(indent) != list_indent):
                out.append("")
            out.append(f"{indent}* {body}")
            list_indent = len(indent)
            i += 1
            continue

        # Numbered lists -> RST auto-enumeration, so renumbering the source cannot desync them.
        numbered = re.match(r'^(\s*)\d+\.\s+(.*)$', line)
        if numbered:
            indent, body = numbered.group(1), _escape_inline(numbered.group(2))
            if out and out[-1].strip() and (list_indent is None or len(indent) != list_indent):
                out.append("")
            out.append(f"{indent}#. {body}")
            list_indent = len(indent)
            i += 1
            continue

        if line.strip():
            list_indent = None
        out.append(_escape_inline(line) if line.strip() else "")
        i += 1

    # Collapse runs of blank lines, which RST treats as significant in some contexts.
    collapsed = []
    for line in out:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip() + "\n"


def write_readme_rst(readme_path, output_path):
    """Reads `readme_path`, converts it, and writes `output_path`. Returns the output path."""
    with open(readme_path, "r", encoding="utf-8") as fh:
        markdown = fh.read()
    body = convert(markdown)
    header = (
        ".. NOTE: generated from README.md by docs/md_to_rst.py during the Sphinx build.\n"
        "   Edit README.md, not this file -- your changes here will be overwritten.\n\n"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(header + body)
    return output_path
