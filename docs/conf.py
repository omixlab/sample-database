import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# The README is the project's main prose documentation, so it is also the docs landing page.
# Sphinx cannot parse Markdown without myst-parser (not a dependency here), so it is converted to
# RST on every build -- generated rather than checked in, so the page cannot drift from the README.
from md_to_rst import write_readme_rst

_DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
write_readme_rst(
    os.path.join(_DOCS_DIR, os.pardir, "README.md"),
    os.path.join(_DOCS_DIR, "readme.rst"),
)

# The validation rule tables, rendered from the live data. Autodoc shows only the module docstring
# and the names of the tables, not their rows -- and the rows are the part a reviewer needs to see.
from rules_to_rst import write_rules_rst

write_rules_rst(os.path.join(_DOCS_DIR, "validation_rules.rst"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sAMPle'
copyright = '2026, omixlab'
author = 'omixlab'

version = '1.0'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_autodoc_typehints',
]

# -- PDF output --------------------------------------------------------------------------------
#
# Built through LaTeX: Sphinx renders the documentation to .tex, then latexmk drives pdflatex to
# produce the PDF. This replaces rst2pdf, which rendered straight to PDF through reportlab and
# needed no TeX toolchain, but produced a different document -- no real sectioning, weaker
# cross-references, and code blocks that were shrunk to fit rather than wrapped.
#
# The toolchain is a conda install, not a system one:
#     mamba install -n sample -c conda-forge texlive-core latexmk
#
# Build with `make docs-pdf`, which is `sphinx -b latex` followed by latexmk.

# (startdocname, targetname, title, author, theme, toctree_only)
#
# `manual` is Sphinx's default theme for a whole-project document: it maps to the LaTeX `report`
# class, giving numbered chapters and a title page. `howto` (the `article` class) is the other
# stock option and suits a single-topic document.
latex_documents = [
    ('index', 'sAMPle.tex', 'sAMPle Documentation', 'omixlab', 'manual'),
]

# Deliberately close to empty. The default template is Sphinx's own `sphinxmanual.cls` wrapper
# around `report`, and overriding its preamble is how a build becomes dependent on TeX packages a
# plain TeX Live does not carry. Only the paper and font sizes are set, both stock LaTeX options.
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    # Long API signatures and code samples overflow the text block otherwise. Unlike rst2pdf's
    # `shrink`, which scaled the overflow down until it fit, this wraps it and keeps it legible.
    'sphinxsetup': 'verbatimwrapslines=true, verbatimforcewraps=true',
}

# The stock theme. Named explicitly so a future Sphinx default change does not silently restyle
# the document.
latex_theme = 'manual'

# XeTeX rather than pdfTeX.
#
# The docstrings contain typographic characters -- curly quotes, en dashes, the occasional Å and
# µ -- because they quote real error messages and units. pdfTeX reaches those through an 8-bit
# font encoding and fails on anything outside it: `Missing character: There is no ' in font
# ec-qtmr`, which halts the build. XeTeX is natively Unicode, so the same source compiles without
# transliterating the documentation to ASCII.
#
# Sphinx emits the correct preamble for whichever engine is named here; the template itself is
# still the stock one.
latex_engine = 'xelatex'

# Autodoc / typehints rendering.
#
# `description` moves the annotations out of the signature line and into the parameter list, so a
# reader sees "peptide_name (str) -- Name to search for" instead of a signature so long it wraps.
# It also means the :param: text written in the docstring and the type from the annotation are
# rendered as one entry rather than two competing ones.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
typehints_document_rtype = True
typehints_use_rtype = True
always_document_param_types = True

# Keep members in source order; these modules are written as a narrative (constants, then the
# functions that use them) and alphabetizing scatters that.
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Napoleon is enabled for Google/NumPy-style docstrings; the codebase uses Sphinx ``:param:``
# fields directly, which need no conversion, so only the field aliases matter here.
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Suppress the duplicate-label noise autodoc emits for same-named sections across
# the generated readme page and the module pages.
suppress_warnings = ['autosectionlabel.*']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def _paramref_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Render SQLAlchemy's ``:paramref:`` role as an inline literal.

    The ORM entity classes in :mod:`sample.models` annotate their columns with SQLAlchemy types,
    and ``sphinx_autodoc_typehints`` follows those annotations into SQLAlchemy's own docstrings to
    resolve them. Those docstrings are written for SQLAlchemy's documentation build, which enables
    the third-party ``sphinx-paramlinks`` extension; ours does not, so every such reference was
    raising ``Unknown interpreted text role "paramref"`` -- around twenty errors per build, all of
    them about a role we have no reason to resolve.

    Registering it as a literal is deliberate rather than a stopgap: the cross-reference target is
    a parameter of a SQLAlchemy function documented on SQLAlchemy's site, not in this project, so
    there is nothing here to link to. Printing the parameter name is the whole of the useful
    content. The leading ``~`` and the dotted path are stripped the way the real role does it, so
    ``:paramref:`~_schema.Column.nullable``` renders as ``nullable``.
    """
    from docutils import nodes

    target = text.lstrip('~').split('(')[0].strip()
    return [nodes.literal(rawtext, target.rsplit('.', 1)[-1])], []


def setup(app):
    """
    Register the local roles and directives this build needs.

    Both entries exist for the same reason: SQLAlchemy's docstrings, reached through the type
    annotations on :mod:`sample.models`, are written against SQLAlchemy's own Sphinx setup. See
    :func:`_paramref_role` for the full explanation.
    """
    from docutils import nodes
    from docutils.parsers.rst import Directive, roles

    class _LegacyDirective(Directive):
        """Render SQLAlchemy's ``.. legacy::`` block as a plain note."""

        has_content = True
        required_arguments = 0
        optional_arguments = 0
        final_argument_whitespace = True

        def run(self):
            node = nodes.note()
            self.state.nested_parse(self.content, self.content_offset, node)
            return [node]

    roles.register_local_role('paramref', _paramref_role)
    app.add_directive('legacy', _LegacyDirective)
