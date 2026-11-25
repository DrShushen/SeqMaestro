"""Sphinx configuration for SeqMaestro's documentation.

Build from the repository root, under `SeqMaestro_Core` (the `docs` extra):

    rm -rf docs/_build && sphinx-build docs docs/_build/html -W --keep-going

**A page links as it would on GitHub.** Every page is written with relative links, so that it reads
as well in a clone or on GitHub as on the site. A link from a page to a file outside `docs/` becomes
a link to that file on GitHub when the site is built, and a link to one of the READMEs the site
includes (`INCLUDED`) becomes a link to its page. A README included in a page keeps its own relative
links, which are resolved from where the README is.
"""

from __future__ import annotations

import re
from pathlib import Path

project = "SeqMaestro"
copyright = "2026, the SeqMaestro authors"
author = "the SeqMaestro authors"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# A code block that names no language is shown as plain text. Most are directory trees and record
# layouts, which Sphinx's default would colour as Python.
highlight_language = "text"

# -- HTML output -------------------------------------------------------------------------------

html_theme = "furo"
html_title = "SeqMaestro"
#: Where GitHub Pages publishes the site, from the public repository. Each page gives its address
#: there as its canonical one, and the README links to the pages there (`tests/test_readme.py`).
html_baseurl = "https://drshushen.github.io/SeqMaestro/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Mermaid -----------------------------------------------------------------------------------
#
# sphinxcontrib-mermaid gives every diagram a fixed 500-pixel box by default, so a wide, short
# flowchart is drawn with a band of empty space above and below it. Sized automatically, and drawn
# at its natural size, each diagram takes the room its shape needs and is scaled down only where it
# is wider than the page. A sequence diagram names its participants once, at the top.

mermaid_height = "auto"
mermaid_width = "auto"
mermaid_init_config = {"startOnLoad": False, "flowchart": {"useMaxWidth": False},
                       "sequence": {"useMaxWidth": False, "mirrorActors": False}}

# -- MyST --------------------------------------------------------------------------------------

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
# `html_image` reads an HTML `<img>` as an image, which is how the user guide shows the harnesses'
# logos: GitHub shows such a tag too, at the height the tag gives, whereas the site sizes it by its
# class (`custom.css`).
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "html_image"]
# Headings up to the third level can be linked to by their text (`page.md#a-heading`).
myst_heading_anchors = 3

# -- The API reference -------------------------------------------------------------------------
#
# Generated from the docstrings of the `seqmaestro` package by sphinx-autodoc2, which parses the source
# without importing it, so the build needs none of the package's dependencies. The docstrings are
# written in markdown, as these pages are, and are rendered by MyST. The prose pages explain why the
# code is built as it is and how its parts fit together, and this reference holds what each module,
# class and function is.

autodoc2_packages = ["../seqmaestro"]
autodoc2_output_dir = "api"
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [(r".*", "myst")]
autodoc2_hidden_objects = ["private", "inherited", "dunder"]

# -- Links -------------------------------------------------------------------------------------

#: The repository a link to a file outside `docs/` goes to on the site.
REPOSITORY = "https://github.com/DrShushen/SeqMaestro"
SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent

#: The READMEs kept beside the code that a page of the site includes, by their path in the
#: repository, with the page that includes each. A link to one of them goes to its page.
INCLUDED = {
    "ENVIRONMENTS.md": "developer/environments.md",
    "experiments/README.md": "developer/experiments.md",
    "experiments/configs/README.md": "developer/configs.md",
    "experiments/datasets/README.md": "developer/descriptions.md",
    "experiments/prepare/README.md": "developer/preparation.md",
    "experiments/adapters/README.md": "developer/adapters.md",
    "experiments/analysis/README.md": "developer/analysis-scripts.md",
}

LINK = re.compile(r"(\]\()([^)\s#]+)((?:#[^)\s]*)?\))")


def relinked(text: str, written_at: Path) -> str:
    """`text` with each relative link resolved from `written_at`, the file it was written in, and
    pointed at what the site has: the page, where the target is a page or an included README, and
    the file on GitHub otherwise."""

    def fix(match: re.Match) -> str:
        target = match.group(2)
        if "://" in target or target.startswith(("mailto:", "/", "#", "{")):
            return match.group(0)
        resolved = (written_at.parent / target).resolve()
        try:
            in_repository = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        page = SOURCE / INCLUDED[in_repository] if in_repository in INCLUDED else resolved
        if page.is_relative_to(SOURCE):
            # A leading slash is MyST's path from the site's root, the same from any page.
            return f"{match.group(1)}/{page.relative_to(SOURCE).as_posix()}{match.group(3)}"
        kind = "tree" if resolved.is_dir() else "blob"
        return f"{match.group(1)}{REPOSITORY}/{kind}/main/{in_repository}{match.group(3)}"

    return LINK.sub(fix, text)


def _page_read(app, docname: str, source: list[str]) -> None:
    source[0] = relinked(source[0], Path(app.env.doc2path(docname)))


def _include_read(app, relative_path: Path, parent_docname: str, content: list[str]) -> None:
    content[0] = relinked(content[0], (SOURCE / relative_path).resolve())


def setup(app):
    app.connect("source-read", _page_read)
    app.connect("include-read", _include_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
