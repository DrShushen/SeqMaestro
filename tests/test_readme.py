"""The README's links lead to pages and files that exist.

The README is the repository's landing page on GitHub, and the documentation's build does not read it,
so nothing else checks its links. It links to each page of the documentation at its address on the
published site (`html_baseurl` in `docs/conf.py`), so a page renamed or removed in `docs/` would leave
a link there that leads nowhere.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text()
SITE = re.search(r'^html_baseurl = "([^"]+)"$', (REPO_ROOT / "docs" / "conf.py").read_text(), re.M).group(1)

#: The target of each markdown link and each image of the README, without its anchor.
TARGETS = [target.split("#")[0] for pair in re.findall(r'\]\(([^)\s]+)\)|src="([^"]+)"', README)
           for target in pair if target]


def source_of(address: str) -> Path:
    """The markdown file in `docs/` that the site's page at `address` is built from."""
    path = address.removeprefix(SITE)
    if path == "" or path.endswith("/"):
        path += "index.html"
    assert path.endswith(".html"), f"{address} is no page of the site"
    return REPO_ROOT / "docs" / (path.removesuffix(".html") + ".md")


def test_each_link_to_the_site_is_a_page_of_the_documentation():
    on_the_site = [target for target in TARGETS if target.startswith(SITE)]
    assert on_the_site, f"the README has no link to the documentation at {SITE}"
    missing = [address for address in on_the_site if not source_of(address).is_file()]
    assert not missing, f"the README links to pages that docs/ does not build: {missing}"


def test_each_relative_link_is_a_file_of_the_repository():
    relative = [target for target in TARGETS if "://" not in target and not target.startswith("mailto:")]
    missing = [target for target in relative if not (REPO_ROOT / target).exists()]
    assert not missing, f"the README links to paths the repository does not have: {missing}"
