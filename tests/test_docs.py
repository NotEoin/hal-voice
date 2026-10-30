"""Tests that the docs index lists every document and every link in it resolves."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
LINK = re.compile(r"\]\(([^)#]+\.md)")


def test_docs_index_lists_every_doc():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    listed = {(DOCS / m).resolve() for m in LINK.findall(index)}
    on_disk = {p.resolve() for p in DOCS.rglob("*.md") if p.name != "README.md"}
    assert on_disk <= listed


def test_docs_index_links_resolve():
    for target in LINK.findall((DOCS / "README.md").read_text(encoding="utf-8")):
        assert (DOCS / target).is_file(), target
