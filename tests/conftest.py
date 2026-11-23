"""
Shared fixtures for the sAMPle test suite.

Two rules the whole suite follows, both enforced here rather than left to each test:

* **No network.** Every test runs offline. Anything that would reach PubMed, Ollama, UniProt or a
  publisher is mocked at its boundary. The ``no_network`` autouse fixture makes a stray real call
  fail loudly instead of quietly succeeding on a developer's machine and hanging in CI.
* **No writes into the repository.** Tests that need files use ``tmp_path``. A test that writes to
  ``data/`` would mutate the release artifacts the pipeline produces.
"""
import json
import os
import socket
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Blocks real socket connections for the duration of every test.

    Tests that legitimately exercise connection handling opt out by monkeypatching the specific
    function they need; this only stops an *unintended* call from silently reaching the internet.

    :param monkeypatch: pytest patcher.
    :returns: ``None``.
    """
    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "A test attempted a real network connection. Mock the boundary instead.")

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=False)
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=False)

    # A cache hit alone does not guarantee no_network's socket block is never hit: HuggingFace's
    # from_pretrained() checks the Hub for updates by default even when the model is already
    # cached locally, and that check is a real HTTP request that would otherwise be blocked above
    # with an opaque low-level socket error rather than a clear "why did this try the network"
    # message. Forcing offline mode makes from_pretrained() use only the local cache, which is all
    # any test needs -- every model this project uses (facebook/esm2_t6_8M_UR50D) is already
    # cached locally from real pipeline runs.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def activity() -> Dict[str, Any]:
    """
    :returns: A minimal, valid activity record — the shape every contract test starts from.
    """
    return {
        "activity_type": "MIC",
        "value_lower": 16.0,
        "value_upper": 16.0,
        "activity_unit": "ug/mL",
        "value_operator": "=",
        "value_raw": "16",
        "target_organism": "Escherichia coli",
        "target_group": "antibacterial",
    }


@pytest.fixture
def peptide(activity: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param activity: A valid activity, attached to the peptide.
    :returns: A minimal, valid peptide record.
    """
    return {
        "peptide_name": "magainin-2",
        "peptide_sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
        "biological_activities": [activity],
    }


@pytest.fixture
def document(peptide: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param peptide: A valid peptide, attached to the document.
    :returns: A minimal, valid extraction document.
    """
    return {"doc_id": "test-doc", "peptides": [peptide]}


@pytest.fixture
def extraction_dir(tmp_path: Any, document: Dict[str, Any]) -> str:
    """
    Builds an on-disk extraction tree of the shape the pipeline stages expect.

    :param tmp_path: pytest temporary directory.
    :param document: Document written into the ``pubmed`` subdirectory.
    :returns: Path to the directory root.
    """
    base = tmp_path / "extractions"
    (base / "pubmed").mkdir(parents=True)
    (base / "pubmed" / "doc1.json").write_text(json.dumps(document), encoding="utf-8")
    return str(base)


@pytest.fixture
def sequences() -> List[str]:
    """
    :returns: Well-characterized peptides used across the suite. Real sequences, so an assertion
        about charge or hydrophobicity can be checked against published values rather than
        against whatever the code currently returns.
    """
    return [
        "GIGKFLHSAKKFGKAFVGEIMNS",      # magainin-2, net positive
        "GIGAVLKVLTTGLPALISWIKRKRQQ",   # melittin, strongly cationic
        "ILPWKWPWWPWRR",                # indolicidin, Trp-rich
        "AAAAAAAAAA",                   # neutral homopolymer
        "DDDDEEEE",                     # strongly anionic
    ]
