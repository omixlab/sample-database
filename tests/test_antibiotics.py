"""Tests for :mod:`sample.antibiotics` — antibiotic name -> Morgan fingerprint for synergy."""
import json
from typing import Any, Dict, Optional

import numpy as np
import pytest

from sample import antibiotics as A


class TestMorganFingerprint:
    def test_parses_a_valid_smiles_to_a_bit_vector(self) -> None:
        fp = A.morgan_fingerprint("CCO", n_bits=64)  # ethanol
        assert fp is not None
        assert fp.shape == (64,)
        assert set(np.unique(fp)) <= {0.0, 1.0}
        assert fp.sum() > 0

    def test_invalid_smiles_returns_none(self) -> None:
        assert A.morgan_fingerprint("not a molecule!!!") is None

    def test_similar_molecules_share_more_bits_than_dissimilar_ones(self) -> None:
        # Two straight-chain alcohols vs an aromatic ring: the alcohols should be more alike.
        ethanol = A.morgan_fingerprint("CCO", n_bits=256)
        propanol = A.morgan_fingerprint("CCCO", n_bits=256)
        benzene = A.morgan_fingerprint("c1ccccc1", n_bits=256)

        def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            return inter / union if union else 0.0

        assert tanimoto(ethanol, propanol) > tanimoto(ethanol, benzene)


class TestFingerprintColumns:
    def test_length_matches_the_bit_count(self) -> None:
        assert A.fingerprint_columns(n_bits=32) == [f"abx_fp_{i}" for i in range(32)]


class TestBuildFingerprints:
    @pytest.fixture
    def cache_file(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
        path = str(tmp_path / "abx_cache.json")
        monkeypatch.setenv("SAMPLE_ANTIBIOTIC_CACHE", path)
        return path

    def test_resolves_and_fingerprints_and_caches(
            self, cache_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_resolve(name: str, timeout: float = 15.0) -> Optional[str]:
            calls.append(name)
            return {"Gentamicin": "CCO", "Ciprofloxacin": "CCCO"}.get(name)

        monkeypatch.setattr(A, "resolve_smiles", fake_resolve)
        monkeypatch.setattr(A.time, "sleep", lambda s: None)

        fps = A.build_fingerprints(["Gentamicin", "Ciprofloxacin", "Gentamicin"], n_bits=64)
        assert fps["Gentamicin"].sum() > 0 and fps["Ciprofloxacin"].sum() > 0
        # Gentamicin appears twice but is resolved once (dedup) and cached.
        assert calls.count("Gentamicin") == 1
        with open(cache_file, encoding="utf-8") as fh:
            assert "Gentamicin" in json.load(fh)

    def test_unresolvable_name_gets_a_zero_fingerprint(
            self, cache_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(A, "resolve_smiles", lambda name, timeout=15.0: None)
        monkeypatch.setattr(A.time, "sleep", lambda s: None)
        fps = A.build_fingerprints(["MysteryDrug"], n_bits=64)
        assert fps["MysteryDrug"].sum() == 0.0
        assert fps["MysteryDrug"].shape == (64,)

    def test_offline_mode_uses_only_the_cache(
            self, cache_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pre-seed a cache; use_network=False must not call resolve_smiles at all.
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump({"Gentamicin": "CCO"}, fh)

        def _boom(*a: Any, **k: Any) -> None:
            raise AssertionError("resolve_smiles must not be called offline")

        monkeypatch.setattr(A, "resolve_smiles", _boom)
        fps = A.build_fingerprints(["Gentamicin", "Uncached"], n_bits=64, use_network=False)
        assert fps["Gentamicin"].sum() > 0     # from the cache
        assert fps["Uncached"].sum() == 0.0    # unknown, zero vector, no lookup

    def test_blank_names_are_ignored(self, cache_file: str,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(A, "resolve_smiles", lambda name, timeout=15.0: None)
        monkeypatch.setattr(A.time, "sleep", lambda s: None)
        fps = A.build_fingerprints(["", "  ", None], n_bits=64)
        assert fps == {}


class TestResolveSmiles:
    def test_parses_the_connectivity_smiles_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        payload = {"PropertyTable": {"Properties": [
            {"CID": 1, "ConnectivitySMILES": "CCO"}]}}

        monkeypatch.setattr(A.urllib.request, "urlopen",
                            lambda url, timeout=15.0: io.StringIO(json.dumps(payload)))
        assert A.resolve_smiles("Ethanol") == "CCO"

    def test_falls_back_to_canonical_smiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        payload = {"PropertyTable": {"Properties": [{"CID": 1, "CanonicalSMILES": "CCO"}]}}
        monkeypatch.setattr(A.urllib.request, "urlopen",
                            lambda url, timeout=15.0: io.StringIO(json.dumps(payload)))
        assert A.resolve_smiles("Ethanol") == "CCO"

    def test_network_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(url: str, timeout: float = 15.0) -> None:
            raise OSError("network down")
        monkeypatch.setattr(A.urllib.request, "urlopen", _boom)
        assert A.resolve_smiles("Ethanol") is None

    def test_empty_name_is_not_queried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(A.urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("queried")))
        assert A.resolve_smiles("   ") is None
