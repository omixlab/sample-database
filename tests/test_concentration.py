"""
Tests for :mod:`sample.concentration` -- the shared, dependency-free molarity-conversion path used
both by ml_models.py's activity threshold and regression targets, and by serialize.py's load-sql
step (value_mM/value_ug_ml/potency columns), so there is exactly one place a unit-conversion
mistake could hide, not two independently-maintained copies. Kept in its own module (no imports
beyond the standard library) specifically so importing it can never pull in a dependency -- like
sample.nlp's ``ahocorasick`` -- that isn't installed in every environment that needs a conversion.
"""
import pytest

from sample import concentration as C


class TestToMicromolar:
    def test_molar_units_convert_directly(self) -> None:
        assert C.to_micromolar(3000.0, "nM", None) == pytest.approx(3.0)
        assert C.to_micromolar(3.0, "uM", None) == pytest.approx(3.0)
        assert C.to_micromolar(0.003, "mM", None) == pytest.approx(3.0)

    def test_unicode_micro_glyphs_are_recognized(self) -> None:
        # U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU both appear in this corpus's
        # activity_unit column and must resolve the same way.
        assert C.to_micromolar(2.0, "µM", None) == pytest.approx(2.0)
        assert C.to_micromolar(2.0, "μM", None) == pytest.approx(2.0)

    def test_mass_unit_needs_molecular_weight(self) -> None:
        # 10 ug/mL of a 2000 g/mol peptide: (10 * 1000) / 2000 = 5 uM.
        assert C.to_micromolar(10.0, "ug/mL", 2000.0) == pytest.approx(5.0)
        assert C.to_micromolar(10.0, "ug/mL", None) is None
        assert C.to_micromolar(10.0, "ug/mL", 0.0) is None

    def test_non_concentration_units_are_not_convertible(self) -> None:
        # A percentage or a dose (mg/kg) is not a concentration; there is nothing to convert.
        assert C.to_micromolar(50.0, "%", 2000.0) is None
        assert C.to_micromolar(5.0, "mg/kg", 2000.0) is None


class TestConcentrationConversions:
    def test_fills_every_figure_from_a_molar_reading(self) -> None:
        # A molar-unit reading gets uM/mM/potency without needing molecular_weight at all; only
        # ug_ml (mass) needs it, and stays None when it is unavailable.
        out = C.concentration_conversions(5.0, "uM", None)
        assert out["uM"] == pytest.approx(5.0)
        assert out["mM"] == pytest.approx(0.005)
        assert out["ug_ml"] is None
        assert out["potency"] == pytest.approx(5.301, abs=1e-3)  # -log10(5e-6)

    def test_fills_every_figure_from_a_mass_reading(self) -> None:
        out = C.concentration_conversions(10.0, "ug/mL", 2000.0)
        assert out["uM"] == pytest.approx(5.0)
        assert out["mM"] == pytest.approx(0.005)
        assert out["ug_ml"] == pytest.approx(10.0)
        assert out["potency"] == pytest.approx(5.301, abs=1e-3)

    def test_unrecognized_unit_returns_all_none(self) -> None:
        out = C.concentration_conversions(50.0, "%", 2000.0)
        assert out == {"uM": None, "mM": None, "ug_ml": None, "potency": None}

    def test_module_imports_nothing_beyond_the_standard_library(self) -> None:
        # The whole point of splitting this out of sample.preprocessing: importing it must never
        # risk pulling in a dependency (ahocorasick, via sample.nlp) that some caller's environment
        # does not have installed. Checked statically against the source rather than sys.modules,
        # since another test module may have already imported sample.nlp earlier in the same run,
        # which would make a sys.modules diff pass even if this module regressed.
        import ast
        import inspect

        source = inspect.getsource(C)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(name.startswith("sample.") for name in imported), imported
