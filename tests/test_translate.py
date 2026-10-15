"""Tests for the translation layer.

argostranslate pulls in torch via stanza, which is far too heavy to install
just to test routing logic — so the backend is stubbed with a fake module that
records what it was asked to do. That exercises the parts that are actually
ours: language normalisation, whether a pair is reachable, and the pivot hop
through English. Detection is tested against the real `langdetect` when it's
installed.
"""
from __future__ import annotations

import sys
import types

import pytest

from core import translate as tr


# ── fake argostranslate ──────────────────────────────────────────────────────

class _FakePackage:
    def __init__(self, from_code, to_code):
        self.from_code = from_code
        self.to_code = to_code
        self.downloaded = False

    def download(self):
        self.downloaded = True
        return f"/tmp/{self.from_code}_{self.to_code}.argosmodel"


@pytest.fixture
def argos(monkeypatch):
    """Install a fake `argostranslate` and return a handle to inspect it."""
    state = types.SimpleNamespace(
        installed=[], available=[], calls=[], installed_paths=[],
        index_ok=True, fail_translate=False,
    )

    package_mod = types.ModuleType("argostranslate.package")
    translate_mod = types.ModuleType("argostranslate.translate")
    root = types.ModuleType("argostranslate")

    def update_package_index():
        if not state.index_ok:
            raise RuntimeError("no network")

    def install_from_path(path):
        state.installed_paths.append(path)

    def translate_fn(text, src, dst):
        if state.fail_translate:
            raise RuntimeError("model exploded")
        state.calls.append((src, dst))
        return f"[{src}->{dst}]{text}"

    package_mod.get_installed_packages = lambda: list(state.installed)
    package_mod.get_available_packages = lambda: list(state.available)
    package_mod.update_package_index = update_package_index
    package_mod.install_from_path = install_from_path
    translate_mod.translate = translate_fn
    root.package = package_mod
    root.translate = translate_mod

    for name, mod in (("argostranslate", root),
                      ("argostranslate.package", package_mod),
                      ("argostranslate.translate", translate_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    # The catalogue refresh is cached across calls; clear it per test.
    tr._index_loaded.cache_clear()
    return state


def _install(state, *pairs):
    state.installed = [_FakePackage(a, b) for a, b in pairs]


# ── language normalisation ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("de", "de"), ("DE", "de"), ("German", "de"), ("german", "de"),
    ("deutsch", "de"), ("de-AT", "de"), ("de_AT", "de"),
    ("en", "en"), ("English", "en"), ("en-GB", "en"),
    ("fr", "fr"), ("français", "fr"), ("francais", "fr"),
    ("zh", "zh"), ("mandarin", "zh"), ("pt", "pt"), ("brazilian", "pt"),
    ("  Spanish  ", "es"),
])
def test_normalise_language(raw, expect):
    assert tr.normalise_language(raw) == expect


@pytest.mark.parametrize("raw", ["", None, "klingon", "xx", "banana", "123"])
def test_normalise_language_rejects_junk(raw):
    assert tr.normalise_language(raw) is None


def test_language_name_falls_back_to_the_code():
    assert tr.language_name("de") == "German"
    assert tr.language_name("qq") == "qq"


def test_every_alias_maps_to_a_known_code():
    for code in tr._NAME_TO_CODE.values():
        assert code in tr.LANGUAGE_NAMES, code


# ── availability probes ──────────────────────────────────────────────────────

def test_available_is_true_with_the_module_present(argos):
    assert tr.available() is True


def test_available_is_false_without_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "argostranslate", None)
    # `import argostranslate` with a None entry raises ImportError.
    assert tr.available() is False


def test_installed_pairs_reads_the_backend(argos):
    _install(argos, ("de", "en"), ("en", "fr"))
    assert tr.installed_pairs() == [("de", "en"), ("en", "fr")]
    assert tr.installed_languages() == {"de", "en", "fr"}


def test_installed_pairs_is_empty_without_the_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "argostranslate", None)
    assert tr.installed_pairs() == []


# ── routing ──────────────────────────────────────────────────────────────────

def test_direct_pair_is_translatable(argos):
    _install(argos, ("de", "en"))
    assert tr.can_translate("de", "en") is True


def test_missing_pair_is_not(argos):
    _install(argos, ("de", "en"))
    assert tr.can_translate("fr", "en") is False


def test_two_hop_route_through_english_counts(argos):
    """Argos models go via English, so de→fr is de→en→fr."""
    _install(argos, ("de", "en"), ("en", "fr"))
    assert tr.can_translate("de", "fr") is True


def test_no_route_when_only_one_leg_exists(argos):
    _install(argos, ("de", "en"))
    assert tr.can_translate("de", "fr") is False


def test_same_language_is_always_translatable(argos):
    assert tr.can_translate("de", "de") is True


# ── translation ──────────────────────────────────────────────────────────────

def test_direct_translation_takes_one_hop(argos):
    _install(argos, ("de", "en"))
    out = tr.translate_text("hallo", "de", "en")
    assert out == "[de->en]hallo"
    assert argos.calls == [("de", "en")]


def test_pivot_translation_takes_two_hops_in_order(argos):
    _install(argos, ("de", "en"), ("en", "fr"))
    out = tr.translate_text("hallo", "de", "fr")
    assert argos.calls == [("de", "en"), ("en", "fr")]
    assert out == "[en->fr][de->en]hallo"


def test_direct_pair_is_preferred_over_the_pivot(argos):
    """A de→fr model, if it exists, must not be ignored for the double hop."""
    _install(argos, ("de", "fr"), ("de", "en"), ("en", "fr"))
    tr.translate_text("hallo", "de", "fr")
    assert argos.calls == [("de", "fr")]


def test_same_language_is_a_no_op(argos):
    _install(argos, ("de", "en"))
    assert tr.translate_text("unchanged", "en", "en") == "unchanged"
    assert argos.calls == []


def test_missing_model_raises_a_readable_error(argos):
    _install(argos, ("de", "en"))
    with pytest.raises(tr.TranslateError, match="French"):
        tr.translate_text("bonjour", "fr", "en")


def test_backend_failure_is_wrapped(argos):
    _install(argos, ("de", "en"))
    argos.fail_translate = True
    with pytest.raises(tr.TranslateError, match="Translation failed"):
        tr.translate_text("hallo", "de", "en")


def test_oversized_input_is_refused_before_the_model_runs(argos):
    _install(argos, ("de", "en"))
    with pytest.raises(tr.TranslateError, match="limit"):
        tr.translate_text("x" * (tr.MAX_CHARS + 1), "de", "en")
    assert argos.calls == []


def test_translating_without_the_backend_explains_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "argostranslate", None)
    with pytest.raises(tr.TranslateError, match="isn't installed"):
        tr.translate_text("hallo", "de", "en")


# ── installing pairs ─────────────────────────────────────────────────────────

def test_install_pair_downloads_the_match(argos):
    argos.available = [_FakePackage("de", "en"), _FakePackage("fr", "en")]
    tr.install_pair("de", "en")
    assert argos.installed_paths == ["/tmp/de_en.argosmodel"]


def test_install_pair_rejects_an_unavailable_combination(argos):
    argos.available = [_FakePackage("de", "en")]
    with pytest.raises(tr.TranslateError, match="no .* model"):
        tr.install_pair("ja", "ko")


def test_install_pair_reports_a_dead_catalogue(argos):
    argos.index_ok = False
    with pytest.raises(tr.TranslateError, match="catalogue"):
        tr.install_pair("de", "en")


def test_available_pairs_survives_a_dead_catalogue(argos):
    argos.index_ok = False
    assert tr.available_pairs() == []


# ── detection ────────────────────────────────────────────────────────────────

langdetect = pytest.importorskip("langdetect", reason="langdetect not installed")


@pytest.mark.parametrize("text,expect", [
    ("Guten Morgen, wie geht es Ihnen heute?", "de"),
    ("Bonjour tout le monde, comment allez-vous aujourd'hui ?", "fr"),
    ("Hola, ¿qué tal estás hoy amigo mío?", "es"),
    ("Good morning everyone, how are you today?", "en"),
])
def test_detect_language(text, expect):
    assert tr.detect_language(text) == expect


@pytest.mark.parametrize("text", ["hi", "ok", "", "   ", "yes"])
def test_detection_refuses_samples_too_short_to_be_reliable(text):
    """langdetect will confidently call a two-word fragment Somali."""
    assert tr.detect_language(text) is None


def test_detection_is_deterministic():
    """langdetect is seeded internally; the same input must not flap."""
    text = "Der schnelle braune Fuchs springt über den faulen Hund."
    assert len({tr.detect_language(text) for _ in range(5)}) == 1


def test_detection_normalises_regional_codes():
    out = tr.detect_language("Bom dia, como você está hoje meu amigo?")
    assert out in ("pt", "es")          # close languages; both are 2-letter
    assert out is None or len(out) == 2
