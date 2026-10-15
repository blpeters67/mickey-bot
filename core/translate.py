"""
Local machine translation, wrapping argostranslate.

Everything runs on the host: no API key, no per-character cost, no third party
seeing the text. The trade-off is that each language *pair* is a separate
~100 MB model that has to be installed before it can be used, so a lot of this
module is about telling the user clearly what's available and what isn't.

Both argostranslate and langdetect are optional. Without them the commands say
so and the rest of the bot is unaffected — the same pattern as faster-whisper,
tesseract and rembg.
"""
from __future__ import annotations

import functools

from .logging_setup import get_logger

log = get_logger(__name__)

# Argos models are trained to and from English, so anything else routes
# through it (de→fr is de→en→fr). That's how the upstream project works;
# quality degrades a little on the double hop.
PIVOT = "en"

MAX_CHARS = 4000

# Friendly names for the codes people are most likely to hit, so the bot can
# say "German" instead of "de". Not exhaustive — unknown codes just print raw.
LANGUAGE_NAMES = {
    "ar": "Arabic", "az": "Azerbaijani", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "ga": "Irish", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "lt": "Lithuanian", "lv": "Latvian", "ms": "Malay", "nb": "Norwegian",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sq": "Albanian",
    "sv": "Swedish", "th": "Thai", "tl": "Tagalog", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
    "zt": "Chinese (traditional)",
}

# Names and non-ISO spellings people type instead of the code.
_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}
_NAME_TO_CODE.update({
    "english": "en", "german": "de", "deutsch": "de", "spanish": "es",
    "español": "es", "espanol": "es", "french": "fr", "français": "fr",
    "francais": "fr", "japanese": "ja", "chinese": "zh", "mandarin": "zh",
    "portuguese": "pt", "brazilian": "pt", "russian": "ru", "italian": "it",
    "korean": "ko", "dutch": "nl", "polish": "pl", "turkish": "tr",
    "arabic": "ar", "hindi": "hi", "ukrainian": "uk", "swedish": "sv",
    "norwegian": "nb", "danish": "da", "finnish": "fi", "greek": "el",
    "hebrew": "he", "czech": "cs", "vietnamese": "vi", "thai": "th",
    "indonesian": "id", "romanian": "ro", "hungarian": "hu",
})


class TranslateError(Exception):
    """Raised with a message meant to be shown to the user verbatim."""


def available() -> bool:
    try:
        import argostranslate  # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False


def detection_available() -> bool:
    try:
        import langdetect  # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def normalise_language(text: str | None) -> str | None:
    """'German' / 'de' / 'DE' / 'deutsch' → 'de'. None if unrecognised."""
    if not text:
        return None
    t = text.strip().lower().replace("_", "-")
    # "en-GB" → "en"; argos models are language-level, not regional.
    base = t.split("-")[0]
    if t in _NAME_TO_CODE:
        return _NAME_TO_CODE[t]
    if base in LANGUAGE_NAMES:
        return base
    if base in _NAME_TO_CODE:
        return _NAME_TO_CODE[base]
    return None


def installed_pairs() -> list[tuple[str, str]]:
    """Language pairs with a model on this host, as (from, to) codes."""
    if not available():
        return []
    try:
        import argostranslate.package as package
        return sorted((p.from_code, p.to_code)
                      for p in package.get_installed_packages())
    except Exception:
        log.warning("couldn't list installed argos packages", exc_info=True)
        return []


def installed_languages() -> set[str]:
    codes: set[str] = set()
    for a, b in installed_pairs():
        codes.update((a, b))
    return codes


def can_translate(src: str, dst: str) -> bool:
    """True if a direct model exists, or a two-hop route through English."""
    if src == dst:
        return True
    pairs = set(installed_pairs())
    if (src, dst) in pairs:
        return True
    return (src, PIVOT) in pairs and (PIVOT, dst) in pairs


@functools.lru_cache(maxsize=1)
def _index_loaded() -> bool:
    """Refresh argos's catalogue of downloadable models. Cached — it's a
    network call and the catalogue barely changes."""
    try:
        import argostranslate.package as package
        package.update_package_index()
        return True
    except Exception:
        log.warning("couldn't refresh the argos package index", exc_info=True)
        return False


def available_pairs() -> list[tuple[str, str]]:
    """Pairs that *could* be installed, from argos's online catalogue."""
    if not available() or not _index_loaded():
        return []
    try:
        import argostranslate.package as package
        return sorted((p.from_code, p.to_code)
                      for p in package.get_available_packages())
    except Exception:
        return []


def install_pair(src: str, dst: str) -> None:
    """Download and install one language pair. Blocking — call in a thread.

    Raises :class:`TranslateError` with a user-facing message on failure.
    """
    if not available():
        raise TranslateError("argostranslate isn't installed on this host.")
    if not _index_loaded():
        raise TranslateError(
            "Couldn't reach the model catalogue — check the host's network.")
    import argostranslate.package as package

    match = next(
        (p for p in package.get_available_packages()
         if p.from_code == src and p.to_code == dst), None)
    if match is None:
        raise TranslateError(
            f"There's no {language_name(src)} → {language_name(dst)} model "
            "available. Argos models mostly go to and from English.")
    try:
        package.install_from_path(match.download())
    except Exception as e:
        raise TranslateError(f"Download failed: {e}") from e


def detect_language(text: str) -> str | None:
    """Best guess at the language of `text`, or None.

    langdetect is statistical and needs a reasonable sample — it will happily
    call a three-word fragment Somali — so very short inputs are refused
    rather than guessed at.
    """
    if not detection_available() or len(text.strip()) < 12:
        return None
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0        # deterministic across runs
        code = detect(text)
    except Exception:
        return None
    return normalise_language(code) or (code.split("-")[0] if code else None)


def translate_text(text: str, src: str, dst: str) -> str:
    """Translate, routing through English when there's no direct model.

    Blocking and CPU-bound — call it in a thread.
    """
    if not available():
        raise TranslateError("argostranslate isn't installed on this host.")
    if len(text) > MAX_CHARS:
        raise TranslateError(
            f"That's {len(text):,} characters — the limit is {MAX_CHARS:,}.")
    if src == dst:
        return text

    import argostranslate.translate as translate_mod

    pairs = set(installed_pairs())
    try:
        if (src, dst) in pairs:
            return translate_mod.translate(text, src, dst)
        if (src, PIVOT) in pairs and (PIVOT, dst) in pairs:
            # Two hops. Quality drops a little; better than refusing.
            return translate_mod.translate(
                translate_mod.translate(text, src, PIVOT), PIVOT, dst)
    except Exception as e:
        log.exception("argos translation failed %s→%s", src, dst)
        raise TranslateError(f"Translation failed: {e}") from e

    raise TranslateError(
        f"No {language_name(src)} → {language_name(dst)} model is installed.")
