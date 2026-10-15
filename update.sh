#!/usr/bin/env bash
#
# update.sh — upgrade every Python dependency to the latest version allowed
# by requirements.txt, in the existing ./venv.
#
# This is the companion to install.sh: install.sh *creates* the environment,
# update.sh *bumps* it. It does NOT rewrite requirements.txt — it just pulls
# the newest releases that satisfy the specifiers already in that file
# (e.g. `yt-dlp>=2024.10.7` grabs whatever the current yt-dlp is; the
# `==0.5.2a179` pin on discord-ext-voice-recv stays put).
#
# The optional extras (audio-separator, faster-whisper, argostranslate, rembg,
# qrcode, …) aren't in requirements.txt — install.sh installs them behind
# flags — so this script upgrades whichever of them are *already present* in
# the venv. That's deliberate: "upgrade what you have" needs no flags and can
# never surprise you by pulling several GB of torch you didn't ask for. To add
# an extra you don't have yet, use install.sh.
#
# Usage:
#   ./update.sh                    # upgrade everything in ./venv
#   ./update.sh --core             # skip the optional extras
#   PYTHON=python3.13 ./update.sh  # force a specific interpreter
#
# Safe to re-run.

set -euo pipefail

# Operate from the repo root (this script's own directory).
cd "$(dirname "$0")"

VENV="venv"
PY="${PYTHON:-python3.12}"
WANT_EXTRAS=1
FAILED_GROUPS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --core|--no-extras) WANT_EXTRAS=0 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "!! Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

echo "==> Music bot dependency update"

# --- 1. Require an existing venv ------------------------------------------
# Updating implies the deps are already installed; if the venv is missing,
# the user wants install.sh, not this.
if [ ! -x "$VENV/bin/python" ]; then
    echo "!! No venv at ./$VENV — run ./install.sh first to create it." >&2
    exit 1
fi

VPIP="$VENV/bin/pip"
VPY="$VENV/bin/python"
PYVER="$("$VPY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "    Using ./$VENV (Python $PYVER)"

# stdlib `audioop` was removed in 3.13; voice-recv + the clip mixer need it.
NEED_AUDIOOP_LTS=0
case "$PYVER" in
    3.13|3.14|3.15|3.16) NEED_AUDIOOP_LTS=1 ;;
esac

# --- 1b. Heal a broken pip -------------------------------------------------
# A half-finished `pip install --upgrade pip` can leave pip's *vendored*
# resolver in a mismatched state (ImportError on pip._vendor.resolvelib).
# pip then can't install anything — not even a new pip — so we re-bootstrap
# it from a clean source. `pip --version` still works when this is broken, so
# we probe the exact import that fails instead.
_pip_ok() { "$VPY" -c 'import pip._internal.resolution.resolvelib.resolver' >/dev/null 2>&1; }
if ! _pip_ok; then
    echo "!! venv pip is broken (vendored resolver mismatch) — re-bootstrapping"
    "$VPY" -m ensurepip --upgrade || true       # offline: stdlib-bundled pip
    if ! _pip_ok; then
        echo "!! ensurepip didn't fix it — force-reinstalling pip via get-pip.py"
        tmp="$(mktemp)"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$tmp"
        "$VPY" "$tmp" --force-reinstall
        rm -f "$tmp"
    fi
    _pip_ok || { echo "!! pip is still broken — fix it manually and re-run." >&2; exit 1; }
    echo "    pip repaired."
fi

# --- 2. Upgrade pip, then every requirement -------------------------------
"$VPY" -m pip install --upgrade pip

echo "==> Upgrading dependencies (requirements.txt)"
# --upgrade-strategy eager also pulls newer transitive deps, so nothing is
# left pinned to a stale sub-dependency.
"$VPIP" install --upgrade --upgrade-strategy eager -r requirements.txt

if [ "$NEED_AUDIOOP_LTS" = "1" ]; then
    echo "==> Python $PYVER: upgrading audioop-lts shim"
    "$VPIP" install --upgrade "audioop-lts>=0.2"
fi

# --- 2b. Upgrade whichever optional extras are already installed -----------
# Each group is upgraded on its own so one failure doesn't strand the rest —
# an extra that won't build is a missing command, not a broken bot.
has_module() { "$VPY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" >/dev/null 2>&1; }

upgrade_group() {
    local label="$1"; shift
    echo "==> Upgrading $label"
    if "$VPIP" install --upgrade "$@"; then
        echo "    ok   $label"
    else
        echo "!!   $label FAILED to upgrade — the previous version is still installed."
        FAILED_GROUPS="$FAILED_GROUPS
      - $label"
    fi
}

if [ "$WANT_EXTRAS" = "1" ]; then
    echo "==> Checking optional extras"
    found=0

    if has_module audio_separator; then
        found=1
        # Keep whichever acceleration flavour is already in place: installing
        # the [cpu] extra over a working CUDA setup would quietly downgrade
        # separation from seconds to minutes per track.
        if has_module onnxruntime_gpu || "$VPIP" show onnxruntime-gpu >/dev/null 2>&1; then
            upgrade_group "karaoke: AI vocal separation (CUDA)" "audio-separator[gpu]"
        else
            upgrade_group "karaoke: AI vocal separation (CPU)" "audio-separator[cpu]"
        fi
    fi
    # Full `if` blocks rather than `cond && { … }`: the short form returns
    # non-zero when the test fails, which `set -e` turns into a silent exit as
    # soon as one of these ends up last in a block.
    if has_module faster_whisper; then
        found=1; upgrade_group "-transcribe (faster-whisper)" "faster-whisper"
    fi
    if has_module argostranslate; then
        found=1; upgrade_group "-translate (argostranslate)" "argostranslate"
    fi
    if has_module langdetect; then
        found=1; upgrade_group "language detection (langdetect)" "langdetect"
    fi
    if has_module rembg; then
        found=1; upgrade_group "-nobg (rembg)" "rembg" "onnxruntime"
    fi
    if has_module qrcode || has_module zxingcpp; then
        found=1; upgrade_group "-qr (qrcode + zxing-cpp)" "qrcode" "zxing-cpp"
    fi

    if [ "$found" = "0" ]; then
        echo "    none installed — add them with ./install.sh --all"
    fi
fi

# --- 3. Verify the clip-critical imports still load -----------------------
# An upgrade can break a pre-release lib (voice-recv warns of exactly this),
# so fail loudly here rather than discovering it at runtime.
echo "==> Verifying imports"
"$VPY" - <<'PY'
import importlib, importlib.util, sys
ok = True
for name in ("discord", "discord.ext.voice_recv", "davey", "nacl"):
    try:
        mod = importlib.import_module(name)
        print(f"   ok    {name} {getattr(mod, '__version__', '')}".rstrip())
    except Exception as exc:
        ok = False
        print(f"   FAIL  {name}: {type(exc).__name__}: {exc}")
try:
    importlib.import_module("cogs.recording")
    print("   ok    cogs.recording (clip command)")
except Exception as exc:
    ok = False
    print(f"   FAIL  cogs.recording: {type(exc).__name__}: {exc}")

# Optional extras: an upgrade breaking one of these shouldn't fail the run
# (the bot still starts), but it must not pass silently either — a torch bump
# that breaks audio-separator would otherwise only surface at karaoke time.
for module, what in (
    ("audio_separator", "karaoke"),
    ("faster_whisper",  "-transcribe"),
    ("argostranslate",  "-translate"),
    ("rembg",           "-nobg"),
):
    if importlib.util.find_spec(module) is None:
        continue
    try:
        importlib.import_module(module)
        print(f"   ok    {module} ({what})")
    except Exception as exc:
        print(f"   WARN  {module} ({what}) no longer imports after the upgrade: "
              f"{type(exc).__name__}: {exc}")
sys.exit(0 if ok else 1)
PY

# --- 4. Show what landed ---------------------------------------------------
echo "==> Installed versions"
"$VPIP" list --format=columns | grep -iE \
    'discord|davey|yt-dlp|pynacl|spotipy|dotenv|aiosqlite|aiohttp|audioop|pytest|audio-separator|faster-whisper|argostranslate|rembg|onnxruntime|torch|qrcode|zxing|langdetect' \
    || true

echo
if [ -n "$FAILED_GROUPS" ]; then
    echo "!! Some optional groups did not upgrade:$FAILED_GROUPS"
    echo "   Their previous versions are still installed, so nothing is broken."
    echo
fi
echo "==> Done. Restart the bot to pick up the new versions:"
echo "    $VPY bot.py"
