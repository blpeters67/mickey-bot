#!/usr/bin/env bash
#
# install.sh — set up the Linux deploy environment for the music bot.
#
# Creates (or reuses) a Python virtualenv in ./venv and installs the bot's
# dependencies: the core set always, the tiny optional extras by default, and
# the heavy AI stack only when you ask for it.
#
# Why the split: audio-separator, argostranslate, faster-whisper and rembg all
# pull torch/ctranslate2/onnxruntime. That's 3-5 GB unpacked and several
# minutes of download. Forcing it on every run would make re-running this
# script — which you're meant to be able to do freely, e.g. to repair pip —
# an expensive operation, and would ambush anyone on a small VPS. The tiny
# extras (qrcode, zxing-cpp, langdetect) have no such downside, so they're in
# by default.
#
# Usage:
#   ./install.sh                    core + tiny extras (qr, langdetect)
#   ./install.sh --all              ...plus every AI extra (needs ~5 GB)
#   ./install.sh --karaoke          ...plus AI vocal separation only
#   ./install.sh --core             core only, nothing optional
#
#   Individual extras:  --karaoke --whisper --translate --rembg --qr
#   Modifiers:          --gpu   install CUDA builds instead of CPU
#                       --yes   assume yes (unattended: no prompts, runs apt)
#                       --no-apt  never touch apt, just report what's missing
#   Interpreter:        PYTHON=python3.13 ./install.sh
#
# Safe to re-run — it upgrades in place.

set -euo pipefail

# Operate from the repo root (this script's own directory) no matter where
# it's invoked from.
cd "$(dirname "$0")"

VENV="venv"
PY="${PYTHON:-python3.12}"

# --- 0. Arguments ----------------------------------------------------------
WANT_QR=1            # tiny, pure-pip — on unless --core
WANT_LANGDETECT=1    # ditto
WANT_KARAOKE=0
WANT_WHISPER=0
WANT_TRANSLATE=0
WANT_REMBG=0
USE_GPU=0
ASSUME_YES=0
ALLOW_APT=1
FAILED_GROUPS=""

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            WANT_KARAOKE=1; WANT_WHISPER=1; WANT_TRANSLATE=1; WANT_REMBG=1
            WANT_QR=1; WANT_LANGDETECT=1 ;;
        --core)
            WANT_QR=0; WANT_LANGDETECT=0
            WANT_KARAOKE=0; WANT_WHISPER=0; WANT_TRANSLATE=0; WANT_REMBG=0 ;;
        --karaoke|--separation) WANT_KARAOKE=1 ;;
        --whisper|--transcribe) WANT_WHISPER=1 ;;
        --translate)            WANT_TRANSLATE=1; WANT_LANGDETECT=1 ;;
        --rembg|--nobg)         WANT_REMBG=1 ;;
        --qr)                   WANT_QR=1 ;;
        --gpu|--cuda)           USE_GPU=1 ;;
        --yes|-y)               ASSUME_YES=1 ;;
        --no-apt)               ALLOW_APT=0 ;;
        -h|--help)              usage ;;
        *) echo "!! Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

HEAVY=$(( WANT_KARAOKE + WANT_WHISPER + WANT_TRANSLATE + WANT_REMBG ))

echo "==> Music bot install"

# Ask a yes/no question. Never blocks on a non-interactive run (cron, CI, a
# piped installer) — those get the safe answer instead of hanging forever.
confirm() {
    if [ "$ASSUME_YES" = "1" ]; then return 0; fi
    if [ ! -t 0 ]; then
        echo "    (non-interactive — skipping. Re-run with --yes to accept.)"
        return 1
    fi
    printf "    %s [y/N] " "$1"
    local reply=""
    read -r reply || return 1
    case "$reply" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

avail_mb() { df -Pm . 2>/dev/null | awk 'NR==2 {print $4}'; }

# --- 1. Pick a Python interpreter -----------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "!! '$PY' not found on PATH — falling back to python3"
    PY=python3
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "!! No usable Python found. Install Python 3.12 and re-run." >&2
    exit 1
fi
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "    Using $PY (Python $PYVER)"

# stdlib `audioop` was removed in 3.13; voice-recv + the clip mixer need it.
# On 3.13+ pull the audioop-lts shim. (Deploy is meant to be 3.12, where it's
# built in — this just keeps the script working if the host is newer.)
NEED_AUDIOOP_LTS=0
case "$PYVER" in
    3.13|3.14|3.15|3.16) NEED_AUDIOOP_LTS=1 ;;
esac

# --- 2. virtualenv ---------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    echo "==> Creating venv at ./$VENV"
    "$PY" -m venv "$VENV"
else
    echo "==> Reusing existing venv at ./$VENV"
fi

# Use the venv's executables directly (no need to `source activate`).
VPIP="$VENV/bin/pip"
VPY="$VENV/bin/python"

# --- 2b. Make sure the venv's pip actually works ---------------------------
# A half-finished `pip install --upgrade pip` leaves the vendored resolvelib
# mismatched, and every later pip call dies with something like
#   ImportError: cannot import name 'RequirementInformation'
#     from 'pip._vendor.resolvelib.structs'
# Upgrading pip can't fix that, because running pip is the broken part. So:
# test it first, then repair with ensurepip, then rebuild the venv outright.
# `pip --version` is NOT enough: the vendored resolver is only imported once
# an actual install starts, so a corrupted one reports a healthy version and
# then dies on the first `pip install`. Import the resolver directly — that's
# the module at the end of the traceback.
pip_works() {
    "$VPY" -m pip --version >/dev/null 2>&1 &&
    "$VPY" -c "import pip._internal.resolution.resolvelib.resolver" >/dev/null 2>&1
}

if ! pip_works; then
    echo "!! The venv's pip is broken — reinstalling it in place"
    # ensurepip on its own isn't enough: it reads the leftover dist-info,
    # decides pip is "already satisfied", and changes nothing. Clearing the
    # package *and* its metadata first makes it unpack its bundled wheel for
    # real — which repairs pip without throwing away everything else that's
    # installed in the venv.
    rm -rf "$VENV"/lib/python*/site-packages/pip \
           "$VENV"/lib/python*/site-packages/pip-*.dist-info
    "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
    pip_works && echo "    Repaired — the rest of the venv is untouched."
fi

if ! pip_works; then
    echo "!! Still broken — rebuilding the venv from scratch"
    # Keep the old one until the new one is proven, so a failure here doesn't
    # leave the user with no environment at all.
    rm -rf "$VENV.broken"
    mv "$VENV" "$VENV.broken"
    "$PY" -m venv "$VENV"
    if pip_works; then
        echo "    Rebuilt. The old one is at ./$VENV.broken — delete it once"
        echo "    the bot starts cleanly:  rm -rf '$VENV.broken'"
    else
        echo "!! A fresh venv still has no working pip. Your system Python's" >&2
        echo "   pip/ensurepip is damaged — on Debian/Ubuntu:" >&2
        echo "     sudo apt-get install --reinstall python3-venv python3-pip" >&2
        mv "$VENV" "$VENV.failed" 2>/dev/null || true
        mv "$VENV.broken" "$VENV" 2>/dev/null || true
        exit 1
    fi
fi

# Now that pip runs, upgrading it is safe. Not fatal if it fails — an older
# pip installs these dependencies perfectly well.
"$VPY" -m pip install --upgrade pip || echo "!! pip self-upgrade failed (continuing)"

# Install one optional group. A failure is reported and collected rather than
# aborting: a missing extra costs you one command, but a half-installed venv
# because rembg couldn't build costs you the whole bot.
pip_group() {
    local label="$1"; shift
    echo "==> Installing $label"
    if "$VPIP" install "$@"; then
        echo "    ok   $label"
    else
        echo "!!   $label FAILED — the bot runs without it; see the log above."
        FAILED_GROUPS="$FAILED_GROUPS
      - $label"
    fi
}

# --- 3. Core dependencies --------------------------------------------------
echo "==> Installing core dependencies (requirements.txt)"
"$VPIP" install -r requirements.txt
if [ "$NEED_AUDIOOP_LTS" = "1" ]; then
    echo "==> Python $PYVER: installing audioop-lts shim"
    "$VPIP" install "audioop-lts>=0.2"
fi

# --- 4. Small optional extras ----------------------------------------------
# Pure-pip, no system libraries, a few hundred KB between them. There's no
# real reason to leave these out, which is why they're the default.
if [ "$WANT_QR" = "1" ]; then
    pip_group "-qr (qrcode + zxing-cpp)" "qrcode>=7.4" "zxing-cpp>=2.2.0"
fi
if [ "$WANT_LANGDETECT" = "1" ]; then
    pip_group "language detection (langdetect)" "langdetect>=1.0.9"
fi

# --- 5. Heavy AI extras ----------------------------------------------------
if [ "$HEAVY" -gt 0 ]; then
    echo
    echo "==> AI extras requested — these are large:"
    # Written as full `if` blocks rather than `[ … ] && …` on purpose: the
    # short form returns non-zero when the test fails, which is harmless here
    # but bites the moment someone moves one of these lines to the end of a
    # function, where `set -e` then kills the script.
    NEED=1000
    if [ "$WANT_KARAOKE" = "1" ]; then
        echo "      karaoke      audio-separator + torch   ~2.5 GB"
        NEED=$(( NEED + 2500 ))
    fi
    if [ "$WANT_WHISPER" = "1" ]; then
        echo "      transcribe   faster-whisper            ~0.5 GB"
        NEED=$(( NEED + 500 ))
    fi
    if [ "$WANT_TRANSLATE" = "1" ]; then
        echo "      translate    argostranslate            ~1.5 GB"
        NEED=$(( NEED + 1500 ))
    fi
    if [ "$WANT_REMBG" = "1" ]; then
        echo "      nobg         rembg + onnxruntime       ~0.4 GB"
        NEED=$(( NEED + 400 ))
    fi
    echo "    Models download separately on first use of each command."

    # Running out of disk halfway through a torch install leaves a wrecked
    # venv, so check before starting rather than after.
    AVAIL="$(avail_mb || echo 0)"
    if [ -n "$AVAIL" ] && [ "$AVAIL" -gt 0 ]; then
        echo "    Disk here: ${AVAIL} MB free, roughly ${NEED} MB needed."
        if [ "$AVAIL" -lt "$NEED" ]; then
            echo "!!  That's probably not enough."
            if ! confirm "Try anyway?"; then
                echo "    Skipping the AI extras. Free some space and re-run."
                HEAVY=0
            fi
        fi
    fi
fi

if [ "$HEAVY" -gt 0 ]; then
    if [ "$WANT_KARAOKE" = "1" ]; then
        # The [cpu]/[gpu] extras select the right onnxruntime build. On a
        # CPU-only host the GPU variant installs fine and then silently never
        # uses the GPU, so picking deliberately matters.
        if [ "$USE_GPU" = "1" ]; then
            pip_group "karaoke: AI vocal separation (CUDA)" "audio-separator[gpu]>=0.24"
        else
            pip_group "karaoke: AI vocal separation (CPU)" "audio-separator[cpu]>=0.24"
        fi
    fi
    if [ "$WANT_WHISPER" = "1" ]; then
        pip_group "-transcribe + voice control (faster-whisper)" "faster-whisper>=1.0.0"
    fi
    if [ "$WANT_TRANSLATE" = "1" ]; then
        pip_group "-translate (argostranslate)" "argostranslate>=1.9"
    fi
    if [ "$WANT_REMBG" = "1" ]; then
        pip_group "-nobg (rembg)" "rembg>=2.0.50" "onnxruntime>=1.16.0"
    fi
fi

# --- 6. Verify the clip-critical imports actually load ---------------------
# A failed cog import is otherwise silent (the bot just lacks the command), so
# fail loudly here if any of the voice/DAVE pieces didn't land.
echo "==> Verifying imports"
"$VPY" - <<'PY'
import importlib, importlib.util, shutil, sys
ok = True
for name in ("discord", "discord.ext.voice_recv", "davey", "nacl", "PIL"):
    try:
        mod = importlib.import_module(name)
        print(f"   ok    {name} {getattr(mod, '__version__', '')}".rstrip())
    except Exception as exc:
        ok = False
        print(f"   FAIL  {name}: {type(exc).__name__}: {exc}")
import discord
print(f"   discord.py == {discord.__version__}")
# Sanity-check the actual cog the clip command lives in.
try:
    importlib.import_module("cogs.recording")
    print("   ok    cogs.recording (clip command)")
except Exception as exc:
    ok = False
    print(f"   FAIL  cogs.recording: {type(exc).__name__}: {exc}")

# Optional extras. Absent is fine — each command says so and the bot runs
# regardless — but knowing which are on saves a round of "why doesn't X work".
print("   optional extras:")
extras = [
    ("audio_separator", "karaoke (AI vocal removal)", "./install.sh --karaoke"),
    ("faster_whisper",  "-transcribe, voice control", "./install.sh --whisper"),
    ("rembg",           "-nobg",                      "./install.sh --rembg"),
    ("qrcode",          "-qr (generate)",             "./install.sh --qr"),
    ("zxingcpp",        "-qr (read)",                 "./install.sh --qr"),
    ("argostranslate",  "-translate",                 "./install.sh --translate"),
    ("langdetect",      "-translate (auto-detect)",   "./install.sh --translate"),
]
for module, what, how in extras:
    try:
        importlib.import_module(module)
        print(f"     on   {what}")
    except Exception:
        print(f"     off  {what:28} — {how}")
if shutil.which("tesseract"):
    print("     on   -ocr")
else:
    print("     off  -ocr                          — sudo apt install tesseract-ocr")

# audio-separator ships a console script; the karaoke cog prefers it over
# `python -m`, so confirm it landed on PATH inside the venv.
if importlib.util.find_spec("audio_separator") is not None:
    print("     ok   audio-separator CLI"
          if shutil.which("audio-separator")
          else "     note audio-separator installed but its CLI isn't on PATH "
               "(the bot falls back to `python -m`)")
sys.exit(0 if ok else 1)
PY

# --- 7. System tools the bot shells out to ---------------------------------
echo "==> Checking system tools"
missing=""
command -v ffmpeg >/dev/null 2>&1 || missing="$missing ffmpeg"
# discord.py loads libopus at runtime for voice; just check the shared lib.
if ! { ldconfig -p 2>/dev/null | grep -q 'libopus\.so'; }; then
    missing="$missing libopus0 libopus-dev"
fi
# tesseract is only needed for -ocr, so it's suggested rather than required.
command -v tesseract >/dev/null 2>&1 || missing="$missing tesseract-ocr"

if [ -n "$missing" ]; then
    echo "!! Missing system package(s):$missing"
    APT_CMD="sudo apt-get update && sudo apt-get install -y$missing"
    echo "     $APT_CMD"
    if [ "$ALLOW_APT" = "0" ]; then
        echo "    (--no-apt given — not running it)"
    elif ! command -v apt-get >/dev/null 2>&1; then
        echo "    (no apt-get here — install these with your package manager)"
    elif [ "$(id -u)" = "0" ]; then
        # Already root, so sudo isn't needed and may not even exist.
        if confirm "Install them now?"; then
            apt-get update && apt-get install -y $missing || \
                echo "!! apt-get failed — install them manually."
        fi
    elif ! command -v sudo >/dev/null 2>&1; then
        echo "    (no sudo here — run the above as root)"
    else
        if confirm "Install them now with sudo?"; then
            sudo apt-get update && sudo apt-get install -y $missing || \
                echo "!! apt-get failed — install them manually."
        fi
    fi
else
    echo "    ffmpeg + libopus + tesseract present"
fi

# --- 8. Summary ------------------------------------------------------------
echo
if [ -n "$FAILED_GROUPS" ]; then
    echo "!! Some optional groups did not install:$FAILED_GROUPS"
    echo "   The bot still runs; those commands will say they're unavailable."
    echo
fi
echo "==> Done."
if [ "$WANT_KARAOKE" = "1" ]; then
    echo "    Karaoke is on. The first play of any track has to be separated"
    echo "    (minutes, on CPU) and is then cached in ./stem_cache — cap it"
    echo "    with STEM_CACHE_MB, default 2048."
elif [ "$HEAVY" = "0" ]; then
    echo "    Tip: ./install.sh --all  adds karaoke, transcription, translation"
    echo "         and background removal (~5 GB)."
fi
echo "    Run the bot with:  $VPY bot.py"
echo "    (or: source $VENV/bin/activate && python bot.py)"
