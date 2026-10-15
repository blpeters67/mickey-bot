"""Unit tests for core.voice_intent — wake-word gating + intent matching.

These lean hard on the mis-transcription cases ("skib", "paws", glued/garbled
wake words), since surviving imperfect whisper output is the whole point.
"""
import pytest

from core import voice_intent as vi
from core.voice_intent import Intent, match_intent, parse, parse_detailed, strip_wake


# ── Wake-word gating ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("hey dj skip", "skip"),
    ("Hey DJ, skip!", "skip"),              # punctuation + case
    ("hey dj play mr brightside", "play mr brightside"),
    ("um hey dj skip", "skip"),             # leading filler
    ("okay hey dj pause", "pause"),
    ("heydj skip", "skip"),                 # whisper glued the phrase
    ("HIDJ, pause the music", "pause the music"),
    ("haydj skip", "skip"),
    ("hey dj", ""),                         # wake only, no command
])
def test_strip_wake_accepts(text, expected):
    assert strip_wake(text) == expected


@pytest.mark.parametrize("text", [
    "skip",                 # no wake word at all
    "i think we should go", # ordinary conversation
    "hey bob skip",         # wrong wake word
    "play despacito",       # command without the wake word
    "",
])
def test_strip_wake_rejects(text):
    assert strip_wake(text) is None


def test_custom_wake_word():
    assert strip_wake("computer skip", "computer") == "skip"
    assert strip_wake("hey dj skip", "computer") is None


@pytest.mark.parametrize("text", ["I said HIDJ pause", "do not say haydj skip"])
def test_fused_alias_does_not_open_a_mid_sentence_wake(text):
    assert parse(text) is None


# ── Multiple wake words (a guild can accept several) ─────────────────────────

def test_multiple_wake_words_each_match():
    wakes = ["hey dj", "computer", "jarvis"]
    assert strip_wake("hey dj skip", wakes) == "skip"
    assert strip_wake("computer pause", wakes) == "pause"
    assert strip_wake("jarvis play mr brightside", wakes) == "play mr brightside"


def test_multiple_wake_words_reject_when_none_match():
    wakes = ["computer", "jarvis"]
    assert strip_wake("hey dj skip", wakes) is None
    assert strip_wake("alexa skip", wakes) is None


def test_multiple_wake_words_tolerate_mishearing():
    # The first list entry should still match through the fuzzy/phonetic path.
    assert strip_wake("computer skib", ["computer", "jarvis"]) == "skib"


def test_empty_or_unusable_wake_list_closes_the_gate():
    # Used to open the gate (everything parsed as a command). A wake list that
    # is empty — or only contains phrases that normalise to nothing, like a
    # Cyrillic word the English model will never transcribe — must not let
    # ordinary conversation through.
    assert strip_wake("skip", []) is None
    assert strip_wake("skip", ["привет"]) is None
    assert parse("skip", []) is None


def test_parse_with_wake_list():
    wakes = ["hey dj", "computer"]
    assert parse("computer skip", wakes).action == "skip"
    assert parse("hey dj play lofi", wakes) == Intent("play", "lofi")
    assert parse("just talking", wakes) is None


def test_parse_requires_wake_by_default():
    assert parse("skip") is None
    assert parse("hey dj skip").action == "skip"


def test_parse_can_skip_wake_for_confirmation_window():
    # Inside a confirm window the bot listens without the wake word.
    assert parse("confirm", require_wake=False).action == "confirm"
    assert parse("yes", require_wake=False).action == "confirm"


# ── Simple verbs, including mis-transcriptions ───────────────────────────────

@pytest.mark.parametrize("phrase,action", [
    ("skip", "skip"),
    ("skib", "skip"),            # mis-heard
    ("skipp", "skip"),
    ("next", "skip"),
    ("next song", "skip"),
    ("skip this one", "skip"),
    ("pause", "pause"),
    ("paws", "pause"),           # homophone → Soundex catch
    ("hold on", "pause"),
    ("resume", "resume"),
    ("continue", "resume"),
    ("unpause", "resume"),
    ("loop", "loop"),
    ("repeat", "loop"),
    ("repeat this", "loop"),
])
def test_simple_verbs(phrase, action):
    intent = match_intent(phrase)
    assert intent is not None, phrase
    assert intent.action == action


def test_bare_play_is_resume():
    # "play" with nothing meaningful after it = resume, not an empty search.
    assert match_intent("play").action == "resume"
    assert match_intent("play it").action == "resume"
    assert match_intent("play the music").action == "resume"


# ── Destructive verbs ────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,action", [
    ("stop", "stop"),
    ("stop it", "stop"),
    ("leave", "stop"),
    ("clear", "clear"),
    ("clear the queue", "clear"),
    ("empty the queue", "clear"),
])
def test_destructive_verbs(phrase, action):
    intent = match_intent(phrase)
    assert intent.action == action
    assert intent.destructive is True


def test_nondestructive_flag():
    assert match_intent("skip").destructive is False


# ── play <song> ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,query", [
    ("play despacito", "despacito"),
    ("play mr brightside", "mr brightside"),
    ("play mister brightside by the killers", "mister brightside by the killers"),
    ("queue some lofi", "some lofi"),
    ("add bohemian rhapsody", "bohemian rhapsody"),
    ("put on some jazz", "some jazz"),
    ("play 50 cent in da club", "50 cent in da club"),   # number stays in query
])
def test_play_song(phrase, query):
    intent = match_intent(phrase)
    assert intent.action == "play"
    assert intent.arg == query


# ── Volume ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,direction", [
    ("louder", "up"),
    ("turn it up", "up"),
    ("crank it", "up"),
    ("quieter", "down"),
    ("turn it down", "down"),
    ("lower", "down"),
])
def test_volume_direction(phrase, direction):
    intent = match_intent(phrase)
    assert intent.action == "volume"
    assert intent.arg == direction


@pytest.mark.parametrize("phrase,level", [
    ("volume 50", "50"),
    ("set volume to 30", "30"),
    ("volume 100", "100"),
])
def test_volume_level(phrase, level):
    intent = match_intent(phrase)
    assert intent.action == "volume"
    assert intent.arg == level


# ── Confirmation window vocabulary ───────────────────────────────────────────

@pytest.mark.parametrize("phrase,action", [
    ("confirm", "confirm"),
    ("yes", "confirm"),
    ("do it", "confirm"),
    ("cancel", "cancel"),
    ("nevermind", "cancel"),
    ("no", "cancel"),
])
def test_confirm_cancel(phrase, action):
    assert match_intent(phrase).action == action


# ── Non-commands stay silent ─────────────────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    "what time is it",
    "did you see the game last night",
    "i love this song",
    "",
])
def test_unrecognized_returns_none(phrase):
    assert match_intent(phrase) is None


def test_full_pipeline_end_to_end():
    assert parse("hey dj, play mr brightside") == Intent("play", "mr brightside")
    assert parse("hey dj skip").action == "skip"
    assert parse("hey dj stop").destructive is True
    assert parse("just chatting normally") is None


# ── Internals: phonetic + fuzzy helpers ──────────────────────────────────────

def test_mishearings_come_from_an_alias_list_not_soundex():
    # Soundex is gone (it matched "loop" to "leave"); homophones are listed.
    assert not hasattr(vi, "_soundex")
    assert match_intent("paws").action == "pause"
    assert match_intent("skib").action == "skip"


def test_ratio_bounds():
    assert vi._ratio("abc", "abc") == 1.0
    assert vi._ratio("", "") == 1.0
    assert 0.0 <= vi._ratio("skip", "skib") <= 1.0


# ══ Regressions from the 2026-10 voice review ═════════════════════════════════
# Each of these was reproduced against the old parser before being fixed.

@pytest.mark.parametrize("text,expected", [
    ("hey dj hold up", Intent("pause")),                    # was volume up
    ("hey dj loop off", Intent("loop", "off")),             # was STOP (soundex loop≈leave)
    ("hey dj loop on", Intent("loop", "on")),               # was STOP
    ("hey dj start over", Intent("restart")),               # was play "over"
    ("hey dj stop looping", Intent("loop", "off")),
    ("hey dj toggle loop", Intent("loop", "toggle")),
    ("hey dj play it again", Intent("restart")),
    ("hey dj play the next song", Intent("skip")),
])
def test_review_misroutes_now_correct(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize("text", [
    "hey dj next week I am away",      # trailing conversation no longer accepted
    "hey dj don't stop",               # was volume DOWN ("don" ≈ "down")
    "hey dj do not skip",
    "hey dj stop skipping",
    "hey dj i love this song",
    "hey dj what's next",
])
def test_review_non_commands_stay_silent(text):
    assert parse(text) is None


@pytest.mark.parametrize("text", [
    "hey d j skip", "hey dee jay skip", "Hey D.J., skip!", "hey deejay skip",
    "heydj skip", "a dj skip", "hi dj skip", "yo dj skip", "um, hey DJ skip",
])
def test_wake_spellings(text):
    assert parse(text) == Intent("skip")


@pytest.mark.parametrize("text", [
    "hey dj please skip", "hey dj can you skip", "hey dj could you skip please",
    "hey dj can you skip this song please", "hey dj just skip it", "hey dj skip now",
])
def test_polite_filler(text):
    assert parse(text) == Intent("skip")


def test_loose_wake_spellings_only_at_the_start():
    # "a DJ" mid-sentence is conversation, not a wake word.
    assert parse("he's a dj, play his stuff") is None
    assert parse_detailed("he's a dj, play his stuff").reason == "no_wake"


def test_wake_mid_sentence_never_acts():
    # A wake phrase inside a sentence is reported for diagnostics but never
    # acts: it's quoted/reported speech far more often than a command.
    r = parse_detailed("so anyway, hey dj skip")
    assert r.intent is None and r.reason == "wake_not_at_start" and not r.wake_at_start
    assert parse("so anyway, hey dj skipp") is None
    assert parse("hey dj pauze") == Intent("pause", confidence=0.75)


@pytest.mark.parametrize("text", [
    "do not say hey dj skip",
    "I said hey dj skip",
    "he told me hey dj skip",
    "never say hey DJ, stop",
    "she was like hey dj pause",
    "and then I said, hey DJ, clear the queue",
    "if you say hey dj skip it skips",
    "that was great hey dj",                       # a trailing wake doesn't arm either
])
def test_quoted_or_negated_wake_does_nothing(text):
    r = parse_detailed(text)
    assert r.intent is None and r.reason == "wake_not_at_start"


@pytest.mark.parametrize("text", [
    "um, hey DJ skip", "hey, hey dj skip", "hey dj hey dj skip", "okay hey dj skip",
    "hey dj I said skip",
])
def test_wake_at_the_start_still_works(text):
    assert parse(text) == Intent("skip")


def test_interpret_window_takes_a_bare_command_but_not_outside_it():
    assert vi.interpret("skip").intent is None
    assert vi.interpret("skip", in_window=True).intent == Intent("skip")
    # A wake-only utterance inside a window re-arms rather than acting.
    assert vi.interpret("hey dj", in_window=True).reason == "wake_only"


def test_wake_only_is_reported_so_the_cog_can_arm_a_window():
    assert parse_detailed("hey dj").reason == "wake_only"
    assert parse_detailed("hey DJ.").reason == "wake_only"
    assert parse_detailed("ok so, hey dj").reason == "wake_only"
    # The follow-up ("…skip") is parsed without a wake word.
    assert parse("skip", require_wake=False) == Intent("skip")


def test_longest_wake_wins():
    assert strip_wake("hey dj skip", ["dj", "hey dj"]) == "skip"


@pytest.mark.parametrize("phrase,canon", [
    ("Hey D.J.", "hey dj"), ("héy dj", "hey dj"), ("  Computer ", "computer"),
])
def test_normalize_wake(phrase, canon):
    assert vi.normalize_wake(phrase) == canon


@pytest.mark.parametrize("phrase", ["", "привет", "!!!", "computer 2",
                                    "one two three four five"])
def test_normalize_wake_rejects_unusable(phrase):
    assert vi.normalize_wake(phrase) is None


# ── song queries keep the original text ──────────────────────────────────────

@pytest.mark.parametrize("text,query", [
    ("Hey DJ, play Beyoncé – Déjà Vu.", "Beyoncé – Déjà Vu"),
    ("hey dj play 夜に駆ける", "夜に駆ける"),                 # used to become "resume"
    ("Hey DJ, play Mr. Brightside by The Killers.", "Mr. Brightside by The Killers"),
    ("hey dj play don't stop me now", "don't stop me now"),  # negation inside a title
    ("hey dj play pretty please", "pretty please"),          # titles keep trailing words
    ("hey dj queue hold on", "hold on"),                     # a title, not "pause"
])
def test_play_queries_from_original_text(text, query):
    assert parse(text) == Intent("play", query)


# ── numbers and volume ───────────────────────────────────────────────────────

@pytest.mark.parametrize("words,value", [
    ("fifty", 50), ("seventy five", 75), ("a hundred and twenty", 120),
    ("one hundred", 100), ("hundred", 100), ("zero", 0), ("50", 50),
    ("50%", 50), ("30 percent", 30), ("two hundred", 200),
])
def test_spoken_numbers(words, value):
    assert vi.parse_number(words.split()) == (value, False)


@pytest.mark.parametrize("words", ["minus ten", "-10", "negative five"])
def test_negative_numbers_are_flagged(words):
    value, negative = vi.parse_number(words.split())
    assert negative and value is not None


@pytest.mark.parametrize("words", ["twenty twenty", "fifty cats", "", "hundred and"])
def test_not_numbers(words):
    assert vi.parse_number(words.split())[0] is None


@pytest.mark.parametrize("text,level", [
    ("hey dj volume fifty", "50"),                   # used to return nothing
    ("hey dj set the volume to a hundred and twenty", "120"),
    ("hey dj turn it to 30 percent", "30"),
    ("hey dj volume 40 please", "40"),
])
def test_spoken_volume_levels(text, level):
    assert parse(text) == Intent("volume", level)


@pytest.mark.parametrize("text", ["hey dj volume -10", "hey dj volume minus ten"])
def test_negative_volume_is_rejected_not_flipped(text):
    # "-10" used to become +10.
    r = parse_detailed(text)
    assert r.intent is None and r.reason == "bad_volume"


def test_loud_spoken_volume_needs_confirmation():
    assert parse("hey dj volume 500").needs_confirmation
    assert not parse("hey dj volume 150").needs_confirmation
    assert parse("hey dj stop").needs_confirmation
    assert not parse("hey dj skip").needs_confirmation


# ── confirmation answers ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,answer", [
    ("yes", "yes"), ("yeah", "yes"), ("sure", "yes"), ("ok go ahead", "yes"),
    ("yes please", "yes"), ("hey dj yes", "yes"), ("yeah do it", "yes"),
    ("yes stop it", "yes"),
    ("no", "no"), ("nope", "no"), ("cancel", "no"), ("never mind", "no"),
    ("yes but do not stop", "no"),       # used to CONFIRM
    ("yeah no", "no"),                   # used to CONFIRM
    ("wait", "no"), ("hold on", "no"),
    ("what was that", None),             # unrelated: neither — keep waiting
    ("yes i think the song is great", None),
    ("undo it", None),
    ("", None),
])
def test_confirmation_grammar(text, answer):
    assert vi.parse_confirmation(text, ["hey dj"], action="stop") == answer


def test_grammar_table_has_no_conflicts():
    # Every phrase maps to exactly one (action, arg): checked at import, and the
    # table must stay closed — nothing outside ACTIONS.
    assert {a for a, _ in vi._PHRASES.values()} <= set(vi.ACTIONS)
