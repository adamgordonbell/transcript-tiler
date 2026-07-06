"""adapters.py — read word-level timestamps from common STT output formats.

Everything normalizes to the internal transcript shape:

    {"words": [{"w": str, "start": float, "end": float}, ...], "windowStart": float}

Formats (auto-detected from the JSON shape, or forced with `fmt=`):

  generic   {"words": [{"w"|"word", "t"|"start", "e"|"end"}, ...]}   (native / Descript-style)
  whisper   openai-whisper `--word_timestamps True` JSON:
            {"segments": [{"words": [{"word", "start", "end", ...}]}], ...}
  whisperx  WhisperX aligned JSON — same segments shape, plus a flat
            "word_segments" list; either is accepted. Words WhisperX could
            not align (no "start") are dropped with a stderr warning.
"""
import json
import sys

FORMATS = ("auto", "generic", "whisper", "whisperx")


def _from_generic(d):
    return [{"w": w.get("w", w.get("word", "")).strip(),
             "start": float(w["t"] if "t" in w else w["start"]),
             "end": float(w["e"] if "e" in w else w["end"])} for w in d["words"]]


def _from_segments(d):
    """whisper / whisperx segments → words. Skips unaligned words (no start)."""
    words, dropped = [], 0
    src = d.get("word_segments") or [w for s in d.get("segments", []) for w in s.get("words", [])]
    for w in src:
        if "start" not in w or "end" not in w:
            dropped += 1
            continue
        words.append({"w": w.get("word", w.get("w", "")).strip(),
                      "start": float(w["start"]), "end": float(w["end"])})
    if dropped:
        print(f"warning: dropped {dropped} words without timestamps", file=sys.stderr)
    return words


def detect(d):
    if "word_segments" in d:
        return "whisperx"
    if "segments" in d:
        return "whisper"
    if "words" in d:
        return "generic"
    raise ValueError("unrecognized transcript JSON: expected 'words', 'segments', or 'word_segments'")


def load_transcript(path, fmt="auto"):
    """path (or '-' for stdin) → normalized transcript dict."""
    d = json.load(sys.stdin if path == "-" else open(path))
    if fmt == "auto":
        fmt = detect(d)
    if fmt == "generic":
        words = _from_generic(d)
    elif fmt in ("whisper", "whisperx"):
        words = _from_segments(d)
    else:
        raise ValueError(f"unknown format {fmt!r} (one of {FORMATS})")
    words.sort(key=lambda w: w["start"])
    return {"words": words, "windowStart": float(d.get("windowStart", 0.0)), "_format": fmt}
