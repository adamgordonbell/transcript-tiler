"""stops.py — emit the FREE words: candidates for removal (fillers, stop words).

Pure interval math over an existing labeling (the output of tile) — no audio
needed. A word is *free at threshold `max_db`* when BOTH of its edges dip to
the noise floor:

  * structural: startKind/endKind ∈ {silence, noise, clip} — a real pause (or
    the clip edge) on that side, not a neighbouring word, AND
  * depth: startDb/endDb ≤ max_db — the dip actually reaches (near) the floor.

Such a word sits in its own pocket of silence and can be cut without clipping
a neighbour. Optionally restrict to a word list (um, uh, ...) — without one,
you get every free word and filter downstream.
"""
FREE_KINDS = {"silence", "noise", "clip"}


def _clean(w):
    return w.lower().strip(".,!?;:\"'`’”…—- ")


def free_words(labeling, max_db=3.0, only=None, min_dur=0.0):
    """[{w, start, end, startDb, endDb, ...}] — words free at `max_db` on both
    edges. `only`: iterable of words (case/punct-insensitive) to restrict to.
    `min_dur`: drop words shorter than this (zero-length artifacts)."""
    want = {_clean(w) for w in only} if only else None
    out = []
    for t in labeling["labels"]:
        if t["kind"] != "word":
            continue
        if want is not None and _clean(t["w"]) not in want:
            continue
        if t["end"] - t["start"] < min_dur:
            continue
        if t.get("startKind") not in FREE_KINDS or t.get("endKind") not in FREE_KINDS:
            continue
        if t.get("startDb", 0.0) > max_db or t.get("endDb", 0.0) > max_db:
            continue
        out.append(dict(t))
    return out
