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

Thresholds, empirically (two full podcast tracks, ~29k words): the edge-dB
distribution is bimodal — pause-adjacent edges sit at median/p95 = 0.0 dB,
word-abutting edges at median ~30 dB, with essentially nothing in between. So
the structural kind check does the real work and `max_db` is a click-guard;
anything in 3–10 dB selects the same words. Default 10 (the production-tuned
value); `near_db` = 30 marks the grey band worth auditioning by ear.

The cut range is the word tile exactly — NO added margin. The tiler already
walks each edge to the noise floor and pads into the gap (start_pad 20ms /
end_pad 15ms, plus the 20ms decay trail), so both cut faces sit at the floor
by construction; that is precisely what the dB check verifies.
"""
FREE_KINDS = {"silence", "noise", "clip"}


def _clean(w):
    return w.lower().strip(".,!?;:\"'`’”…—- ")


def free_words(labeling, max_db=10.0, near_db=30.0, only=None, min_dur=0.0):
    """[{w, start, end, startDb, endDb, category, ...}] — words with a real
    pause on both sides, whose worst edge is under `near_db`. category: "free"
    (worst edge ≤ max_db — clean cut) or "near" (in (max_db, near_db] — close,
    listen first). `only`: iterable of words (case/punct-insensitive) to
    restrict to. `min_dur`: drop words shorter than this."""
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
        worst = max(t.get("startDb", 0.0), t.get("endDb", 0.0))
        if worst > near_db:
            continue
        out.append({**t, "category": "free" if worst <= max_db else "near"})
    return out
