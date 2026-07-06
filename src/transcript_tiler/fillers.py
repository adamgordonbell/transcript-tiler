"""fillers.py — emit the FREE words: candidates for removal (fillers, disfluencies).

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


def free_words(labeling, max_db=10.0, near_db=30.0, seam_db=None, only=None, min_dur=0.0):
    """[{w, start, end, startDb, endDb, category, freeSide, ...}].

    Two-sided (default): words with a real pause on both sides, worst edge
    under `near_db`. category: "free" (worst edge ≤ max_db — clean cut) or
    "near" (in (max_db, near_db] — close, listen first). freeSide: "both".

    One-sided (`seam_db` set): ALSO admit words with a pause on exactly one
    side (that edge ≤ max_db) whose OTHER boundary — the word-abutting seam —
    dips within `seam_db` of the floor. The cut lands at the inter-word valley;
    the neighbour's onset/tail stays intact, so the seam sounds like a normal
    pause→word transition. Unlike the bimodal two-sided case, seam dBs spread
    continuously (0→50 dB on the reference episode), so this threshold is a
    real dial: 10 = conservative (seam dips as deep as a free edge), 15 =
    the knee. These hits get freeSide: "start"|"end" and category "one-sided".

    Each hit also carries the extent of its flanking pause tiles when present —
    `silenceStart` (start of the preceding silence/noise tile) and `silenceEnd`
    (end of the following one) — so a cutter can absorb adjacent silence
    without re-reading the labeling.

    `only`: iterable of words (case/punct-insensitive) to restrict to.
    `min_dur`: drop words shorter than this."""
    want = {_clean(w) for w in only} if only else None
    labels = labeling["labels"]

    def flanks(i):
        f = {}
        if i > 0 and labels[i - 1]["kind"] != "word":
            f["silenceStart"] = labels[i - 1]["start"]
        if i + 1 < len(labels) and labels[i + 1]["kind"] != "word":
            f["silenceEnd"] = labels[i + 1]["end"]
        return f

    out = []
    for i, t in enumerate(labels):
        if t["kind"] != "word":
            continue
        if want is not None and _clean(t["w"]) not in want:
            continue
        if t["end"] - t["start"] < min_dur:
            continue
        s_pause = t.get("startKind") in FREE_KINDS
        e_pause = t.get("endKind") in FREE_KINDS
        sdb, edb = t.get("startDb", 0.0), t.get("endDb", 0.0)
        if s_pause and e_pause:
            worst = max(sdb, edb)
            if worst <= near_db:
                out.append({**t, **flanks(i), "freeSide": "both",
                            "category": "free" if worst <= max_db else "near"})
            continue
        if seam_db is None:
            continue
        s_free = s_pause and sdb <= max_db
        e_free = e_pause and edb <= max_db
        if s_free != e_free:                       # exactly one free pause-side
            seam = edb if s_free else sdb
            if seam <= seam_db:
                out.append({**t, **flanks(i), "freeSide": "start" if s_free else "end",
                            "category": "one-sided"})
    return out
