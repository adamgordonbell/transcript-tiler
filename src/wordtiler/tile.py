"""tile.py — refine an STT transcript's word timestamps into a complete labeling.

Given a clip's audio and word-level marks from any speech-to-text tool (which
drift — a mark can sit well off the actual speech), it lays a COMPLETE labeling
over the timeline: every instant is exactly one label, kind ∈ word | silence |
noise. Word labels carry corrected start/end anchors; silence/noise labels are
the gaps. Boundaries are shared transitions — a word's end IS the next label's
start — so nothing strands in silence and nothing is left uncovered.

Energy + silero VAD + spectral flatness only (see features.py) — no forced
aligners, no re-transcription.

  # library
  from wordtiler.tile import tile_audio, enrich
  tiles = tile_audio("clip.wav", [{"w": "It's", "start": 0.22, "end": 0.52}, ...])

Input words: [{"w"|"word", "t"|"start", "e"|"end"}, ...]
Output: {"windowStart", "words":[{w,start,end,startKind,endKind,startDb,endDb}],
         "silences":[[s,e]], "noise":[[s,e]], "labels":[{kind,w,start,end}]}
         — labels is the source of truth; the rest is derived.
"""
import os
import json
import hashlib
import numpy as np
import soundfile as sf
from .features import frame_rms, vad_probs, frame_flatness, classify_frames, load_mono, HZ, SPEECH, NONSP


def tile_full(words, rms, vad, flat, hz=HZ, trail=0.02, merge=0.08):
    """Complete word|sil|noise cover of the clip, as an ordered list of
    {kind, w, start, end}. `words` is [(w, t, e), ...] (transcript marks). rms/vad/flat
    are the 100Hz frame features for this audio (see tile_audio for the wrapper).

    ROBUST assignment uses MERGED speech segments (brief intra-word dips glued, so a
    fricative can't split a word). FINE bounds come from per-segment ENERGY BURSTS,
    threshold normalized to each segment's own peak (the clip floor sits in dead
    silence and can't see inter-word room tone). Words are matched to bursts MONOTONICALLY
    by burst centre (marks drift, so edge-distance mis-snaps a late mark onto the next
    burst). A word with no nearby burst is ISOLATED — the energy detector missed it (a
    near-silent /f/) — and keeps its transcript mark. Gaps between bursts become sil, or
    noise where NONSP frames dominate; a word's end trails `trail`s into a following gap."""
    n = len(rms)
    dur = n / hz if n else 0.0
    if not words:
        return [{"kind": "silence", "w": "", "start": 0.0, "end": dur}] if n else []
    floor = max(float(np.percentile(rms, 20)) * 1.5, 1e-4)
    lab = classify_frames(rms, vad, flat, floor)
    sp = lab == SPEECH
    raw, i = [], 0                                       # raw speech runs (frame idx)
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]:
                j += 1
            raw.append([i, j]); i = j
        else:
            i += 1
    if not raw:
        return [{"kind": "word", "w": w, "start": t, "end": e} for (w, t, e) in words]
    mg = int(round(merge * hz))                          # robust segments = runs glued over short dips
    merged = []
    for s in raw:
        if merged and s[0] - merged[-1][1] <= mg:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    segs = [(a / hz, b / hz) for a, b in merged]
    cents = [(t + e) / 2 for (_, t, e) in words]
    w_seg = []                                           # word → merged segment it belongs to (centre-in, else nearest)
    NEAR = 0.18                                          # admit a word to a burst within this of its edge; else isolated
    for c in cents:
        k = next((si for si, (s, e) in enumerate(segs) if s <= c <= e), None)
        if k is None:
            k = min(range(len(segs)), key=lambda si: abs(sum(segs[si]) / 2 - c))
        w_seg.append(k)
    res = [None] * len(words)
    MIN_SIL = 0.04                                       # a gap shorter than this is an intra-word dip, not silence
    SIL_FRAC = 0.28                                      # 'speech' = RMS above 28% of the segment's peak level

    def relmin(a, b, lo_t, hi_t):                        # frame of least energy between two word centres (abut split)
        lo = max(int(round(min(a, b) * hz)), int(round(lo_t * hz)))
        hi = min(int(round(max(a, b) * hz)), int(round(hi_t * hz)))
        return (lo + int(np.argmin(rms[lo:hi + 1]))) / hz if hi > lo else (a + b) / 2

    def bursts(segS, segE):
        """split a segment into speech BURSTS by a peak-relative energy threshold. Dips
        shorter than MIN_SIL are glued (intra-word fricative); each surviving gap is real
        silence. Burst edges = the actual speech onset/offset, independent of the marks."""
        lo, hi = int(round(segS * hz)), int(round(segE * hz))
        seg = np.asarray(rms[lo:hi + 1], float)
        if not len(seg):
            return [(segS, segE)]
        thr = floor + SIL_FRAC * (float(seg.max()) - floor)
        above = seg >= thr
        bl, i, m = [], 0, len(above)
        while i < m:
            if above[i]:
                j = i
                while j < m and above[j]:
                    j += 1
                bl.append([lo + i, lo + j]); i = j
            else:
                i += 1
        glue = int(round(MIN_SIL * hz))
        out = []
        for s in bl:
            if out and s[0] - out[-1][1] < glue:
                out[-1][1] = s[1]
            else:
                out.append(list(s))
        return [(a / hz, b / hz) for a, b in out] or [(segS, segE)]

    for si, (segS, segE) in enumerate(segs):
        wis = sorted([wi for wi in range(len(words)) if w_seg[wi] == si], key=lambda wi: cents[wi])
        if not wis:
            continue
        bl = bursts(segS, segE)
        bc = [sum(b) / 2 for b in bl]                    # burst centres
        # MONOTONIC assignment: words & bursts both run left-to-right, so a later word
        # can't claim an earlier burst. Match by burst CENTRE; a word whose nearest
        # admissible burst is still far (centre outside it ±NEAR) is ISOLATED.
        assign, bp = {}, 0
        for wi in wis:
            c = cents[wi]
            k = min(range(bp, len(bl)), key=lambda bi: abs(bc[bi] - c)) if bp < len(bl) else None
            if k is not None and bl[k][0] - NEAR <= c <= bl[k][1] + NEAR:
                assign.setdefault(k, []).append(wi); bp = k
            else:                                        # isolated → local burst around the mark, else the mark
                t, e = words[wi][1], words[wi][2]
                loc = [b for b in bursts(t - 0.06, e + 0.06) if b[0] <= c <= b[1]]
                res[wi] = [words[wi][0], *(min(loc, key=lambda b: abs(sum(b) / 2 - c)) if loc else (t, e))]
        for k, group in assign.items():                  # one burst → its word(s), abut at relmin if shared
            a, b = bl[k]
            group.sort(key=lambda wi: cents[wi])
            bnd = [a] + [relmin(cents[x], cents[y], a, b) for x, y in zip(group, group[1:])] + [b]
            for idx, wi in enumerate(group):
                res[wi] = [words[wi][0], bnd[idx], bnd[idx + 1]]
        for k, (a, b) in enumerate(bl):                  # orphan burst (fricative half) → swallow a word ≤0.15s away
            if k in assign:
                continue
            wi = min(wis, key=lambda wi: abs(cents[wi] - (a + b) / 2))
            if res[wi] and min(abs(cents[wi] - a), abs(cents[wi] - b)) <= 0.15:
                res[wi][1] = min(res[wi][1], a); res[wi][2] = max(res[wi][2], b)
    for wi, (w, t, e) in enumerate(words):
        if res[wi] is None:
            res[wi] = [w, t, e]
    order = sorted(range(len(words)), key=lambda wi: res[wi][1])

    def gap_kind(a, b):
        lo, hi = int(round(a * hz)), int(round(b * hz))
        if hi <= lo:
            return "silence"
        return "noise" if np.count_nonzero(lab[lo:hi] == NONSP) > (hi - lo) / 2 else "silence"

    tiles, prev = [], 0.0
    for oi, wi in enumerate(order):
        w, ws, we = res[wi]
        ws = max(ws, prev)
        if ws - prev > 1e-3:
            tiles.append({"kind": gap_kind(prev, ws), "w": "", "start": prev, "end": ws})
        nxt = res[order[oi + 1]][1] if oi + 1 < len(order) else dur
        we = max(we, ws)
        if nxt - we > 1e-3:                              # gap follows → trail into it (capture decay)
            we = min(we + trail, nxt)
        tiles.append({"kind": "word", "w": w, "start": ws, "end": we})
        prev = we
    if dur - prev > 1e-3:
        tiles.append({"kind": gap_kind(prev, dur), "w": "", "start": prev, "end": dur})
    return tiles


_FRIC_END = ("s", "z", "x", "sh", "ch", "ce", "se", "ze", "ge", "ss", "zz", "f", "fe", "ve", "ph", "th", "gh")


def _fricative_final(w):
    """does the word end in a sibilant/fricative? — then a trailing fricative (its /s/, /z/…)
    can follow an interior stop-closure and is part of the word ('bucks' = /bʌk-s/). A nasal/
    stop/vowel ending ('some', 'up') has no such tail, so energy after it is breath, not word."""
    return w.lower().strip(".,!?;:\"'`’”").endswith(_FRIC_END)


def _fine(x, sr, hop_ms=5):
    """Fine-resolution (hop_ms) RMS frames + the noise-floor estimate. Shared by the
    edge walk and the edge-freedom measurement so both read the SAME floor."""
    hop = max(1, int(sr * hop_ms / 1000))
    nf = len(x) // hop
    fr = np.sqrt(np.maximum((x[:nf * hop].reshape(nf, hop) ** 2).mean(1), 1e-12)) if nf else np.zeros(0)
    floor = max(float(np.percentile(fr, 20)) * 1.5, 1e-4) if nf else 1e-4
    return fr, floor, hop / sr


def _edge_db(fr, floor, t, dt, win=0.02):
    """EDGE FREEDOM, as a number: the quietest fine-RMS within ±`win` of boundary time
    `t`, in dB ABOVE the noise floor (clamped ≥0). ~0 dB = the edge dips to the floor (a
    real pause → the word is free on this side); large = no dip there (continuous voicing
    into the neighbour → embedded, not removable on its own). One value per boundary, so
    a word's endDb equals the next word's startDb (they share the boundary)."""
    if len(fr) == 0:
        return 0.0
    lo = max(0, int(round((t - win) / dt)))
    hi = min(len(fr), int(round((t + win) / dt)) + 1)
    if hi <= lo:
        return 0.0
    m = float(np.min(fr[lo:hi]))
    return round(max(0.0, 20.0 * np.log10(max(m, 1e-12) / floor)), 1)


def refine_edges(tiles, x, sr, hop_ms=5, k=1.1, max_reach=0.30, start_pad=0.020, end_pad=0.015, min_pause=0.06, end_reach=0.35):
    """Push each word↔gap edge outward at FINE (hop_ms) resolution to the floor band,
    capturing the soft attack/release the 100Hz burst threshold cuts too tight.

    The coarse 10ms/peak-relative tiler leaves silence-adjacent edges ~25-33ms inside
    the word (measured vs hand-golden); silero VAD is too coarse to fix it (32ms window,
    overshoots +87ms). Two regimes, from the golden:
      * END — the release tail rides ABOVE the floor, so a 5ms-hop RMS walk to `k`*floor
        tracks it (p90 ~32ms), + a small `end_pad` to seat the mean.
      * START — the soft attack sits BELOW the floor band (an RMS walk can't see it), but
        the miss is a uniform offset (MAD ~9ms), so a fixed `start_pad` back into the gap
        nails it.
    Word↔word abut edges (relmin) are left alone — min-loudness there is the internal dip,
    not the boundary. Continuous cover preserved (the gap edge moves with the word edge)."""
    if not tiles:
        return tiles
    fr, floor, dt = _fine(x, sr, hop_ms)
    nf = len(fr)
    if nf < 3:
        return tiles
    band = floor * k
    idx = lambda t: int(round(t / dt))

    def reaches_floor(a, b):                                 # does the gap hold a real pause?
        seg = fr[idx(a):idx(b)] <= band
        best = run = 0
        for v in seg:
            run = run + 1 if v else 0
            best = max(best, run)
        return best * dt >= min_pause

    for n, t in enumerate(tiles):                            # 1) REAL-pause edges: walk to floor
        if t["kind"] != "word":
            continue
        prev = tiles[n - 1] if n > 0 else None
        nxt = tiles[n + 1] if n + 1 < len(tiles) else None
        if prev and prev["kind"] != "word" and (prev is tiles[0] or reaches_floor(prev["start"], prev["end"])):
            reach = idx(max(prev["start"], t["start"] - max_reach))   # walk START back through the
            i = idx(t["start"])                              # above-band attack to the floor, then pad.
            while i > reach and fr[max(0, i - 1)] > band:    # the cap is generous (raw marks can land
                i -= 1                                       # a full vowel-initial onset late, e.g. the
            #   'ac' of 'actually'); it self-stops at the floor crossing & is clamped to the gap start
            t["start"] = prev["end"] = max(prev["start"], i * dt - start_pad)
        if nxt and nxt["kind"] != "word" and (nxt is tiles[-1] or reaches_floor(nxt["start"], nxt["end"])):
            reach = idx(min(nxt["end"], t["end"] + end_reach))        # walk END forward through the
            i = idx(t["end"])                                # release; for a FRICATIVE-final word
            last = i                                         # jump a short interior stop-closure to
            jump = _fricative_final(t["w"])                  # capture its trailing /s/,/z/…; else any
            while i < reach and i < nf:                      # floor dip is the word end (no tail)
                if fr[i] > band:
                    i += 1
                    last = i
                else:                                        # floor dip: closure or the real pause?
                    j = i
                    while j < reach and j < nf and fr[j] <= band:
                        j += 1
                    if not jump or j >= nf or (j - i) * dt >= min_pause:  # sustained floor / no tail expected
                        break                                # → real word end
                    i = j                                    # short closure (e.g. /k/ in "bucks") → jump
            t["end"] = nxt["start"] = min(nxt["end"], last * dt + end_pad)

    out = []                                                 # 2) run-together gaps: dissolve at valley
    for n, t in enumerate(tiles):
        prev = tiles[n - 1] if n > 0 else None
        nxt = tiles[n + 1] if n + 1 < len(tiles) else None
        if (t["kind"] == "word" or not prev or not nxt or prev["kind"] != "word"
                or nxt["kind"] != "word" or reaches_floor(t["start"], t["end"])):
            out.append(t)                                    # word, clip-edge, or real pause → keep
            continue
        a, b = idx(t["start"]), idx(t["end"])                # never reaches floor → not a pause:
        mid = a + int(0.45 * (b - a))                        # boundary = valley, biased to the next
        seg = fr[mid:max(b, mid + 1)]                        # word's onset (trailing consonant stays
        bnd = (mid + int(np.argmin(seg))) * dt if len(seg) else (t["start"] + t["end"]) / 2  # with prev)
        prev["end"] = nxt["start"] = bnd                     # drop the gap → word↔word boundary

    for n in range(len(out) - 1):                            # safety: no crossed edges
        if out[n]["end"] > out[n + 1]["start"]:
            m = (out[n]["end"] + out[n + 1]["start"]) / 2
            out[n]["end"] = out[n + 1]["start"] = m
    return [t for t in out if t["kind"] == "word" or t["end"] - t["start"] > 1e-4]


def tile_words(words, rms, vad, flat, hz=HZ):
    """word tiles only, (w, start, end) — the corrected anchors, no sil/noise."""
    return [(t["w"], t["start"], t["end"]) for t in tile_full(words, rms, vad, flat, hz)
            if t["kind"] == "word"]


ALGO_VERSION = "tile-v5.1"                                # bump to invalidate the cache when the algo/output changes (v5.1: emit per-chunk noise-floor dBFS as "floors")


def _cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "wordtiler")


def _norm(words):
    """input words → [(w, start, end)] (accepts w/word and t/start, e/end keys)."""
    return [(w.get("w", w.get("word", "")).strip(),
             float(w["t"] if "t" in w else w["start"]),
             float(w["e"] if "e" in w else w["end"])) for w in words]


def features(x, sr):
    """100Hz (rms, vad, flat) frame features for audio samples — what tile_full needs."""
    rms = frame_rms(x, sr)
    vad, _, _ = vad_probs(x, sr, len(rms))
    flat = frame_flatness(x, sr)
    return rms, vad, flat


def tile_audio(wav, words, **kw):
    """convenience for a SHORT clip: load all of `wav`, frame it, tile `words`."""
    x, sr = load_mono(wav)
    return refine_edges(tile_full(_norm(words), *features(x, sr), **kw), x, sr)


def _stitch(tiles, dur):
    """sort per-chunk tiles into absolute time, fill any uncovered gap (inter-chunk /
    edges) with silence, and merge abutting non-word tiles — a seamless single cover."""
    out, prev = [], 0.0
    for t in sorted(tiles, key=lambda t: t["start"]):
        s = max(t["start"], prev)
        if s - prev > 1e-3:
            out.append({"kind": "silence", "w": "", "start": prev, "end": s})
        e = max(t["end"], s)
        out.append({**t, "start": s, "end": e})
        prev = e
    if dur - prev > 1e-3:
        out.append({"kind": "silence", "w": "", "start": prev, "end": dur})
    merged = []
    for t in out:
        if (merged and t["kind"] != "word" and merged[-1]["kind"] == t["kind"]
                and abs(merged[-1]["end"] - t["start"]) < 1e-6):
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    return merged


def tile_file(wav, words, chunk_s=600, gap=0.8, margin=0.4, progress=False):
    """Tile a LONG file (an hour, three hours) without loading it whole. The word list is
    split at big transcript gaps (≥`gap`s, i.e. clear silence) into chunks of ≳`chunk_s`;
    each chunk reads ONLY its own audio span (+`margin` of context) via a partial decode,
    is framed and tiled, then offset back to absolute time. Because every cut lands in
    silence (no word straddles a boundary), the stitched result equals whole-file tiling."""
    nw = _norm(words)
    if not nw:
        return []
    info = sf.info(wav)
    sr, dur = info.samplerate, info.frames / info.samplerate
    groups, cur = [], [0]                                # break a chunk only at a real gap once it's long enough
    for i in range(1, len(nw)):
        if nw[i][1] - nw[i - 1][2] > gap and nw[i - 1][2] - nw[cur[0]][1] > chunk_s:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    tiles, floors = [], []
    for gi, grp in enumerate(groups):
        a = max(0.0, nw[grp[0]][1] - margin)
        b = min(dur, nw[grp[-1]][2] + margin)
        x, _ = sf.read(wav, start=int(a * sr), stop=int(b * sr), dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        local = [(w, t - a, e - a) for (w, t, e) in (nw[k] for k in grp)]
        fr, floor, dt = _fine(x, sr)                                          # for edge-freedom depth
        floors.append({"start": round(a, 3), "end": round(b, 3),
                       "dbfs": round(20 * np.log10(max(floor, 1e-12)), 1)})
        for t in refine_edges(tile_full(local, *features(x, sr)), x, sr):    # soft attack/release → floor
            e = {"kind": t["kind"], "w": t["w"], "start": t["start"] + a, "end": t["end"] + a}
            if t["kind"] == "word":                                           # floor-margin at each boundary
                e["startDb"] = _edge_db(fr, floor, t["start"], dt)
                e["endDb"] = _edge_db(fr, floor, t["end"], dt)
            tiles.append(e)
        if progress:
            print(f"  chunk {gi + 1}/{len(groups)}  [{a:7.1f}-{b:7.1f}s]  {len(grp)} words", flush=True)
    return _stitch(tiles, dur), floors


def _cache_key(wav, words):
    st = os.stat(wav)
    h = hashlib.sha256(ALGO_VERSION.encode())
    h.update(f"{os.path.abspath(wav)}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(json.dumps(_norm(words), sort_keys=True).encode())
    return h.hexdigest()[:16]


def enrich(wav, transcript, use_cache=True, progress=False):
    """richer transcript dict from a normalized transcript dict (+ its audio):
    {"words": [...], "windowStart"?: float} — see adapters.load_transcript for
    turning Whisper/WhisperX/etc. output into this shape. Scales to multi-hour
    audio (chunked) and caches by (audio, transcript, algo) hash — a re-run on
    unchanged inputs is instant."""
    words = transcript["words"]
    cp = os.path.join(_cache_dir(), _cache_key(wav, words) + ".json")
    if use_cache and os.path.isfile(cp):
        out = json.load(open(cp)); out["_cached"] = True
        return out
    tiles, floors = tile_file(wav, words, progress=progress)
    # EDGE FREEDOM (structural): each word's start/end kind is its neighbour tile's kind —
    # silence/noise/clip ⇒ that side is FREE (a real pause); `word` ⇒ EMBEDDED (abuts the
    # neighbour with no pause). Pairs with startDb/endDb (the depth of the dip) from tile_file.
    for i, t in enumerate(tiles):
        if t["kind"] != "word":
            continue
        t["startKind"] = tiles[i - 1]["kind"] if i > 0 else "clip"
        t["endKind"] = tiles[i + 1]["kind"] if i + 1 < len(tiles) else "clip"
    rnd = lambda t: round(float(t), 3)
    _free = ("startKind", "endKind", "startDb", "endDb")
    word = lambda t: {"w": t["w"], "start": rnd(t["start"]), "end": rnd(t["end"]),
                      **{f: t[f] for f in _free if f in t}}
    label = lambda t: {"kind": t["kind"], "w": t["w"], "start": rnd(t["start"]), "end": rnd(t["end"]),
                       **({f: t[f] for f in _free if f in t} if t["kind"] == "word" else {})}
    out = {
        "windowStart": transcript.get("windowStart", 0.0),
        "words": [word(t) for t in tiles if t["kind"] == "word"],
        "silences": [[rnd(t["start"]), rnd(t["end"])] for t in tiles if t["kind"] == "silence"],
        "noise": [[rnd(t["start"]), rnd(t["end"])] for t in tiles if t["kind"] == "noise"],
        "labels": [label(t) for t in tiles],
        # per-chunk estimated noise floor (dBFS) — the 0 dB reference all startDb/endDb
        # values are relative to. Self-estimated (20th-pct fine RMS × 1.5); on gated
        # audio it sits near the clamp (~-77 dBFS), on ungated audio at room tone.
        "floors": floors,
    }
    os.makedirs(_cache_dir(), exist_ok=True)
    json.dump(out, open(cp, "w"), indent=2)
    return out
