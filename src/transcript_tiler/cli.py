"""transcript_tiler CLI.

  transcript_tiler tile  <audio> <transcript.json> [-o OUT] [--from FMT] [--format json|textgrid|audacity] [--no-cache]
  transcript-tiler fillers <labels.json> [--max-db 10] [--only um,uh] [--min-dur 0.03]
"""
import argparse
import json
import os
import sys

from .adapters import load_transcript, FORMATS
from .export import WRITERS
from .fillers import free_words


def cmd_tile(args):
    from .tile import enrich   # lazy: torch import is slow, keep `fillers` snappy
    transcript = load_transcript(args.transcript, fmt=args.frm)
    out = enrich(args.audio, transcript, use_cache=not args.no_cache, progress=True)
    writer, ext = WRITERS[args.format]
    base = os.path.splitext(args.transcript if args.transcript != "-" else "stdin")[0]
    dst = args.out or base + ext
    writer(out, dst)
    tag = " (cached)" if out.get("_cached") else ""
    print(f"wrote {dst}{tag}  ({len(out['words'])} words, "
          f"{len(out['silences'])} silences, {len(out['noise'])} noise)")


def _load_labeling(path):
    labeling = json.load(sys.stdin if path == "-" else open(path))
    if "labels" not in labeling:
        sys.exit("not a labeling file (no 'labels' key) — run `transcript_tiler tile` first")
    return labeling


def cmd_fillers(args):
    labeling = _load_labeling(args.labels)
    only = [w for w in args.only.split(",") if w] if args.only else None
    hits = free_words(labeling, max_db=args.max_db, near_db=args.near_db,
                      seam_db=args.seam_db, only=only, min_dur=args.min_dur)
    if not args.near:
        hits = [h for h in hits if h["category"] != "near"]
    json.dump(hits, sys.stdout if not args.out else open(args.out, "w"), indent=2)
    from collections import Counter
    counts = Counter(h["category"] for h in hits)
    if args.out:
        print(f"wrote {args.out}  ({dict(counts)})")
    else:
        print(file=sys.stdout)


def cmd_stats(args):
    """Calibration view: the estimated floor + the edge-dB distribution, split
    by boundary type — so a user can pick --max-db/--seam-db for THEIR audio."""
    labeling = _load_labeling(args.labels)
    words = [t for t in labeling["labels"] if t["kind"] == "word"]
    pause, abut = [], []
    for t in words:
        for kind_key, db_key in (("startKind", "startDb"), ("endKind", "endDb")):
            if db_key not in t:
                continue
            (pause if t.get(kind_key) in ("silence", "noise", "clip") else abut).append(t[db_key])
    floors = labeling.get("floors", [])
    if floors:
        vals = sorted(f["dbfs"] for f in floors)
        print(f"noise floor (0 dB reference): median {vals[len(vals)//2]} dBFS "
              f"across {len(floors)} chunk(s), range [{vals[0]}, {vals[-1]}]")
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
    for name, xs in (("pause-adjacent edges", pause), ("word-abutting edges", abut)):
        if not xs:
            continue
        print(f"{name} (n={len(xs)}): median {q(xs,.5):.1f}  p75 {q(xs,.75):.1f}  "
              f"p90 {q(xs,.9):.1f}  p95 {q(xs,.95):.1f} dB above floor")
    if pause:
        p95 = q(pause, .95)
        print(f"\nsuggested --max-db: {max(10.0, round(p95 + 2))}"
              f"  (pause-edge p95 {p95:.1f} + headroom; healthy audio is bimodal —"
              f" pause edges near 0, abut edges ≫)")


def main():
    ap = argparse.ArgumentParser(prog="transcript-tiler",
                                 description="Refine STT word timestamps into a complete word|silence|noise tiling.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tile", help="audio + transcript → labeling (corrected boundaries + edge freedom)")
    t.add_argument("audio", help="source audio (clip or multi-hour track)")
    t.add_argument("transcript", help="word-timestamped transcript JSON ('-' for stdin)")
    t.add_argument("-o", "--out", help="output path (default: <transcript> + format extension)")
    t.add_argument("--from", dest="frm", default="auto", choices=FORMATS,
                   help="input format (default: auto-detect)")
    t.add_argument("--format", default="json", choices=sorted(WRITERS),
                   help="output format (default: json — the rich native form)")
    t.add_argument("--no-cache", action="store_true", help="recompute even if cached")
    t.set_defaults(fn=cmd_tile)

    s = sub.add_parser("fillers", aliases=["stops"], help="labeling → words FREE enough to cut (fillers, disfluencies)")
    s.add_argument("labels", help="labeling JSON from `transcript_tiler tile` ('-' for stdin)")
    s.add_argument("--max-db", type=float, default=10.0,
                   help="'free' threshold: worst edge within this of the noise floor (default 10)")
    s.add_argument("--near-db", type=float, default=30.0,
                   help="'near' ceiling: worst edge in (max-db, near-db] = audition by ear (default 30)")
    s.add_argument("--near", action="store_true", help="include 'near' words too (default: free only)")
    s.add_argument("--seam-db", type=float, default=None,
                   help="also emit ONE-SIDED words (pause on one side) whose word-abutting seam "
                        "dips within this of the floor (10 = conservative, 15 = the knee; off by default)")
    s.add_argument("--only", help="comma-separated word list to restrict to (e.g. um,uh,like)")
    s.add_argument("--min-dur", type=float, default=0.0, help="drop words shorter than this (s)")
    s.add_argument("-o", "--out", help="write JSON here instead of stdout")
    s.set_defaults(fn=cmd_fillers)

    st = sub.add_parser("stats", help="labeling → floor + edge-dB distribution (calibrate thresholds for your audio)")
    st.add_argument("labels", help="labeling JSON from `transcript_tiler tile` ('-' for stdin)")
    st.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
