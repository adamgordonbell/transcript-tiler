"""wordtiler CLI.

  wordtiler tile  <audio> <transcript.json> [-o OUT] [--from FMT] [--format json|textgrid|audacity] [--no-cache]
  wordtiler stops <labels.json> [--max-db 3.0] [--only um,uh] [--min-dur 0.03]
"""
import argparse
import json
import os
import sys

from .adapters import load_transcript, FORMATS
from .export import WRITERS
from .stops import free_words


def cmd_tile(args):
    from .tile import enrich   # lazy: torch import is slow, keep `stops` snappy
    transcript = load_transcript(args.transcript, fmt=args.frm)
    out = enrich(args.audio, transcript, use_cache=not args.no_cache, progress=True)
    writer, ext = WRITERS[args.format]
    base = os.path.splitext(args.transcript if args.transcript != "-" else "stdin")[0]
    dst = args.out or base + ext
    writer(out, dst)
    tag = " (cached)" if out.get("_cached") else ""
    print(f"wrote {dst}{tag}  ({len(out['words'])} words, "
          f"{len(out['silences'])} silences, {len(out['noise'])} noise)")


def cmd_stops(args):
    labeling = json.load(sys.stdin if args.labels == "-" else open(args.labels))
    if "labels" not in labeling:
        sys.exit("not a labeling file (no 'labels' key) — run `wordtiler tile` first")
    only = [w for w in args.only.split(",") if w] if args.only else None
    hits = free_words(labeling, max_db=args.max_db, near_db=args.near_db,
                      only=only, min_dur=args.min_dur)
    if not args.near:
        hits = [h for h in hits if h["category"] == "free"]
    json.dump(hits, sys.stdout if not args.out else open(args.out, "w"), indent=2)
    nf = sum(1 for h in hits if h["category"] == "free")
    if args.out:
        print(f"wrote {args.out}  ({nf} free, {len(hits) - nf} near)")
    else:
        print(file=sys.stdout)


def main():
    ap = argparse.ArgumentParser(prog="wordtiler",
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

    s = sub.add_parser("stops", help="labeling → words FREE enough to cut (fillers/stop words)")
    s.add_argument("labels", help="labeling JSON from `wordtiler tile` ('-' for stdin)")
    s.add_argument("--max-db", type=float, default=10.0,
                   help="'free' threshold: worst edge within this of the noise floor (default 10)")
    s.add_argument("--near-db", type=float, default=30.0,
                   help="'near' ceiling: worst edge in (max-db, near-db] = audition by ear (default 30)")
    s.add_argument("--near", action="store_true", help="include 'near' words too (default: free only)")
    s.add_argument("--only", help="comma-separated word list to restrict to (e.g. um,uh,like)")
    s.add_argument("--min-dur", type=float, default=0.0, help="drop words shorter than this (s)")
    s.add_argument("-o", "--out", help="write JSON here instead of stdout")
    s.set_defaults(fn=cmd_stops)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
