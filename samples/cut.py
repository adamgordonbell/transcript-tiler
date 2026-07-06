"""cut.py — remove the words `fillers` found, with micro-fades, via ffmpeg.

  uv run python samples/cut.py in.wav fillers.json out.wav [--fade-ms 5] [--absorb-trailing]

The gist: the complement of the cut spans is a list of KEEP segments; each is
atrim'd with a short fade-in/out (default 5 ms) at its faces, then concat'd.
Because every cut face sits at the noise floor (that's what `fillers`
verified), the fades are insurance against dither/DC ticks, not doing the
real work — plain butt joints are usually already clean.

By default only the word span is removed, so the pause it sat in survives and
grows by the word's duration (its two flanking silences join). --absorb-trailing
extends each cut through the word's trailing silence tile (the `silenceEnd`
field in the fillers output), so what remains is just the word's leading
silence — the pause keeps roughly its natural length instead of growing.
"""
import argparse
import json
import subprocess
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description="remove filler spans from audio with micro-fades")
    ap.add_argument("wav")
    ap.add_argument("fillers_json", help="output of `transcript-tiler fillers`")
    ap.add_argument("out")
    ap.add_argument("--fade-ms", type=float, default=5.0)
    ap.add_argument("--absorb-trailing", action="store_true",
                    help="also remove each word's trailing silence tile (pause keeps its natural length)")
    args = ap.parse_args()

    fade = args.fade_ms / 1000
    info = sf.info(args.wav)
    dur = info.frames / info.samplerate
    spans = []
    for h in json.load(open(args.fillers_json)):
        end = h.get("silenceEnd", h["end"]) if args.absorb_trailing else h["end"]
        spans.append((h["start"], min(end, dur)))
    keeps, pos = [], 0.0
    for a, b in sorted(spans):
        if a - pos > 1e-3:
            keeps.append((pos, a))
        pos = max(pos, b)
    if dur - pos > 1e-3:
        keeps.append((pos, dur))
    parts, chain = [], []
    for i, (a, b) in enumerate(keeps):
        f = min(fade, (b - a) / 2)
        chain.append(f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={f:.4f},afade=t=out:st={b - a - f:.4f}:d={f:.4f}[s{i}]")
        parts.append(f"[s{i}]")
    graph = ";".join(chain) + f";{''.join(parts)}concat=n={len(keeps)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.wav,
                    "-filter_complex", graph, "-map", "[out]", args.out], check=True)
    removed = sum(b - a for a, b in spans)
    print(f"wrote {args.out}: kept {len(keeps)} segments, removed {len(spans)} spans ({removed:.2f}s)")


if __name__ == "__main__":
    main()
