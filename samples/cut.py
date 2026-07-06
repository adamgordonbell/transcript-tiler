"""cut.py — remove the words `stops` found, with micro-fades, via ffmpeg.

  uv run python samples/cut.py in.wav stops.json out.wav [fade_ms]

The gist: the complement of the cut spans is a list of KEEP segments; each is
atrim'd with a short fade-in/out (default 5 ms) at its faces, then concat'd.
Because every cut face sits at the noise floor (that's what `stops` verified),
the fades are insurance against dither/DC ticks, not doing the real work —
plain butt joints are usually already clean.
"""
import json
import subprocess
import sys
import soundfile as sf


def main(wav, stops_json, out, fade_ms=5.0):
    fade = float(fade_ms) / 1000
    dur = sf.info(wav).frames / sf.info(wav).samplerate
    cuts = sorted((h["start"], h["end"]) for h in json.load(open(stops_json)))
    keeps, pos = [], 0.0
    for a, b in cuts:
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
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
                    "-filter_complex", graph, "-map", "[out]", out], check=True)
    removed = sum(b - a for a, b in cuts)
    print(f"wrote {out}: kept {len(keeps)} segments, removed {len(cuts)} spans ({removed:.2f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
