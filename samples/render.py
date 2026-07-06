"""render.py — draw a labeling over its waveform as an SVG (the README header).

  uv run python samples/render.py samples/sample.wav samples/sample.labels.json docs/sample-labeling.svg

Waveform in gray; word tiles blue, silence tiles pale, noise tiles orange
(colorblind-safe hues), with per-edge freedom dB printed at each boundary.
"""
import json
import sys
import numpy as np
import soundfile as sf

W, H = 980, 300
WAVE_H, TILE_Y, TILE_H = 190, 210, 44
BG, WAVE, WORD, SIL, NOISE, TXT, SUB = ("#0d1117", "#8b949e", "#388bfd", "#21262d",
                                        "#d29922", "#e6edf3", "#8b949e")


def peaks(x, n):
    step = max(1, len(x) // n)
    return [(float(x[i:i + step].min()), float(x[i:i + step].max())) for i in range(0, step * n, step)]


def main(wav, labels_json, out):
    x, sr = sf.read(wav, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    lab = json.load(open(labels_json))["labels"]
    dur = lab[-1]["end"]
    px = lambda t: 10 + (W - 20) * t / dur
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="ui-monospace,Menlo,monospace">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="8"/>']
    mid, amp = 12 + WAVE_H / 2, WAVE_H / 2 - 8
    scale = amp / (max(abs(x.min()), x.max()) or 1)
    pts = peaks(x, W - 20)
    for i, (lo, hi) in enumerate(pts):
        e.append(f'<line x1="{10 + i}" y1="{mid - hi * scale:.1f}" x2="{10 + i}" '
                 f'y2="{mid - lo * scale:.1f}" stroke="{WAVE}" stroke-width="1"/>')
    for t in lab:                                    # tile band + word shading on the wave
        a, b = px(t["start"]), px(t["end"])
        if t["kind"] == "word":
            e.append(f'<rect x="{a:.1f}" y="12" width="{b - a:.1f}" height="{WAVE_H}" '
                     f'fill="{WORD}" opacity="0.13"/>')
        fill = {"word": WORD, "silence": SIL, "noise": NOISE}[t["kind"]]
        e.append(f'<rect x="{a:.1f}" y="{TILE_Y}" width="{max(b - a, 1):.1f}" height="{TILE_H}" '
                 f'fill="{fill}" rx="4" stroke="{BG}" stroke-width="1.5"/>')
        if t["kind"] == "word":
            e.append(f'<text x="{(a + b) / 2:.1f}" y="{TILE_Y + 27}" fill="{TXT}" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{t["w"]}</text>')
    seen = set()
    for t in lab:                                    # per-boundary freedom dB, printed once
        if t["kind"] != "word":
            continue
        for tt, db in ((t["start"], t.get("startDb")), (t["end"], t.get("endDb"))):
            if db is None or round(tt, 3) in seen:
                continue
            seen.add(round(tt, 3))
            e.append(f'<line x1="{px(tt):.1f}" y1="12" x2="{px(tt):.1f}" y2="{TILE_Y + TILE_H}" '
                     f'stroke="{TXT}" stroke-width="0.6" opacity="0.35" stroke-dasharray="3,3"/>')
            lx = min(max(px(tt), 26), W - 26)        # keep edge labels inside the canvas
            e.append(f'<text x="{lx:.1f}" y="{TILE_Y + TILE_H + 22}" fill="{SUB}" '
                     f'font-size="11" text-anchor="middle">{db:g}dB</text>')
    e.append(f'<text x="12" y="{H - 8}" fill="{SUB}" font-size="11">word</text>')
    e.append(f'<rect x="52" y="{H - 18}" width="12" height="12" fill="{WORD}" rx="2"/>')
    e.append(f'<text x="72" y="{H - 8}" fill="{SUB}" font-size="11">silence</text>')
    e.append(f'<rect x="124" y="{H - 18}" width="12" height="12" fill="{SIL}" stroke="{SUB}" stroke-width="0.5" rx="2"/>')
    e.append(f'<text x="{W - 12}" y="{H - 8}" fill="{SUB}" font-size="11" text-anchor="end">'
             f'boundary labels = dB above noise floor (0 = a real pause)</text>')
    e.append("</svg>")
    open(out, "w").write("\n".join(e))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
