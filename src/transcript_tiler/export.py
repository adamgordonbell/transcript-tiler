"""export.py — write the labeling out as JSON (native), Praat TextGrid, or
Audacity labels.

The native JSON is the rich format (edge freedom lives only there). TextGrid
carries the tiling losslessly — an IntervalTier is by definition a complete
cover — as two tiers: "words" (silence/noise = empty text) and "kind"
(word|silence|noise). Audacity labels are the flat tab-separated form.
"""
import json


def to_json(labeling, path):
    with open(path, "w") as f:
        json.dump({k: v for k, v in labeling.items() if not k.startswith("_")}, f, indent=2)


def _tg_escape(s):
    return s.replace('"', '""')


def to_textgrid(labeling, path):
    labels = labeling["labels"]
    xmax = labels[-1]["end"] if labels else 0.0
    tiers = [
        ("words", [(t["start"], t["end"], t["w"] if t["kind"] == "word" else "") for t in labels]),
        ("kind", [(t["start"], t["end"], t["kind"]) for t in labels]),
    ]
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for ti, (name, ivs) in enumerate(tiers, 1):
        out += [f"    item [{ti}]:", '        class = "IntervalTier"', f'        name = "{name}"',
                "        xmin = 0", f"        xmax = {xmax}", f"        intervals: size = {len(ivs)}"]
        for ii, (a, b, txt) in enumerate(ivs, 1):
            out += [f"        intervals [{ii}]:", f"            xmin = {a}", f"            xmax = {b}",
                    f'            text = "{_tg_escape(txt)}"']
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def to_audacity(labeling, path):
    """Audacity label track: start<TAB>end<TAB>text, one line per tile.
    Import via Tracks > Edit Labels or File > Import > Labels."""
    with open(path, "w") as f:
        for t in labeling["labels"]:
            txt = t["w"] if t["kind"] == "word" else f"[{t['kind']}]"
            f.write(f"{t['start']:.3f}\t{t['end']:.3f}\t{txt}\n")


WRITERS = {"json": (to_json, ".labels.json"),
           "textgrid": (to_textgrid, ".TextGrid"),
           "audacity": (to_audacity, ".labels.txt")}
