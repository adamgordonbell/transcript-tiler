# wordtiler

Refine the sloppy word timestamps from your speech-to-text tool into an exact,
gap-free **word | silence | noise tiling** of the audio — using energy, silero
VAD, and spectral flatness only. No forced alignment, no re-transcription.

Every instant of the track gets exactly one label; boundaries are shared
transitions (a word's end *is* the next label's start). Each word also carries
**edge freedom**: whether each side borders a real pause (`startKind`/`endKind`)
and how deeply the signal dips to the noise floor there (`startDb`/`endDb`,
0 dB = a true pause) — so downstream tools can decide which words are safely
cuttable.

Against a hand-labeled golden set, this tiler scored **0 gross / 0 severe
boundary errors** (p90 44 ms) where raw STT marks and forced aligners
(MMS/wav2vec2, WhisperX) sat at 7–10 gross errors per set.

## Install

```bash
uv sync            # or: pip install .
```

## Use

```bash
# Whisper / WhisperX / generic word-timestamp JSON in, labeling out
uv run wordtiler tile clip.wav clip.whisper.json            # → clip.whisper.labels.json
uv run wordtiler tile clip.wav clip.json --format textgrid  # → Praat/ELAN interop
uv run wordtiler tile clip.wav clip.json --format audacity  # → Audacity label track

# words free enough to cut (fillers / stop words) — pure interval math, no audio
uv run wordtiler stops clip.labels.json --only um,uh --max-db 3
uv run wordtiler stops clip.labels.json                     # every free word
```

Scales to multi-hour files (chunked partial decode, split at silences) and
caches by content hash — re-runs on unchanged input are instant.

### Input formats (auto-detected)

| format | shape |
|---|---|
| `whisper` | `{"segments":[{"words":[{"word","start","end"}]}]}` — `whisper --word_timestamps True` |
| `whisperx` | same, or the flat `"word_segments"` list |
| `generic` | `{"words":[{"w"\|"word", "t"\|"start", "e"\|"end"}]}` |

### Output (native JSON)

```json
{
  "labels": [
    {"kind": "silence", "w": "",     "start": 0.0,  "end": 6.32},
    {"kind": "word",    "w": "Yeah", "start": 6.32, "end": 6.565,
     "startKind": "silence", "endKind": "word", "startDb": 0.0, "endDb": 34.1}
  ],
  "words": ["…derived view…"], "silences": [], "noise": []
}
```

`labels` is the source of truth — a complete ordered cover. `--format textgrid`
exports the tiling for Praat/ELAN/MFA comparison (edge freedom is JSON-only;
no standard slot for it).

## Library

```python
from wordtiler.adapters import load_transcript
from wordtiler.tile import enrich
from wordtiler.stops import free_words

labeling = enrich("clip.wav", load_transcript("clip.whisper.json"))
cuttable = free_words(labeling, max_db=3.0, only=["um", "uh"])
```

## How it works

1. **Coarse tiling** (10 ms grid): silero VAD + RMS + spectral flatness →
   3-class frames; merged speech segments split into per-segment energy
   *bursts* (threshold relative to each segment's own peak); words matched to
   bursts monotonically by centre. Gaps become silence, or noise where
   non-speech (breath/click) frames dominate.
2. **Edge refinement** (5 ms grid): word↔gap edges walk out to the noise
   floor, capturing soft attacks/releases the coarse threshold cuts short —
   with a fricative-aware jump so a trailing /s/ after a stop closure stays
   part of its word.
3. **Edge freedom**: each word boundary is annotated with its neighbour kind
   and the floor-margin dB of the quietest point near the boundary.
