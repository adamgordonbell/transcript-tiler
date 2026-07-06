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

## Recipe: transcript → cuttable filler list

```bash
# 0. get a word-timestamped transcript (any of Whisper/WhisperX/generic)
whisper track.wav --model medium --word_timestamps True --output_format json

# 1. build the labeling (corrected word boundaries + edge freedom)
uv run wordtiler tile track.wav track.json                  # → track.labels.json

# 2. calibrate for YOUR audio: see the noise floor + edge-dB distribution
uv run wordtiler stats track.labels.json

# 3. emit the words free enough to cut
uv run wordtiler stops track.labels.json --only um,uh                # clean both sides
uv run wordtiler stops track.labels.json --only um,uh --seam-db 10   # + one-sided cuts
```

There is no separate "measure the silence level" step — the noise floor is
self-estimated per chunk (20th-percentile fine RMS × 1.5) and every
`startDb`/`endDb` in the output is *relative to it* (0 dB = at the floor).
The estimated floor is reported in the output (`"floors"`, dBFS) and by
`stats`, so you can sanity-check it.

**Gated vs ungated audio.** On gated tracks (edges pumped to digital zero)
the floor sits at the gate residue (~−77 dBFS) and free edges score exactly
0 dB — the distribution is sharply bimodal and the defaults just work. On
ungated audio the floor lands on your room tone and free-edge dBs will sit a
little above 0. Run `stats` first: healthy audio shows pause-adjacent edges
clustered near 0 dB and word-abutting edges ≫ (median ~30); set `--max-db`
just above the pause-edge p95 (the `stats` output suggests a value).

Other outputs:

```bash
uv run wordtiler tile clip.wav clip.json --format textgrid  # → Praat/ELAN interop
uv run wordtiler tile clip.wav clip.json --format audacity  # → Audacity label track
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
