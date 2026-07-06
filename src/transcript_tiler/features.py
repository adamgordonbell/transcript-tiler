"""features.py — 100Hz frame features for word-boundary tiling.

RMS energy, silero VAD probability, and spectral flatness on a shared 10ms frame
grid, plus the 3-class frame classifier (silence | speech | nonspeech) the tiler
builds on. Pure feature extraction — no plotting, no CLI.
"""
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from silero_vad import load_silero_vad
import torch

HZ = 100                       # frame grid (10 ms)
SR16 = 16000
VAD_WIN = 512                  # silero v5 native window @16k (32 ms)

# 3-class labels
SIL, SPEECH, NONSP = 0, 1, 2
NAMES = {SIL: "silence", SPEECH: "speech", NONSP: "nonspeech"}


def load_mono(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def frame_rms(x, sr, hz=HZ):
    hop = sr / hz
    n = int(np.floor(len(x) / hop))
    win = int(round(hop))
    out = np.zeros(n)
    for i in range(n):
        a = int(round(i * hop))
        seg = x[a:a + win]
        if len(seg):
            out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
    return out


def frame_flatness(x, sr, hz=HZ, win_ms=25):
    """spectral flatness per frame: geomean(power)/mean(power). ~1 = noise-like
    (breath/fricative/room), ~0 = tonal (voiced speech)."""
    hop = sr / hz
    n = int(np.floor(len(x) / hop))
    w = int(round(win_ms / 1000 * sr))
    window = np.hanning(w)
    out = np.zeros(n)
    for i in range(n):
        c = int(round(i * hop))
        a = max(0, c - w // 2)
        seg = x[a:a + w]
        if len(seg) < w:
            seg = np.pad(seg, (0, w - len(seg)))
        sp = np.abs(np.fft.rfft(seg * window)) ** 2 + 1e-12
        out[i] = np.exp(np.mean(np.log(sp))) / np.mean(sp)
    return out


_vad_model = None


def vad_probs(x, sr, n_frames, hz=HZ):
    """silero per-32ms prob, mapped onto the 100 Hz frame grid (step fill)."""
    global _vad_model
    x16 = resample_poly(x, SR16, sr).astype(np.float32)
    if _vad_model is None:
        _vad_model = load_silero_vad()
    model = _vad_model
    model.reset_states()
    probs, centers = [], []
    for i in range(0, len(x16) - VAD_WIN, VAD_WIN):
        chunk = torch.from_numpy(x16[i:i + VAD_WIN])
        with torch.no_grad():
            p = model(chunk, SR16).item()
        probs.append(p)
        centers.append((i + VAD_WIN / 2) / SR16)   # seconds
    probs, centers = np.array(probs), np.array(centers)
    if not len(probs):
        return np.zeros(n_frames), probs, centers
    t = np.arange(n_frames) / hz                          # nearest-center, vectorized
    j = np.clip(np.searchsorted(centers, t), 1, len(centers) - 1)
    j = np.where(t - centers[j - 1] <= centers[j] - t, j - 1, j)
    return probs[j], probs, centers


def classify_frames(rms, vad, flat, floor, vad_thr=0.5, flat_thr=0.45):
    lab = np.full(len(rms), SIL, dtype=int)
    for i in range(len(rms)):
        if vad[i] >= vad_thr:
            lab[i] = SPEECH
        elif rms[i] >= floor:
            # energy present but not speech -> breath/click/room if noise-like
            lab[i] = NONSP if flat[i] >= flat_thr else SPEECH
        else:
            lab[i] = SIL
    return lab
