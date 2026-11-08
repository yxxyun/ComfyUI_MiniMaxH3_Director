"""Cross-segment cleanup for merged Director audio exports.

Each segment's audio is decoded independently, so every segment arrives at its
own loudness with its own sub-bass layer from the audio VAE. A bare
``torch.cat`` therefore puts a level step at every join and keeps a constant
rumble under the whole timeline. On a measured 12-segment export the segment
median RMS spanned 23.6 dB and 6 of 11 joins stepped more than 6 dB.

No crossfade here on purpose: the waveform is already sample-continuous across
a join, and overlapping the two sides only carves an audible notch at every
boundary. Level is smoothed with a gain ramp instead, so the samples are never
scaled against each other.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.audio_postprocess")

# Per-segment correction cap, so a mostly-silent segment is not boosted into hiss.
MAX_GAIN_DB = 9.0
# Half-width of the gain ramp centred on a join, in seconds.
BOUNDARY_RAMP_SEC = 0.08
# Sub-bass from the audio VAE sits near 20-80 Hz; dialogue does not.
HIGHPASS_HZ = 80.0
# AAC decode overshoots the sample peak, so do not sit at 0 dBFS.
PEAK_CEILING_DBFS = -2.0
# Segments below this median frame level count as silence and are left alone.
SILENT_FLOOR_DBFS = -80.0


def _frame_rms_db(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-frame RMS in dBFS for a ``[..., T]`` tensor."""
    hop = max(1, int(round(sample_rate * 0.02)))
    frames = int(waveform.shape[-1]) // hop
    if frames <= 0:
        return waveform.new_empty(0)
    trimmed = waveform[..., : frames * hop].reshape(*waveform.shape[:-1], frames, hop)
    return 20.0 * torch.log10(trimmed.pow(2).mean(dim=-1).clamp_min(1e-20).sqrt())


def _spans(frame_counts, total: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for count in frame_counts or []:
        length = min(max(0, int(count)), total - cursor)
        if length <= 0:
            continue
        spans.append((cursor, cursor + length))
        cursor += length
    return spans


def segment_gains_db(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_counts,
    *,
    max_gain_db: float = MAX_GAIN_DB,
) -> list[float]:
    """Gain per span that pulls each audible segment toward the median level."""
    total = int(waveform.shape[-1])
    spans = _spans(frame_counts, total)
    levels: list[float | None] = []
    for start, end in spans:
        levels_in = _frame_rms_db(waveform[..., start:end], sample_rate)
        levels.append(float(levels_in.median()) if levels_in.numel() else None)

    audible = [level for level in levels if level is not None and level >= SILENT_FLOOR_DBFS]
    if len(audible) < 2:
        return [0.0] * len(spans)
    # Interpolated median: torch.median returns the lower of two middles, which
    # would bias an even number of segments all in one direction.
    target = float(torch.quantile(torch.tensor(audible), 0.5))

    gains = []
    for level in levels:
        if level is None or level < SILENT_FLOOR_DBFS:
            gains.append(0.0)
            continue
        gains.append(max(-max_gain_db, min(max_gain_db, target - level)))
    return gains


def apply_gain_ramp(
    waveform: torch.Tensor,
    frame_counts,
    gains_db: list[float],
    sample_rate: int,
    *,
    ramp_sec: float = BOUNDARY_RAMP_SEC,
) -> torch.Tensor:
    """Apply per-segment gains, ramping linearly in dB across each join."""
    total = int(waveform.shape[-1])
    spans = _spans(frame_counts, total)
    if not spans or len(gains_db) != len(spans):
        return waveform

    curve = torch.zeros(total, dtype=torch.float32)
    for (start, end), gain in zip(spans, gains_db):
        curve[start:end] = float(gain)

    ramp = max(1, int(round(sample_rate * ramp_sec)))
    for index in range(1, len(spans)):
        boundary = spans[index][0]
        half = min(ramp, (spans[index][0] - spans[index - 1][0]) // 4, (spans[index][1] - spans[index][0]) // 4)
        low, high = max(0, boundary - half), min(total, boundary + half)
        if high - low <= 1:
            continue
        curve[low:high] = torch.linspace(gains_db[index - 1], gains_db[index], high - low)

    linear = torch.pow(10.0, curve / 20.0).to(dtype=waveform.dtype, device=waveform.device)
    return waveform * linear


def highpass(waveform: torch.Tensor, sample_rate: int, cutoff_hz: float = HIGHPASS_HZ) -> torch.Tensor:
    """Drop the sub-bass layer; keep the waveform when torchaudio is absent."""
    if cutoff_hz <= 0 or sample_rate <= 0:
        return waveform
    try:
        import torchaudio
    except ImportError as exc:  # optional dependency
        log.warning("Audio high-pass skipped, torchaudio unavailable (%s).", exc)
        return waveform
    return torchaudio.functional.highpass_biquad(waveform, int(sample_rate), float(cutoff_hz))


def limit_peak(waveform: torch.Tensor, ceiling_dbfs: float = PEAK_CEILING_DBFS) -> torch.Tensor:
    """Scalar gain to keep the loudest sample under the ceiling.

    A scalar preserves the inter-segment balance a per-sample limiter would undo.
    """
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    target = float(10.0 ** (ceiling_dbfs / 20.0))
    if peak <= 0 or peak <= target:
        return waveform
    return waveform * (target / peak)


def cleanup_merged_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_counts,
    *,
    level_match: bool = True,
    highpass_hz: float = HIGHPASS_HZ,
    ceiling_dbfs: float = PEAK_CEILING_DBFS,
) -> torch.Tensor:
    """Level-match, high-pass and peak-limit a concatenated ``[1, C, T]`` waveform."""
    if waveform.ndim != 3 or waveform.numel() == 0:
        return waveform
    work = waveform.to(torch.float32)

    if level_match and frame_counts and len(frame_counts) > 1:
        gains = segment_gains_db(work, sample_rate, frame_counts)
        if any(abs(gain) > 0.01 for gain in gains):
            work = apply_gain_ramp(work, frame_counts, gains, sample_rate)
            log.info("Segment audio level match: %s dB.", " ".join(f"{gain:+.1f}" for gain in gains))

    work = highpass(work, sample_rate, highpass_hz)
    work = limit_peak(work, ceiling_dbfs)
    return work.to(dtype=waveform.dtype)
