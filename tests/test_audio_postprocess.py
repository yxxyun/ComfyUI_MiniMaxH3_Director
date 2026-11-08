"""Run with ComfyUI on PYTHONPATH (torch only, no FFmpeg needed)."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director import audio_postprocess  # noqa: E402


def make_segment(level_dbfs, sample_rate, seconds, freq=300.0):
    """Tone at a requested RMS level."""
    samples = int(sample_rate * seconds)
    times = torch.arange(samples, dtype=torch.float32) / sample_rate
    tone = torch.sin(2 * torch.pi * freq * times)
    amplitude = float(10 ** (level_dbfs / 20) * 2 ** 0.5)
    return (tone * amplitude).reshape(1, 1, samples)


class SegmentLevelTests(unittest.TestCase):
    sample_rate = 32000

    def test_both_directions_move_toward_median(self):
        parts = [make_segment(-20.0, self.sample_rate, 1.0),
                 make_segment(-26.0, self.sample_rate, 1.0)]
        merged = torch.cat(parts, dim=-1)
        gains = audio_postprocess.segment_gains_db(
            merged, self.sample_rate, [p.shape[-1] for p in parts],
        )
        # Median of (-20, -26) is -23: the loud side comes down, the quiet side goes up.
        self.assertAlmostEqual(gains[0], -3.0, delta=0.5)
        self.assertAlmostEqual(gains[1], 3.0, delta=0.5)

    def test_median_target_is_not_overall_loudness(self):
        """Segments already at the same level must not be moved at all."""
        parts = [make_segment(-22.0, self.sample_rate, 1.0)] * 3
        merged = torch.cat(parts, dim=-1)
        gains = audio_postprocess.segment_gains_db(
            merged, self.sample_rate, [p.shape[-1] for p in parts],
        )
        for gain in gains:
            self.assertAlmostEqual(gain, 0.0, delta=0.1)

    def test_correction_is_clamped(self):
        parts = [make_segment(-10.0, self.sample_rate, 1.0),
                 make_segment(-60.0, self.sample_rate, 1.0)]
        merged = torch.cat(parts, dim=-1)
        gains = audio_postprocess.segment_gains_db(
            merged, self.sample_rate, [p.shape[-1] for p in parts],
        )
        self.assertLessEqual(max(abs(g) for g in gains), audio_postprocess.MAX_GAIN_DB)
    def test_silent_segment_is_not_boosted(self):
        parts = [make_segment(-18.0, self.sample_rate, 1.0),
                 torch.zeros(1, 1, self.sample_rate)]
        merged = torch.cat(parts, dim=-1)
        gains = audio_postprocess.segment_gains_db(
            merged, self.sample_rate, [p.shape[-1] for p in parts],
        )
        self.assertEqual(gains[1], 0.0)

    def test_single_segment_is_untouched(self):
        merged = make_segment(-25.0, self.sample_rate, 1.0)
        self.assertEqual(
            audio_postprocess.segment_gains_db(merged, self.sample_rate, [merged.shape[-1]]),
            [0.0],
        )


class BoundaryRampTests(unittest.TestCase):
    sample_rate = 32000

    def test_join_stays_sample_continuous(self):
        """A gain ramp must not notch the join the way a crossfade would."""
        parts = [make_segment(-14.0, self.sample_rate, 1.0),
                 make_segment(-26.0, self.sample_rate, 1.0)]
        merged = torch.cat(parts, dim=-1)
        counts = [p.shape[-1] for p in parts]
        gains = audio_postprocess.segment_gains_db(merged, self.sample_rate, counts)
        ramped = audio_postprocess.apply_gain_ramp(
            merged, counts, gains, self.sample_rate,
        )

        boundary = counts[0]
        window = slice(boundary - 320, boundary + 320)
        # Peak around the join should stay in the same order of magnitude as the
        # steady state: a crossfade would drop it by 20 dB or more.
        steady = float(merged[..., boundary - 6400:boundary - 320].abs().max())
        near = float(ramped[..., window].abs().max())
        self.assertGreater(near, steady * 0.3)

    def test_gain_curve_reaches_both_sides(self):
        parts = [make_segment(-20.0, self.sample_rate, 1.0) * 0 + 0.1,
                 make_segment(-20.0, self.sample_rate, 1.0) * 0 + 0.1]
        merged = torch.cat(parts, dim=-1)
        counts = [p.shape[-1] for p in parts]
        ramped = audio_postprocess.apply_gain_ramp(
            merged, counts, [0.0, 6.0], self.sample_rate,
        )
        early = float(ramped[..., 1000].abs().max())
        late = float(ramped[..., -1000].abs().max())
        self.assertAlmostEqual(early, 0.1, delta=0.005)
        self.assertAlmostEqual(late, 0.1 * 10 ** (6.0 / 20), delta=0.01)


class PeakLimitTests(unittest.TestCase):
    def test_peak_is_pulled_to_ceiling(self):
        wave = torch.full((1, 2, 1000), 0.95)
        limited = audio_postprocess.limit_peak(wave, -2.0)
        self.assertAlmostEqual(float(limited.abs().max()), 10 ** (-2.0 / 20), places=5)

    def test_quiet_signal_is_untouched(self):
        wave = torch.full((1, 2, 1000), 0.1)
        self.assertTrue(torch.equal(audio_postprocess.limit_peak(wave, -2.0), wave))


class CleanupTests(unittest.TestCase):
    sample_rate = 32000

    def test_cleanup_keeps_shape_and_dtype(self):
        parts = [make_segment(-18.0, self.sample_rate, 0.5),
                 make_segment(-30.0, self.sample_rate, 0.5)]
        merged = torch.cat(parts, dim=-1).repeat(1, 2, 1)
        out = audio_postprocess.cleanup_merged_audio(
            merged, self.sample_rate, [p.shape[-1] for p in parts],
        )
        self.assertEqual(out.shape, merged.shape)
        self.assertEqual(out.dtype, merged.dtype)
        self.assertLessEqual(float(out.abs().max()), 10 ** (-2.0 / 20) + 1e-6)

    def test_level_match_can_be_disabled(self):
        parts = [make_segment(-12.0, self.sample_rate, 0.5),
                 make_segment(-16.0, self.sample_rate, 0.5)]
        merged = torch.cat(parts, dim=-1).repeat(1, 2, 1)
        counts = [p.shape[-1] for p in parts]
        # Ceiling well above the material so only level matching can change it.
        matched = audio_postprocess.cleanup_merged_audio(
            merged, self.sample_rate, counts, ceiling_dbfs=6.0,
        )
        unmatched = audio_postprocess.cleanup_merged_audio(
            merged, self.sample_rate, counts, level_match=False, ceiling_dbfs=6.0,
        )
        head, tail = counts[0] // 2, counts[1] // 2
        # -12 comes down, -16 goes up, so the step at the join shrinks.
        self.assertLess(
            float(matched[..., :head].abs().max()),
            float(unmatched[..., :head].abs().max()),
        )
        self.assertGreater(
            float(matched[..., -tail:].abs().max()),
            float(unmatched[..., -tail:].abs().max()),
        )


if __name__ == "__main__":
    unittest.main()
