from __future__ import annotations

import torch


class LeadingSilenceTrimmer:
    """Trim only leading silence from an int16 PCM stream."""

    def __init__(self, threshold: int, fade_in_samples: int = 64) -> None:
        self.threshold = max(0, threshold)
        self.fade_in_samples = max(0, fade_in_samples)
        self._started = False
        self._pending = b""

    def process(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""

        data = self._pending + chunk
        self._pending = b""

        if len(data) % 2 == 1:
            self._pending = data[-1:]
            data = data[:-1]

        if not data:
            return b""

        if self._started:
            return data

        samples = memoryview(data).cast("h")
        first_non_silent_idx = None
        for idx, sample in enumerate(samples):
            if abs(sample) >= self.threshold:
                first_non_silent_idx = idx
                break

        if first_non_silent_idx is None:
            return b""

        self._started = True
        trimmed = data[first_non_silent_idx * 2 :]
        return self._apply_fade_in(trimmed)

    def _apply_fade_in(self, data: bytes) -> bytes:
        if self.fade_in_samples == 0 or len(data) < 2:
            return data

        pcm = bytearray(data)
        samples = memoryview(pcm).cast("h")
        fade_count = min(self.fade_in_samples, len(samples))

        if fade_count <= 1:
            return bytes(pcm)

        denominator = fade_count - 1
        for idx in range(fade_count):
            # Short linear ramp on the very first audible samples.
            samples[idx] = int(samples[idx] * idx / denominator)

        return bytes(pcm)


def coerce_pcm_bytes(chunk: object) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, bytearray):
        return bytes(chunk)
    if isinstance(chunk, memoryview):
        return chunk.tobytes()
    if isinstance(chunk, torch.Tensor):
        return chunk.detach().cpu().numpy().tobytes()
    if hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    raise TypeError(f"Unsupported stream chunk type: {type(chunk).__name__}")
