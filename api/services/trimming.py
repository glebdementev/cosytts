from __future__ import annotations

import torch


class LeadingSilenceTrimmer:
    """Trim only leading silence from an int16 PCM stream."""

    def __init__(self, threshold: int) -> None:
        self.threshold = max(0, threshold)
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
        return data[first_non_silent_idx * 2 :]


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
