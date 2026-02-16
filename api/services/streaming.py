from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass

import soundfile as sf
import torch

from api.schemas.tts import StreamingTtsRequest
from api.services.trimming import LeadingSilenceTrimmer, coerce_pcm_bytes

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402
from fastcosyvoice import FastCosyVoice3  # noqa: E402

logger = logging.getLogger("cosyvoice3")

MAX_RETRIES = 3
TTFB_TIMEOUT_S = 1.0


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


@dataclass
class _RequestState:
    registered_voices: set[str]


class CosyVoice3StreamingTtsService:
    def __init__(self) -> None:
        self.model_dir = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
        self.use_fast_pipeline = _env_bool("USE_FASTCOSYVOICE3", "true")
        self.fp16 = _env_bool("FP16", "true")
        self.load_trt_flow = _env_bool("LOAD_TRT_FLOW", "false")
        self.load_trt_llm = _env_bool("LOAD_TRT_LLM", "false")
        self.trt_llm_dtype = os.getenv("TRT_LLM_DTYPE", "bfloat16")
        self.trt_llm_kv_cache_tokens = int(os.getenv("TRT_LLM_KV_CACHE_TOKENS", "8192"))
        self.trt_concurrent = int(os.getenv("TRT_CONCURRENT", "1"))
        self.instruction = os.getenv("PROMPT_INSTRUCTION", "You are a helpful assistant.")
        self.text_frontend = _env_bool("TEXT_FRONTEND", "true")
        self.auto_stress = _env_bool("AUTO_STRESS", "false")
        self.use_spk2info_cache = _env_bool("USE_SPK2INFO_CACHE", "true")
        self.trim_leading_silence = _env_bool("TRIM_LEADING_SILENCE", "true")
        self.silence_threshold = int(os.getenv("SILENCE_THRESHOLD_INT16", "256"))
        self.voices_dir = os.getenv(
            "VOICES_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "voices")),
        )

        torch.set_float32_matmul_precision("high")
        self._state = _RequestState(registered_voices=set())
        self._inference_lock = threading.Lock()
        self.ready = False

        if self.use_fast_pipeline:
            self.model = FastCosyVoice3(
                model_dir=self.model_dir,
                fp16=self.fp16,
                load_trt=self.load_trt_flow,
                load_trt_llm=self.load_trt_llm,
                trt_concurrent=self.trt_concurrent,
                trt_llm_dtype=self.trt_llm_dtype,
                trt_llm_kv_cache_tokens=self.trt_llm_kv_cache_tokens,
            )
        else:
            self.model = CosyVoice3(
                model_dir=self.model_dir,
                fp16=self.fp16,
                load_trt=self.load_trt_flow,
                load_vllm=False,
                trt_concurrent=self.trt_concurrent,
            )

        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self.ready = True

    def health(self) -> bool:
        return self.model is not None and self.ready

    def _load_voice_profile(self, voice: str) -> tuple[str, str]:
        if "/" in voice or "\\" in voice:
            raise ValueError("voice must be a simple name without path separators")

        wav_path = os.path.join(self.voices_dir, f"{voice}.wav")
        txt_path = os.path.join(self.voices_dir, f"{voice}.txt")

        if not os.path.exists(wav_path):
            raise ValueError(f"Unknown voice '{voice}': missing {voice}.wav in voices directory")
        if not os.path.exists(txt_path):
            raise ValueError(f"Unknown voice '{voice}': missing {voice}.txt in voices directory")

        try:
            _, sample_rate = sf.read(wav_path, dtype="float32")
        except Exception as exc:
            raise ValueError(f"Failed to read voice reference audio: {wav_path}") from exc

        if sample_rate not in (16000, 24000):
            raise ValueError(
                f"Voice '{voice}' sample rate must be 16000 or 24000 Hz, got {sample_rate} Hz"
            )

        try:
            with open(txt_path, "r", encoding="utf-8") as handle:
                reference_text = handle.read().strip()
        except OSError as exc:
            raise ValueError(f"Failed to read voice reference text: {txt_path}") from exc

        if not reference_text:
            raise ValueError(f"Voice '{voice}' reference text file is empty: {txt_path}")

        return wav_path, reference_text

    def _ensure_voice_registered(self, prompt_text: str, wav_path: str, spk_id: str) -> None:
        if self.use_spk2info_cache and spk_id not in self._state.registered_voices:
            self.model.add_zero_shot_spk(prompt_text, wav_path, spk_id)
            self._state.registered_voices.add(spk_id)

    # ------------------------------------------------------------------
    # Raw inference generators (no retry, no TTFB check)
    # ------------------------------------------------------------------

    def _raw_stream_fast(
        self,
        request: StreamingTtsRequest,
        prompt_text: str,
        wav_path: str,
        spk_id: str,
        trimmer: LeadingSilenceTrimmer | None,
    ) -> Generator[bytes, None, None]:
        for chunk in self.model.inference_zero_shot_stream(
            tts_text=request.text,
            prompt_text=prompt_text,
            prompt_wav=wav_path,
            zero_shot_spk_id=spk_id,
            text_frontend=self.text_frontend,
            auto_stress=self.auto_stress,
        ):
            chunk_bytes = coerce_pcm_bytes(chunk)
            if trimmer is None:
                if chunk_bytes:
                    yield chunk_bytes
                continue
            trimmed = trimmer.process(chunk_bytes)
            if trimmed:
                yield trimmed

    def _raw_stream_standard(
        self,
        request: StreamingTtsRequest,
        prompt_text: str,
        wav_path: str,
        spk_id: str,
        trimmer: LeadingSilenceTrimmer | None,
    ) -> Generator[bytes, None, None]:
        for model_output in self.model.inference_zero_shot(
            tts_text=request.text,
            prompt_text=prompt_text,
            prompt_wav=wav_path,
            zero_shot_spk_id=spk_id,
            stream=True,
        ):
            speech = model_output.get("tts_speech")
            if speech is None:
                continue
            speech = speech.squeeze().clamp(-1.0, 1.0)
            audio_int16 = (speech * 32767).to(torch.int16)
            chunk = audio_int16.cpu().numpy().tobytes()
            if trimmer is None:
                yield chunk
                continue
            trimmed = trimmer.process(chunk)
            if trimmed:
                yield trimmed

    # ------------------------------------------------------------------
    # Safe cancellation helpers
    # ------------------------------------------------------------------

    def _drain_generator(self, gen: Generator[bytes, None, None]) -> None:
        """Consume a generator to completion so model cleanup code runs.

        CosyVoice3 (non-fast) stores per-request state in UUID-keyed dicts
        and only cleans them up after the last yield.  Simply closing the
        generator would skip that cleanup and leak memory.

        FastCosyVoice3 uses function-local state with daemon threads, so
        closing is safe, but draining is also harmless.
        """
        try:
            for _ in gen:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API — streaming with TTFB retry
    # ------------------------------------------------------------------

    def stream_synthesize(self, request: StreamingTtsRequest) -> Generator[bytes, None, None]:
        wav_path, reference_text = self._load_voice_profile(request.voice)
        prompt_text = f"{self.instruction}<|endofprompt|>{reference_text}"
        spk_id = request.voice if self.use_spk2info_cache else ""

        self._ensure_voice_registered(prompt_text, wav_path, spk_id)

        with self._inference_lock:
            yield from self._stream_with_retries(
                request, prompt_text, wav_path, spk_id,
            )

    def _stream_with_retries(
        self,
        request: StreamingTtsRequest,
        prompt_text: str,
        wav_path: str,
        spk_id: str,
    ) -> Generator[bytes, None, None]:
        for attempt in range(1, MAX_RETRIES + 1):
            trimmer = (
                LeadingSilenceTrimmer(self.silence_threshold)
                if self.trim_leading_silence
                else None
            )

            if self.use_fast_pipeline:
                gen = self._raw_stream_fast(
                    request, prompt_text, wav_path, spk_id, trimmer,
                )
            else:
                gen = self._raw_stream_standard(
                    request, prompt_text, wav_path, spk_id, trimmer,
                )

            first_chunk, ttfb_ok = self._await_first_chunk(gen, attempt)

            if ttfb_ok and first_chunk is not None:
                yield first_chunk
                yield from gen
                return

            # TTFB exceeded — cancel this attempt
            self._drain_generator(gen)

        raise RuntimeError(
            f"TTS TTFB exceeded {TTFB_TIMEOUT_S:.0f}s on all "
            f"{MAX_RETRIES} attempts for voice={request.voice}"
        )

    def _await_first_chunk(
        self,
        gen: Generator[bytes, None, None],
        attempt: int,
    ) -> tuple[bytes | None, bool]:
        """Pull the first chunk and check TTFB.

        Returns (first_chunk, ttfb_ok).  If the generator is exhausted
        before yielding anything, returns (None, True) — empty output is
        not a timeout.
        """
        t0 = time.perf_counter()

        first_chunk: bytes | None = None
        for chunk in gen:
            first_chunk = chunk
            break

        elapsed_s = time.perf_counter() - t0

        if first_chunk is None:
            return None, True

        if elapsed_s > TTFB_TIMEOUT_S:
            logger.error(
                "TTS TTFB %.0fms (>%.0fms) on attempt %d — retrying",
                elapsed_s * 1000,
                TTFB_TIMEOUT_S * 1000,
                attempt,
            )
            return first_chunk, False

        return first_chunk, True
