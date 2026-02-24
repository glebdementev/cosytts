from __future__ import annotations

import logging
import os
import sys
import threading
import time
import queue
from collections.abc import Generator
from dataclasses import dataclass, field

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
    registered_voices: set[str] = field(default_factory=set)


@dataclass
class SynthesisResult:
    """Eagerly-validated first chunk + lazy remainder stream."""

    stream: Generator[bytes, None, None]
    ttfb_ms: float
    attempts: int


class CosyVoice3StreamingTtsService:
    def __init__(self) -> None:
        self.model_dir = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
        self.llm_model_path = os.getenv("LLM_MODEL_PATH", "")
        self.flow_model_path = os.getenv("FLOW_MODEL_PATH", "")
        self.hift_model_path = os.getenv("HIFT_MODEL_PATH", "")
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
        self.silence_threshold = int(os.getenv("SILENCE_THRESHOLD_INT16", "512"))
        self.voices_dir = os.getenv(
            "VOICES_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "voices")),
        )

        torch.set_float32_matmul_precision("high")
        self._state = _RequestState()
        self._inference_lock = threading.Lock()
        self.ready = False

        if self.use_fast_pipeline:
            self.model = FastCosyVoice3(
                model_dir=self.model_dir,
                llm_model_path=self.llm_model_path or None,
                flow_model_path=self.flow_model_path or None,
                hift_model_path=self.hift_model_path or None,
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
                llm_model_path=self.llm_model_path or None,
                flow_model_path=self.flow_model_path or None,
                hift_model_path=self.hift_model_path or None,
                fp16=self.fp16,
                load_trt=self.load_trt_flow,
                load_vllm=False,
                trt_concurrent=self.trt_concurrent,
            )

        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))

        warmup_voice = os.getenv("WARMUP_VOICE", "")
        if warmup_voice:
            self._warmup(warmup_voice)

        self.ready = True

    def health(self) -> bool:
        return self.model is not None and self.ready

    def _warmup(self, voice: str) -> None:
        """Run a short inference to trigger CUDA/PyTorch JIT compilation.

        Without this, the first real request pays a ~20-30s cold-start
        penalty while GPU kernels are compiled on demand.
        """
        logger.info("Warmup: running inference with voice=%s ...", voice)
        t0 = time.perf_counter()
        try:
            wav_path, reference_text = self._load_voice_profile(voice)
            prompt_text = f"{self.instruction}<|endofprompt|>{reference_text}"
            spk_id = voice if self.use_spk2info_cache else ""
            self._ensure_voice_registered(prompt_text, wav_path, spk_id)

            warmup_request = StreamingTtsRequest(text="Тест.", voice=voice)

            if self.use_fast_pipeline:
                gen = self._raw_stream_fast(
                    warmup_request, prompt_text, wav_path, spk_id, trimmer=None,
                )
            else:
                gen = self._raw_stream_standard(
                    warmup_request, prompt_text, wav_path, spk_id, trimmer=None,
                )
            for _ in gen:
                pass

            elapsed_s = time.perf_counter() - t0
            logger.info("Warmup complete in %.1fs", elapsed_s)
        except Exception:
            elapsed_s = time.perf_counter() - t0
            logger.exception("Warmup failed after %.1fs (non-fatal)", elapsed_s)

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

    def _start_stream_worker(
        self,
        gen: Generator[bytes, None, None],
        cancel_event: threading.Event,
    ) -> tuple[queue.Queue, threading.Thread, object]:
        end_sentinel: object = object()
        output_queue: queue.Queue = queue.Queue()

        def _run() -> None:
            try:
                for chunk in gen:
                    if cancel_event.is_set():
                        if self.use_fast_pipeline:
                            break
                        # For non-fast pipeline, keep draining for cleanup.
                        continue
                    output_queue.put(chunk)
            except Exception as exc:
                output_queue.put(exc)
            finally:
                if cancel_event.is_set() and self.use_fast_pipeline:
                    try:
                        gen.close()
                    except Exception:
                        pass
                output_queue.put(end_sentinel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return output_queue, thread, end_sentinel

    def _cancel_failed_attempt(
        self,
        cancel_event: threading.Event,
        worker: threading.Thread,
        attempt: int,
    ) -> None:
        cancel_event.set()
        if self.use_fast_pipeline:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(
                    "TTS worker still running after cancel on attempt %d",
                    attempt,
                )
            return
        worker.join()

    def _queue_stream(
        self,
        output_queue: queue.Queue,
        end_sentinel: object,
    ) -> Generator[bytes, None, None]:
        while True:
            item = output_queue.get()
            if item is end_sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    # ------------------------------------------------------------------
    # Public API — eager first-chunk with TTFB retry
    # ------------------------------------------------------------------

    def stream_synthesize(self, request: StreamingTtsRequest) -> SynthesisResult:
        """Start synthesis, eagerly validate TTFB, retry if slow.

        This is NOT a generator — it blocks until the first audio chunk
        arrives (or all retries are exhausted), so the caller can decide
        to return an HTTP error before committing a 200 response.

        Returns a SynthesisResult whose .stream() yields audio bytes.
        Raises RuntimeError if all attempts exceed the TTFB timeout.
        """
        wav_path, reference_text = self._load_voice_profile(request.voice)
        prompt_text = f"{self.instruction}<|endofprompt|>{reference_text}"
        spk_id = request.voice if self.use_spk2info_cache else ""

        self._ensure_voice_registered(prompt_text, wav_path, spk_id)

        with self._inference_lock:
            return self._synthesize_with_retries(
                request, prompt_text, wav_path, spk_id,
            )

    def _synthesize_with_retries(
        self,
        request: StreamingTtsRequest,
        prompt_text: str,
        wav_path: str,
        spk_id: str,
    ) -> SynthesisResult:
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
            cancel_event = threading.Event()
            output_queue, worker, end_sentinel = self._start_stream_worker(gen, cancel_event)

            t0 = time.perf_counter()
            try:
                first_item = output_queue.get(timeout=TTFB_TIMEOUT_S)
            except queue.Empty:
                first_item = None
            elapsed_s = time.perf_counter() - t0
            ttfb_ms = elapsed_s * 1000.0

            if first_item is None:
                logger.error(
                    "TTS TTFB exceeded %.0fms on attempt %d — retrying",
                    TTFB_TIMEOUT_S * 1000,
                    attempt,
                )
                self._cancel_failed_attempt(cancel_event, worker, attempt)
                continue

            if first_item is end_sentinel:
                logger.info("TTS produced no audio on attempt %d", attempt)
                return SynthesisResult(
                    stream=iter(()),  # type: ignore[arg-type]
                    ttfb_ms=ttfb_ms,
                    attempts=attempt,
                )

            if isinstance(first_item, Exception):
                raise first_item

            if elapsed_s > TTFB_TIMEOUT_S:
                logger.error(
                    "TTS TTFB %.0fms (>%.0fms) on attempt %d — retrying",
                    ttfb_ms,
                    TTFB_TIMEOUT_S * 1000,
                    attempt,
                )
                self._cancel_failed_attempt(cancel_event, worker, attempt)
                continue

            logger.info(
                "TTS TTFB %.0fms voice=%s (attempt %d)",
                ttfb_ms, request.voice, attempt,
            )
            rest_stream = self._queue_stream(output_queue, end_sentinel)
            return SynthesisResult(
                stream=self._chain_stream(first_item, rest_stream),
                ttfb_ms=ttfb_ms,
                attempts=attempt,
            )

        raise RuntimeError(
            f"TTS TTFB exceeded {TTFB_TIMEOUT_S:.0f}s on all "
            f"{MAX_RETRIES} attempts for voice={request.voice}"
        )

    def _chain_stream(
        self,
        first_item: bytes,
        rest_stream: Generator[bytes, None, None],
    ) -> Generator[bytes, None, None]:
        yield first_item
        yield from rest_stream
