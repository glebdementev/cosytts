import os
import sys
import threading
from collections.abc import Generator
from dataclasses import dataclass

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402
from fastcosyvoice import FastCosyVoice3  # noqa: E402


class StreamingTtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)


@dataclass
class _RequestState:
    registered_voices: set[str]


class CosyVoice3StreamingTtsService:
    def __init__(self) -> None:
        self.model_dir = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
        self.use_fast_pipeline = os.getenv("USE_FASTCOSYVOICE3", "true").lower() in {"1", "true", "yes"}
        self.fp16 = os.getenv("FP16", "true").lower() in {"1", "true", "yes"}
        self.load_trt_flow = os.getenv("LOAD_TRT_FLOW", "false").lower() in {"1", "true", "yes"}
        self.load_trt_llm = os.getenv("LOAD_TRT_LLM", "false").lower() in {"1", "true", "yes"}
        self.trt_llm_dtype = os.getenv("TRT_LLM_DTYPE", "bfloat16")
        self.trt_llm_kv_cache_tokens = int(os.getenv("TRT_LLM_KV_CACHE_TOKENS", "8192"))
        self.trt_concurrent = int(os.getenv("TRT_CONCURRENT", "1"))
        self.instruction = os.getenv("PROMPT_INSTRUCTION", "You are a helpful assistant.")
        self.text_frontend = os.getenv("TEXT_FRONTEND", "true").lower() in {"1", "true", "yes"}
        self.auto_stress = os.getenv("AUTO_STRESS", "false").lower() in {"1", "true", "yes"}
        self.use_spk2info_cache = os.getenv("USE_SPK2INFO_CACHE", "true").lower() in {"1", "true", "yes"}
        self.voices_dir = os.getenv(
            "VOICES_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "voices")),
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

        if sample_rate != 16000:
            raise ValueError(f"Voice '{voice}' sample rate must be 16000 Hz, got {sample_rate} Hz")

        try:
            with open(txt_path, "r", encoding="utf-8") as handle:
                reference_text = handle.read().strip()
        except OSError as exc:
            raise ValueError(f"Failed to read voice reference text: {txt_path}") from exc

        if not reference_text:
            raise ValueError(f"Voice '{voice}' reference text file is empty: {txt_path}")

        return wav_path, reference_text

    def stream_synthesize(self, request: StreamingTtsRequest) -> Generator[bytes, None, None]:
        wav_path, reference_text = self._load_voice_profile(request.voice)
        prompt_text = f"{self.instruction}<|endofprompt|>{reference_text}"
        spk_id = request.voice if self.use_spk2info_cache else ""

        if self.use_spk2info_cache and spk_id not in self._state.registered_voices:
            self.model.add_zero_shot_spk(prompt_text, wav_path, spk_id)
            self._state.registered_voices.add(spk_id)

        with self._inference_lock:
            if self.use_fast_pipeline:
                yield from self.model.inference_zero_shot_stream(
                    tts_text=request.text,
                    prompt_text=prompt_text,
                    prompt_wav=wav_path,
                    zero_shot_spk_id=spk_id,
                    text_frontend=self.text_frontend,
                    auto_stress=self.auto_stress,
                )
                return

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
                yield audio_int16.cpu().numpy().tobytes()


app = FastAPI(title="CosyVoice3 Streaming TTS Server")


@app.on_event("startup")
def startup() -> None:
    app.state.service = CosyVoice3StreamingTtsService()


@app.get("/health", response_model=None)
def health():  # type: ignore[no-untyped-def]
    service: CosyVoice3StreamingTtsService = app.state.service
    if not service.health():
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}


@app.post("/synthesize/stream")
def synthesize_stream(request: StreamingTtsRequest) -> StreamingResponse:
    service: CosyVoice3StreamingTtsService = app.state.service
    if not service.health():
        raise HTTPException(status_code=503, detail="CosyVoice3 service is not ready.")

    try:
        stream = service.stream_synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        stream,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(service.sample_rate),
            "X-Channels": "1",
            "X-Sample-Width": "2",
        },
    )
