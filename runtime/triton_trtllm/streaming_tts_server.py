import base64
import io
import os
import queue
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from functools import partial

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


class StreamingTtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    reference_text: str = ""
    reference_audio_base64: str | None = None


@dataclass
class _RequestState:
    completed_requests: queue.Queue


def _stream_callback(state: _RequestState, result, error) -> None:
    state.completed_requests.put((result, error))


class TritonStreamingTtsService:
    def __init__(self) -> None:
        self.server_addr = os.getenv("TRITON_SERVER_ADDR", "localhost:8001")
        self.model_name = os.getenv("TRITON_MODEL_NAME", "cosyvoice2")
        self.sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
        self.request_timeout_s = int(os.getenv("TTS_REQUEST_TIMEOUT_S", "300"))
        self.use_spk2info_cache = os.getenv("USE_SPK2INFO_CACHE", "false").lower() in {"1", "true", "yes"}

    def health(self) -> bool:
        client = grpcclient.InferenceServerClient(url=self.server_addr, verbose=False)
        try:
            return (
                client.is_server_live()
                and client.is_server_ready()
                and client.is_model_ready(self.model_name)
            )
        finally:
            client.close()

    def _decode_reference_audio(self, encoded_audio: str) -> np.ndarray:
        try:
            raw = base64.b64decode(encoded_audio, validate=True)
        except Exception as exc:
            raise ValueError("reference_audio_base64 is not valid base64") from exc

        try:
            waveform, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception as exc:
            raise ValueError("reference_audio_base64 is not a valid audio file") from exc

        if sample_rate != 16000:
            raise ValueError("reference audio sample rate must be 16000 Hz")

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform

    def _build_inputs(self, request: StreamingTtsRequest):
        if self.use_spk2info_cache:
            target_text = np.array([request.text], dtype=object).reshape((1, 1))
            text_input = grpcclient.InferInput("target_text", [1, 1], "BYTES")
            text_input.set_data_from_numpy(target_text)
            return [text_input]

        if not request.reference_audio_base64:
            raise ValueError("reference_audio_base64 is required when USE_SPK2INFO_CACHE=false")

        waveform = self._decode_reference_audio(request.reference_audio_base64)
        waveform = waveform.reshape(1, -1).astype(np.float32)
        wav_len = np.array([[waveform.shape[1]]], dtype=np.int32)

        reference_text = np.array([request.reference_text], dtype=object).reshape((1, 1))
        target_text = np.array([request.text], dtype=object).reshape((1, 1))

        inputs = [
            grpcclient.InferInput("reference_wav", waveform.shape, np_to_triton_dtype(waveform.dtype)),
            grpcclient.InferInput("reference_wav_len", wav_len.shape, np_to_triton_dtype(wav_len.dtype)),
            grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
            grpcclient.InferInput("target_text", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(waveform)
        inputs[1].set_data_from_numpy(wav_len)
        inputs[2].set_data_from_numpy(reference_text)
        inputs[3].set_data_from_numpy(target_text)
        return inputs

    @staticmethod
    def _to_pcm16le(chunk: np.ndarray) -> bytes:
        if chunk is None:
            return b""
        mono = np.asarray(chunk).reshape(-1)
        audio_int16 = np.clip(mono, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
        return audio_int16.tobytes()

    def stream_synthesize(self, request: StreamingTtsRequest) -> Generator[bytes, None, None]:
        inputs = self._build_inputs(request)
        outputs = [grpcclient.InferRequestedOutput("waveform")]
        state = _RequestState(completed_requests=queue.Queue())
        request_id = str(uuid.uuid4())

        client = grpcclient.InferenceServerClient(url=self.server_addr, verbose=False)
        try:
            client.start_stream(callback=partial(_stream_callback, state))
            client.async_stream_infer(
                self.model_name,
                inputs,
                request_id=request_id,
                outputs=outputs,
                enable_empty_final_response=True,
            )

            while True:
                try:
                    result, error = state.completed_requests.get(timeout=self.request_timeout_s)
                except queue.Empty as exc:
                    raise RuntimeError("Timed out waiting for streaming chunk from Triton") from exc

                if error:
                    if isinstance(error, InferenceServerException):
                        raise RuntimeError(error.message()) from error
                    raise RuntimeError(str(error))

                response = result.get_response()
                final = False
                params = getattr(response, "parameters", {})
                if "triton_final_response" in params:
                    final = params["triton_final_response"].bool_param
                if final:
                    break

                chunk = result.as_numpy("waveform")
                payload = self._to_pcm16le(chunk)
                if payload:
                    yield payload
        finally:
            try:
                client.stop_stream()
            finally:
                client.close()


app = FastAPI(title="CosyVoice Triton Streaming TTS Server")


@app.on_event("startup")
def startup() -> None:
    app.state.service = TritonStreamingTtsService()


@app.get("/health", response_model=None)
def health():  # type: ignore[no-untyped-def]
    service: TritonStreamingTtsService = app.state.service
    if not service.health():
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}


@app.post("/synthesize/stream")
def synthesize_stream(request: StreamingTtsRequest) -> StreamingResponse:
    service: TritonStreamingTtsService = app.state.service
    if not service.health():
        raise HTTPException(status_code=503, detail="Triton server is not ready.")

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
