from __future__ import annotations

import logging
import time
from collections.abc import Generator

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas.tts import StreamingTtsRequest
from api.services.streaming import CosyVoice3StreamingTtsService

logger = logging.getLogger("cosyvoice3")


def health_response(
    service: CosyVoice3StreamingTtsService | None,
    init_error: str | None,
) -> JSONResponse | dict[str, str]:
    if init_error:
        return JSONResponse(status_code=503, content={"status": "error", "detail": init_error})
    if service is None or not service.health():
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}


def synthesize_stream_response(
    service: CosyVoice3StreamingTtsService | None,
    init_error: str | None,
    request: StreamingTtsRequest,
    request_started_at: float,
) -> StreamingResponse:
    if init_error:
        raise HTTPException(status_code=503, detail=init_error)
    if service is None or not service.health():
        raise HTTPException(status_code=503, detail="CosyVoice3 service is not ready.")

    try:
        stream = service.stream_synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wrapped_stream = _stream_with_ttfb(
        stream=stream,
        request_started_at=request_started_at,
        voice=request.voice,
    )

    return StreamingResponse(
        wrapped_stream,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(service.sample_rate),
            "X-Channels": "1",
            "X-Sample-Width": "2",
        },
    )


def _stream_with_ttfb(
    stream: Generator[bytes, None, None],
    request_started_at: float,
    voice: str,
) -> Generator[bytes, None, None]:
    first_chunk_sent = False

    for chunk in stream:
        if not first_chunk_sent:
            ttfb_ms = (time.perf_counter() - request_started_at) * 1000.0
            if ttfb_ms > 1000:
                logger.error("TTS TTFB %.1fms (>1000ms) voice=%s", ttfb_ms, voice)
            elif ttfb_ms > 500:
                logger.warning("TTS TTFB %.1fms (>500ms) voice=%s", ttfb_ms, voice)
            else:
                logger.info("TTS TTFB %.1fms voice=%s", ttfb_ms, voice)
            first_chunk_sent = True
        yield chunk
