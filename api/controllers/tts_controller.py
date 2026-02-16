from __future__ import annotations

import logging
import time

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
        result = service.stream_synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    e2e_ttfb_ms = (time.perf_counter() - request_started_at) * 1000.0
    if result.attempts > 1:
        logger.warning(
            "TTS e2e TTFB %.0fms voice=%s (%d attempts)",
            e2e_ttfb_ms, request.voice, result.attempts,
        )

    return StreamingResponse(
        result.stream,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(service.sample_rate),
            "X-Channels": "1",
            "X-Sample-Width": "2",
        },
    )
