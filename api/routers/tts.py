import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.controllers.tts_controller import synthesize_stream_response
from api.schemas.tts import StreamingTtsRequest

router = APIRouter()


@router.post("/synthesize/stream")
def synthesize_stream(request: Request, payload: StreamingTtsRequest) -> StreamingResponse:
    request_started_at = time.perf_counter()
    return synthesize_stream_response(
        service=request.app.state.service,
        init_error=request.app.state.init_error,
        request=payload,
        request_started_at=request_started_at,
    )
