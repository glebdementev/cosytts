from fastapi import APIRouter, Request

from api.controllers.tts_controller import health_response

router = APIRouter()


@router.get("/health", response_model=None)
def health(request: Request):  # type: ignore[no-untyped-def]
    return health_response(service=request.app.state.service, init_error=request.app.state.init_error)
