import logging

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.tts import router as tts_router
from api.services.streaming import CosyVoice3StreamingTtsService


app = FastAPI(title="CosyVoice3 Streaming TTS Server")
logger = logging.getLogger("cosyvoice3")


class _ExcludeHealthAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health " not in record.getMessage()


def _configure_logging() -> None:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(filter_, _ExcludeHealthAccessFilter) for filter_ in access_logger.filters):
        access_logger.addFilter(_ExcludeHealthAccessFilter())


_configure_logging()

app.include_router(health_router)
app.include_router(tts_router)


@app.on_event("startup")
def startup() -> None:
    app.state.service = None
    app.state.init_error = None
    try:
        app.state.service = CosyVoice3StreamingTtsService()
    except Exception as exc:
        app.state.init_error = str(exc)
        logger.exception("CosyVoice3 service failed to initialize")
