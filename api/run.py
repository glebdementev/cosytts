from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "api.streaming_tts_server:app",
        host="0.0.0.0",
        port=8090,
        log_level="info",
    )


if __name__ == "__main__":
    main()
