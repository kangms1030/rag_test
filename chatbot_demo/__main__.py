"""python -m chatbot_demo 실행 진입점."""

from __future__ import annotations

import argparse

import uvicorn

from .app.main import create_app
from .config.settings import load_settings


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="school-network-chatbot-demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=settings.demo_port)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
