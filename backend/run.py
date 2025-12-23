#!/usr/bin/env python3
"""Run the Ambient Desktop Agent backend server."""

import uvicorn
from app.config import settings


def main():
    """Start the server."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )


if __name__ == "__main__":
    main()

