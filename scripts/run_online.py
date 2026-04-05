#!/usr/bin/env python
"""Start the online BCI simulation server.

Usage:
    python scripts/run_online.py --model results/fbcsp_svm_2a/models/fbcsp_svm_2a_sub1.pkl
    python scripts/run_online.py --model results/fbcsp_svm_2a/models/fbcsp_svm_2a_sub1.pkl --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online.classifier import OnlineClassifier
from src.online.replay import ReplayStream
from src.online.server import BCIServer
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Online BCI simulation server")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pkl")
    parser.add_argument("--extractor", type=str, default=None, help="Path to extractor .pkl (auto-detected if omitted)")
    parser.add_argument("--replay-data", type=str, default=None, help="Path to replay_data.npz (auto-detected if omitted)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP server port for web frontend")
    parser.add_argument("--max-trials", type=int, default=None, help="Stop after N trials")
    args = parser.parse_args()

    setup_logging()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    # Auto-detect extractor
    extractor_path = args.extractor
    if extractor_path is None:
        extractor_path = OnlineClassifier.infer_extractor_path(model_path)
        if extractor_path:
            logger.info("Auto-detected extractor: %s", extractor_path)

    # Auto-detect replay data
    replay_path = args.replay_data
    if replay_path is None:
        replay_path = ReplayStream.infer_path(model_path)
        if replay_path is None:
            logger.error("No replay data found. Specify --replay-data.")
            sys.exit(1)
        logger.info("Auto-detected replay data: %s", replay_path)

    classifier = OnlineClassifier(model_path, extractor_path)
    replay = ReplayStream(replay_path)
    server = BCIServer(
        classifier, replay,
        host=args.host,
        port=args.port,
        http_port=args.http_port,
        max_trials=args.max_trials,
    )

    logger.info("Starting BCI server on %s:%d ...", args.host, args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
