"""WebSocket server for online BCI simulation.

Runs a trial state machine and pushes state updates to connected browsers.
Also serves the web frontend via a simple HTTP server on a separate port.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from functools import partial
from pathlib import Path
from typing import Any

import websockets

from src.online.classifier import OnlineClassifier, LABEL_MAP
from src.online.replay import ReplayStream
from src.utils.logging import get_logger

logger = get_logger(__name__)

WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web_frontend"

# Trial timing (seconds)
TIMING = {
    "cue": 2.0,
    "imagine": 3.0,
    "feedback": 2.0,
    "rest": 1.5,
}


class BCIServer:
    """WebSocket server driving a BCI trial loop.

    Parameters
    ----------
    classifier : OnlineClassifier
        Loaded model + extractor for single-trial prediction.
    replay : ReplayStream
        Source of (epoch, true_label) trials.
    host : str
        Bind address.
    port : int
        WebSocket port.
    http_port : int
        HTTP port for serving web frontend.
    max_trials : int or None
        Stop after N trials.  ``None`` = unlimited (loop).
    """

    def __init__(
        self,
        classifier: OnlineClassifier,
        replay: ReplayStream,
        host: str = "localhost",
        port: int = 8765,
        http_port: int = 8080,
        max_trials: int | None = None,
    ):
        self.classifier = classifier
        self.replay = replay
        self.host = host
        self.port = port
        self.http_port = http_port
        self.max_trials = max_trials

        self._clients: set = set()
        self._running = False
        self._trial_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def _broadcast(self, data: dict[str, Any]) -> None:
        if not self._clients:
            return
        msg = json.dumps(data, ensure_ascii=False)
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except websockets.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------
    # Trial state machine
    # ------------------------------------------------------------------

    async def _run_trials(self) -> None:
        self._running = True
        trial_num = 0
        correct = 0

        for epoch, true_label in self.replay:
            if not self._running:
                break
            if self.max_trials and trial_num >= self.max_trials:
                break

            trial_num += 1
            true_name = LABEL_MAP.get(true_label, str(true_label))

            # --- CUE ---
            await self._broadcast({
                "state": "cue",
                "trial_num": trial_num,
                "cue_direction": true_name,
                "cue_label": true_label,
            })
            await asyncio.sleep(TIMING["cue"])

            # --- IMAGINE ---
            await self._broadcast({
                "state": "imagine",
                "trial_num": trial_num,
                "cue_direction": true_name,
            })
            await asyncio.sleep(TIMING["imagine"])

            # --- CLASSIFY ---
            result = self.classifier.predict_trial(epoch)
            is_correct = result["label"] == true_label
            if is_correct:
                correct += 1
            accuracy = correct / trial_num

            # --- FEEDBACK ---
            await self._broadcast({
                "state": "feedback",
                "trial_num": trial_num,
                "true_label": true_label,
                "true_name": true_name,
                "prediction": result["label"],
                "prediction_name": result["label_name"],
                "confidence": result["confidence"],
                "correct": is_correct,
                "latency_ms": result["latency_ms"],
                "accuracy": round(accuracy, 4),
                "correct_count": correct,
            })
            logger.info(
                "Trial %d: cue=%s pred=%s conf=%.1f%% %s (%.1fms) — running acc %.1f%%",
                trial_num, true_name, result["label_name"],
                result["confidence"] * 100,
                "✓" if is_correct else "✗",
                result["latency_ms"],
                accuracy * 100,
            )
            await asyncio.sleep(TIMING["feedback"])

            # --- REST ---
            await self._broadcast({
                "state": "rest",
                "trial_num": trial_num,
            })
            await asyncio.sleep(TIMING["rest"])

        # Session complete
        await self._broadcast({
            "state": "done",
            "total_trials": trial_num,
            "total_correct": correct,
            "final_accuracy": round(correct / max(trial_num, 1), 4),
        })
        self._running = False
        logger.info("Session complete: %d/%d (%.1f%%)", correct, trial_num,
                     correct / max(trial_num, 1) * 100)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _ws_handler(self, websocket) -> None:
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)

        # Send welcome
        await websocket.send(json.dumps({
            "state": "connected",
            "total_trials": self.replay.n_trials,
            "label_map": LABEL_MAP,
        }))

        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                cmd = msg.get("command")
                if cmd == "start" and not self._running:
                    self._trial_task = asyncio.create_task(self._run_trials())
                elif cmd == "stop":
                    self._running = False
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s", remote)

    # ------------------------------------------------------------------
    # HTTP file server (separate thread)
    # ------------------------------------------------------------------

    def _start_http_server(self) -> None:
        """Start a simple HTTP server for the web frontend in a daemon thread."""
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_ROOT))
        httpd = http.server.HTTPServer((self.host, self.http_port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        logger.info("HTTP server serving %s at http://%s:%d", WEB_ROOT, self.host, self.http_port)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start both the WebSocket server and HTTP file server."""
        self._start_http_server()

        server = await websockets.serve(
            self._ws_handler,
            self.host,
            self.port,
        )
        logger.info(
            "WebSocket server on ws://%s:%d — Open http://%s:%d in browser",
            self.host, self.port,
            self.host, self.http_port,
        )
        await asyncio.Future()  # run forever
