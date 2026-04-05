/**
 * MI-BCI Online Simulation — WebSocket client
 *
 * Connects to the Python backend, receives trial state updates,
 * and drives the visual feedback UI.
 */

(function () {
    "use strict";

    // ── DOM references ──
    const $ = (sel) => document.querySelector(sel);
    const screens = document.querySelectorAll(".screen");
    const statusDot = $("#connection-status");
    const statusText = $("#connection-text");
    const trialCounter = $("#trial-counter");
    const accuracyDisplay = $("#accuracy-display");
    const btnStart = $("#btn-start");
    const btnRestart = $("#btn-restart");

    let ws = null;

    // ── Screen switching ──
    function showScreen(id) {
        screens.forEach((s) => s.classList.remove("active"));
        const target = $(`#screen-${id}`);
        if (target) target.classList.add("active");
    }

    // ── Connection ──
    function connect() {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        // WebSocket runs on port 8765, HTTP on 8080
        const wsPort = 8765;
        const url = `${proto}://${location.hostname}:${wsPort}`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            statusDot.className = "status-dot connected";
            statusText.textContent = "Connected";
            btnStart.disabled = false;
        };

        ws.onclose = () => {
            statusDot.className = "status-dot disconnected";
            statusText.textContent = "Disconnected";
            btnStart.disabled = true;
            // Reconnect after 2s
            setTimeout(connect, 2000);
        };

        ws.onerror = () => {
            ws.close();
        };

        ws.onmessage = (event) => {
            let data;
            try {
                data = JSON.parse(event.data);
            } catch {
                return;
            }
            handleState(data);
        };
    }

    // ── State handler ──
    function handleState(data) {
        const state = data.state;

        switch (state) {
            case "connected":
                showScreen("idle");
                break;

            case "cue":
                trialCounter.textContent = `Trial: ${data.trial_num}`;
                showCue(data.cue_direction);
                showScreen("cue");
                break;

            case "imagine":
                showImagine(data.cue_direction);
                showScreen("imagine");
                break;

            case "feedback":
                showFeedback(data);
                showScreen("feedback");
                break;

            case "rest":
                showScreen("rest");
                break;

            case "done":
                showDone(data);
                showScreen("done");
                break;
        }
    }

    // ── Cue display ──
    function showCue(direction) {
        const arrow = $("#cue-arrow");
        const text = $("#cue-text");
        arrow.className = `arrow ${direction === "left_hand" ? "left" : "right"}`;
        text.textContent = direction === "left_hand" ? "Left Hand" : "Right Hand";
    }

    // ── Imagine display ──
    function showImagine(direction) {
        const arrow = $("#imagine-arrow");
        arrow.className = `arrow faded ${direction === "left_hand" ? "left" : "right"}`;
    }

    // ── Feedback display ──
    function showFeedback(data) {
        const icon = $("#feedback-icon");
        icon.className = `feedback-icon ${data.correct ? "correct" : "wrong"}`;

        const predName = data.prediction_name === "left_hand" ? "Left Hand" : "Right Hand";
        $("#feedback-prediction").textContent = `Predicted: ${predName}`;

        const pct = Math.round(data.confidence * 100);
        $("#confidence-bar").style.width = `${pct}%`;
        $("#confidence-text").textContent = `${pct}%`;

        $("#feedback-latency").textContent = `Latency: ${data.latency_ms}ms`;

        accuracyDisplay.textContent = `Accuracy: ${Math.round(data.accuracy * 100)}% (${data.correct_count}/${data.trial_num})`;
    }

    // ── Done display ──
    function showDone(data) {
        const pct = Math.round(data.final_accuracy * 100);
        $("#done-stats").innerHTML =
            `${data.total_correct} / ${data.total_trials} correct<br>` +
            `Final accuracy: <strong>${pct}%</strong>`;
    }

    // ── Button handlers ──
    btnStart.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ command: "start" }));
            btnStart.disabled = true;
            trialCounter.textContent = "Trial: 0";
            accuracyDisplay.textContent = "Accuracy: —";
            showScreen("rest");
        }
    });

    btnRestart.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ command: "start" }));
            trialCounter.textContent = "Trial: 0";
            accuracyDisplay.textContent = "Accuracy: —";
            showScreen("rest");
        }
    });

    // ── Init ──
    connect();
})();
