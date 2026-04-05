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
    let reconnectDelay = 1000;
    const MAX_RECONNECT_DELAY = 30000;
    let reconnecting = false;

    // ── Screen switching ──
    function showScreen(id) {
        screens.forEach((s) => s.classList.remove("active"));
        const target = $(`#screen-${id}`);
        if (target) target.classList.add("active");
    }

    // ── Connection with exponential backoff ──
    function connect() {
        if (reconnecting) return;
        reconnecting = true;

        const proto = location.protocol === "https:" ? "wss" : "ws";
        const wsPort = parseInt(document.documentElement.dataset.wsPort) || 8765;
        const url = `${proto}://${location.hostname}:${wsPort}`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            reconnectDelay = 1000; // reset backoff
            reconnecting = false;
            statusDot.className = "status-dot connected";
            statusText.textContent = "Connected";
            btnStart.disabled = false;
        };

        ws.onclose = () => {
            reconnecting = false;
            statusDot.className = "status-dot disconnected";
            statusText.textContent = "Disconnected";
            btnStart.disabled = true;
            showScreen("idle");
            // Exponential backoff
            setTimeout(connect, reconnectDelay);
            reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
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

        const pct = Math.round(Number(data.confidence) * 100) || 0;
        $("#confidence-bar").style.width = `${pct}%`;
        $("#confidence-text").textContent = `${pct}%`;
        $("#confidence-bar-container").setAttribute("aria-valuenow", pct);

        $("#feedback-latency").textContent = `Latency: ${Number(data.latency_ms).toFixed(1)}ms`;

        const acc = Math.round(Number(data.accuracy) * 100) || 0;
        accuracyDisplay.textContent = `Accuracy: ${acc}% (${Number(data.correct_count)}/${Number(data.trial_num)})`;
    }

    // ── Done display ──
    function showDone(data) {
        const pct = Math.round(Number(data.final_accuracy) * 100) || 0;
        const stats = $("#done-stats");
        stats.textContent = "";
        stats.appendChild(document.createTextNode(
            `${Number(data.total_correct)} / ${Number(data.total_trials)} correct`
        ));
        stats.appendChild(document.createElement("br"));
        const strong = document.createElement("strong");
        strong.textContent = `Final accuracy: ${pct}%`;
        stats.appendChild(strong);
    }

    // ── Button handlers ──
    btnStart.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ command: "start" }));
            btnStart.disabled = true;
            trialCounter.textContent = "Trial: 0";
            accuracyDisplay.textContent = "Accuracy: \u2014";
            showScreen("rest");
        }
    });

    btnRestart.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ command: "start" }));
            trialCounter.textContent = "Trial: 0";
            accuracyDisplay.textContent = "Accuracy: \u2014";
            showScreen("rest");
        }
    });

    // ── Init ──
    connect();
})();
