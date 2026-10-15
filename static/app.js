"use strict";

const els = {
  phase: document.getElementById("phase"),
  clock: document.getElementById("clock"),
  clockLabel: document.getElementById("clock-label"),
  fps: document.getElementById("fps"),
  gate: document.getElementById("gate"),
  distance: document.getElementById("distance"),
  markers: document.getElementById("markers"),
  closing: document.getElementById("closing"),
  source: document.getElementById("source"),
  error: document.getElementById("error"),
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
  lapBody: document.querySelector("#laps tbody"),
  summary: document.getElementById("summary"),
  sumTotal: document.getElementById("sum-total"),
  sumCount: document.getElementById("sum-count"),
  sumBest: document.getElementById("sum-best"),
  sumAvg: document.getElementById("sum-avg"),
};

const fmt = (s) => (s == null ? "-" : Number(s).toFixed(3));

async function post(url) {
  await fetch(url, { method: "POST" });
  refresh();
}

els.start.addEventListener("click", () => post("/api/race/start"));
els.stop.addEventListener("click", () => post("/api/race/stop"));

let renderedLaps = -1;

function renderLaps(laps) {
  if (laps.length === renderedLaps) return;
  renderedLaps = laps.length;
  els.lapBody.innerHTML = laps
    .map(
      (lap) =>
        `<tr><td>${lap.index}</td><td>${fmt(lap.lap_time)}</td><td>${fmt(
          lap.total_time,
        )}</td></tr>`,
    )
    .join("");
}

function renderSummary(summary) {
  if (!summary) {
    els.summary.classList.add("hidden");
    return;
  }
  els.summary.classList.remove("hidden");
  els.sumTotal.textContent = fmt(summary.total_time);
  els.sumCount.textContent = summary.lap_count;
  els.sumBest.textContent = fmt(summary.best_lap);
  els.sumAvg.textContent = fmt(summary.average_lap);
}

async function refresh() {
  let state;
  try {
    const response = await fetch("/api/race/state");
    state = await response.json();
  } catch (err) {
    return;
  }

  els.phase.textContent = state.phase.toUpperCase();
  els.phase.dataset.phase = state.phase;

  if (state.phase === "countdown") {
    els.clock.textContent = state.countdown_remaining.toFixed(1);
    els.clockLabel.textContent = "get ready";
  } else {
    els.clock.textContent = fmt(state.elapsed);
    els.clockLabel.textContent =
      { idle: "ready", running: "racing", stopped: "finished" }[state.phase] || "";
  }

  els.fps.textContent = `${state.fps} fps`;
  els.gate.textContent = `gate: ${state.gate_visible ? "locked" : "searching"}`;
  els.gate.className = `diag ${state.gate_visible ? "ok" : "bad"}`;
  els.distance.textContent =
    state.distance_m == null ? "distance: -- m" : `distance: ${state.distance_m.toFixed(2)} m`;
  els.markers.textContent = `markers: ${state.markers}`;
  els.closing.textContent = `closing: ${state.closing_speed} m/s`;
  els.source.textContent = `source: ${state.source}`;

  if (state.error) {
    els.error.textContent = state.error;
    els.error.classList.remove("hidden");
  } else {
    els.error.classList.add("hidden");
  }

  els.start.disabled = state.phase === "countdown" || state.phase === "running";
  els.stop.disabled = !(state.phase === "countdown" || state.phase === "running");

  renderLaps(state.laps);
  renderSummary(state.summary);
}

refresh();
setInterval(refresh, 200);
