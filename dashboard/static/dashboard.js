const waveformChart = new Chart(document.getElementById("waveform-chart"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "Vibration (raw)", data: [], borderWidth: 1, pointRadius: 0 }] },
  options: { animation: false, scales: { x: { display: false } } },
});

const fftChart = new Chart(document.getElementById("fft-chart"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "FFT magnitude (0-500 Hz)", data: [], borderWidth: 1, pointRadius: 0 }] },
  options: { animation: false },
});

async function pollLatest() {
  let state;
  try {
    const res = await fetch("/latest");
    state = await res.json();
  } catch (err) {
    console.error("poll failed", err);
    return;
  }

  if (state.latest_window && state.latest_window.length) {
    waveformChart.data.labels = state.latest_window.map((_, i) => i);
    waveformChart.data.datasets[0].data = state.latest_window;
    waveformChart.update();
  }

  if (state.latest_fft && state.latest_fft.freqs && state.latest_fft.freqs.length) {
    fftChart.data.labels = state.latest_fft.freqs.map((f) => f.toFixed(0));
    fftChart.data.datasets[0].data = state.latest_fft.magnitude;
    fftChart.update();
  }

  const predictionEl = document.getElementById("prediction");
  const probabilityEl = document.getElementById("probability");
  if (state.prediction) {
    predictionEl.textContent = state.prediction === "imbalance" ? "FAULTY -- imbalance" : "NORMAL";
    predictionEl.style.color = state.prediction === "imbalance" ? "#c0392b" : "#2e7d32";
    probabilityEl.textContent = `confidence: ${(state.probability * 100).toFixed(1)}%`;
  } else {
    predictionEl.textContent = "--";
    predictionEl.style.color = "#333";
    probabilityEl.textContent = state.calibrated ? "" : "waiting for calibration";
  }

  if (state.features) {
    document.getElementById("feat-rms").textContent = state.features.rms.toFixed(3);
    document.getElementById("feat-freq").textContent = state.features.dominant_freq.toFixed(1) + " Hz";
    document.getElementById("feat-energy").textContent = state.features.spectral_energy.toExponential(2);
    document.getElementById("feat-kurt").textContent = state.features.kurtosis.toFixed(3);
  }
  document.getElementById("feat-rpm").textContent = state.rpm != null ? state.rpm.toFixed(1) : "--";

  document.getElementById("calibration-status").textContent = state.calibrating
    ? "Calibrating..."
    : state.calibrated
      ? "Calibrated"
      : "Not calibrated";
}

document.getElementById("calibrate-start").addEventListener("click", async () => {
  await fetch("/calibrate/start", { method: "POST" });
  document.getElementById("calibrate-start").disabled = true;
  document.getElementById("calibrate-stop").disabled = false;
});

document.getElementById("calibrate-stop").addEventListener("click", async () => {
  const res = await fetch("/calibrate/stop", { method: "POST" });
  const data = await res.json();
  document.getElementById("calibrate-start").disabled = false;
  document.getElementById("calibrate-stop").disabled = true;
  alert(data.status === "calibrated" ? `Calibrated on ${data.n_windows} windows` : data.error);
});

setInterval(pollLatest, 1000);
pollLatest();