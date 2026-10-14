/**
 * Predictive Machine Vibration Monitoring
 * Client-side Real-time Telemetry, Motor Control & Chart.js Visualizer
 */

// Global Chart configurations
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "'JetBrains Mono', monospace";

// Waveform Chart (Time Domain)
const waveformCtx = document.getElementById('waveform-chart').getContext('2d');
const waveformChart = new Chart(waveformCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Radial Acceleration (g)',
      data: [],
      borderColor: '#58a6ff',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1,
    }]
  },
  options: {
    responsive: true,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      x: {
        display: true,
        title: { display: true, text: 'Sample Index' },
        ticks: { maxTicksLimit: 8 },
        grid: { color: 'rgba(48, 54, 61, 0.5)' }
      },
      y: {
        title: { display: true, text: 'Acceleration (g)' },
        grid: { color: 'rgba(48, 54, 61, 0.5)' }
      }
    }
  }
});

// FFT Chart (Frequency Domain)
const fftCtx = document.getElementById('fft-chart').getContext('2d');
const fftChart = new Chart(fftCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'FFT Magnitude',
      data: [],
      borderColor: '#3fb950',
      backgroundColor: 'rgba(46, 160, 67, 0.1)',
      borderWidth: 1.5,
      fill: true,
      pointRadius: 0,
    }]
  },
  options: {
    responsive: true,
    animation: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        title: { display: true, text: 'Frequency (Hz)' },
        ticks: { maxTicksLimit: 12 },
        grid: { color: 'rgba(48, 54, 61, 0.5)' }
      },
      y: {
        title: { display: true, text: 'Magnitude' },
        beginAtZero: true,
        grid: { color: 'rgba(48, 54, 61, 0.5)' }
      }
    }
  }
});

// UI Elements
const connDot = document.getElementById('conn-dot');
const connText = document.getElementById('conn-text');
const calibStatusText = document.getElementById('calib-status-text');
const ratePill = document.getElementById('rate-pill');
const staleWarning = document.getElementById('stale-warning');
const sensorError = document.getElementById('sensor-error');
const errorMessageText = document.getElementById('error-message-text');

const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');

const featRms = document.getElementById('feat-rms');
const featFreq = document.getElementById('feat-freq');
const featEnergy = document.getElementById('feat-energy');
const featKurt = document.getElementById('feat-kurt');
const featRpm = document.getElementById('feat-rpm');

const btnMotorStart = document.getElementById('btn-motor-start');
const btnMotorStop = document.getElementById('btn-motor-stop');
const motorSpeedSlider = document.getElementById('motor-speed-slider');
const motorSpeedVal = document.getElementById('motor-speed-val');

const btnCalibStart = document.getElementById('calibrate-start');
const btnCalibStop = document.getElementById('calibrate-stop');
const calibFeedback = document.getElementById('calib-feedback');

// Telemetry Polling
async function pollLatest() {
  try {
    const res = await fetch('/latest');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const state = await res.json();

    // Connection & Stale Status
    if (state.is_stale) {
      connDot.className = 'dot stale';
      connText.textContent = 'STALE';
      staleWarning.style.display = 'flex';
    } else {
      connDot.className = 'dot live';
      connText.textContent = 'LIVE';
      staleWarning.style.display = 'none';
    }

    // Hardware or Pipeline Error
    if (state.error_message) {
      sensorError.style.display = 'flex';
      errorMessageText.textContent = `Error: ${state.error_message}`;
    } else {
      sensorError.style.display = 'none';
    }

    // Rate pill
    if (state.sample_rate) {
      ratePill.textContent = `${state.sample_rate} Hz`;
    }

    // Calibration state
    if (state.calibrating) {
      calibStatusText.textContent = `Calibrating (${state.calibration_buffer ? state.calibration_buffer.length : 0} windows)`;
      calibStatusText.style.color = '#58a6ff';
      btnCalibStart.disabled = true;
      btnCalibStop.disabled = false;
    } else if (state.calibrated) {
      calibStatusText.textContent = 'Calibrated';
      calibStatusText.style.color = '#3fb950';
      btnCalibStart.disabled = false;
      btnCalibStop.disabled = true;
    } else {
      calibStatusText.textContent = 'Not Calibrated';
      calibStatusText.style.color = '#8b949e';
      btnCalibStart.disabled = false;
      btnCalibStop.disabled = true;
    }

    // Classification Verdict & Confidence
    if (state.prediction) {
      if (state.prediction === 'imbalance') {
        predictionEl.textContent = 'FAULTY -- ROTOR IMBALANCE';
        predictionEl.className = 'verdict-text faulty';
      } else {
        predictionEl.textContent = 'NORMAL OPERATION';
        predictionEl.className = 'verdict-text normal';
      }

      if (state.probability !== null && state.probability !== undefined) {
        const confPercent = (state.probability * 100).toFixed(1);
        confidenceEl.textContent = `Model Confidence (Fault Probability): ${confPercent}%`;
      } else {
        confidenceEl.textContent = 'Model Confidence: --';
      }
    } else {
      if (state.calibrating) {
        predictionEl.textContent = 'CALIBRATING BASELINE...';
        predictionEl.className = 'verdict-text';
        confidenceEl.textContent = 'Accumulating healthy reference windows';
      } else if (!state.calibrated) {
        predictionEl.textContent = 'AWAITING CALIBRATION';
        predictionEl.className = 'verdict-text';
        confidenceEl.textContent = 'Perform healthy baseline calibration first';
      } else {
        predictionEl.textContent = 'AWAITING SAMPLES';
        predictionEl.className = 'verdict-text';
        confidenceEl.textContent = 'Streaming active';
      }
    }

    // Feature Telemetry Values
    if (state.features) {
      featRms.textContent = state.features.rms !== undefined ? state.features.rms.toFixed(3) : '--';
      featFreq.textContent = state.features.dominant_freq !== undefined ? state.features.dominant_freq.toFixed(1) : '--';
      featEnergy.textContent = state.features.spectral_energy !== undefined ? state.features.spectral_energy.toExponential(2) : '--';
      featKurt.textContent = state.features.kurtosis !== undefined ? state.features.kurtosis.toFixed(3) : '--';
    } else {
      featRms.textContent = '--';
      featFreq.textContent = '--';
      featEnergy.textContent = '--';
      featKurt.textContent = '--';
    }

    // RPM
    featRpm.textContent = (state.rpm !== null && state.rpm !== undefined) ? state.rpm.toFixed(1) : '--';

    // Waveform Chart Update
    if (state.latest_window && state.latest_window.length > 0) {
      waveformChart.data.labels = state.latest_window.map((_, i) => i);
      waveformChart.data.datasets[0].data = state.latest_window;
      waveformChart.update('none');
    }

    // FFT Chart Update
    if (state.latest_fft && state.latest_fft.freqs && state.latest_fft.freqs.length > 0) {
      fftChart.data.labels = state.latest_fft.freqs.map(f => f.toFixed(0));
      fftChart.data.datasets[0].data = state.latest_fft.magnitude;
      fftChart.update('none');
    }

  } catch (err) {
    connDot.className = 'dot disconnected';
    connText.textContent = 'OFFLINE';
    staleWarning.style.display = 'none';
  }
}

// Motor Control Event Handlers
btnMotorStart.addEventListener('click', async () => {
  const speed = parseInt(motorSpeedSlider.value, 10);
  try {
    const res = await fetch('/motor/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'START', speed: speed }),
    });
    const data = await res.json();
    calibFeedback.textContent = `Motor started at speed ${data.speed || speed}`;
  } catch (e) {
    calibFeedback.textContent = `Motor start failed: ${e}`;
  }
});

btnMotorStop.addEventListener('click', async () => {
  try {
    const res = await fetch('/motor/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'STOP' }),
    });
    calibFeedback.textContent = 'Motor stopped.';
  } catch (e) {
    calibFeedback.textContent = `Motor stop failed: ${e}`;
  }
});

motorSpeedSlider.addEventListener('input', () => {
  motorSpeedVal.textContent = motorSpeedSlider.value;
});

motorSpeedSlider.addEventListener('change', async () => {
  const speed = parseInt(motorSpeedSlider.value, 10);
  try {
    await fetch('/motor/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'SPEED', speed: speed }),
    });
  } catch (e) {}
});

// Calibration Event Handlers
btnCalibStart.addEventListener('click', async () => {
  try {
    calibFeedback.textContent = 'Starting calibration... Ensure stable healthy motor rotation.';
    await fetch('/calibrate/start', { method: 'POST' });
    btnCalibStart.disabled = true;
    btnCalibStop.disabled = false;
  } catch (e) {
    calibFeedback.textContent = `Calibration start failed: ${e}`;
  }
});

btnCalibStop.addEventListener('click', async () => {
  try {
    calibFeedback.textContent = 'Fitting baseline statistics and computing feature scale floors...';
    const res = await fetch('/calibrate/stop', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      calibFeedback.textContent = `Baseline successfully calibrated on ${data.n_windows} windows.`;
      btnCalibStart.disabled = false;
      btnCalibStop.disabled = true;
    } else {
      calibFeedback.textContent = `Calibration error: ${data.error}`;
      btnCalibStart.disabled = false;
    }
  } catch (e) {
    calibFeedback.textContent = `Calibration request failed: ${e}`;
  }
});

// Polling interval: 500 ms for responsive updates
setInterval(pollLatest, 500);
pollLatest();