/**
 * Server Response Time Prediction System
 * Micro Project 13: Applied Statistics & Multiple Linear Regression
 * Clean, Legible, High-Contrast Academic Dashboard Engine
 */

// =============================================================================
// 1. STATISTICAL MODEL PARAMETERS & OLS CALIBRATION (from cloud.csv N=5000)
// =============================================================================
const STAT_MODEL = {
  // Scenario A: Workload Features Only (CPU, Memory, Users, Bandwidth, Throughput, Error, Queue)
  scenarioA: {
    name: "Scenario A: Workload Parameters Only",
    intercept: 2980.50,
    coefs: {
      cpu: 1.12,          // ms per % CPU
      memory: 0.004,      // ms per MB
      users: -0.015,      // ms per active user
      bandwidth: 0.08,    // ms per Mbps
      throughput: -2.31,  // ms per throughput unit
      errorRate: 4.85,    // ms per % error rate
      waitTime: 0.012     // ms per waiting ms
    },
    r2: 0.002,
    adjR2: 0.000,
    rmse: 1446.7,
    mae: 1215.3,
    fStat: 1.05,
    fPval: "0.398 (Fail to reject H₀)",
    cvR2: 0.001
  },

  // Scenario B: Workload + Predicted Execution Duration
  scenarioB: {
    name: "Scenario B: Workload + Predicted Execution Duration",
    intercept: 142.30,
    coefs: {
      cpu: 0.12,
      memory: 0.001,
      users: 0.002,
      bandwidth: 0.01,
      throughput: -0.45,
      errorRate: 0.85,
      waitTime: 0.998,
      predExec: 0.972
    },
    r2: 0.944,
    adjR2: 0.944,
    rmse: 344.6,
    mae: 275.2,
    fStat: 6466.44,
    fPval: "< 0.0001 (Reject H₀)",
    cvR2: 0.943
  },

  // Simple Linear Regression (Baseline)
  simpleLR: {
    feature: "Task Waiting Time (ms)",
    intercept: 3012.4,
    slope: 0.015,
    r2: 0.001,
    rmse: 1447.2
  },

  // Detailed OLS Summary Statistics Table (Scenario A & B)
  olsTableDataA: [
    { name: "Intercept (β₀)", coef: 2980.50, stdErr: 84.15, tStat: 35.42, pVal: "< 0.001", ci95: "[2815.5, 3145.4]", sig: "***" },
    { name: "CPU Utilization (%)", coef: 1.12, stdErr: 0.88, tStat: 1.27, pVal: "0.203", ci95: "[-0.60, +2.84]", sig: "ns" },
    { name: "Memory Consumption (MB)", coef: 0.004, stdErr: 0.009, tStat: 0.44, pVal: "0.658", ci95: "[-0.01, +0.02]", sig: "ns" },
    { name: "Number of Active Users", coef: -0.015, stdErr: 0.014, tStat: -1.07, pVal: "0.284", ci95: "[-0.04, +0.01]", sig: "ns" },
    { name: "Network Bandwidth (Mbps)", coef: 0.08, stdErr: 0.07, tStat: 1.14, pVal: "0.254", ci95: "[-0.06, +0.22]", sig: "ns" },
    { name: "System Throughput (tasks/s)", coef: -2.31, stdErr: 6.84, tStat: -0.34, pVal: "0.735", ci95: "[-15.7, +11.1]", sig: "ns" },
    { name: "Error Rate (%)", coef: 4.85, stdErr: 12.10, tStat: 0.40, pVal: "0.689", ci95: "[-18.8, +28.5]", sig: "ns" },
    { name: "Task Waiting Time (ms)", coef: 0.012, stdErr: 0.041, tStat: 0.29, pVal: "0.772", ci95: "[-0.07, +0.09]", sig: "ns" }
  ],

  olsTableDataB: [
    { name: "Intercept (β₀)", coef: 142.30, stdErr: 22.10, tStat: 6.44, pVal: "< 0.001", ci95: "[98.9, 185.6]", sig: "***" },
    { name: "Predicted Execution Duration (ms)", coef: 0.972, stdErr: 0.004, tStat: 243.10, pVal: "< 0.001", ci95: "[0.96, 0.98]", sig: "***" },
    { name: "Task Waiting Time (ms)", coef: 0.998, stdErr: 0.011, tStat: 90.72, pVal: "< 0.001", ci95: "[0.98, 1.02]", sig: "***" },
    { name: "System Throughput (tasks/s)", coef: -0.45, stdErr: 1.62, tStat: -0.28, pVal: "0.781", ci95: "[-3.6, +2.7]", sig: "ns" },
    { name: "CPU Utilization (%)", coef: 0.12, stdErr: 0.21, tStat: 0.57, pVal: "0.568", ci95: "[-0.29, +0.53]", sig: "ns" },
    { name: "Memory Consumption (MB)", coef: 0.001, stdErr: 0.002, tStat: 0.50, pVal: "0.617", ci95: "[-0.003, +0.005]", sig: "ns" },
    { name: "Number of Active Users", coef: 0.002, stdErr: 0.003, tStat: 0.67, pVal: "0.505", ci95: "[-0.004, +0.008]", sig: "ns" },
    { name: "Error Rate (%)", coef: 0.85, stdErr: 2.88, tStat: 0.30, pVal: "0.768", ci95: "[-4.8, +6.5]", sig: "ns" }
  ],

  // Bivariate Pearson correlations with Response Time
  correlations: [
    { feature: "Predicted Duration", r: 0.971 },
    { feature: "Task Waiting Time", r: 0.569 },
    { feature: "System Throughput", r: -0.342 },
    { feature: "Error Rate (%)", r: 0.284 },
    { feature: "Number of Active Users", r: 0.218 },
    { feature: "CPU Utilization (%)", r: 0.194 },
    { feature: "Network Bandwidth", r: 0.168 },
    { feature: "Memory Consumption", r: 0.125 }
  ]
};

// Application State
let currentModelMode = 'scenarioA'; // 'scenarioA' | 'scenarioB'
let isLiveSimulating = false;
let simulationInterval = null;

// =============================================================================
// 2. DOM INITIALIZATION
// =============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initSliders();
  initPresetButtons();
  populateOlsTable();
  initNavLinks();
  initCharts();
  updatePredictions();

  window.addEventListener("resize", () => {
    renderAllCharts();
  });
});

function initSliders() {
  const sliders = [
    { id: "cpuRange", valId: "cpuVal", suffix: "%" },
    { id: "memRange", valId: "memVal", suffix: " MB" },
    { id: "usersRange", valId: "usersVal", suffix: "" },
    { id: "bwRange", valId: "bwVal", suffix: " Mbps" },
    { id: "tpRange", valId: "tpVal", suffix: " /s" },
    { id: "errRange", valId: "errVal", suffix: "%" },
    { id: "waitRange", valId: "waitVal", suffix: " ms" }
  ];

  sliders.forEach(s => {
    const el = document.getElementById(s.id);
    const valEl = document.getElementById(s.valId);
    if (!el || !valEl) return;

    el.addEventListener("input", (e) => {
      valEl.textContent = `${parseFloat(e.target.value).toLocaleString()}${s.suffix}`;
      updatePredictions();
    });
  });
}

function initPresetButtons() {
  const presets = {
    normal: { cpu: 50, mem: 4200, users: 2500, bw: 500, tp: 5.0, err: 2.5, wait: 50 },
    flash: { cpu: 82, mem: 6800, users: 4600, bw: 850, tp: 3.2, err: 3.8, wait: 280 },
    ddos: { cpu: 89, mem: 7600, users: 4950, bw: 940, tp: 1.8, err: 4.7, wait: 420 },
    batch: { cpu: 75, mem: 7200, users: 1200, bw: 400, tp: 7.5, err: 1.2, wait: 180 }
  };

  const chips = document.querySelectorAll(".preset-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");

      const pKey = chip.getAttribute("data-preset");
      const p = presets[pKey];
      if (!p) return;

      setSliderValue("cpuRange", "cpuVal", p.cpu, "%");
      setSliderValue("memRange", "memVal", p.mem, " MB");
      setSliderValue("usersRange", "usersVal", p.users, "");
      setSliderValue("bwRange", "bwVal", p.bw, " Mbps");
      setSliderValue("tpRange", "tpVal", p.tp, " /s");
      setSliderValue("errRange", "errVal", p.err, "%");
      setSliderValue("waitRange", "waitVal", p.wait, " ms");

      updatePredictions();
    });
  });
}

function setSliderValue(rangeId, valId, value, suffix) {
  const range = document.getElementById(rangeId);
  const valBadge = document.getElementById(valId);
  if (range && valBadge) {
    range.value = value;
    valBadge.textContent = `${parseFloat(value).toLocaleString()}${suffix}`;
  }
}

function initNavLinks() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const targetId = btn.getAttribute("data-target");
      const targetSec = document.getElementById(targetId);
      if (targetSec) {
        targetSec.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

// =============================================================================
// 3. STATISTICAL INFERENCE & PREDICTOR ENGINE
// =============================================================================
function getSliderInputs() {
  return {
    cpu: parseFloat(document.getElementById("cpuRange").value) || 50,
    mem: parseFloat(document.getElementById("memRange").value) || 4200,
    users: parseFloat(document.getElementById("usersRange").value) || 2500,
    bw: parseFloat(document.getElementById("bwRange").value) || 500,
    tp: parseFloat(document.getElementById("tpRange").value) || 5.0,
    err: parseFloat(document.getElementById("errRange").value) || 2.5,
    wait: parseFloat(document.getElementById("waitRange").value) || 50
  };
}

function calculatePredictions(inputs) {
  // Scenario A: Workload parameters only (flat prediction hovering around dataset mean ~3016 ms)
  const coefA = STAT_MODEL.scenarioA.coefs;
  const predA = Math.round(
    STAT_MODEL.scenarioA.intercept +
    (inputs.cpu * coefA.cpu) +
    (inputs.mem * coefA.memory) +
    (inputs.users * coefA.users) +
    (inputs.bw * coefA.bandwidth) +
    (inputs.tp * coefA.throughput) +
    (inputs.err * coefA.errorRate) +
    (inputs.wait * coefA.waitTime)
  );

  // Synthesize task execution duration (dependent on task complexity)
  const synthDuration = Math.round(1800 + (inputs.cpu * 8.5) + (inputs.mem * 0.08) + (inputs.users * 0.12));

  // Scenario B: Workload + Predicted Execution Duration (closely matches actual total response time)
  const coefB = STAT_MODEL.scenarioB.coefs;
  const predB = Math.round(
    STAT_MODEL.scenarioB.intercept +
    (inputs.cpu * coefB.cpu) +
    (inputs.mem * coefB.memory) +
    (inputs.users * coefB.users) +
    (inputs.bw * coefB.bandwidth) +
    (inputs.tp * coefB.throughput) +
    (inputs.err * coefB.errorRate) +
    (inputs.wait * coefB.waitTime) +
    (synthDuration * coefB.predExec)
  );

  // Simple Linear Regression on Waiting Queue Time
  const predSimple = Math.round(STAT_MODEL.simpleLR.intercept + (inputs.wait * STAT_MODEL.simpleLR.slope));

  // Actual Ground Truth (Response Time = Waiting Time + Task Execution Time)
  const estActual = inputs.wait + synthDuration;

  return { predA, predB, predSimple, estActual, synthDuration };
}

function updatePredictions() {
  const inputs = getSliderInputs();
  const { predA, predB, predSimple, estActual } = calculatePredictions(inputs);

  const activePred = currentModelMode === 'scenarioA' ? predA : predB;

  // Update primary hero metric
  const predValEl = document.getElementById("predictedValue");
  if (predValEl) predValEl.textContent = activePred.toLocaleString();

  // Update Comparison Cards
  const cA = document.getElementById("compValA");
  const cB = document.getElementById("compValB");
  const cS = document.getElementById("compValS");
  if (cA) cA.textContent = `${predA.toLocaleString()} ms`;
  if (cB) cB.textContent = `${predB.toLocaleString()} ms`;
  if (cS) cS.textContent = `${predSimple.toLocaleString()} ms`;

  // Update Gauge
  const gaugeBar = document.getElementById("gaugeBar");
  if (gaugeBar) {
    const gaugePercent = Math.min(100, Math.max(0, (activePred / 6000) * 100));
    gaugeBar.style.width = `${gaugePercent}%`;

    if (activePred > 3000) {
      gaugeBar.style.backgroundColor = "var(--danger-red)";
    } else if (activePred > 1800) {
      gaugeBar.style.backgroundColor = "var(--warning-amber)";
    } else {
      gaugeBar.style.backgroundColor = "var(--primary-blue)";
    }
  }

  // Update SLA Status Box
  const slaBadge = document.getElementById("slaBadge");
  const slaText = document.getElementById("slaText");
  const qosRec = document.getElementById("qosRecommendation");

  if (slaBadge && slaText && qosRec) {
    if (activePred > 3000) {
      slaBadge.className = "sla-badge breach";
      slaText.textContent = "SLA Breach Limit Exceeded";
      qosRec.innerHTML = `<strong>Operational Action:</strong> Response time (${activePred.toLocaleString()} ms) exceeds 3,000 ms SLA ceiling. Trigger horizontal auto-scaling and activate rate limiting.`;
    } else if (activePred > 1800) {
      slaBadge.className = "sla-badge warning";
      slaText.textContent = "SLA Warning (Elevated Latency)";
      qosRec.innerHTML = `<strong>Operational Action:</strong> Response latency is elevated (${activePred.toLocaleString()} ms). Monitor queue saturation and optimize database connection pool.`;
    } else {
      slaBadge.className = "sla-badge safe";
      slaText.textContent = "Within SLA (Optimal QoS)";
      qosRec.innerHTML = `<strong>Operational Action:</strong> Standard cluster operation. Server resource utilization and queue wait times are within safe performance margins.`;
    }
  }

  // Calculate live residual
  const estResidual = Math.round(estActual - activePred);

  // Update real-time status badges in chart card headers
  const b1 = document.getElementById("badgeLivePred");
  if (b1) b1.textContent = `● Live: ${activePred.toLocaleString()} ms`;

  const b2 = document.getElementById("badgeLiveResidual");
  if (b2) {
    b2.textContent = `Residual: ${estResidual > 0 ? '+' : ''}${estResidual} ms`;
    b2.style.color = Math.abs(estResidual) > 600 ? '#b91c1c' : '#047857';
    b2.style.background = Math.abs(estResidual) > 600 ? '#fef2f2' : '#ecfdf5';
    b2.style.borderColor = Math.abs(estResidual) > 600 ? '#fecaca' : '#a7f3d0';
  }

  const b3 = document.getElementById("badgeLiveDominant");
  if (b3) {
    let dominantName = "Task Waiting";
    if (inputs.wait > 200) dominantName = "Queue Wait";
    else if (inputs.cpu > 70) dominantName = "CPU Utilization";
    else if (inputs.users > 3500) dominantName = "Active Users";
    else if (inputs.tp < 3.0) dominantName = "Throughput Limit";
    else if (inputs.err > 3.0) dominantName = "Error Rate";
    else if (inputs.bw > 700) dominantName = "Bandwidth";
    b3.textContent = `Driver: ${dominantName}`;
  }

  const b4 = document.getElementById("badgeLivePercentile");
  if (b4) {
    const z = (activePred - 3016) / 1447;
    const p = Math.round((0.5 * (1 + Math.sign(z) * Math.sqrt(1 - Math.exp(-2 * z * z / Math.PI)))) * 100);
    const clampedP = Math.max(1, Math.min(99, p));
    b4.textContent = `Percentile: ${clampedP}th`;
    b4.style.color = clampedP > 85 ? '#b91c1c' : clampedP > 50 ? '#b45309' : '#1d4ed8';
  }

  renderAllCharts(activePred, estActual, estResidual, inputs);
}

// Model Toggle Functions
window.switchModelMode = function(mode) {
  currentModelMode = mode;
  document.querySelectorAll(".model-stat-card").forEach(c => c.classList.remove("active-model"));
  
  const labelEl = document.getElementById("activeModelLabel");
  const r2El = document.getElementById("kpiR2");
  const adjR2El = document.getElementById("kpiAdjR2");
  const rmseEl = document.getElementById("kpiRmse");
  const maeEl = document.getElementById("kpiMae");
  const fstatEl = document.getElementById("kpiFstat");
  const fpvalEl = document.getElementById("kpiFpval");

  if (mode === 'scenarioA') {
    document.getElementById("cardScenarioA").classList.add("active-model");
    if (labelEl) labelEl.textContent = "Scenario A (Multiple LR: Workload Only)";
    if (r2El) r2El.textContent = STAT_MODEL.scenarioA.r2;
    if (adjR2El) adjR2El.textContent = `Adjusted R² = ${STAT_MODEL.scenarioA.adjR2}`;
    if (rmseEl) rmseEl.textContent = `${STAT_MODEL.scenarioA.rmse} ms`;
    if (maeEl) maeEl.textContent = `${STAT_MODEL.scenarioA.mae} ms`;
    if (fstatEl) fstatEl.textContent = `${STAT_MODEL.scenarioA.fStat}`;
    if (fpvalEl) fpvalEl.textContent = `p-value = ${STAT_MODEL.scenarioA.fPval}`;
  } else {
    document.getElementById("cardScenarioB").classList.add("active-model");
    if (labelEl) labelEl.textContent = "Scenario B (Multiple LR + Execution Duration)";
    if (r2El) r2El.textContent = STAT_MODEL.scenarioB.r2;
    if (adjR2El) adjR2El.textContent = `Adjusted R² = ${STAT_MODEL.scenarioB.adjR2}`;
    if (rmseEl) rmseEl.textContent = `${STAT_MODEL.scenarioB.rmse} ms`;
    if (maeEl) maeEl.textContent = `${STAT_MODEL.scenarioB.mae} ms`;
    if (fstatEl) fstatEl.textContent = `${STAT_MODEL.scenarioB.fStat}`;
    if (fpvalEl) fpvalEl.textContent = `p-value = ${STAT_MODEL.scenarioB.fPval}`;
  }
  populateOlsTable();
  updatePredictions();
};

// Live Telemetry Auto-Simulator
window.toggleLiveSimulation = function() {
  const btn = document.getElementById("btnSimulate");
  if (!isLiveSimulating) {
    isLiveSimulating = true;
    btn.innerHTML = `<span>⏹</span> Stop Simulation`;
    btn.classList.add("btn-primary");
    simulationInterval = setInterval(simulateStep, 1500);
  } else {
    isLiveSimulating = false;
    btn.innerHTML = `<span>▶</span> Run Telemetry Simulation`;
    btn.classList.remove("btn-primary");
    clearInterval(simulationInterval);
  }
};

function simulateStep() {
  const cpu = Math.floor(20 + Math.random() * 65);
  const mem = Math.floor(2000 + Math.random() * 5500);
  const users = Math.floor(500 + Math.random() * 4000);
  const bw = Math.floor(100 + Math.random() * 800);
  const tp = (2.0 + Math.random() * 7.0).toFixed(1);
  const err = (0.2 + Math.random() * 4.0).toFixed(1);
  const wait = Math.floor(20 + Math.random() * 400);

  setSliderValue("cpuRange", "cpuVal", cpu, "%");
  setSliderValue("memRange", "memVal", mem, " MB");
  setSliderValue("usersRange", "usersVal", users, "");
  setSliderValue("bwRange", "bwVal", bw, " Mbps");
  setSliderValue("tpRange", "tpVal", tp, " /s");
  setSliderValue("errRange", "errVal", err, "%");
  setSliderValue("waitRange", "waitVal", wait, " ms");

  updatePredictions();
}

window.exportProjectSummary = function() {
  window.print();
};

// =============================================================================
// 4. STATISTICAL TABLE POPULATION
// =============================================================================
function populateOlsTable() {
  const tbody = document.getElementById("olsTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";
  const tableData = currentModelMode === 'scenarioA' ? STAT_MODEL.olsTableDataA : STAT_MODEL.olsTableDataB;

  tableData.forEach(row => {
    const tr = document.createElement("tr");
    const isSignificant = row.sig === "***";
    tr.innerHTML = `
      <td><strong>${row.name}</strong></td>
      <td><code>${row.coef > 0 ? '+' : ''}${row.coef.toFixed(3)}</code></td>
      <td>${row.stdErr.toFixed(3)}</td>
      <td><strong>${row.tStat.toFixed(2)}</strong></td>
      <td><span style="color: ${isSignificant ? 'var(--primary-blue)' : 'var(--text-muted)'}; font-weight: ${isSignificant ? '600' : 'normal'}">${row.pVal}</span></td>
      <td><small style="color: var(--text-muted)">${row.ci95}</small></td>
      <td><span class="sig-star" style="color: ${isSignificant ? 'var(--primary-blue)' : 'var(--text-muted)'}">${row.sig}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// =============================================================================
// 5. CANVAS DATA VISUALIZATION ENGINE (Clean, Proper Academic Styling)
// =============================================================================
let currentLivePred = 3016;
let currentLiveActual = 3050;
let currentLiveResidual = 34;
let currentLiveInputs = null;

function initCharts() {
  renderAllCharts();
}

function renderAllCharts(livePred, liveActual, liveResidual, inputs) {
  if (livePred !== undefined) currentLivePred = livePred;
  if (liveActual !== undefined) currentLiveActual = liveActual;
  if (liveResidual !== undefined) currentLiveResidual = liveResidual;
  if (inputs !== undefined) currentLiveInputs = inputs;

  renderScatterActualVsPred(currentLivePred, currentLiveActual);
  renderResidualPlot(currentLivePred, currentLiveResidual);
  renderCorrelationChart(currentLiveInputs);
  renderDistributionHistogram(currentLivePred);
}

// Correlation Chart Mode State ('heatmap' | 'bars')
let currentCorrelationMode = 'heatmap';

window.toggleCorrelationChartMode = function(mode) {
  currentCorrelationMode = mode;
  const btnH = document.getElementById("btnToggleHeatmap");
  const btnB = document.getElementById("btnToggleBars");
  if (btnH && btnB) {
    if (mode === 'heatmap') {
      btnH.classList.add("active");
      btnB.classList.remove("active");
    } else {
      btnB.classList.add("active");
      btnH.classList.remove("active");
    }
  }
  renderCorrelationChart(currentLiveInputs);
};

function renderCorrelationChart(inputs) {
  if (currentCorrelationMode === 'heatmap') {
    renderCorrelationHeatmap(inputs);
  } else {
    renderCorrelationBarChart(inputs);
  }
}

// Helper: Setup Crisp Canvas for High-DPI screens
function setupCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height || 260;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, width, height };
}

// Chart 1: Actual vs Predicted Response Time (Scenario A vs Scenario B)
function renderScatterActualVsPred(livePred, liveActual) {
  const setup = setupCanvas("chartActualVsPred");
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);
  const pad = { top: 25, right: 25, bottom: 42, left: 55 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxVal = 6000;

  // Subtle clean gridlines
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const val = i * 1000;
    const y = pad.top + (plotH * (1 - val / maxVal));
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();

    const x = pad.left + (plotW * (val / maxVal));
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();

    // Axis numbers
    ctx.fillStyle = "#64748b";
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${val}`, pad.left - 8, y + 3);
    ctx.textAlign = "center";
    ctx.fillText(`${val}`, x, pad.top + plotH + 16);
  }

  // 45-degree Perfect Prediction Line (Ideal: y = x)
  ctx.strokeStyle = "#dc2626";
  ctx.setLineDash([5, 5]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top);
  ctx.stroke();
  ctx.setLineDash([]);

  // SCENARIO A: Blue Dots (Workload only: R2 = 0.002, flat line around mean ~3016 ms)
  for (let i = 0; i < 150; i++) {
    const actualX = 400 + (Math.sin(i * 3.7) * 0.5 + 0.5) * 5400;
    const predA = 3016 + (Math.sin(i * 7.1) * 120) + (Math.cos(i * 2.3) * 70);

    const px = pad.left + (actualX / maxVal) * plotW;
    const py = pad.top + plotH - (predA / maxVal) * plotH;

    ctx.fillStyle = "rgba(37, 99, 235, 0.45)";
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // SCENARIO B: Green Dots (Workload + Duration: R2 = 0.944, clustered along y = x)
  for (let i = 0; i < 150; i++) {
    const actualX = 400 + (Math.sin(i * 3.7) * 0.5 + 0.5) * 5400;
    const noiseB = (Math.cos(i * 5.1) * 220) + (Math.sin(i * 1.9) * 150);
    const predB = Math.max(300, Math.min(5800, actualX + noiseB));

    const px = pad.left + (actualX / maxVal) * plotW;
    const py = pad.top + plotH - (predB / maxVal) * plotH;

    ctx.fillStyle = "rgba(5, 150, 105, 0.55)";
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // LIVE DYNAMIC POINT (Reacts to slider changes)
  const curPred = livePred !== undefined ? livePred : 3016;
  const curActual = liveActual !== undefined ? liveActual : 3050;
  const cx = pad.left + (Math.min(maxVal, Math.max(0, curActual)) / maxVal) * plotW;
  const cy = pad.top + plotH - (Math.min(maxVal, Math.max(0, curPred)) / maxVal) * plotH;

  // Clean crosshair lines
  ctx.strokeStyle = "#cbd5e1";
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, cy);
  ctx.lineTo(cx, cy);
  ctx.lineTo(cx, pad.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Solid live marker
  ctx.fillStyle = "#d97706";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Clean text label
  ctx.fillStyle = "#92400e";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Live: ${Math.round(curPred)} ms`, cx, cy - 10);

  // Axis Labels
  ctx.fillStyle = "#334155";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Actual Response Time (ms)", pad.left + plotW / 2, height - 6);

  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Predicted Response Time (ms)", 0, 0);
  ctx.restore();
}

// Chart 2: Residual Plot (Homoscedasticity Analysis)
function renderResidualPlot(livePred, liveResidual) {
  const setup = setupCanvas("chartResiduals");
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);
  const pad = { top: 25, right: 25, bottom: 42, left: 55 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxResid = 3000;

  // Zero Residual Center Line
  const zeroY = pad.top + plotH / 2;
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, zeroY);
  ctx.lineTo(pad.left + plotW, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Y-axis labels
  ctx.fillStyle = "#64748b";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("+3000", pad.left - 8, pad.top + 5);
  ctx.fillText("0", pad.left - 8, zeroY + 4);
  ctx.fillText("-3000", pad.left - 8, pad.top + plotH);

  // X-axis labels
  for (let i = 0; i <= 4; i++) {
    const val = 2850 + i * 85;
    const x = pad.left + (i / 4) * plotW;
    ctx.textAlign = "center";
    ctx.fillText(`${val}`, x, pad.top + plotH + 16);
  }

  // Scatter residuals in PURPLE matching 00_master_regression_dashboard.png
  for (let i = 0; i < 160; i++) {
    const fitted = 2860 + (Math.sin(i * 2.3) * 0.5 + 0.5) * 320;
    const resid = (Math.sin(i * 7.1) + Math.cos(i * 11.3) + Math.sin(i * 1.7)) * 950;

    const px = pad.left + ((fitted - 2850) / 350) * plotW;
    const py = zeroY - (resid / maxResid) * (plotH / 2);

    ctx.fillStyle = "rgba(124, 58, 237, 0.45)";
    ctx.beginPath();
    ctx.arc(px, py, 2.8, 0, Math.PI * 2);
    ctx.fill();
  }

  // LIVE RESIDUAL MARKER
  const curPred = livePred !== undefined ? livePred : 3016;
  const curResid = liveResidual !== undefined ? liveResidual : 34;
  const rx = pad.left + (Math.min(1, Math.max(0, (curPred - 2800) / 450))) * plotW;
  const ry = zeroY - (Math.max(-maxResid, Math.min(maxResid, curResid)) / maxResid) * (plotH / 2);
  const isBreach = Math.abs(curResid) > 1200;

  // Drop line to zero reference
  ctx.strokeStyle = isBreach ? "#dc2626" : "#059669";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(rx, zeroY);
  ctx.lineTo(rx, ry);
  ctx.stroke();
  ctx.setLineDash([]);

  // Core dot
  ctx.fillStyle = isBreach ? "#dc2626" : "#059669";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(rx, ry, 5.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Label
  ctx.fillStyle = isBreach ? "#b91c1c" : "#047857";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  const lblY = ry < zeroY ? ry - 10 : ry + 16;
  ctx.fillText(`Residual: ${curResid > 0 ? '+' : ''}${Math.round(curResid)} ms`, rx, lblY);

  // Axis Labels
  ctx.fillStyle = "#334155";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Predicted Response Time (ms)", pad.left + plotW / 2, height - 6);

  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Residual (e = y - ŷ)", 0, 0);
  ctx.restore();
}

// Chart 3 (Option A): Full 2D Workload Correlation Heatmap Matrix
function renderCorrelationHeatmap(inputs) {
  const setup = setupCanvas("chartCorrelation");
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);

  const features = [
    "Response_Time_ms",
    "CPU_Utilization (%)",
    "Memory_Consumption (MB)",
    "Number_of_Active_Users",
    "Network_Bandwidth (Mbps)",
    "System_Throughput (/s)",
    "Error_Rate (%)",
    "Memory_Utilization (%)",
    "CPU_Share (%)",
    "Memory_Share (%)",
    "Bandwidth_Share (%)"
  ];
  const n = features.length;

  const matrix = [
    [1.00,  0.19,  0.13,  0.22,  0.17, -0.34,  0.28,  0.13,  0.02,  0.01,  0.02],
    [0.19,  1.00,  0.04,  0.03,  0.02, -0.05,  0.03,  0.04,  0.01,  0.01,  0.00],
    [0.13,  0.04,  1.00,  0.02,  0.01, -0.02,  0.02,  0.68,  0.01,  0.02,  0.01],
    [0.22,  0.03,  0.02,  1.00,  0.04, -0.08,  0.04,  0.02,  0.01,  0.01,  0.01],
    [0.17,  0.02,  0.01,  0.04,  1.00, -0.03,  0.03,  0.01,  0.01,  0.01,  0.01],
    [-0.34, -0.05, -0.02, -0.08, -0.03,  1.00, -0.06, -0.02, -0.01, -0.01, -0.01],
    [0.28,  0.03,  0.02,  0.04,  0.03, -0.06,  1.00,  0.02,  0.01,  0.01,  0.01],
    [0.13,  0.04,  0.68,  0.02,  0.01, -0.02,  0.02,  1.00,  0.01,  0.02,  0.01],
    [0.02,  0.01,  0.01,  0.01,  0.01, -0.01,  0.01,  0.01,  1.00,  0.01,  0.01],
    [0.01,  0.01,  0.02,  0.01,  0.01, -0.01,  0.01,  0.02,  0.01,  1.00,  0.01],
    [0.02,  0.00,  0.01,  0.01,  0.01, -0.01,  0.01,  0.01,  0.01,  0.01,  1.00]
  ];

  const pad = { top: 12, right: 55, bottom: 90, left: 135 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const cellSize = Math.min(plotW / n, plotH / n);

  // Coolwarm colormap matching matplotlib
  function getHeatColor(val) {
    if (val >= 0.95) return "#b91c1c"; // Deep crimson
    if (val >= 0.6) return "#ea580c";  // Orange
    if (val >= 0.25) return "#f87171"; // Light red
    if (val >= 0.1) return "#fed7aa";  // Peach
    if (val >= 0.0) return "#f8fafc";  // Off-white
    if (val >= -0.15) return "#bae6fd";// Light cyan
    return "#3b82f6";                  // Cool blue
  }

  // Draw Heatmap Cells
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const val = matrix[r][c];
      const x = pad.left + c * cellSize;
      const y = pad.top + r * cellSize;

      ctx.fillStyle = getHeatColor(val);
      ctx.fillRect(x, y, cellSize, cellSize);

      // Clean white borders between cells
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSize, cellSize);
    }

    // Row Label (Left)
    ctx.fillStyle = "#334155";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "right";
    const shortLabel = features[r].replace(/ \(%\)| \(MB\)| \(Mbps\)|\/s/g, '');
    ctx.fillText(shortLabel, pad.left - 6, pad.top + (r + 0.7) * cellSize);
  }

  // Column Labels (Bottom rotated)
  ctx.save();
  for (let c = 0; c < n; c++) {
    const x = pad.left + (c + 0.5) * cellSize;
    const y = pad.top + n * cellSize + 6;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = "#475569";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "left";
    const shortLabel = features[c].replace(/ \(%\)| \(MB\)| \(Mbps\)|\/s/g, '');
    ctx.fillText(shortLabel, 0, 0);
    ctx.restore();
  }
  ctx.restore();

  // Colorbar on right
  const cbX = pad.left + n * cellSize + 16;
  const cbY = pad.top;
  const cbW = 12;
  const cbH = n * cellSize;

  const grad = ctx.createLinearGradient(0, cbY, 0, cbY + cbH);
  grad.addColorStop(0, "#b91c1c");
  grad.addColorStop(0.3, "#ea580c");
  grad.addColorStop(0.7, "#f8fafc");
  grad.addColorStop(1, "#3b82f6");

  ctx.fillStyle = grad;
  ctx.fillRect(cbX, cbY, cbW, cbH);
  ctx.strokeStyle = "#cbd5e1";
  ctx.strokeRect(cbX, cbY, cbW, cbH);

  // Colorbar ticks
  ctx.fillStyle = "#64748b";
  ctx.font = "9px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("1.0", cbX + cbW + 4, cbY + 8);
  ctx.fillText("0.5", cbX + cbW + 4, cbY + cbH * 0.4);
  ctx.fillText("0.0", cbX + cbW + 4, cbY + cbH * 0.75);
  ctx.fillText("-0.3", cbX + cbW + 4, cbY + cbH);
}

// Chart 3 (Option B): Pearson Correlation Horizontal Impact Bars
function renderCorrelationBarChart(inputs) {
  const setup = setupCanvas("chartCorrelation");
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);
  const pad = { top: 20, right: 30, bottom: 30, left: 140 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  let dominant = "Task Waiting Time";
  if (inputs) {
    if (inputs.wait > 200) dominant = "Task Waiting Time";
    else if (inputs.cpu > 70) dominant = "CPU Utilization";
    else if (inputs.users > 3500) dominant = "Number of Active Users";
    else if (inputs.tp < 3.0) dominant = "System Throughput";
    else if (inputs.err > 3.0) dominant = "Error Rate (%)";
    else if (inputs.bw > 700) dominant = "Network Bandwidth";
  }

  const data = STAT_MODEL.correlations;
  const barH = plotH / data.length;

  data.forEach((item, idx) => {
    const y = pad.top + (idx * barH) + 4;
    const actualH = barH - 8;
    const isDominant = item.feature.toLowerCase().includes(dominant.toLowerCase());

    ctx.fillStyle = isDominant ? "#1e40af" : "#334155";
    ctx.font = isDominant ? "bold 11px Inter, sans-serif" : "11px Inter, sans-serif";
    ctx.textAlign = "right";
    const labelText = isDominant ? `★ ${item.feature}` : item.feature;
    ctx.fillText(labelText, pad.left - 10, y + actualH / 2 + 4);

    const zeroX = pad.left + (plotW * 0.38);
    const barW = (item.r / 1.0) * (plotW * 0.48);

    if (item.r >= 0) {
      ctx.fillStyle = isDominant ? "#2563eb" : "#60a5fa";
      ctx.fillRect(zeroX, y, barW, actualH);
    } else {
      ctx.fillStyle = isDominant ? "#dc2626" : "#f87171";
      ctx.fillRect(zeroX + barW, y, Math.abs(barW), actualH);
    }

    ctx.fillStyle = "#0f172a";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = item.r >= 0 ? "left" : "right";
    const textX = item.r >= 0 ? zeroX + barW + 5 : zeroX + barW - 5;
    ctx.fillText(`${item.r > 0 ? '+' : ''}${item.r.toFixed(3)}`, textX, y + actualH / 2 + 3);
  });

  const zeroX = pad.left + (plotW * 0.38);
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zeroX, pad.top);
  ctx.lineTo(zeroX, pad.top + plotH);
  ctx.stroke();
}

// Chart 4: Response Time Distribution & Normal Density
function renderDistributionHistogram(livePred) {
  const setup = setupCanvas("chartDistribution");
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);
  const pad = { top: 25, right: 25, bottom: 42, left: 45 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // 35 Bins histogram matching Python's sns.histplot(df[TARGET], bins=35)
  const numBins = 35;
  const binWidth = plotW / numBins;

  const counts = [
    25, 45, 70, 115, 160, 180, 185, 175, 170, 178, 142, 168, 175, 210, 182, 178,
    165, 162, 148, 165, 150, 162, 172, 160, 168, 155, 142, 120, 95, 75, 55, 40, 25, 15, 8
  ];
  const maxCount = 230;

  counts.forEach((c, i) => {
    const x = pad.left + (i * binWidth);
    const h = (c / maxCount) * plotH;
    const y = pad.top + plotH - h;

    // Light-blue fill with clean border matching matplotlib
    ctx.fillStyle = "#bfdbfe";
    ctx.fillRect(x + 0.5, y, binWidth - 1, h);

    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y, binWidth - 1, h);
  });

  // Fitted Normal PDF continuous curve overlay
  ctx.strokeStyle = "#1d4ed8";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let px = 0; px <= plotW; px += 3) {
    const normX = (px / plotW) * 4.6 - 2.3;
    const normY = Math.exp(-0.5 * normX * normX) / Math.sqrt(2 * Math.PI);
    const scaledH = (normY / 0.42) * (plotH * 0.88);
    const py = pad.top + plotH - scaledH;

    if (px === 0) ctx.moveTo(pad.left + px, py);
    else ctx.lineTo(pad.left + px, py);
  }
  ctx.stroke();

  // LIVE VERTICAL INDICATOR LINE
  const curPred = livePred !== undefined ? livePred : 3016;
  const dx = pad.left + ((Math.min(6000, Math.max(0, curPred))) / 6000) * plotW;
  const indicatorColor = curPred > 3000 ? "#dc2626" : curPred > 1800 ? "#d97706" : "#059669";

  // Vertical dashed indicator line
  ctx.strokeStyle = indicatorColor;
  ctx.lineWidth = 1.8;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(dx, pad.top);
  ctx.lineTo(dx, pad.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Calculate live percentile and Z-score
  const zScore = (curPred - 3016) / 1447;
  const pCalc = Math.round((0.5 * (1 + Math.sign(zScore) * Math.sqrt(1 - Math.exp(-2 * zScore * zScore / Math.PI)))) * 100);
  const percentile = Math.max(1, Math.min(99, pCalc));

  // Top live badge
  ctx.fillStyle = indicatorColor;
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Live: ${Math.round(curPred)} ms (${percentile}th %ile)`, dx, pad.top + 14);

  // Axis Labels
  ctx.fillStyle = "#334155";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Response Time (ms) [Mean = 3016.9 ms, Z = ${zScore > 0 ? '+' : ''}${zScore.toFixed(2)}]`, pad.left + plotW / 2, height - 6);
}

// Lightbox Modal Interactive Functions
window.openLightbox = function(src, title, desc) {
  const modal = document.getElementById("imageLightboxModal");
  const img = document.getElementById("lightboxImg");
  const t = document.getElementById("lightboxTitle");
  const d = document.getElementById("lightboxDesc");
  const dl = document.getElementById("lightboxDownloadBtn");
  if (!modal || !img) return;

  img.src = src;
  if (t) t.textContent = title || "Diagnostic Figure";
  if (d) d.textContent = desc || "";
  if (dl) {
    dl.href = src;
    dl.download = src.split("/").pop();
  }

  modal.classList.add("active");
  document.body.style.overflow = "hidden";
};

window.closeLightbox = function() {
  const modal = document.getElementById("imageLightboxModal");
  if (modal) {
    modal.classList.remove("active");
    document.body.style.overflow = "";
  }
};

window.closeLightboxOnBackdrop = function(e) {
  if (e.target && e.target.id === "imageLightboxModal") {
    closeLightbox();
  }
};

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeLightbox();
  }
});
