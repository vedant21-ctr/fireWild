/**
 * FireWild Intelligence — Unified Application Logic & Router
 * Implements client-side hash routing, Spread Explorer canvas, Model Lab diagnostics,
 * CLBI Live Scenario Simulator, and segment table audit.
 */

// Global Application State
const AppState = {
  currentRoute: 'overview',
  ndwsBenchmark: null,
  spatialPatch: null,
  spatialThreshold: 0.028115,
  isSweepPlaying: false,
  sweepInterval: null,
  hoveredCell: null,
  clbiSegments: [],
  clbiFilteredSegments: [],
  clbiSortColumn: 'vulnerability_rank',
  clbiSortDir: 'asc',
};

// Scenario Presets Definitions
const ScenarioPresets = {
  ridge: {
    name: 'Vulnerable Ridge',
    slope: 35.0,
    elevation: 1200.0,
    distance_to_fire: 350.0,
    barrier_width_m: 2.0,
    burn_prob: 0.85,
    attack_angle: 15.0,
    attack_dot_product: 0.95
  },
  headfire: {
    name: 'Direct Head-Fire Attack',
    slope: 28.0,
    elevation: 950.0,
    distance_to_fire: 450.0,
    barrier_width_m: 6.5,
    burn_prob: 0.92,
    attack_angle: 10.0,
    attack_dot_product: 0.98
  },
  barrier: {
    name: 'Wide Cleared Barrier',
    slope: 12.0,
    elevation: 850.0,
    distance_to_fire: 1200.0,
    barrier_width_m: 9.0,
    burn_prob: 0.35,
    attack_angle: 70.0,
    attack_dot_product: 0.34
  }
};

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  initRouter();
  initSpreadExplorer();
  initModelLab();
  initRiskSimulator();
  initModals();

  // Load backend data in parallel
  loadBenchmarkMetrics();
  loadSpatialDemoPatch();
  loadCLBIDemoSegments();
});

// ==========================================================================
// Module 1: Client-Side Hash Router
// ==========================================================================
function initRouter() {
  const handleRouteChange = () => {
    let hash = window.location.hash.replace('#', '').trim();
    if (!hash || hash === 'demo') hash = 'overview';

    const validRoutes = ['overview', 'spread-explorer', 'model-lab', 'risk-simulator', 'research'];
    if (!validRoutes.includes(hash)) hash = 'overview';

    AppState.currentRoute = hash;

    // Toggle active view sections
    document.querySelectorAll('.page-view').forEach(view => {
      view.classList.remove('active');
    });
    const targetView = document.getElementById(`view-${hash}`);
    if (targetView) targetView.classList.add('active');

    // Toggle active navigation links
    document.querySelectorAll('.primary-nav .nav-item').forEach(item => {
      const page = item.getAttribute('data-page');
      if (page === hash) {
        item.classList.add('active');
        item.setAttribute('aria-current', 'page');
      } else {
        item.classList.remove('active');
        item.removeAttribute('aria-current');
      }
    });

    // Update document title
    const titles = {
      'overview': 'Overview | FireWild Intelligence',
      'spread-explorer': 'Spread Explorer | FireWild Intelligence',
      'model-lab': 'Model Lab | FireWild Intelligence',
      'risk-simulator': 'Line Risk Simulator | FireWild Intelligence',
      'research': 'Research & Methodology | FireWild Intelligence'
    };
    document.title = titles[hash] || 'FireWild Intelligence';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'instant' });

    // Trigger canvas redraw if opening Spread Explorer
    if (hash === 'spread-explorer' && AppState.spatialPatch) {
      setTimeout(() => renderSpreadCanvas(), 50);
    }
  };

  window.addEventListener('hashchange', handleRouteChange);
  handleRouteChange();
}

// ==========================================================================
// Module 2: Spread Explorer (Canvas & Threshold Calibration)
// ==========================================================================
function initSpreadExplorer() {
  const canvas = document.getElementById('spread-canvas');
  const slider = document.getElementById('spread-threshold-slider');
  const valReadout = document.getElementById('spread-threshold-val');
  const crosshair = document.getElementById('canvas-crosshair');
  const playBtn = document.getElementById('btn-playback-toggle');
  const playIcon = document.getElementById('icon-play-btn');
  const playLabel = document.getElementById('label-play-btn');
  const stepFwdBtn = document.getElementById('btn-playback-step-fwd');
  const stepBackBtn = document.getElementById('btn-playback-step-back');
  const resetBtn = document.getElementById('btn-playback-reset');
  const reloadBtn = document.getElementById('btn-canvas-reload');
  const centerBtn = document.getElementById('btn-canvas-center');

  // Slider change
  slider.addEventListener('input', (e) => {
    AppState.spatialThreshold = parseFloat(e.target.value);
    valReadout.textContent = AppState.spatialThreshold.toFixed(4);
    renderSpreadCanvas();
  });

  // Canvas Mouse Interaction for Cell Inspection
  canvas.addEventListener('mousemove', (e) => {
    if (!AppState.spatialPatch) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellW = rect.width / 64;
    const cellH = rect.height / 64;

    const col = Math.min(63, Math.max(0, Math.floor(x / cellW)));
    const row = Math.min(63, Math.max(0, Math.floor(y / cellH)));

    // Position crosshair
    crosshair.style.display = 'block';
    crosshair.style.left = `${(col + 0.5) * cellW}px`;
    crosshair.style.top = `${(row + 0.5) * cellH}px`;

    updateCellInspector(row, col);
  });

  canvas.addEventListener('mouseleave', () => {
    crosshair.style.display = 'none';
  });

  // Simulated Threshold Sweep Playback
  const toggleSweep = () => {
    if (AppState.isSweepPlaying) {
      clearInterval(AppState.sweepInterval);
      AppState.isSweepPlaying = false;
      playLabel.textContent = 'Play Sweep';
      playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    } else {
      AppState.isSweepPlaying = true;
      playLabel.textContent = 'Pause';
      playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';

      let direction = 1;
      AppState.sweepInterval = setInterval(() => {
        let current = AppState.spatialThreshold;
        if (current >= 0.085) direction = -1;
        if (current <= 0.015) direction = 1;

        current += direction * 0.002;
        AppState.spatialThreshold = parseFloat(current.toFixed(4));
        slider.value = AppState.spatialThreshold;
        valReadout.textContent = AppState.spatialThreshold.toFixed(4);
        renderSpreadCanvas();
      }, 90);
    }
  };

  playBtn.addEventListener('click', toggleSweep);

  stepFwdBtn.addEventListener('click', () => {
    AppState.spatialThreshold = Math.min(0.100, parseFloat((AppState.spatialThreshold + 0.005).toFixed(4)));
    slider.value = AppState.spatialThreshold;
    valReadout.textContent = AppState.spatialThreshold.toFixed(4);
    renderSpreadCanvas();
  });

  stepBackBtn.addEventListener('click', () => {
    AppState.spatialThreshold = Math.max(0.010, parseFloat((AppState.spatialThreshold - 0.005).toFixed(4)));
    slider.value = AppState.spatialThreshold;
    valReadout.textContent = AppState.spatialThreshold.toFixed(4);
    renderSpreadCanvas();
  });

  resetBtn.addEventListener('click', () => {
    AppState.spatialThreshold = 0.0281;
    slider.value = AppState.spatialThreshold;
    valReadout.textContent = '0.0281';
    renderSpreadCanvas();
  });

  reloadBtn.addEventListener('click', () => {
    loadSpatialDemoPatch();
  });

  centerBtn.addEventListener('click', () => {
    updateCellInspector(32, 32);
    crosshair.style.display = 'block';
    const rect = canvas.getBoundingClientRect();
    crosshair.style.left = `${rect.width / 2}px`;
    crosshair.style.top = `${rect.height / 2}px`;
  });
}

async function loadSpatialDemoPatch() {
  try {
    const res = await fetch('/api/v1/ndws/demo/patch');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    AppState.spatialPatch = data;
    renderSpreadCanvas();
    updateCellInspector(32, 32);
  } catch (err) {
    console.error('Failed to load spatial patch:', err);
    showToast('Failed to load spatial patch demo data', 'error');
  }
}

function renderSpreadCanvas() {
  const canvas = document.getElementById('spread-canvas');
  if (!canvas || !AppState.spatialPatch) return;
  const ctx = canvas.getContext('2d');
  const grid = AppState.spatialPatch.probabilities_2d;
  if (!grid || !grid.length) return;

  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  const tau = AppState.spatialThreshold;
  let activeCells = 0;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = grid[r][c];
      const isSpread = p >= tau;
      if (isSpread) activeCells++;

      // Color mapping
      let color;
      if (p < 0.010) {
        color = '#E8ECE9'; // Ambient unburned
      } else if (p < 0.025) {
        color = '#88A795'; // Low
      } else if (p < 0.050) {
        color = '#D4A045'; // Moderate
      } else if (p < 0.080) {
        color = '#D95A2B'; // Elevated
      } else {
        color = '#BA2525'; // Extreme
      }

      ctx.fillStyle = color;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);

      // Threshold contour highlight
      if (isSpread) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }
  }

  // Update counter badges
  const totalCells = rows * cols;
  const activePct = ((activeCells / totalCells) * 100).toFixed(1);
  const activeCellsElem = document.getElementById('spread-active-cells');
  const activePctElem = document.getElementById('spread-active-pct');
  const progressFill = document.getElementById('spread-progress-fill');

  if (activeCellsElem) activeCellsElem.textContent = activeCells.toLocaleString();
  if (activePctElem) activePctElem.textContent = `${activePct}%`;
  if (progressFill) progressFill.style.width = `${activePct}%`;
}

function updateCellInspector(row, col) {
  if (!AppState.spatialPatch) return;
  const grid = AppState.spatialPatch.probabilities_2d;
  if (!grid || !grid[row] || grid[row][col] === undefined) return;

  const p = grid[row][col];
  const tau = AppState.spatialThreshold;
  const isSpread = p >= tau;

  // Approximate distance to center fire in kilometers
  const distKm = (Math.sqrt(Math.pow((row - 32) / 1.5, 2) + Math.pow(col - 32, 2))).toFixed(1);

  let category = 'Ambient / Unburned';
  if (p >= 0.080) category = 'Extreme Fire Front';
  else if (p >= 0.050) category = 'Elevated Spread Risk';
  else if (p >= 0.025) category = 'Moderate Spread Zone';
  else if (p >= 0.010) category = 'Low Spread Potential';

  document.getElementById('insp-cell-coords').textContent = `Row ${row}, Col ${col}`;
  document.getElementById('insp-prob').textContent = p.toFixed(4);
  document.getElementById('insp-dist').textContent = `${distKm} km`;
  document.getElementById('insp-category').textContent = category;

  const statusElem = document.getElementById('insp-status');
  if (isSpread) {
    statusElem.innerHTML = `<span class="text-danger font-bold">Active Spread (&ge; ${tau.toFixed(4)})</span>`;
  } else {
    statusElem.innerHTML = `<span class="text-muted">Unburned Ambient (&lt; ${tau.toFixed(4)})</span>`;
  }

  // Simulated context metrics
  document.getElementById('insp-ctx-elevation').textContent = `${Math.round(450 + (row * 6) + (col * 3))} m`;
  document.getElementById('insp-ctx-wind').textContent = `${Math.round(180 + Math.sin(col) * 45)}° (SW)`;
  document.getElementById('insp-ctx-erc').textContent = `${(25.4 + p * 35).toFixed(1)}`;
  document.getElementById('insp-ctx-ndvi').textContent = `${(0.42 - p * 0.15).toFixed(2)}`;
}

// ==========================================================================
// Module 3: Model Lab & Diagnostic Plot Viewer
// ==========================================================================
function initModelLab() {
  const plotButtons = document.querySelectorAll('.plot-tab-btn');
  const activeImg = document.getElementById('active-diagnostic-img');
  const activeDesc = document.getElementById('active-diagnostic-desc');
  const zoomBtn = document.getElementById('btn-expand-plot');

  plotButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      plotButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const plotPath = btn.getAttribute('data-plot');
      const plotDesc = btn.getAttribute('data-desc');

      activeImg.src = plotPath;
      activeDesc.textContent = plotDesc;
    });
  });

  zoomBtn.addEventListener('click', () => {
    const modal = document.getElementById('plot-modal');
    const modalImg = document.getElementById('modal-img');
    const modalDesc = document.getElementById('modal-desc');

    modalImg.src = activeImg.src;
    modalDesc.textContent = activeDesc.textContent;
    modal.classList.add('open');
  });
}

// Alias for contract test compatibility
async function loadNDWSBenchmarkMetrics() {
  return loadBenchmarkMetrics();
}

async function loadBenchmarkMetrics() {
  try {
    const res = await fetch('/api/v1/ndws/benchmark/metrics');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    AppState.ndwsBenchmark = data;

    // Update hero ribbon stats
    if (data.best_model_details) {
      const best = data.best_model_details;
      const valPerf = best.validation_performance || {};
      const evalPerf = best.held_out_evaluation_performance || {};

      const heroValAp = document.getElementById('hero-stat-val-ap');
      const heroEvalAp = document.getElementById('hero-stat-eval-ap');

      if (heroValAp && valPerf.average_precision) {
        heroValAp.textContent = Number(valPerf.average_precision).toFixed(4);
      }
      if (heroEvalAp && evalPerf.average_precision) {
        heroEvalAp.textContent = Number(evalPerf.average_precision).toFixed(4);
      }

      // Update operational fixed-FPR benchmark table
      if (evalPerf.recall_at_1pct_fpr !== undefined) {
        const elRec1 = document.getElementById('fpr-lgbm-eval-rec1');
        const elFpr1 = document.getElementById('fpr-lgbm-eval-fpr1');
        const elTau1 = document.getElementById('fpr-lgbm-eval-tau1');
        const elRec5 = document.getElementById('fpr-lgbm-eval-rec5');
        const elFpr5 = document.getElementById('fpr-lgbm-eval-fpr5');
        const elTau5 = document.getElementById('fpr-lgbm-eval-tau5');
        if (elRec1) elRec1.textContent = (evalPerf.recall_at_1pct_fpr * 100).toFixed(2) + '%';
        if (elFpr1) elFpr1.textContent = (evalPerf.achieved_fpr_at_1pct * 100).toFixed(3) + '%';
        if (elTau1) elTau1.textContent = Number(evalPerf.threshold_at_1pct_fpr).toFixed(6);
        if (elRec5) elRec5.textContent = (evalPerf.recall_at_5pct_fpr * 100).toFixed(2) + '%';
        if (elFpr5) elFpr5.textContent = (evalPerf.achieved_fpr_at_5pct * 100).toFixed(3) + '%';
        if (elTau5) elTau5.textContent = Number(evalPerf.threshold_at_5pct_fpr).toFixed(6);
      }
      if (valPerf.recall_at_1pct_fpr !== undefined) {
        const elRec1 = document.getElementById('fpr-lgbm-val-rec1');
        const elFpr1 = document.getElementById('fpr-lgbm-val-fpr1');
        const elTau1 = document.getElementById('fpr-lgbm-val-tau1');
        const elRec5 = document.getElementById('fpr-lgbm-val-rec5');
        const elFpr5 = document.getElementById('fpr-lgbm-val-fpr5');
        const elTau5 = document.getElementById('fpr-lgbm-val-tau5');
        if (elRec1) elRec1.textContent = (valPerf.recall_at_1pct_fpr * 100).toFixed(2) + '%';
        if (elFpr1) elFpr1.textContent = (valPerf.achieved_fpr_at_1pct * 100).toFixed(3) + '%';
        if (elTau1) elTau1.textContent = Number(valPerf.threshold_at_1pct_fpr).toFixed(6);
        if (elRec5) elRec5.textContent = (valPerf.recall_at_5pct_fpr * 100).toFixed(2) + '%';
        if (elFpr5) elFpr5.textContent = (valPerf.achieved_fpr_at_5pct * 100).toFixed(3) + '%';
        if (elTau5) elTau5.textContent = Number(valPerf.threshold_at_5pct_fpr).toFixed(6);
      }
      if (data.models_comparison) {
        const logModel = data.models_comparison.find(m => m.model_id === 'Logistic_Regression_L2');
        if (logModel && logModel.recall_at_1pct_fpr !== undefined) {
          const elRec1 = document.getElementById('fpr-log-val-rec1');
          const elFpr1 = document.getElementById('fpr-log-val-fpr1');
          const elTau1 = document.getElementById('fpr-log-val-tau1');
          const elRec5 = document.getElementById('fpr-log-val-rec5');
          const elFpr5 = document.getElementById('fpr-log-val-fpr5');
          const elTau5 = document.getElementById('fpr-log-val-tau5');
          if (elRec1) elRec1.textContent = (logModel.recall_at_1pct_fpr * 100).toFixed(2) + '%';
          if (elFpr1) elFpr1.textContent = (logModel.achieved_fpr_at_1pct * 100).toFixed(3) + '%';
          if (elTau1) elTau1.textContent = Number(logModel.threshold_at_1pct_fpr).toFixed(6);
          if (elRec5) elRec5.textContent = (logModel.recall_at_5pct_fpr * 100).toFixed(2) + '%';
          if (elFpr5) elFpr5.textContent = (logModel.achieved_fpr_at_5pct * 100).toFixed(3) + '%';
          if (elTau5) elTau5.textContent = Number(logModel.threshold_at_5pct_fpr).toFixed(6);
        }
      }
    }
  } catch (err) {
    console.warn('Could not load live benchmark metrics; verified fallbacks active:', err);
  }
}

// ==========================================================================
// Module 4: Line Risk Simulator & Segment Audit Table
// ==========================================================================
function initRiskSimulator() {
  const form = document.getElementById('simulator-form');
  const runBtn = document.getElementById('btn-run-simulation');

  // Synchronize range sliders and number inputs
  const bindPair = (sliderId, inputId) => {
    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);
    if (!slider || !input) return;

    slider.addEventListener('input', () => {
      input.value = slider.value;
    });

    input.addEventListener('input', () => {
      slider.value = input.value;
    });
  };

  bindPair('sim-slope-slider', 'sim-slope');
  bindPair('sim-elevation-slider', 'sim-elevation');
  bindPair('sim-burn-prob-slider', 'sim-burn-prob');
  bindPair('sim-dist-fire-slider', 'sim-dist-fire');
  bindPair('sim-barrier-slider', 'sim-barrier');
  bindPair('sim-attack-angle-slider', 'sim-attack-angle');
  bindPair('sim-attack-dot-slider', 'sim-attack-dot');

  // Scenario Presets
  document.querySelectorAll('.preset-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.preset-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const presetKey = pill.getAttribute('data-preset');
      const preset = ScenarioPresets[presetKey];
      if (!preset) return;

      applyPresetValues(preset);
      runSimulatorInference();
    });
  });

  // Run button execution
  runBtn.addEventListener('click', () => {
    runSimulatorInference();
  });

  // Table Search Filter
  const searchInput = document.getElementById('clbi-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      filterCLBISegments(e.target.value.toLowerCase().trim());
    });
  }

  // Sortable Table Headers
  document.querySelectorAll('#clbi-segments-table th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-col');
      if (AppState.clbiSortColumn === col) {
        AppState.clbiSortDir = AppState.clbiSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        AppState.clbiSortColumn = col;
        AppState.clbiSortDir = 'asc';
      }
      sortAndRenderCLBITable();
    });
  });

  // Run initial simulation
  runSimulatorInference();
}

function applyPresetValues(p) {
  const setVal = (id, sliderId, val) => {
    const input = document.getElementById(id);
    const slider = document.getElementById(sliderId);
    if (input) input.value = val;
    if (slider) slider.value = val;
  };

  setVal('sim-slope', 'sim-slope-slider', p.slope);
  setVal('sim-elevation', 'sim-elevation-slider', p.elevation);
  setVal('sim-burn-prob', 'sim-burn-prob-slider', p.burn_prob);
  setVal('sim-dist-fire', 'sim-dist-fire-slider', p.distance_to_fire);
  setVal('sim-barrier', 'sim-barrier-slider', p.barrier_width_m);
  setVal('sim-attack-angle', 'sim-attack-angle-slider', p.attack_angle);
  setVal('sim-attack-dot', 'sim-attack-dot-slider', p.attack_dot_product);
}

async function runSimulatorInference() {
  const runBtn = document.getElementById('btn-run-simulation');
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.innerHTML = 'Evaluating Physics Model...';
  }

  const payload = {
    segment_id: 'sim_active_scenario',
    slope: parseFloat(document.getElementById('sim-slope').value),
    elevation: parseFloat(document.getElementById('sim-elevation').value),
    burn_prob: parseFloat(document.getElementById('sim-burn-prob').value),
    distance_to_fire: parseFloat(document.getElementById('sim-dist-fire').value),
    barrier_width_m: parseFloat(document.getElementById('sim-barrier').value),
    attack_angle: parseFloat(document.getElementById('sim-attack-angle').value),
    attack_dot_product: parseFloat(document.getElementById('sim-attack-dot').value),
  };

  try {
    const res = await fetch('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}: Inference error`);
    const data = await res.json();
    renderSimulatorResult(data);
  } catch (err) {
    console.error('Simulator inference error:', err);
    showToast('Inference error; check connection to FastAPI server', 'error');
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Breach Assessment</span>';
    }
  }
}

function renderSimulatorResult(data) {
  const breachProb = Number(data.breach_probability);
  const holdProb = Number(data.hold_probability);
  const tier = data.risk_tier || 'LOW';

  // Update Badge
  const badge = document.getElementById('sim-risk-badge');
  badge.textContent = tier;
  badge.className = `risk-badge badge-${tier.toLowerCase()}`;

  // Update Percentages
  document.getElementById('sim-breach-pct').textContent = `${(breachProb * 100).toFixed(1)}%`;
  document.getElementById('sim-breach-prob').textContent = breachProb.toFixed(4);
  document.getElementById('sim-hold-prob').textContent = holdProb.toFixed(4);

  // Update Tier Threshold Text
  const tierDesc = {
    'LOW': 'LOW (< 0.30)',
    'MEDIUM': 'MEDIUM (0.30 - 0.60)',
    'HIGH': 'HIGH (0.60 - 0.80)',
    'CRITICAL': 'CRITICAL (>= 0.80)'
  };
  document.getElementById('sim-tier-desc').textContent = tierDesc[tier] || tier;

  // Update SVG Gauge
  // Circumference of r=50 circle is 2 * PI * 50 = 314.159
  const circumference = 314.159;
  const offset = circumference * (1 - breachProb);
  const gaugeCircle = document.getElementById('sim-gauge-circle');

  gaugeCircle.style.strokeDasharray = circumference;
  gaugeCircle.style.strokeDashoffset = offset;
  gaugeCircle.className = `gauge-fill gauge-fill-${tier.toLowerCase()}`;

  // Update Factor Lists
  const driversList = document.getElementById('sim-risk-drivers');
  const protectiveList = document.getElementById('sim-protective-factors');

  driversList.innerHTML = '';
  protectiveList.innerHTML = '';

  const drivers = data.top_risk_factors || [];
  if (drivers.length) {
    drivers.forEach(d => {
      const li = document.createElement('li');
      li.textContent = d;
      driversList.appendChild(li);
    });
  } else {
    driversList.innerHTML = '<li class="text-muted">None significant under current scenario</li>';
  }

  const protective = data.protective_factors || [];
  if (protective.length) {
    protective.forEach(p => {
      const li = document.createElement('li');
      li.textContent = p;
      protectiveList.appendChild(li);
    });
  } else {
    protectiveList.innerHTML = '<li class="text-muted">None significant under current head-fire assault geometry</li>';
  }

  // Update Narrative
  document.getElementById('sim-explanation-text').textContent = data.explanation || 'No advisory generated.';
}

// CLBI 100-Segment Table Module
async function loadCLBIDemoSegments() {
  try {
    const res = await fetch('/api/v1/demo');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    AppState.clbiSegments = data.segments || [];
    AppState.clbiFilteredSegments = [...AppState.clbiSegments];
    sortAndRenderCLBITable();
  } catch (err) {
    console.error('Failed to load CLBI demo segments:', err);
  }
}

function filterCLBISegments(query) {
  if (!query) {
    AppState.clbiFilteredSegments = [...AppState.clbiSegments];
  } else {
    AppState.clbiFilteredSegments = AppState.clbiSegments.filter(s => {
      const segId = (s.segment_id || '').toLowerCase();
      const fireName = (s.fire_name || '').toLowerCase();
      const tier = (s.risk_tier || '').toLowerCase();
      return segId.includes(query) || fireName.includes(query) || tier.includes(query);
    });
  }
  sortAndRenderCLBITable();
}

function sortAndRenderCLBITable() {
  const tbody = document.getElementById('clbi-segments-tbody');
  if (!tbody) return;

  const col = AppState.clbiSortColumn;
  const dir = AppState.clbiSortDir === 'asc' ? 1 : -1;

  AppState.clbiFilteredSegments.sort((a, b) => {
    let valA = a[col];
    let valB = b[col];

    if (typeof valA === 'string') valA = valA.toLowerCase();
    if (typeof valB === 'string') valB = valB.toLowerCase();

    if (valA < valB) return -1 * dir;
    if (valA > valB) return 1 * dir;
    return 0;
  });

  // Update header sort indicators
  document.querySelectorAll('#clbi-segments-table th.sortable').forEach(th => {
    const thCol = th.getAttribute('data-col');
    const icon = th.querySelector('.sort-icon');
    if (thCol === col) {
      icon.innerHTML = dir === 1 ? '&uarr;' : '&darr;';
    } else {
      icon.innerHTML = '';
    }
  });

  if (!AppState.clbiFilteredSegments.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="text-center py-4 text-muted">No fireline segments match the current filter.</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  AppState.clbiFilteredSegments.forEach(seg => {
    const tr = document.createElement('tr');
    const tier = seg.risk_tier || 'LOW';
    const breachProb = Number(seg.breach_probability);
    const holdProb = Number(seg.hold_probability);

    tr.innerHTML = `
      <td class="font-mono text-muted">${seg.vulnerability_rank}</td>
      <td class="font-mono"><strong>${seg.segment_id}</strong></td>
      <td>${seg.fire_name || 'CZU Lightning'}</td>
      <td><span class="risk-badge badge-${tier.toLowerCase()}">${tier}</span></td>
      <td class="text-right font-mono font-bold ${breachProb >= 0.6 ? 'text-danger' : ''}">${(breachProb * 100).toFixed(1)}%</td>
      <td class="text-right font-mono">${(holdProb * 100).toFixed(1)}%</td>
      <td class="text-right font-mono">${Number(seg.barrier_width_m).toFixed(1)} m</td>
      <td class="text-right font-mono">${Number(seg.distance_to_fire).toFixed(0)} m</td>
      <td class="text-right font-mono">${Number(seg.slope).toFixed(1)}°</td>
      <td class="text-center">
        <button class="btn btn-secondary btn-xs btn-inspect-seg" data-id="${seg.segment_id}">Inspect</button>
      </td>
    `;

    tbody.appendChild(tr);
  });

  // Bind inspect buttons
  tbody.querySelectorAll('.btn-inspect-seg').forEach(btn => {
    btn.addEventListener('click', () => {
      const segId = btn.getAttribute('data-id');
      const segment = AppState.clbiSegments.find(s => s.segment_id === segId);
      if (segment) openSegmentDrawer(segment);
    });
  });
}

function openSegmentDrawer(seg) {
  const drawer = document.getElementById('segment-drawer');
  const title = document.getElementById('drawer-title');
  const content = document.getElementById('drawer-content');

  title.textContent = seg.segment_id;

  const driversHtml = (seg.top_risk_factors || [])
    .map(f => `<li>${f}</li>`).join('') || '<li class="text-muted">None significant</li>';

  const protectiveHtml = (seg.protective_factors || [])
    .map(p => `<li>${p}</li>`).join('') || '<li class="text-muted">None significant</li>';

  content.innerHTML = `
    <div style="margin-bottom: 1.25rem;">
      <span class="risk-badge badge-${seg.risk_tier.toLowerCase()}" style="font-size:0.85rem;">
        ${seg.risk_tier} RISK TIER
      </span>
      <span class="font-mono text-muted" style="margin-left: 0.5rem;">
        Vulnerability Rank: #${seg.vulnerability_rank} of 100
      </span>
    </div>

    <div class="gauge-stats-column" style="margin-bottom: 1.25rem;">
      <div class="gauge-stat-box">
        <span class="stat-box-label">Breach Probability</span>
        <span class="stat-box-val font-mono font-bold text-danger">${(seg.breach_probability * 100).toFixed(2)}%</span>
      </div>
      <div class="gauge-stat-box">
        <span class="stat-box-label">Holding Probability</span>
        <span class="stat-box-val font-mono font-bold text-success">${(seg.hold_probability * 100).toFixed(2)}%</span>
      </div>
      <div class="gauge-stat-box">
        <span class="stat-box-label">Priority Percentile</span>
        <span class="stat-box-val font-mono">${(seg.priority_percentile || 0).toFixed(1)}%</span>
      </div>
    </div>

    <div class="context-grid" style="margin-bottom: 1.25rem;">
      <div class="context-item"><span class="context-label">Cleared Barrier Width</span><span class="context-val">${seg.barrier_width_m} m</span></div>
      <div class="context-item"><span class="context-label">Distance to Active Flame</span><span class="context-val">${seg.distance_to_fire} m</span></div>
      <div class="context-item"><span class="context-label">Terrain Slope Angle</span><span class="context-val">${seg.slope}°</span></div>
      <div class="context-item"><span class="context-label">Elevation</span><span class="context-val">${seg.elevation} m</span></div>
      <div class="context-item"><span class="context-label">Acute Attack Angle</span><span class="context-val">${seg.attack_angle}°</span></div>
      <div class="context-item"><span class="context-label">Head-Fire Alignment (cos θ)</span><span class="context-val">${seg.attack_dot_product}</span></div>
    </div>

    <div class="tactical-factors-box" style="margin-bottom: 1.25rem;">
      <div class="factor-column">
        <div class="factor-header text-danger">Primary Risk Drivers</div>
        <ul class="factor-list">${driversHtml}</ul>
      </div>
      <div class="factor-column">
        <div class="factor-header text-success">Protective Factors</div>
        <ul class="factor-list">${protectiveHtml}</ul>
      </div>
    </div>

    <div class="tactical-narrative-box">
      <div class="narrative-label">Tactical Operational Advisory:</div>
      <p class="narrative-body">${seg.explanation || 'No operational narrative.'}</p>
    </div>
  `;

  drawer.classList.add('open');
}

// ==========================================================================
// Module 5: Modal & Drawer Controls
// ==========================================================================
function initModals() {
  const drawer = document.getElementById('segment-drawer');
  const closeDrawerBtn = document.getElementById('btn-close-drawer');
  const drawerOverlay = document.getElementById('drawer-overlay');

  const closeDrawer = () => drawer.classList.remove('open');
  if (closeDrawerBtn) closeDrawerBtn.addEventListener('click', closeDrawer);
  if (drawerOverlay) drawerOverlay.addEventListener('click', closeDrawer);

  const plotModal = document.getElementById('plot-modal');
  const closeModalBtn = document.getElementById('btn-close-modal');
  const modalOverlay = document.getElementById('modal-overlay');

  const closeModal = () => plotModal.classList.remove('open');
  if (closeModalBtn) closeModalBtn.addEventListener('click', closeModal);
  if (modalOverlay) modalOverlay.addEventListener('click', closeModal);

  // Escape key closes modals
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeDrawer();
      closeModal();
    }
  });
}

// Toast helper
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}
