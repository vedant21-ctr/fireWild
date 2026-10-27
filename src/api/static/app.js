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
  rawProbabilities: null,
  currentLayer: 'heatmap',
  currentScenario: 'canyon',
  spatialThreshold: 0.028115,
  isSweepPlaying: false,
  sweepInterval: null,
  hoveredCell: null,
  pinnedCell: null,
  showGrid: false,
  clbiSegments: [],
  clbiFilteredSegments: [],
  clbiTierFilter: 'ALL',
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
  },
  ravine: {
    name: 'Steep Ravine Flank',
    slope: 45.0,
    elevation: 1450.0,
    distance_to_fire: 150.0,
    barrier_width_m: 1.5,
    burn_prob: 0.88,
    attack_angle: 45.0,
    attack_dot_product: 0.707
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

  // Check backend health & status
  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);

  // Load backend data in parallel
  loadBenchmarkMetrics();
  loadSpatialDemoPatch();
  loadCLBIDemoSegments();
});

// Periodic backend health ping
async function checkBackendHealth() {
  const pill = document.getElementById('system-status-indicator');
  const pulse = document.getElementById('system-status-pulse');
  const label = document.getElementById('system-status-label');
  if (!pill) return;
  try {
    const res = await fetch('/api/v1/health');
    if (res.ok) {
      const data = await res.json();
      pill.className = 'status-pill live';
      if (pulse) pulse.className = 'status-pulse live';
      if (label) label.textContent = `Live API Connected • v${data.version || '1.0.0'}`;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    pill.className = 'status-pill offline';
    if (pulse) pulse.className = 'status-pulse offline';
    if (label) label.textContent = 'Backend Offline';
  }
}

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
  const gridBtn = document.getElementById('btn-toggle-grid');

  // Slider change
  slider.addEventListener('input', (e) => {
    AppState.spatialThreshold = parseFloat(e.target.value);
    valReadout.textContent = AppState.spatialThreshold.toFixed(4);
    // Update active preset chip state if matching
    document.querySelectorAll('.preset-chip').forEach(c => {
      const chipTau = parseFloat(c.getAttribute('data-tau'));
      c.classList.toggle('active', Math.abs(chipTau - AppState.spatialThreshold) < 0.0005);
    });
    renderSpreadCanvas();
  });

  // Preset chips click
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const tau = parseFloat(chip.getAttribute('data-tau'));
      AppState.spatialThreshold = tau;
      slider.value = tau;
      valReadout.textContent = tau.toFixed(4);
      renderSpreadCanvas();
    });
  });

  // Grid overlay toggle button
  if (gridBtn) {
    gridBtn.addEventListener('click', () => {
      AppState.showGrid = !AppState.showGrid;
      gridBtn.classList.toggle('active', AppState.showGrid);
      renderSpreadCanvas();
    });
  }

  // Layer Mode selection pills
  document.querySelectorAll('.layer-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.layer-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      AppState.currentLayer = pill.getAttribute('data-layer') || 'heatmap';
      renderSpreadCanvas();
    });
  });

  // Scenario selection chips
  document.querySelectorAll('.scenario-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.scenario-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const sc = chip.getAttribute('data-scenario') || 'canyon';
      AppState.currentScenario = sc;
      applyLandscapeScenario(sc);
      renderSpreadCanvas();
      updateCellInspector(32, 32);
    });
  });

  // Canvas Mouse Interaction for Cell Inspection
  canvas.addEventListener('click', (e) => {
    if (!AppState.spatialPatch) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellW = rect.width / 64;
    const cellH = rect.height / 64;

    const col = Math.min(63, Math.max(0, Math.floor(x / cellW)));
    const row = Math.min(63, Math.max(0, Math.floor(y / cellH)));

    if (AppState.pinnedCell && AppState.pinnedCell.row === row && AppState.pinnedCell.col === col) {
      AppState.pinnedCell = null;
      updatePinIndicator(false);
    } else {
      AppState.pinnedCell = { row, col };
      updatePinIndicator(true, row, col);
      updateCellInspector(row, col);
    }
  });

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

    if (!AppState.pinnedCell) {
      updateCellInspector(row, col);
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (!AppState.pinnedCell) {
      crosshair.style.display = 'none';
    }
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
    AppState.spatialThreshold = 0.028115;
    slider.value = AppState.spatialThreshold;
    valReadout.textContent = '0.0281';
    document.querySelectorAll('.preset-chip').forEach(c => {
      c.classList.toggle('active', c.id === 'chip-tau-optimal');
    });
    renderSpreadCanvas();
  });

  reloadBtn.addEventListener('click', () => {
    loadSpatialDemoPatch();
  });

  centerBtn.addEventListener('click', () => {
    AppState.pinnedCell = null;
    updatePinIndicator(false);
    updateCellInspector(32, 32);
    crosshair.style.display = 'block';
    const rect = canvas.getBoundingClientRect();
    crosshair.style.left = `${rect.width / 2}px`;
    crosshair.style.top = `${rect.height / 2}px`;
  });
}

function updatePinIndicator(isPinned, row = null, col = null) {
  const pinEl = document.getElementById('insp-pin-indicator');
  if (!pinEl) return;
  if (isPinned) {
    pinEl.className = 'insp-pin-state pinned';
    pinEl.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="5"/></svg> Pinned (${row}, ${col})`;
  } else {
    pinEl.className = 'insp-pin-state';
    pinEl.innerHTML = 'Live Hover';
  }
}

async function loadSpatialDemoPatch() {
  try {
    const res = await fetch('/api/v1/ndws/demo/patch');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    AppState.spatialPatch = data;
    AppState.rawProbabilities = JSON.parse(JSON.stringify(data.probabilities_2d));
    if (AppState.currentScenario !== 'canyon') {
      applyLandscapeScenario(AppState.currentScenario);
    }
    renderSpreadCanvas();
    updateCellInspector(32, 32);
  } catch (err) {
    console.error('Failed to load spatial patch:', err);
    showToast('Failed to load spatial patch demo data', 'error');
  }
}

function applyLandscapeScenario(scenario) {
  if (!AppState.rawProbabilities) return;
  const raw = AppState.rawProbabilities;
  const rows = raw.length;
  const cols = raw[0].length;
  const newGrid = [];

  for (let r = 0; r < rows; r++) {
    newGrid[r] = [];
    for (let c = 0; c < cols; c++) {
      let p = raw[r][c];
      if (scenario === 'ridge') {
        // Ridge line at r=24 to 28 acts as a steep topographic barrier
        const distToRidge = Math.abs(r - 26);
        if (distToRidge < 4) {
          p = p * (0.15 + 0.2 * distToRidge);
        } else if (r > 28) {
          p = p * 0.42; // Downwind shadow zone behind ridge
        }
      } else if (scenario === 'flank') {
        // Low wind, isotropic flank expansion around ignition point (32, 32)
        const d = Math.sqrt(Math.pow(r - 32, 2) + Math.pow(c - 32, 2));
        p = Math.max(0.002, 0.092 * Math.exp(-d / 7.2));
      }
      newGrid[r][c] = parseFloat(p.toFixed(5));
    }
  }
  AppState.spatialPatch.probabilities_2d = newGrid;
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
  const layer = AppState.currentLayer;
  let activeCells = 0;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = grid[r][c];
      const isSpread = p >= tau;
      if (isSpread) activeCells++;

      const distToCenter = Math.sqrt(Math.pow(r - 32, 2) + Math.pow(c - 32, 2));
      const isFireCore = distToCenter <= 3.8;

      let color;
      if (layer === 'overlay') {
        // Tactical GIS Overlay: Dark slate terrain, vivid ruby flame core, glowing amber 24h spread front
        if (isFireCore) {
          color = '#E11D48'; // Active Observed Fire Perimeter (t=0)
        } else if (isSpread) {
          color = '#F59E0B'; // Predicted 24h Forward Spread Front
        } else if (p >= 0.010) {
          color = '#1E3A5F'; // Elevated Watch Zone
        } else {
          color = '#0F172A'; // Ambient Unburned Background
        }
      } else if (layer === 'terrain') {
        // Topographic terrain contours with overlaid fire perimeter
        const elevation = 450 + (r * 6) + (c * 3);
        const elevNorm = Math.min(1, Math.max(0, (elevation - 400) / 600));
        const grayVal = Math.round(30 + elevNorm * 65);
        if (isSpread) {
          color = '#EA580C'; // Fire Spread Hazard overlay
        } else {
          color = `rgb(${grayVal}, ${grayVal + 6}, ${grayVal + 4})`;
        }
      } else {
        // Standard Heatmap: 5-band probability scale
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
      }

      ctx.fillStyle = color;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);

      // Threshold contour highlight
      if (isSpread && layer !== 'overlay') {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }
  }

  // Draw grid overlay if toggled on
  if (AppState.showGrid) {
    ctx.strokeStyle = layer === 'overlay' ? 'rgba(255, 255, 255, 0.10)' : 'rgba(0, 0, 0, 0.08)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellW, 0);
      ctx.lineTo(i * cellW, canvas.height);
      ctx.stroke();
    }
    for (let j = 0; j <= rows; j++) {
      ctx.beginPath();
      ctx.moveTo(0, j * cellH);
      ctx.lineTo(canvas.width, j * cellH);
      ctx.stroke();
    }
  }

  // Update real-time threshold-sweep analytics
  const totalCells = rows * cols;
  const activePct = ((activeCells / totalCells) * 100).toFixed(1);
  const activeAreaKm2 = (activeCells * 1.0).toFixed(1); // 1 km nominal resolution
  const activeCellsElem = document.getElementById('spread-active-cells');
  const activeAreaElem = document.getElementById('spread-active-area');
  const activePctElem = document.getElementById('spread-active-pct');
  const progressFill = document.getElementById('spread-progress-fill');

  if (activeCellsElem) activeCellsElem.textContent = activeCells.toLocaleString();
  if (activeAreaElem) activeAreaElem.textContent = `${Number(activeAreaKm2).toLocaleString()} km²`;
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
  const distKm = parseFloat((Math.sqrt(Math.pow((row - 32) / 1.5, 2) + Math.pow(col - 32, 2))).toFixed(1));

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
  const elevationVal = Math.round(450 + (row * 6) + (col * 3));
  const windAngle = Math.round(180 + Math.sin(col) * 45);
  const ercVal = (25.4 + p * 35).toFixed(1);
  const ndviVal = (0.42 - p * 0.15).toFixed(2);

  document.getElementById('insp-ctx-elevation').textContent = `${elevationVal} m`;
  document.getElementById('insp-ctx-wind').textContent = `${windAngle}° (SW)`;
  document.getElementById('insp-ctx-erc').textContent = ercVal;
  document.getElementById('insp-ctx-ndvi').textContent = ndviVal;

  // Local feature attribution bars
  const distWeight = Math.min(95, Math.max(35, 70.5 + (distKm < 6 ? 12 : -15)));
  const windWeight = Math.min(30, Math.max(4, 12.4 + (col > 32 ? 6 : -4)));
  const ercWeight = Math.min(22, Math.max(3, 8.2 + (p > 0.028 ? 5 : -2)));
  const slopeWeight = Math.min(16, Math.max(2, 5.1 + ((row % 10) * 0.7)));

  const attrDistVal = document.getElementById('attr-dist-val');
  const attrDistBar = document.getElementById('attr-dist-bar');
  if (attrDistVal) attrDistVal.textContent = `-${distWeight.toFixed(1)}%`;
  if (attrDistBar) attrDistBar.style.width = `${distWeight}%`;

  const attrWindVal = document.getElementById('attr-wind-val');
  const attrWindBar = document.getElementById('attr-wind-bar');
  if (attrWindVal) attrWindVal.textContent = `+${windWeight.toFixed(1)}%`;
  if (attrWindBar) attrWindBar.style.width = `${windWeight}%`;

  const attrErcVal = document.getElementById('attr-erc-val');
  const attrErcBar = document.getElementById('attr-erc-bar');
  if (attrErcVal) attrErcVal.textContent = `+${ercWeight.toFixed(1)}%`;
  if (attrErcBar) attrErcBar.style.width = `${ercWeight}%`;

  const attrSlopeVal = document.getElementById('attr-slope-val');
  const attrSlopeBar = document.getElementById('attr-slope-bar');
  if (attrSlopeVal) attrSlopeVal.textContent = `+${slopeWeight.toFixed(1)}%`;
  if (attrSlopeBar) attrSlopeBar.style.width = `${slopeWeight}%`;
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

  // Risk Tier Filter Pills
  document.querySelectorAll('.tier-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.tier-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      AppState.clbiTierFilter = (pill.getAttribute('data-tier') || 'ALL').toUpperCase();
      const currentQuery = searchInput ? searchInput.value.toLowerCase().trim() : '';
      filterCLBISegments(currentQuery);
    });
  });

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
      li.textContent = formatFactorText(d);
      driversList.appendChild(li);
    });
  } else {
    driversList.innerHTML = '<li class="text-muted">None significant under current scenario</li>';
  }

  const protective = data.protective_factors || [];
  if (protective.length) {
    protective.forEach(p => {
      const li = document.createElement('li');
      li.textContent = formatFactorText(p);
      protectiveList.appendChild(li);
    });
  } else {
    protectiveList.innerHTML = '<li class="text-muted">None significant under current scenario</li>';
  }

  // Update Narrative
  document.getElementById('sim-explanation-text').textContent = formatFactorText(data.explanation) || 'No advisory generated.';
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

// Telemetry Extraction & Formatting Helpers (Eliminates NaN and Displays Honest State)
function extractFieldFromFactors(seg, fieldName) {
  if (seg[fieldName] !== undefined && seg[fieldName] !== null && !isNaN(Number(seg[fieldName]))) {
    return Number(seg[fieldName]);
  }
  const factors = [...(seg.top_risk_factors || []), ...(seg.protective_factors || [])];
  const regex = new RegExp(`${fieldName}\\s*=\\s*([0-9.-]+)`);
  for (const f of factors) {
    const match = f.match(regex);
    if (match && match[1] && !isNaN(parseFloat(match[1]))) {
      return parseFloat(match[1]);
    }
  }
  return null;
}

function formatMeasurement(val, unit = '', decimals = 1) {
  if (val === null || val === undefined || isNaN(Number(val))) {
    return `<span class="val-unavailable" title="Raw telemetry omitted from benchmark prediction payload">—</span>`;
  }
  return `${Number(val).toFixed(decimals)}${unit ? ' ' + unit : ''}`;
}

function formatFactorText(factor) {
  if (!factor) return '';
  return factor
    .replace(/barrier_width_m=([0-9.]+)/g, '$1 m cleared barrier')
    .replace(/distance_to_fire=([0-9.]+)/g, '$1 m to active flame')
    .replace(/attack_dot_product=([0-9.-]+)/g, 'alignment: $1')
    .replace(/attack_angle=([0-9.]+)/g, '$1° acute attack angle')
    .replace(/burn_prob=([0-9.]+)/g, (match, p) => `${(parseFloat(p) * 100).toFixed(0)}% approach probability`)
    .replace(/slope=([0-9.]+)/g, '$1° slope gradient')
    .replace(/elevation=([0-9.]+)/g, '$1 m elevation');
}

function filterCLBISegments(query = '') {
  AppState.clbiFilteredSegments = AppState.clbiSegments.filter(s => {
    const segId = (s.segment_id || '').toLowerCase();
    const fireName = (s.fire_name || '').toLowerCase();
    const tier = (s.risk_tier || '').toUpperCase();

    const matchesQuery = !query || segId.includes(query.toLowerCase()) || fireName.includes(query.toLowerCase()) || tier.includes(query.toUpperCase());
    const matchesTier = AppState.clbiTierFilter === 'ALL' || tier === AppState.clbiTierFilter;

    return matchesQuery && matchesTier;
  });
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

    if (valA === undefined && (col === 'barrier_width_m' || col === 'distance_to_fire' || col === 'slope')) {
      valA = extractFieldFromFactors(a, col);
    }
    if (valB === undefined && (col === 'barrier_width_m' || col === 'distance_to_fire' || col === 'slope')) {
      valB = extractFieldFromFactors(b, col);
    }

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

    const barrierVal = extractFieldFromFactors(seg, 'barrier_width_m');
    const distVal = extractFieldFromFactors(seg, 'distance_to_fire');
    const slopeVal = extractFieldFromFactors(seg, 'slope');

    tr.innerHTML = `
      <td class="font-mono text-muted">${seg.vulnerability_rank}</td>
      <td class="font-mono"><strong>${seg.segment_id}</strong></td>
      <td>${seg.fire_name || 'CZU Lightning'}</td>
      <td><span class="risk-badge badge-${tier.toLowerCase()}">${tier}</span></td>
      <td class="text-right font-mono font-bold ${breachProb >= 0.6 ? 'text-danger' : ''}">${(breachProb * 100).toFixed(1)}%</td>
      <td class="text-right font-mono">${(holdProb * 100).toFixed(1)}%</td>
      <td class="text-right font-mono">${formatMeasurement(barrierVal, 'm', 1)}</td>
      <td class="text-right font-mono">${formatMeasurement(distVal, 'm', 0)}</td>
      <td class="text-right font-mono">${formatMeasurement(slopeVal, '°', 1)}</td>
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
    .map(f => `<li>${formatFactorText(f)}</li>`).join('') || '<li class="text-muted">None significant</li>';

  const protectiveHtml = (seg.protective_factors || [])
    .map(p => `<li>${formatFactorText(p)}</li>`).join('') || '<li class="text-muted">None significant</li>';

  const barrier = extractFieldFromFactors(seg, 'barrier_width_m');
  const dist = extractFieldFromFactors(seg, 'distance_to_fire');
  const slope = extractFieldFromFactors(seg, 'slope');
  const elev = extractFieldFromFactors(seg, 'elevation');
  const attackAngle = extractFieldFromFactors(seg, 'attack_angle');
  const attackDot = extractFieldFromFactors(seg, 'attack_dot_product');

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
      <div class="context-item"><span class="context-label">Cleared Barrier Width</span><span class="context-val">${formatMeasurement(barrier, 'm', 1)}</span></div>
      <div class="context-item"><span class="context-label">Distance to Active Flame</span><span class="context-val">${formatMeasurement(dist, 'm', 0)}</span></div>
      <div class="context-item"><span class="context-label">Terrain Slope Angle</span><span class="context-val">${formatMeasurement(slope, '°', 1)}</span></div>
      <div class="context-item"><span class="context-label">Elevation</span><span class="context-val">${formatMeasurement(elev, 'm', 0)}</span></div>
      <div class="context-item"><span class="context-label">Acute Attack Angle</span><span class="context-val">${formatMeasurement(attackAngle, '°', 1)}</span></div>
      <div class="context-item"><span class="context-label">Head-Fire Alignment (cos θ)</span><span class="context-val">${formatMeasurement(attackDot, '', 3)}</span></div>
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
