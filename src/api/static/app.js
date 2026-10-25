// FireWild Command Center - Phase 1 Dashboard Application Logic
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadNDWSBenchmarkMetrics();
  loadNDWSSpatialPatch();
  loadCLBIDemoSegments();
  initSimulator();
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      const targetId = tab.getAttribute('data-target');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
}

// Global State
let gCLBISegments = [];
let gCurrentSort = { column: 'vulnerability_rank', dir: 'asc' };
let gOptimalThreshold = 0.028115;
let gSpatialMatrix = null;
let gCurrentHeatmapThreshold = 0.028115;

// ==========================================
// MODULE 1: NDWS BENCHMARK METRICS
// ==========================================
async function loadNDWSBenchmarkMetrics() {
  const tableBody = document.getElementById('ndws-models-tbody');
  const featureList = document.getElementById('ndws-feature-bars');

  try {
    const res = await fetch('/api/v1/ndws/benchmark/metrics');
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch metrics`);
    const data = await res.json();

    const best = data.best_model_details;
    if (best) {
      gOptimalThreshold = best.optimal_decision_threshold || 0.028115;
      gCurrentHeatmapThreshold = gOptimalThreshold;
      const valPerf = best.validation_performance || {};
      const evalPerf = best.held_out_evaluation_performance || {};

      document.getElementById('stat-best-model').textContent = best.model_id.replace(/_/g, ' ');
      document.getElementById('stat-val-ap').textContent = (valPerf.average_precision || 0).toFixed(4);
      document.getElementById('stat-eval-ap').textContent = (evalPerf.average_precision || 0).toFixed(4);
      document.getElementById('stat-fp-reduction').textContent = evalPerf.false_positive_reduction_vs_logistic || "-73.3% False Alarms";
      document.getElementById('stat-optimal-thresh').textContent = `Optimal τ* = ${gOptimalThreshold.toFixed(4)}`;

      // Update Heatmap Slider initial position if present
      const threshSlider = document.getElementById('heatmap-threshold-slider');
      const threshDisplay = document.getElementById('heatmap-thresh-value');
      if (threshSlider && threshDisplay) {
        threshSlider.value = gOptimalThreshold.toFixed(4);
        threshDisplay.textContent = gOptimalThreshold.toFixed(4);
      }

      // Feature Importance (Split Gain Attribution)
      const features = best.feature_importance_ranking || [];
      if (featureList && features.length > 0) {
        featureList.innerHTML = '';
        features.forEach(feat => {
          const row = document.createElement('div');
          row.className = 'feature-row';
          row.innerHTML = `
            <div class="feature-name" title="${feat.feature_name}">${feat.feature_name}</div>
            <div class="feature-bar-bg">
              <div class="feature-bar-fill" style="width: ${Math.min(100, feat.percentage_share)}%"></div>
            </div>
            <div class="feature-percent">${feat.percentage_share.toFixed(1)}%</div>
          `;
          featureList.appendChild(row);
        });
      }
    }

    // Model Comparison Leaderboard Table
    const models = data.models_comparison || [];
    if (tableBody) {
      tableBody.innerHTML = '';
      models.forEach((m, idx) => {
        const tr = document.createElement('tr');
        const isBest = m.is_best_model;
        tr.innerHTML = `
          <td><strong>#${idx + 1}</strong></td>
          <td>
            <strong>${m.display_name}</strong>
            ${isBest ? '<span class="badge badge-best" style="margin-left: 0.5rem">RECOMMENDED</span>' : ''}
          </td>
          <td style="font-family: var(--font-mono); font-weight: 700; color: ${isBest ? 'var(--accent-orange)' : 'var(--text-main)'}">
            ${m.average_precision.toFixed(4)}
          </td>
          <td style="font-family: var(--font-mono)">${m.roc_auc.toFixed(4)}</td>
          <td style="font-family: var(--font-mono)">${m.brier_score.toFixed(4)}</td>
          <td style="font-family: var(--font-mono)">${(m.precision_at_threshold * 100).toFixed(2)}%</td>
          <td style="font-family: var(--font-mono)">${(m.recall_at_threshold * 100).toFixed(2)}%</td>
          <td style="font-family: var(--font-mono); font-weight: 600">${m.f1_at_threshold.toFixed(4)}</td>
          <td style="font-family: var(--font-mono); color: var(--text-muted)">${m.decision_threshold.toFixed(4)}</td>
        `;
        tableBody.appendChild(tr);
      });
    }

  } catch (err) {
    console.error('Error loading NDWS metrics:', err);
    if (tableBody) {
      tableBody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--accent-red); padding: 1.5rem">Error loading benchmark data: ${err.message}</td></tr>`;
    }
  }
}

// ==========================================
// MODULE 2: UPGRADED SPATIAL HEATMAP
// ==========================================
async function loadNDWSSpatialPatch() {
  const canvas = document.getElementById('spatial-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('canvas-tooltip');
  const threshSlider = document.getElementById('heatmap-threshold-slider');
  const threshDisplay = document.getElementById('heatmap-thresh-value');
  const activeCellsDisplay = document.getElementById('heatmap-active-cells');

  try {
    const res = await fetch('/api/v1/ndws/demo/patch');
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch spatial patch`);
    const data = await res.json();
    gSpatialMatrix = data.probabilities_2d;

    updateHeatmap();

    // Slider listener
    if (threshSlider) {
      threshSlider.addEventListener('input', (e) => {
        gCurrentHeatmapThreshold = parseFloat(e.target.value);
        if (threshDisplay) threshDisplay.textContent = gCurrentHeatmapThreshold.toFixed(4);
        updateHeatmap();
      });
    }

    // Canvas Hover Inspector
    canvas.addEventListener('mousemove', (e) => {
      if (!gSpatialMatrix) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const pxX = Math.floor((e.clientX - rect.left) * scaleX);
      const pxY = Math.floor((e.clientY - rect.top) * scaleY);

      const cellW = canvas.width / 64;
      const cellH = canvas.height / 64;

      const col = Math.floor(pxX / cellW);
      const row = Math.floor(pxY / cellH);

      if (row >= 0 && row < 64 && col >= 0 && col < 64) {
        const val = gSpatialMatrix[row][col];
        const isSpread = val >= gCurrentHeatmapThreshold;
        tooltip.style.display = 'block';
        tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        tooltip.style.top = `${e.clientY - rect.top + 15}px`;
        tooltip.innerHTML = `
          <div><strong>Grid Coord:</strong> [r:${row}, c:${col}]</div>
          <div><strong>Spread Prob:</strong> ${val.toFixed(4)}</div>
          <div><strong>Status:</strong> ${isSpread ? '<span style="color:var(--accent-orange); font-weight:700">SPREAD BOUNDARY (≥τ)</span>' : '<span style="color:var(--accent-green)">BACKGROUND (&lt;τ)</span>'}</div>
        `;
      }
    });

    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });

  } catch (err) {
    console.error('Error loading spatial demo patch:', err);
  }

  function updateHeatmap() {
    if (!gSpatialMatrix) return;
    renderCanvas(ctx, gSpatialMatrix, gCurrentHeatmapThreshold);

    // Calculate count and percentage above threshold
    let spreadCount = 0;
    const totalCells = 64 * 64;
    for (let r = 0; r < 64; r++) {
      for (let c = 0; c < 64; c++) {
        if (gSpatialMatrix[r][c] >= gCurrentHeatmapThreshold) {
          spreadCount++;
        }
      }
    }
    const pct = ((spreadCount / totalCells) * 100).toFixed(1);
    if (activeCellsDisplay) {
      activeCellsDisplay.innerHTML = `Highlighting: <strong>${spreadCount} / ${totalCells}</strong> cells (${pct}% of grid ≥ τ)`;
    }
  }
}

// Reset Heatmap to Optimal Threshold
window.resetHeatmapThreshold = function() {
  const threshSlider = document.getElementById('heatmap-threshold-slider');
  const threshDisplay = document.getElementById('heatmap-thresh-value');
  gCurrentHeatmapThreshold = gOptimalThreshold;
  if (threshSlider) threshSlider.value = gOptimalThreshold.toFixed(4);
  if (threshDisplay) threshDisplay.textContent = gOptimalThreshold.toFixed(4);
  const canvas = document.getElementById('spatial-canvas');
  if (canvas && gSpatialMatrix) {
    const ctx = canvas.getContext('2d');
    renderCanvas(ctx, gSpatialMatrix, gCurrentHeatmapThreshold);
    const activeCellsDisplay = document.getElementById('heatmap-active-cells');
    let spreadCount = 0;
    for (let r = 0; r < 64; r++) {
      for (let c = 0; c < 64; c++) {
        if (gSpatialMatrix[r][c] >= gCurrentHeatmapThreshold) spreadCount++;
      }
    }
    if (activeCellsDisplay) {
      activeCellsDisplay.innerHTML = `Highlighting: <strong>${spreadCount} / 4096</strong> cells (${((spreadCount / 4096) * 100).toFixed(1)}% of grid ≥ τ)`;
    }
  }
};

// Inferno Colormap with Adaptive Contrast
function getInfernoColor(prob, threshold) {
  if (prob < threshold) {
    // Dim background cells below selected visualization threshold
    return [10, 14, 22, 255];
  }

  // Enhanced fire gradient for active cells
  const t = Math.min(1.0, prob * 3.5);
  let r = 0, g = 0, b = 0;

  if (t < 0.25) {
    const f = t / 0.25;
    r = Math.floor(45 + f * 50);
    g = Math.floor(15 + f * 15);
    b = Math.floor(75 + f * 40);
  } else if (t < 0.5) {
    const f = (t - 0.25) / 0.25;
    r = Math.floor(95 + f * 95);
    g = Math.floor(30 + f * 10);
    b = Math.floor(115 - f * 45);
  } else if (t < 0.75) {
    const f = (t - 0.5) / 0.25;
    r = Math.floor(190 + f * 55);
    g = Math.floor(40 + f * 85);
    b = Math.floor(70 - f * 50);
  } else {
    const f = (t - 0.75) / 0.25;
    r = Math.floor(245 + f * 10);
    g = Math.floor(125 + f * 125);
    b = Math.floor(20 + f * 160);
  }

  return [r, g, b, 255];
}

function renderCanvas(ctx, matrix, threshold) {
  if (!matrix) return;
  const size = 64;
  const imgData = ctx.createImageData(384, 384);
  const data = imgData.data;
  const scale = 384 / size; // 6px per grid cell

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const prob = matrix[r][c];
      const [cr, cg, cb, ca] = getInfernoColor(prob, threshold);

      for (let sy = 0; sy < scale; sy++) {
        for (let sx = 0; sx < scale; sx++) {
          const px = Math.floor(c * scale + sx);
          const py = Math.floor(r * scale + sy);
          const idx = (py * 384 + px) * 4;
          data[idx] = cr;
          data[idx + 1] = cg;
          data[idx + 2] = cb;
          data[idx + 3] = ca;
        }
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

// ==========================================
// MODULE 3: UPGRADED CLBI TABLE WITH SORTING
// ==========================================
async function loadCLBIDemoSegments() {
  const tableBody = document.getElementById('clbi-segments-tbody');
  const filterSelect = document.getElementById('clbi-filter-risk');
  const searchInput = document.getElementById('clbi-search-input');

  try {
    const res = await fetch('/api/v1/demo');
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch CLBI demo`);
    const data = await res.json();
    gCLBISegments = data.segments || [];

    // Calculate summary statistics dynamically from returned array
    const total = gCLBISegments.length;
    let critical = 0, high = 0, medium = 0, low = 0;
    gCLBISegments.forEach(s => {
      if (s.risk_tier === 'CRITICAL') critical++;
      else if (s.risk_tier === 'HIGH') high++;
      else if (s.risk_tier === 'MEDIUM') medium++;
      else if (s.risk_tier === 'LOW') low++;
    });

    const actionable = critical + high;
    const actionablePct = total > 0 ? Math.round((actionable / total) * 100) : 0;

    document.getElementById('clbi-stat-total').textContent = total;
    document.getElementById('clbi-stat-critical').textContent = critical;
    document.getElementById('clbi-stat-high').textContent = high;
    document.getElementById('clbi-stat-actionable').textContent = `${actionable} (${actionablePct}%)`;

    renderCLBITable();

    if (filterSelect) filterSelect.addEventListener('change', renderCLBITable);
    if (searchInput) searchInput.addEventListener('input', renderCLBITable);

    // Attach sort header listeners
    document.querySelectorAll('#clbi-table th.sortable').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.getAttribute('data-sort');
        if (gCurrentSort.column === col) {
          gCurrentSort.dir = gCurrentSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          gCurrentSort.column = col;
          gCurrentSort.dir = (col === 'breach_probability' || col === 'priority_percentile') ? 'desc' : 'asc';
        }
        updateSortIcons();
        renderCLBITable();
      });
    });

  } catch (err) {
    console.error('Error loading CLBI demo:', err);
    if (tableBody) {
      tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--accent-red); padding: 1.5rem">Error loading containment lines: ${err.message}</td></tr>`;
    }
  }
}

function updateSortIcons() {
  document.querySelectorAll('#clbi-table th.sortable').forEach(th => {
    const col = th.getAttribute('data-sort');
    const label = th.getAttribute('data-label');
    if (gCurrentSort.column === col) {
      const arrow = gCurrentSort.dir === 'asc' ? ' ▲' : ' ▼';
      th.textContent = label + arrow;
      th.style.color = 'var(--accent-orange)';
    } else {
      th.textContent = label;
      th.style.color = '';
    }
  });
}

function renderCLBITable() {
  const tableBody = document.getElementById('clbi-segments-tbody');
  const filterRisk = document.getElementById('clbi-filter-risk').value;
  const search = (document.getElementById('clbi-search-input').value || '').toLowerCase();

  let filtered = gCLBISegments.filter(seg => {
    const matchesRisk = (filterRisk === 'ALL' || seg.risk_tier === filterRisk);
    const matchesSearch = seg.segment_id.toLowerCase().includes(search) || (seg.fire_name && seg.fire_name.toLowerCase().includes(search));
    return matchesRisk && matchesSearch;
  });

  // Sort filtered array
  const { column, dir } = gCurrentSort;
  filtered.sort((a, b) => {
    let valA = a[column];
    let valB = b[column];

    if (valA === null || valA === undefined) valA = 0;
    if (valB === null || valB === undefined) valB = 0;

    if (typeof valA === 'string') {
      return dir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return dir === 'asc' ? valA - valB : valB - valA;
  });

  tableBody.innerHTML = '';
  document.getElementById('clbi-count-display').textContent = `Showing ${filtered.length} of ${gCLBISegments.length} candidate segments`;

  if (filtered.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted); padding: 2rem">No fireline segments match the selected criteria.</td></tr>`;
    return;
  }

  filtered.forEach(seg => {
    const tr = document.createElement('tr');
    const badgeClass = `badge badge-${seg.risk_tier.toLowerCase()}`;
    const bpPct = Math.round(seg.breach_probability * 100);

    let progressColor = 'var(--risk-low)';
    if (seg.risk_tier === 'CRITICAL') progressColor = 'var(--risk-critical)';
    else if (seg.risk_tier === 'HIGH') progressColor = 'var(--risk-high)';
    else if (seg.risk_tier === 'MEDIUM') progressColor = 'var(--risk-med)';

    tr.innerHTML = `
      <td><strong>${seg.segment_id}</strong></td>
      <td>${seg.fire_name || 'CZU Scenario'}</td>
      <td>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${bpPct}%; background: ${progressColor}"></div>
        </div>
        <span style="font-family: var(--font-mono); font-weight:600">${seg.breach_probability.toFixed(3)}</span>
      </td>
      <td><span class="${badgeClass}">${seg.risk_tier}</span></td>
      <td style="font-family: var(--font-mono)">${seg.vulnerability_rank || '-'}</td>
      <td style="font-family: var(--font-mono)">${seg.priority_percentile ? seg.priority_percentile.toFixed(1) + '%' : '-'}</td>
      <td>
        <button class="btn-outline" onclick="inspectSegment('${seg.segment_id}')">
          Inspect
        </button>
      </td>
    `;
    tableBody.appendChild(tr);
  });
}

// Inspect Modal
window.inspectSegment = function(segId) {
  const seg = gCLBISegments.find(s => s.segment_id === segId);
  if (!seg) return;

  const modal = document.getElementById('segment-modal');
  document.getElementById('modal-title').textContent = `Segment Telemetry: ${seg.segment_id}`;

  const detailsContainer = document.getElementById('modal-details');
  const badgeClass = `badge badge-${seg.risk_tier.toLowerCase()}`;

  detailsContainer.innerHTML = `
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.25rem;">
      <div style="background:var(--bg-secondary); padding:1rem; border-radius:var(--radius-sm); border:1px solid var(--border-color);">
        <div class="metric-label">Risk Classification</div>
        <div><span class="${badgeClass}" style="font-size: 0.95rem">${seg.risk_tier} RISK TIER</span></div>
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
          Vulnerability Rank: <strong>#${seg.vulnerability_rank || '-'}</strong> (Percentile: <strong>${seg.priority_percentile ? seg.priority_percentile.toFixed(1) + '%' : '-'}</strong>)
        </div>
      </div>
      <div style="background:var(--bg-secondary); padding:1rem; border-radius:var(--radius-sm); border:1px solid var(--border-color);">
        <div class="metric-label">Holding Probability</div>
        <div style="font-family: var(--font-mono); font-size: 1.3rem; font-weight:800; color:var(--text-main)">
          ${(seg.hold_probability * 100).toFixed(1)}%
        </div>
        <div style="font-size: 0.8rem; color: var(--accent-orange); margin-top: 0.25rem;">
          Breach Risk: <strong>${(seg.breach_probability * 100).toFixed(1)}%</strong>
        </div>
      </div>
    </div>

    <div style="margin-bottom: 1.25rem;">
      <div class="metric-label">Model Reasoning</div>
      <p style="font-size: 0.88rem; background: var(--bg-secondary); padding: 0.9rem; border-radius: var(--radius-sm); border: 1px solid var(--border-color); line-height:1.6">
        ${seg.explanation || 'Segment scored via 7-feature logistic regression decision boundary.'}
      </p>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem;">
      <div>
        <div class="metric-label" style="color: #f87171">Primary Risk Accelerators</div>
        <ul style="font-size: 0.82rem; padding-left: 1.2rem; color: var(--text-main); display:flex; flex-direction:column; gap:0.35rem;">
          ${(seg.top_risk_factors || []).map(f => `<li>${f}</li>`).join('') || '<li>None identified</li>'}
        </ul>
      </div>
      <div>
        <div class="metric-label" style="color: #4ade80">Primary Protective Buffers</div>
        <ul style="font-size: 0.82rem; padding-left: 1.2rem; color: var(--text-main); display:flex; flex-direction:column; gap:0.35rem;">
          ${(seg.protective_factors || []).map(f => `<li>${f}</li>`).join('') || '<li>None identified</li>'}
        </ul>
      </div>
    </div>
  `;

  modal.style.display = 'flex';
};

window.closeModal = function() {
  document.getElementById('segment-modal').style.display = 'none';
};

// ==========================================
// MODULE 4: REDESIGNED SCENARIO SIMULATOR
// ==========================================
const SIM_PRESETS = {
  vulnerable_ridge: {
    slope: 32.0,
    elevation: 850.0,
    distance_to_fire: 45.0,
    barrier_width_m: 3.0,
    burn_prob: 0.72,
    attack_angle: 25.0,
    attack_dot_product: 0.85,
    note: "Steep topography, narrow cleared fuel break, and close active flame front with head-fire alignment."
  },
  direct_attack: {
    slope: 16.0,
    elevation: 480.0,
    distance_to_fire: 80.0,
    barrier_width_m: 5.0,
    burn_prob: 0.60,
    attack_angle: 10.0,
    attack_dot_product: 0.95,
    note: "Direct flame assault with near-perpendicular attack alignment and moderate fuel barrier."
  },
  wide_barrier: {
    slope: 6.0,
    elevation: 320.0,
    distance_to_fire: 250.0,
    barrier_width_m: 18.0,
    burn_prob: 0.35,
    attack_angle: 60.0,
    attack_dot_product: -0.20,
    note: "Gentle valley terrain with substantial 18-meter cleared defensible buffer and distant perimeter."
  }
};

function initSimulator() {
  // Synchronize range sliders with number inputs
  syncInput('sim-slope-range', 'sim-slope-num');
  syncInput('sim-elev-range', 'sim-elev-num');
  syncInput('sim-dist-range', 'sim-dist-num');
  syncInput('sim-barrier-range', 'sim-barrier-num');
  syncInput('sim-burn-range', 'sim-burn-num');
  syncInput('sim-angle-range', 'sim-angle-num');
  syncInput('sim-dot-range', 'sim-dot-num');

  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const presetKey = btn.getAttribute('data-preset');
      applyPreset(presetKey);
    });
  });

  // Simulator Form Submission
  const form = document.getElementById('sim-form');
  if (form) {
    form.addEventListener('submit', runSimulatorScenario);
  }

  // Run initial simulation on load
  runSimulatorScenario();
}

function syncInput(rangeId, numId) {
  const range = document.getElementById(rangeId);
  const num = document.getElementById(numId);
  if (!range || !num) return;

  range.addEventListener('input', () => {
    num.value = range.value;
  });

  num.addEventListener('input', () => {
    range.value = num.value;
  });
}

function applyPreset(key) {
  const p = SIM_PRESETS[key];
  if (!p) return;

  setInputValue('sim-slope-range', 'sim-slope-num', p.slope);
  setInputValue('sim-elev-range', 'sim-elev-num', p.elevation);
  setInputValue('sim-dist-range', 'sim-dist-num', p.distance_to_fire);
  setInputValue('sim-barrier-range', 'sim-barrier-num', p.barrier_width_m);
  setInputValue('sim-burn-range', 'sim-burn-num', p.burn_prob);
  setInputValue('sim-angle-range', 'sim-angle-num', p.attack_angle);
  setInputValue('sim-dot-range', 'sim-dot-num', p.attack_dot_product);

  const presetDesc = document.getElementById('sim-preset-desc');
  if (presetDesc) presetDesc.textContent = p.note;

  runSimulatorScenario();
}

function setInputValue(rangeId, numId, val) {
  const r = document.getElementById(rangeId);
  const n = document.getElementById(numId);
  if (r) r.value = val;
  if (n) n.value = val;
}

async function runSimulatorScenario(e) {
  if (e) e.preventDefault();

  const runBtn = document.getElementById('btn-run-sim');
  const probDisplay = document.getElementById('sim-prob-display');
  const tierBadge = document.getElementById('sim-tier-badge');
  const holdDisplay = document.getElementById('sim-hold-display');
  const explanationBox = document.getElementById('sim-explanation-box');
  const riskList = document.getElementById('sim-risk-factors');
  const protectiveList = document.getElementById('sim-protective-factors');

  if (runBtn) {
    runBtn.disabled = true;
    runBtn.innerHTML = 'Evaluating Scenario...';
  }

  const payload = {
    segment_id: "sim_scenario",
    slope: parseFloat(document.getElementById('sim-slope-num').value),
    elevation: parseFloat(document.getElementById('sim-elev-num').value),
    distance_to_fire: parseFloat(document.getElementById('sim-dist-num').value),
    barrier_width_m: parseFloat(document.getElementById('sim-barrier-num').value),
    burn_prob: parseFloat(document.getElementById('sim-burn-num').value),
    attack_angle: parseFloat(document.getElementById('sim-angle-num').value),
    attack_dot_product: parseFloat(document.getElementById('sim-dot-num').value)
  };

  try {
    const res = await fetch('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Color code probability
    const bpPct = (data.breach_probability * 100).toFixed(1);
    let color = 'var(--risk-low)';
    if (data.risk_tier === 'CRITICAL') color = 'var(--risk-critical)';
    else if (data.risk_tier === 'HIGH') color = 'var(--risk-high)';
    else if (data.risk_tier === 'MEDIUM') color = 'var(--risk-med)';

    probDisplay.textContent = `${bpPct}%`;
    probDisplay.style.color = color;

    tierBadge.className = `badge badge-${data.risk_tier.toLowerCase()}`;
    tierBadge.textContent = `${data.risk_tier} RISK TIER`;

    holdDisplay.textContent = `Hold Confidence: ${(data.hold_probability * 100).toFixed(1)}%`;

    explanationBox.textContent = data.explanation;

    // Render Factor Badges
    if (riskList) {
      riskList.innerHTML = '';
      (data.top_risk_factors || []).forEach(f => {
        const item = document.createElement('div');
        item.className = 'factor-tag factor-tag-risk';
        item.innerHTML = `⚠️ ${f}`;
        riskList.appendChild(item);
      });
      if ((data.top_risk_factors || []).length === 0) {
        riskList.innerHTML = '<div style="font-size:0.75rem; color:var(--text-dim)">None elevated</div>';
      }
    }

    if (protectiveList) {
      protectiveList.innerHTML = '';
      (data.protective_factors || []).forEach(f => {
        const item = document.createElement('div');
        item.className = 'factor-tag factor-tag-protective';
        item.innerHTML = `🛡️ ${f}`;
        protectiveList.appendChild(item);
      });
      if ((data.protective_factors || []).length === 0) {
        protectiveList.innerHTML = '<div style="font-size:0.75rem; color:var(--text-dim)">None active</div>';
      }
    }

  } catch (err) {
    console.error('Simulation error:', err);
    if (explanationBox) {
      explanationBox.textContent = `Scenario submission failed: ${err.message}`;
      explanationBox.style.color = 'var(--accent-red)';
    }
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.innerHTML = 'Run Benchmark Scenario';
    }
  }
}
