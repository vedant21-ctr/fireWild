/**
 * format.js — Display helpers for numeric values that may be missing or invalid.
 *
 * Loaded before app.js. Missing, null, empty, non-numeric, NaN and infinite values are
 * shown as a styled "Unavailable" label instead of "NaN", "Infinity" or a misleading 0.
 * Also exported for Node so tests/test_clbi_measurements.py can exercise it.
 */

const UNAVAILABLE_LABEL = 'Unavailable';
const UNAVAILABLE_HTML = `<span class="value-unavailable" title="Measurement not provided">${UNAVAILABLE_LABEL}</span>`;

/** Finite number, or null for missing / null / empty / boolean / non-numeric / non-finite input. */
function toFiniteNumber(value) {
  if (value === null || value === undefined || typeof value === 'boolean') return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

/** "6.5 m"-style text for a valid measurement, the unavailable label otherwise. */
function formatMeasurement(value, decimals, unit) {
  const n = toFiniteNumber(value);
  return n === null ? UNAVAILABLE_HTML : `${n.toFixed(decimals)}${unit}`;
}

/** Probability in [0, 1] as a percentage, the unavailable label otherwise. */
function formatProbability(value, decimals) {
  const n = toFiniteNumber(value);
  return n === null || n < 0 || n > 1 ? UNAVAILABLE_HTML : `${(n * 100).toFixed(decimals)}%`;
}

function isMissingValue(value) {
  if (typeof value === 'string') return value.trim() === '';
  return toFiniteNumber(value) === null;
}

/** Sort comparator (dir = 1 ascending, -1 descending); missing values always sort last. */
function compareValues(a, b, dir) {
  const missingA = isMissingValue(a);
  const missingB = isMissingValue(b);
  if (missingA || missingB) return missingA === missingB ? 0 : (missingA ? 1 : -1);
  if (typeof a === 'string') a = a.toLowerCase();
  if (typeof b === 'string') b = b.toLowerCase();
  if (a < b) return -dir;
  if (a > b) return dir;
  return 0;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { UNAVAILABLE_LABEL, toFiniteNumber, formatMeasurement, formatProbability, compareValues };
}
