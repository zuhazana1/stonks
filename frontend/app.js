/* ── Stonks Frontend ── */

const API = '';          // same origin
let charts = {};         // { main, rsi, macd }
let insightSrc = null;
let summarySrc  = null;
let pollTimer   = null;
let fakeProgress = 0;

// ── Initialise ──────────────────────────────────────────────────────────────

function init() {
  startPolling();
}

function startPolling() {
  pollTimer = setInterval(async () => {
    try {
      const data = await fetchJSON('/api/status');
      document.getElementById('loader-msg').textContent = data.progress_msg || 'Loading…';

      if (data.status === 'loading') {
        fakeProgress = Math.min(fakeProgress + Math.random() * 2.5, 87);
        setProgress(fakeProgress);
      } else if (data.status === 'ready') {
        setProgress(100);
        document.getElementById('loader-msg').textContent = 'Ready!';
        clearInterval(pollTimer);
        setTimeout(() => launchApp(data), 700);
      } else if (data.status === 'error') {
        document.getElementById('loader-msg').textContent =
          '⚠ Error loading data — check the server console.';
        clearInterval(pollTimer);
      }
    } catch (_) { /* server still starting */ }
  }, 2000);
}

function setProgress(pct) {
  document.getElementById('loader-fill').style.width = pct + '%';
}

async function launchApp(status) {
  // Fade out loader, show app
  document.getElementById('loading-screen').classList.add('out');
  const app = document.getElementById('app');
  app.style.display = 'flex';
  setTimeout(() => { document.getElementById('loading-screen').style.display = 'none'; }, 520);

  const ts = new Date(status.timestamp * 1000);
  document.getElementById('last-updated').textContent =
    'Data as of ' + ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const stocks = await fetchJSON('/api/top-performers');
  renderList(stocks);
  streamSummary();

  // Auto-select first stock
  if (stocks.length > 0) selectStock(stocks[0].ticker, stocks[0]);
}

// ── Stock List ───────────────────────────────────────────────────────────────

function renderList(stocks) {
  const el = document.getElementById('stock-list');
  el.innerHTML = '';
  stocks.forEach((s, i) => {
    const item = document.createElement('div');
    item.className = 'si';
    item.dataset.ticker = s.ticker;

    const chgCls  = s.change_1d >= 0 ? 'pos' : 'neg';
    const chgSign = s.change_1d >= 0 ? '+' : '';
    const mlCls   = s.predicted_return_5d >= 0 ? 'pos' : 'neg';
    const mlSign  = s.predicted_return_5d >= 0 ? '+' : '';

    item.innerHTML = `
      <div class="si-l">
        <span class="si-rank">#${i + 1}</span>
        <span class="si-ticker">${s.ticker}</span>
        <span class="si-price">$${s.current_price.toFixed(2)}</span>
      </div>
      <div class="si-r">
        <div class="si-chg ${chgCls}">${chgSign}${s.change_1d.toFixed(2)}%</div>
        <div class="si-ml ${mlCls}">ML: ${mlSign}${s.predicted_return_5d.toFixed(1)}%</div>
      </div>`;

    item.addEventListener('click', () => selectStock(s.ticker, s));
    el.appendChild(item);
  });
}

// ── Select Stock ─────────────────────────────────────────────────────────────

async function selectStock(ticker, metrics) {
  // Highlight in list
  document.querySelectorAll('.si').forEach(el => {
    el.classList.toggle('active', el.dataset.ticker === ticker);
  });

  // Header
  const hdr = document.getElementById('stock-hdr');
  hdr.classList.remove('hidden');

  document.getElementById('sh-ticker').textContent = ticker;
  document.getElementById('sh-price').textContent  = '$' + metrics.current_price.toFixed(2);

  const chgEl = document.getElementById('sh-change');
  const sign  = metrics.change_1d >= 0 ? '+' : '';
  chgEl.textContent = sign + metrics.change_1d.toFixed(2) + '% today';
  chgEl.className   = metrics.change_1d >= 0 ? 'pos' : 'neg';

  const m3el   = document.getElementById('sh-3mo');
  const m3sign = metrics.change_3mo >= 0 ? '+' : '';
  m3el.textContent = m3sign + metrics.change_3mo.toFixed(2) + '%';
  m3el.className   = metrics.change_3mo >= 0 ? 'pos' : 'neg';

  // Show chart area, hide placeholder
  document.getElementById('chart-area').classList.remove('hidden');
  document.getElementById('placeholder').style.display = 'none';

  renderStats(metrics);
  streamInsight(ticker);

  const [ohlcv, indicators] = await Promise.all([
    fetchJSON(`/api/stock/${ticker}/ohlcv`),
    fetchJSON(`/api/stock/${ticker}/indicators`),
  ]);
  renderCharts(ohlcv, indicators);
}

// ── Key Stats ─────────────────────────────────────────────────────────────────

function renderStats(m) {
  const rsiLabel = m.rsi > 70 ? 'Overbought' : m.rsi < 30 ? 'Oversold' : 'Neutral';
  const macdDir  = m.macd > m.macd_signal ? '↑ Bullish' : '↓ Bearish';
  const bbLabel  = m.bb_pct > 0.8 ? 'Near upper' : m.bb_pct < 0.2 ? 'Near lower' : 'Mid-range';
  const volLabel = m.volume_ratio > 1.5 ? 'High ↑' : m.volume_ratio < 0.7 ? 'Low ↓' : 'Normal';

  const rows = [
    ['1D Return',       pct(m.change_1d)],
    ['5D Return',       pct(m.change_5d)],
    ['1M Return',       pct(m.change_1mo)],
    ['3M Return',       pct(m.change_3mo)],
    ['RSI (14)',        `${m.rsi.toFixed(1)} · ${rsiLabel}`],
    ['MACD',            macdDir],
    ['BB Position',     `${(m.bb_pct * 100).toFixed(0)}% · ${bbLabel}`],
    ['Volume vs Avg',   `${m.volume_ratio.toFixed(2)}× · ${volLabel}`],
    ['ML Forecast 5D',  pct(m.predicted_return_5d)],
    ['Score',           m.composite_score.toFixed(2)],
  ];

  document.getElementById('key-stats').innerHTML = rows.map(([lbl, val]) => {
    const cls = val.startsWith('+') ? 'pos' : val.startsWith('-') ? 'neg' : '';
    return `<div class="stat-row">
      <span class="stat-lbl">${lbl}</span>
      <span class="stat-val ${cls}">${val}</span>
    </div>`;
  }).join('');
}

function pct(v) { return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'; }

// ── Charts ───────────────────────────────────────────────────────────────────

function renderCharts(ohlcv, ind) {
  Object.values(charts).forEach(c => c.remove());
  charts = {};
  renderMain(ohlcv, ind);
  renderRSI(ind.rsi);
  renderMACD(ind.macd, ind.macd_signal);
  syncTimeScales();
}

function baseOpts(el) {
  return {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
    grid:   { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#30363d', scaleMargins: { top: 0.08, bottom: 0.08 } },
    timeScale:       { borderColor: '#30363d', timeVisible: true },
    width:  el.clientWidth,
    height: el.clientHeight,
  };
}

function renderMain(ohlcv, ind) {
  const el = document.getElementById('c-main');
  el.innerHTML = '';
  const chart = LightweightCharts.createChart(el, baseOpts(el));
  charts.main = chart;

  // Candlestick
  const candles = chart.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149',
    borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor: '#3fb950',   wickDownColor: '#f85149',
  });
  candles.setData(ohlcv.map(d => ({
    time: d.time, open: d.open, high: d.high, low: d.low, close: d.close,
  })));

  // Volume (secondary scale)
  const vol = chart.addHistogramSeries({
    priceFormat: { type: 'volume' }, priceScaleId: 'vol',
  });
  chart.priceScale('vol').applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 }, visible: false,
  });
  vol.setData(ohlcv.map(d => ({
    time: d.time, value: d.volume,
    color: d.close >= d.open ? 'rgba(63,185,80,.22)' : 'rgba(248,81,73,.22)',
  })));

  // Bollinger Bands
  if (ind.bb_upper?.length) {
    const bbStyle = { lineWidth: 1, lineStyle: 2, color: 'rgba(88,166,255,.45)' };
    chart.addLineSeries({ ...bbStyle }).setData(ind.bb_upper);
    chart.addLineSeries({ ...bbStyle }).setData(ind.bb_lower);
  }

  chart.timeScale().fitContent();
  new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth })).observe(el);
}

function renderRSI(rsiData) {
  const el = document.getElementById('c-rsi');
  el.innerHTML = '';
  const chart = LightweightCharts.createChart(el, baseOpts(el));
  charts.rsi = chart;

  chart.addLineSeries({ color: '#e3b341', lineWidth: 1.5 }).setData(rsiData);

  // Reference lines 70 / 30
  if (rsiData.length) {
    chart.addLineSeries({ color: 'rgba(248,81,73,.4)',  lineWidth: 1, lineStyle: 3 })
         .setData(rsiData.map(d => ({ time: d.time, value: 70 })));
    chart.addLineSeries({ color: 'rgba(63,185,80,.4)',  lineWidth: 1, lineStyle: 3 })
         .setData(rsiData.map(d => ({ time: d.time, value: 30 })));
  }

  chart.timeScale().fitContent();
  new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth })).observe(el);
}

function renderMACD(macdData, signalData) {
  const el = document.getElementById('c-macd');
  el.innerHTML = '';
  const chart = LightweightCharts.createChart(el, baseOpts(el));
  charts.macd = chart;

  chart.addLineSeries({ color: '#58a6ff', lineWidth: 1.5 }).setData(macdData);
  chart.addLineSeries({ color: '#f97583', lineWidth: 1.5 }).setData(signalData);

  // Histogram
  if (macdData.length && signalData.length) {
    const macdMap = {};
    macdData.forEach(d => { macdMap[d.time] = d.value; });
    const hist = signalData
      .filter(d => macdMap[d.time] != null)
      .map(d => {
        const v = macdMap[d.time] - d.value;
        return { time: d.time, value: v,
                 color: v >= 0 ? 'rgba(63,185,80,.5)' : 'rgba(248,81,73,.5)' };
      });
    chart.addHistogramSeries({ priceScaleId: 'right' }).setData(hist);
  }

  chart.timeScale().fitContent();
  new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth })).observe(el);
}

// Synchronise crosshair + scroll across all three charts
function syncTimeScales() {
  const all = [charts.main, charts.rsi, charts.macd].filter(Boolean);

  all.forEach((src, si) => {
    src.subscribeCrosshairMove(param => {
      all.forEach((tgt, ti) => {
        if (ti === si || !param.time) return;
        tgt.setCrosshairPosition(0, param.time, tgt.series?.[0]);
      });
    });
    src.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (!range) return;
      all.forEach((tgt, ti) => {
        if (ti === si) return;
        tgt.timeScale().setVisibleLogicalRange(range);
      });
    });
  });
}

// ── AI Streaming ──────────────────────────────────────────────────────────────

function streamInsight(ticker) {
  if (insightSrc) { insightSrc.close(); insightSrc = null; }

  const panel = document.getElementById('ai-panel');
  const span  = document.createElement('span');
  span.className = 'cursor';
  panel.innerHTML = '';
  panel.appendChild(span);

  let text = '';
  const src = new EventSource(`${API}/api/insight/${ticker}`);
  insightSrc = src;

  src.onmessage = e => {
    if (e.data === '[DONE]') { src.close(); span.classList.remove('cursor'); return; }
    try {
      const { text: chunk } = JSON.parse(e.data);
      text += chunk;
      span.textContent = text;
    } catch (_) {}
  };
  src.onerror = () => {
    src.close();
    if (!text) panel.innerHTML = '<p class="muted">AI insight unavailable — check ANTHROPIC_API_KEY</p>';
  };
}

function streamSummary() {
  if (summarySrc) { summarySrc.close(); summarySrc = null; }

  const el = document.getElementById('banner-text');
  el.textContent = '';
  let text = '';

  const src = new EventSource(`${API}/api/market-summary/stream`);
  summarySrc = src;

  src.onmessage = e => {
    if (e.data === '[DONE]') { src.close(); return; }
    try {
      const { text: chunk } = JSON.parse(e.data);
      text += chunk;
      el.textContent = text;
    } catch (_) {}
  };
  src.onerror = () => {
    src.close();
    if (!text) el.textContent = 'Market summary unavailable.';
  };
}

// ── Utilities ─────────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
