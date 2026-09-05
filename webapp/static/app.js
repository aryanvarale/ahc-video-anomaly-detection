/* SENTINEL — live aerial anomaly detection demo */

const $ = (id) => document.getElementById(id);
const el = {
  drop: $('dropzone'), file: $('fileInput'), browse: $('browseBtn'),
  viewer: $('viewer'), video: $('video'), image: $('image'),
  scan: $('scanline'), live: $('liveBadge'),
  verdict: $('verdictOverlay'), vLabel: $('verdictLabel'), vConf: $('verdictConf'),
  status: $('statusPill'), statusText: $('statusText'), modelName: $('modelName'),
  source: $('sourceName'), list: $('detectionList'), count: $('eventCount'),
  bars: $('confBars'), chunkPos: $('chunkPos'),
  tChunks: $('tChunks'), tP50: $('tP50'), tP95: $('tP95'), tRtf: $('tRtf'),
  bar: $('progressBar'), pct: $('progressPct'), eta: $('progressEta'),
  canvas: $('heatCanvas'), bands: $('eventBands'), playhead: $('playhead'),
  timeLabel: $('timeLabel'), durLabel: $('durLabel'), reset: $('resetBtn'),
  toast: $('toast'),
};

const CLASSES = [
  'normal', 'traffic_accident', 'traffic_congestion', 'stalled_or_broken_down_vehicle',
  'vehicle_blocking_traffic', 'fire', 'smoke', 'waterlogging_or_flood',
  'wrong_way_driving', 'road_spill_or_debris', 'fighting_or_violence',
  'loitering_or_suspicious_presence',
];

const pretty = (c) => c.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
const clock = (s) => {
  if (s == null || !isFinite(s)) return '--:--';
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
};

const state = { duration: 0, total: 0, done: 0, kind: null, times: [], t0: 0, es: null };

/* ── status ─────────────────────────────────────────────────────────── */
function setStatus(mode, text) {
  el.status.className = `pill pill-${mode}`;
  el.statusText.textContent = text;
}

function toast(msg) {
  el.toast.textContent = msg;
  el.toast.classList.remove('hidden');
  requestAnimationFrame(() => el.toast.classList.add('show'));
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.toast.classList.remove('show');
    setTimeout(() => el.toast.classList.add('hidden'), 350);
  }, 4200);
}

async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    el.modelName.textContent = (d.model || '').replace(/^.*\//, '') || 'unknown';
    if (d.model_online) setStatus('online', 'MODEL ONLINE');
    else setStatus('off', 'MODEL OFFLINE');
    return d.model_online;
  } catch {
    setStatus('off', 'NO BACKEND');
    return false;
  }
}

/* ── confidence bars ────────────────────────────────────────────────── */
function initBars() {
  el.bars.innerHTML = '';
  CLASSES.forEach((c) => {
    const row = document.createElement('div');
    row.className = 'conf-row' + (c === 'normal' ? ' normal' : '');
    row.dataset.cls = c;
    row.innerHTML =
      `<span class="conf-label">${pretty(c)}</span>` +
      `<span class="conf-val">0%</span>` +
      `<span class="conf-track"><span class="conf-fill"></span></span>`;
    el.bars.appendChild(row);
  });
}

function paintBars(top) {
  const map = Object.fromEntries(top.map((t) => [t.cls, t.p]));
  const best = top[0]?.cls;
  CLASSES.forEach((c) => {
    const row = el.bars.querySelector(`[data-cls="${c}"]`);
    if (!row) return;
    const p = map[c] || 0;
    row.querySelector('.conf-fill').style.width = `${(p * 100).toFixed(1)}%`;
    row.querySelector('.conf-val').textContent = `${Math.round(p * 100)}%`;
    row.classList.toggle('top', c === best);
    row.classList.toggle('alert', c !== 'normal' && p > 0.5);
  });
}

/* ── timeline heat strip ────────────────────────────────────────────── */
let ctx = null;
function initCanvas() {
  const c = el.canvas, r = c.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  c.width = r.width * dpr; c.height = r.height * dpr;
  ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, r.width, r.height);
  el.bands.innerHTML = '';
}

function paintChunk(idx, pAnom) {
  if (!ctx || !state.total) return;
  const r = el.canvas.getBoundingClientRect();
  const w = Math.max(r.width / state.total, 1.2);
  const x = (idx / state.total) * r.width;
  const h = Math.max(pAnom * r.height, 1.5);
  // cool → hot as anomaly probability rises
  const hue = 190 - pAnom * 190;
  ctx.fillStyle = `hsla(${hue},92%,${52 + pAnom * 8}%,${0.30 + pAnom * 0.62})`;
  ctx.fillRect(x, r.height - h, w + 0.6, h);
}

function drawBands(events) {
  el.bands.innerHTML = '';
  if (!state.duration) return;
  events.forEach((ev) => {
    if (ev.start_time_sec == null) return;
    const b = document.createElement('div');
    b.className = 'event-band';
    b.style.left = `${(ev.start_time_sec / state.duration) * 100}%`;
    b.style.width = `${Math.max(((ev.end_time_sec - ev.start_time_sec) / state.duration) * 100, 0.7)}%`;
    b.title = `${pretty(ev.class_name)} · ${clock(ev.start_time_sec)}–${clock(ev.end_time_sec)}`;
    el.bands.appendChild(b);
  });
}

/* ── detections ─────────────────────────────────────────────────────── */
function renderEvents(events) {
  el.count.textContent = events.length;
  el.count.classList.add('bump');
  setTimeout(() => el.count.classList.remove('bump'), 320);

  if (!events.length) {
    el.list.innerHTML =
      `<div class="empty-state"><div class="empty-ring" style="animation:none;border-top-color:rgba(60,232,164,.7)"></div>` +
      `<p style="color:var(--green)">No anomalies detected</p>` +
      `<span class="mono muted">scene classified as normal throughout</span></div>`;
    return;
  }

  el.list.innerHTML = '';
  events.forEach((ev, i) => {
    const card = document.createElement('div');
    card.className = 'detection';
    card.style.animationDelay = `${i * 55}ms`;
    const t = ev.start_time_sec == null
      ? 'whole clip'
      : `${clock(ev.start_time_sec)}–${clock(ev.end_time_sec)}`;
    const conf = Math.round((ev.confidence || 0) * 100);
    card.innerHTML =
      `<img class="det-thumb" src="${ev._thumb || ''}" alt="" onerror="this.style.visibility='hidden'">` +
      `<div class="det-body">` +
        `<div class="det-top"><span class="det-class">${pretty(ev.class_name)}</span>` +
        `<span class="det-time">${t}</span></div>` +
        `<div class="det-expl">${ev.explanation || ''}</div>` +
        `<div class="det-conf"><span class="det-conf-track"><span class="det-conf-fill"></span></span>` +
        `<span>${conf}%</span></div>` +
      `</div>`;
    el.list.appendChild(card);
    requestAnimationFrame(() => {
      card.querySelector('.det-conf-fill').style.width = `${conf}%`;
    });
    if (ev.start_time_sec != null) {
      card.style.cursor = 'pointer';
      card.onclick = () => { el.video.currentTime = ev.start_time_sec; el.video.play(); };
    }
  });
}

/* ── analysis run ───────────────────────────────────────────────────── */
async function analyse(file) {
  if (!file) return;

  el.drop.classList.add('hidden');
  el.viewer.classList.remove('hidden');
  el.source.textContent = file.name;
  el.list.innerHTML = '';
  el.count.textContent = '0';
  el.reset.classList.add('hidden');
  el.verdict.classList.add('hidden');
  Object.assign(state, { duration: 0, total: 0, done: 0, times: [], t0: performance.now() });
  initBars();

  const isImage = /^image\//.test(file.type);
  state.kind = isImage ? 'image' : 'video';
  const url = URL.createObjectURL(file);
  if (isImage) {
    el.image.src = url; el.image.classList.remove('hidden'); el.video.classList.add('hidden');
  } else {
    el.video.src = url; el.video.classList.remove('hidden'); el.image.classList.add('hidden');
    el.video.onloadedmetadata = () => { el.durLabel.textContent = clock(el.video.duration); };
  }

  setStatus('busy', 'UPLOADING');
  let job;
  try {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || 'upload failed');
    job = await r.json();
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
    setStatus('off', 'ERROR');
    resetUI();
    return;
  }

  setStatus('busy', 'ANALYSING');
  el.scan.classList.remove('hidden');
  el.live.classList.remove('hidden');
  requestAnimationFrame(initCanvas);

  const es = new EventSource(`/api/stream/${job.job_id}`);
  state.es = es;
  let lastEvents = [], firstThumb = null;

  es.onmessage = (msg) => {
    const d = JSON.parse(msg.data);

    if (d.type === 'meta') {
      state.duration = d.duration || 0;
      state.total = d.chunks || 1;
      el.tChunks.innerHTML = `0<i>/${state.total}</i>`;
      el.durLabel.textContent = d.duration ? clock(d.duration) : '—';
      requestAnimationFrame(initCanvas);
    }

    else if (d.type === 'chunk') {
      state.done++;
      state.times.push(d.ms);
      paintChunk(d.idx, d.p_anom);
      paintBars(d.top);

      if (d.thumb_b64) firstThumb = `data:image/jpeg;base64,${d.thumb_b64}`;
      else if (d.thumb) firstThumb = `data:image/jpeg;base64,${d.thumb}`;

      // playhead sweep
      if (state.duration) {
        el.playhead.classList.add('on');
        el.playhead.style.left = `${(d.end / state.duration) * 100}%`;
        el.timeLabel.textContent = `${d.end.toFixed(1)}s`;
      }
      el.chunkPos.textContent = state.duration
        ? `${d.start.toFixed(1)}–${d.end.toFixed(1)}s` : 'frame';

      // running telemetry
      const pct = Math.round((state.done / state.total) * 100);
      el.bar.style.width = `${pct}%`;
      el.pct.textContent = `${pct}%`;
      el.tChunks.innerHTML = `${state.done}<i>/${state.total}</i>`;
      const sorted = [...state.times].sort((a, b) => a - b);
      el.tP50.innerHTML = `${Math.round(sorted[Math.floor(sorted.length * 0.5)])}<i>ms</i>`;
      el.tP95.innerHTML = `${Math.round(sorted[Math.floor(sorted.length * 0.95)] || sorted.at(-1))}<i>ms</i>`;
      const elapsed = (performance.now() - state.t0) / 1000;
      const rate = state.done / Math.max(elapsed, 0.001);
      const left = (state.total - state.done) / Math.max(rate, 0.001);
      el.eta.textContent = state.done < state.total ? `~${Math.ceil(left)}s left` : '';

      // top verdict overlay
      const best = d.top[0];
      if (best) {
        el.verdict.classList.remove('hidden');
        el.vLabel.textContent = pretty(best.cls);
        el.vConf.textContent = `${Math.round(best.p * 100)}%`;
        el.verdict.classList.toggle('alert', best.cls !== 'normal' && best.p > 0.6);
        el.verdict.classList.toggle('clear', best.cls === 'normal');
      }
    }

    else if (d.type === 'events') {
      lastEvents = d.events.map((e) => ({ ...e, _thumb: firstThumb }));
      drawBands(lastEvents);
      renderEvents(lastEvents);
    }

    else if (d.type === 'done') {
      const rt = d.runtime || {};
      if (rt.p50_ms != null) el.tP50.innerHTML = `${Math.round(rt.p50_ms)}<i>ms</i>`;
      if (rt.p95_ms != null) el.tP95.innerHTML = `${Math.round(rt.p95_ms)}<i>ms</i>`;
      el.tRtf.textContent = rt.rtf != null ? `${rt.rtf}×` : '—';
      el.bar.style.width = '100%';
      el.pct.textContent = '100%';
      el.eta.textContent = `${(rt.total_ms / 1000).toFixed(1)}s total`;
      finish(lastEvents.length);
    }

    else if (d.type === 'error') {
      toast(d.message);
      setStatus('off', 'ERROR');
      finish(null, true);
    }
  };

  es.addEventListener('eof', () => es.close());
  es.onerror = () => { es.close(); };
}

function finish(nEvents, failed) {
  state.es?.close();
  el.scan.classList.add('hidden');
  el.live.classList.add('hidden');
  el.playhead.classList.remove('on');
  el.reset.classList.remove('hidden');
  if (!failed) setStatus(nEvents ? 'busy' : 'online', nEvents ? `${nEvents} DETECTION${nEvents > 1 ? 'S' : ''}` : 'CLEAR');
  if (state.kind === 'video') el.video.play().catch(() => {});
}

function resetUI() {
  state.es?.close();
  el.viewer.classList.add('hidden');
  el.drop.classList.remove('hidden');
  el.scan.classList.add('hidden');
  el.live.classList.add('hidden');
  el.verdict.classList.add('hidden');
  el.reset.classList.add('hidden');
  el.video.pause(); el.video.removeAttribute('src'); el.image.removeAttribute('src');
  el.source.textContent = 'no source';
  el.bands.innerHTML = ''; el.bar.style.width = '0'; el.pct.textContent = '0%';
  el.eta.textContent = ''; el.count.textContent = '0';
  el.tChunks.innerHTML = '0<i>/0</i>'; el.tP50.innerHTML = '—<i>ms</i>';
  el.tP95.innerHTML = '—<i>ms</i>'; el.tRtf.textContent = '—';
  el.chunkPos.textContent = '—'; el.timeLabel.textContent = '0.0s'; el.durLabel.textContent = '--:--';
  el.list.innerHTML =
    `<div class="empty-state"><div class="empty-ring"></div><p>Awaiting footage</p>` +
    `<span class="mono muted">events will appear here as they are found</span></div>`;
  initBars();
  if (ctx) initCanvas();
  checkHealth();
}

/* ── video playhead while playing ───────────────────────────────────── */
el.video.addEventListener('timeupdate', () => {
  if (!state.duration || el.video.paused) return;
  el.playhead.classList.add('on');
  el.playhead.style.left = `${(el.video.currentTime / state.duration) * 100}%`;
  el.timeLabel.textContent = `${el.video.currentTime.toFixed(1)}s`;
});

/* ── input wiring ───────────────────────────────────────────────────── */
el.browse.onclick = () => el.file.click();
el.drop.onclick = (e) => { if (e.target === el.drop || e.target.closest('.dz-inner') === e.target) el.file.click(); };
el.file.onchange = (e) => analyse(e.target.files[0]);
el.reset.onclick = resetUI;

['dragenter', 'dragover'].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); if (ev === 'drop' || e.target === el.drop) el.drop.classList.remove('drag'); }));
el.drop.addEventListener('drop', (e) => analyse(e.dataTransfer.files[0]));
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());
addEventListener('resize', () => { if (state.total) initCanvas(); });

initBars();
checkHealth();
setInterval(() => { if (!state.es || state.es.readyState === 2) checkHealth(); }, 15000);
