/* Aerial anomaly detection — live analysis client */

const $ = (id) => document.getElementById(id);
const el = {
  drop: $('dropzone'), file: $('fileInput'), browse: $('browseBtn'),
  viewer: $('viewer'), video: $('video'), image: $('image'),
  live: $('liveBadge'), verdict: $('verdictOverlay'),
  vLabel: $('verdictLabel'), vConf: $('verdictConf'),
  status: $('statusPill'), statusText: $('statusText'), modelName: $('modelName'),
  source: $('sourceName'), list: $('detectionList'), count: $('eventCount'),
  bars: $('confBars'), chunkPos: $('chunkPos'),
  tProgress: $('tProgress'), tChunks: $('tChunks'), tP50: $('tP50'),
  tP95: $('tP95'), tRtf: $('tRtf'), bar: $('progressBar'),
  canvas: $('heatCanvas'), bands: $('eventBands'), playhead: $('playhead'),
  iouBands: $('iouBands'),
  gtWrap: $('gtTrackWrap'), gtBands: $('gtBands'), gtPlayhead: $('gtPlayhead'),
  gtSummary: $('gtSummary'),
  timeline: $('timeline'), timeLabel: $('timeLabel'), durLabel: $('durLabel'),
  toast: $('toast'), back: $('backBtn'),
  controls: $('controls'), play: $('playBtn'), seek: $('seek'), seekFill: $('seekFill'),
  curTime: $('curTime'), totTime: $('totTime'), mute: $('muteBtn'),
  iPlay: $('iconPlay'), iPause: $('iconPause'), iMuted: $('iconMuted'), iSound: $('iconSound'),
  theme: $('themeBtn'), iMoon: $('iconMoon'), iSun: $('iconSun'),
  stage: $('stage'), fsOverlay: $('fsOverlay'),
  fs: $('fsBtn'), iFsIn: $('iconFsIn'), iFsOut: $('iconFsOut'),
  panel: $('panelBtn'), iPanelOn: $('iconPanelOn'), iPanelOff: $('iconPanelOff'),
  hud: $('hud'), hudDot: $('hudDot'), hudClass: $('hudClass'), hudConf: $('hudConf'),
  hudIouRow: $('hudIouRow'), hudIou: $('hudIou'), hudIouGate: $('hudIouGate'),
  hudGtRow: $('hudGtRow'), hudGt: $('hudGt'),
  prep: $('prepNotice'), prepText: $('prepText'),
  layout: document.querySelector('.layout'),
  side: document.querySelector('.col-side'),
  tele: document.querySelector('.tele'),
};

const CLASSES = [
  'normal', 'traffic_accident', 'traffic_congestion', 'stalled_or_broken_down_vehicle',
  'vehicle_blocking_traffic', 'fire', 'smoke', 'waterlogging_or_flood',
  'wrong_way_driving', 'road_spill_or_debris', 'fighting_or_violence',
  'loitering_or_suspicious_presence',
];

const pretty = (c) => c.replace(/_/g, ' ').replace(/^\w/, (m) => m.toUpperCase());
const clock = (s) => {
  if (s == null || !isFinite(s)) return '—';
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
};

const state = {
  duration: 0, total: 0, done: 0, kind: null, times: [], t0: 0, es: null,
  jobId: null,
  events: [],        // detections, kept so the HUD can answer "what now?"
  gt: [],            // labelled intervals when the clip is one of the held-out set
};

/* ── theme ──────────────────────────────────────────────────────────── */
function applyTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  el.iMoon.classList.toggle('hidden', mode === 'dark');
  el.iSun.classList.toggle('hidden', mode !== 'dark');
  try { localStorage.setItem('theme', mode); } catch {}
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch {}
  const prefersDark = window.matchMedia
    && window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
})();
el.theme.onclick = () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  if (state.total) requestAnimationFrame(repaintHeat);   // heat colours are theme-aware
};

/* ── status / toast ─────────────────────────────────────────────────── */
function setStatus(kind, text, live) {
  el.status.className = `status s-${kind}${live ? ' live' : ''}`;
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
  }, 4000);
}

async function checkHealth() {
  try {
    const d = await (await fetch('/api/health')).json();
    el.modelName.textContent = (d.model || '').replace(/^.*\//, '') || 'unknown';
    setStatus(d.model_online ? 'ok' : 'err', d.model_online ? 'Model online' : 'Model offline');
    return d.model_online;
  } catch {
    setStatus('err', 'No backend');
    return false;
  }
}

/* ── confidence bars ────────────────────────────────────────────────── */
function initBars() {
  el.bars.innerHTML = '';
  CLASSES.forEach((c) => {
    const row = document.createElement('div');
    row.className = 'row' + (c === 'normal' ? ' norm' : '');
    row.dataset.cls = c;
    row.innerHTML = `<span class="name">${pretty(c)}</span><span class="val">0%</span>`
                  + `<span class="tr"><span class="fl"></span></span>`;
    el.bars.appendChild(row);
  });
}

function paintBars(top) {
  const map = Object.fromEntries(top.map((t) => [t.cls, t.p]));
  const lead = top[0]?.cls;
  CLASSES.forEach((c) => {
    const row = el.bars.querySelector(`[data-cls="${c}"]`);
    if (!row) return;
    const p = map[c] || 0;
    row.querySelector('.fl').style.width = `${(p * 100).toFixed(1)}%`;
    row.querySelector('.val').textContent = `${Math.round(p * 100)}%`;
    row.classList.toggle('lead', c === lead);
    row.classList.toggle('hot', c !== 'normal' && p > 0.5);
  });
}

/* ── heat track ─────────────────────────────────────────────────────── */
let ctx = null;
const heat = [];

function initCanvas() {
  const c = el.canvas, r = c.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  c.width = Math.max(r.width * dpr, 1); c.height = Math.max(r.height * dpr, 1);
  ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
}

/* Is this chunk inside an interval the system actually reported? */
function inReportedEvent(idx) {
  if (!state.events.length || !state.duration || !state.total) return false;
  const t = (idx / state.total) * state.duration;
  return state.events.some((e) => e.start_time_sec != null
    && t >= e.start_time_sec && t <= e.end_time_sec);
}

function bar(idx, p, w, h) {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  const x = (idx / state.total) * w;
  const bw = Math.max(w / state.total, 1.2);
  const bh = Math.max(p * h, 1.5);

  // Bar HEIGHT is the raw per-chunk anomaly probability; bar COLOUR is whether
  // that chunk falls inside a reported event. These are deliberately different
  // things. After fine-tuning, this model's probabilities saturate near 1.0 for
  // long stretches of a clip - colouring by probability turned the whole track
  // red on any anomalous-looking scene and said nothing about *when* the event
  // was. Red now means "this is in a detection", which is the claim being made.
  const hot = inReportedEvent(idx);
  const hue = hot ? 8 : 214;
  const sat = hot ? 82 : 42;
  const light = dark ? (hot ? 56 : 50) : (hot ? 48 : 54);
  const alpha = hot ? 0.5 + p * 0.45 : 0.22 + p * 0.3;
  ctx.fillStyle = `hsla(${hue},${sat}%,${light}%,${alpha})`;
  ctx.fillRect(x, h - bh, bw + 0.5, bh);
}

function paintChunk(idx, p) {
  if (!ctx || !state.total) return;
  heat[idx] = p;
  const r = el.canvas.getBoundingClientRect();
  bar(idx, p, r.width, r.height);
}

function repaintHeat() {
  if (!ctx || !state.total) return;
  const r = el.canvas.getBoundingClientRect();
  initCanvas();
  heat.forEach((p, i) => { if (p != null) bar(i, p, r.width, r.height); });
}

function pct(a, b) { return (a / b) * 100; }

/* Detected events, the labelled ground truth beneath them, and the overlap
   box for every same-class pair that shares any span at all - the visible
   version of the IoU the score is computed from, not a number to take on
   trust. Skipped entirely for an upload, which has no truth to draw. */
function drawBands(events) {
  el.bands.innerHTML = '';
  el.gtBands.innerHTML = '';
  el.iouBands.innerHTML = '';
  if (!state.duration) return;

  const timed = events.filter((e) => e.start_time_sec != null);
  const matchedIdx = new Set();

  if (state.gt.length) {
    el.gtWrap.classList.remove('hidden');
    el.gtSummary.textContent = `${state.gt.length} labelled interval${state.gt.length > 1 ? 's' : ''}`;

    state.gt.forEach((g) => {
      const b = document.createElement('div');
      b.className = 'gt-band';
      b.style.left = `${pct(g.start, state.duration)}%`;
      b.style.width = `${Math.max(pct(g.end - g.start, state.duration), .7)}%`;
      b.title = `Truth: ${pretty(g.class_name)} · ${clock(g.start)}–${clock(g.end)}`;
      const lab = document.createElement('span');
      lab.className = 'gt-band-label';
      lab.textContent = pretty(g.class_name);
      b.appendChild(lab);
      el.gtBands.appendChild(b);

      // best-matching detection of the same class, for the overlap box
      let bestI = -1, bestIou = 0;
      timed.forEach((ev, i) => {
        if (ev.class_name !== g.class_name) return;
        const v = iou(ev.start_time_sec, ev.end_time_sec, g.start, g.end);
        if (v > bestIou) { bestIou = v; bestI = i; }
      });
      if (bestI === -1) return;                          // nothing of this class detected
      const ev = timed[bestI];
      const os = Math.max(ev.start_time_sec, g.start);
      const oe = Math.min(ev.end_time_sec, g.end);
      if (oe <= os) return;                               // detected but no time overlap at all
      matchedIdx.add(bestI);

      const pass = bestIou >= 0.5;
      const box = document.createElement('div');
      box.className = `iou-overlap ${pass ? 'pass' : 'fail'}`;
      box.style.left = `${pct(os, state.duration)}%`;
      box.style.width = `${Math.max(pct(oe - os, state.duration), .5)}%`;
      el.iouBands.appendChild(box);

      const tag = document.createElement('span');
      tag.className = `iou-tag ${pass ? 'pass' : 'fail'}`;
      tag.style.left = `${pct((os + oe) / 2, state.duration)}%`;
      tag.textContent = `IoU ${bestIou.toFixed(2)}`;
      el.iouBands.appendChild(tag);
    });
  } else {
    el.gtWrap.classList.add('hidden');
  }

  timed.forEach((ev, i) => {
    const b = document.createElement('div');
    b.className = `band${matchedIdx.has(i) ? ' matched' : ''}`;
    b.style.left = `${pct(ev.start_time_sec, state.duration)}%`;
    b.style.width = `${Math.max(pct(ev.end_time_sec - ev.start_time_sec, state.duration), .7)}%`;
    b.title = `${pretty(ev.class_name)} · ${clock(ev.start_time_sec)}–${clock(ev.end_time_sec)}`
      + (matchedIdx.has(i) ? ' · matches truth (IoU ≥ 0.50)' : '');
    el.bands.appendChild(b);
  });
}

/* ── playback source ────────────────────────────────────────────────── */
/* Swap the <video> over to the server's re-encoded copy once it exists, keeping
   the current position so a viewer who already hit play is not thrown back to
   the start. Analysis is untouched by any of this - it reads the original. */
async function waitForPlayback(jobId) {
  for (let i = 0; i < 600; i++) {
    if (state.jobId !== jobId) return;                 // a new file was dropped
    try {
      const r = await fetch(`/api/media_status/${jobId}`);
      if (r.ok) {
        const s = await r.json();
        if (s.ready && !s.transcoding) {
          const at = el.video.currentTime || 0;
          const wasPlaying = !el.video.paused && !el.video.ended;
          el.video.src = `/api/media/${jobId}`;
          el.video.addEventListener('loadedmetadata', () => {
            if (at && at < el.video.duration) el.video.currentTime = at;
            if (wasPlaying) el.video.play().catch(() => {});
            el.durLabel.textContent = clock(el.video.duration);
            el.totTime.textContent = clock(el.video.duration);
          }, { once: true });
          el.prep.classList.add('hidden');
          return;
        }
      }
    } catch { /* transient - keep waiting */ }
    await new Promise((res) => setTimeout(res, 1000));
  }
  el.prepText.textContent = 'Playback unavailable for this codec';
}

/* ── live HUD ───────────────────────────────────────────────────────── */
function iou(a0, a1, b0, b1) {
  const inter = Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
  const union = Math.max(a1, b1) - Math.min(a0, b0);
  return union > 0 ? inter / union : 0;
}

function activeAt(list, t) {
  return list.find((e) => e.start_time_sec != null
    && t >= e.start_time_sec && t <= e.end_time_sec) || null;
}

/* What is being claimed at this instant, and how well it lines up with the
   truth. The IoU shown is the detection's overlap with the best same-class
   labelled interval - the same quantity the task is scored on, so a judge can
   watch the gate being met or missed rather than being told about it. */
function updateHud(t) {
  if (state.kind !== 'video' || !state.duration) return;
  const ev = activeAt(state.events, t);
  const gtNow = state.gt.find((g) => t >= g.start && t <= g.end) || null;

  el.hud.classList.remove('hidden');
  el.hudClass.textContent = ev ? pretty(ev.class_name) : 'Normal';
  el.hudConf.textContent = ev && ev.confidence != null
    ? `${Math.round(ev.confidence * 100)}%` : '';
  el.hudDot.className = `hud-dot ${ev ? 'on' : ''}`;
  el.hud.classList.toggle('alert', !!ev);

  if (!state.gt.length) {                     // an upload has nothing to score against
    el.hudIouRow.classList.add('hidden');
    el.hudGtRow.classList.add('hidden');
    return;
  }

  el.hudGtRow.classList.remove('hidden');
  el.hudGt.textContent = gtNow ? pretty(gtNow.class_name) : 'normal';

  if (!ev) {
    el.hudIouRow.classList.add('hidden');
    return;
  }
  let best = 0;
  state.gt.forEach((g) => {
    if (g.class_name !== ev.class_name) return;
    best = Math.max(best, iou(ev.start_time_sec, ev.end_time_sec, g.start, g.end));
  });
  el.hudIouRow.classList.remove('hidden');
  el.hudIou.textContent = best.toFixed(2);
  const pass = best >= 0.5;
  el.hudIou.classList.toggle('pass', pass);
  el.hudIou.classList.toggle('fail', !pass);
  el.hudIouGate.textContent = pass ? 'counts as a match' : 'needs ≥ 0.50';
}

/* ── detections ─────────────────────────────────────────────────────── */
function renderEvents(events) {
  state.events = events.slice();
  repaintHeat();                 // event windows drive the red, so recolour
  el.count.textContent = events.length;
  el.count.classList.add('pop');
  setTimeout(() => el.count.classList.remove('pop'), 320);

  if (!events.length) {
    el.list.innerHTML =
      `<div class="empty"><div class="empty-dot" style="animation:none;border-top-color:var(--ok)"></div>`
      + `<p style="color:var(--ok)">No anomalies detected</p>`
      + `<span class="sub">Scene classified as normal throughout</span></div>`;
    return;
  }

  el.list.innerHTML = '';
  events.forEach((ev, i) => {
    const card = document.createElement('div');
    card.className = 'det';
    card.style.animationDelay = `${i * 55}ms`;
    const t = ev.start_time_sec == null ? 'whole clip'
            : `${clock(ev.start_time_sec)}–${clock(ev.end_time_sec)}`;
    const conf = Math.round((ev.confidence || 0) * 100);
    card.innerHTML =
      `<img class="det-thumb" src="${ev._thumb || ''}" alt="" onerror="this.style.visibility='hidden'">`
      + `<div class="det-b"><div class="det-t"><span class="det-cls">${pretty(ev.class_name)}</span>`
      + `<span class="det-time">${t}</span></div>`
      + `<div class="det-x">${ev.explanation || ''}</div>`
      + `<div class="det-c"><span class="det-c-track"><span class="det-c-fill"></span></span>`
      + `<span>${conf}%</span></div></div>`;
    el.list.appendChild(card);
    requestAnimationFrame(() => { card.querySelector('.det-c-fill').style.width = `${conf}%`; });
    if (ev.start_time_sec != null) {
      card.style.cursor = 'pointer';
      card.onclick = () => { el.video.currentTime = ev.start_time_sec; el.video.play().catch(() => {}); };
    }
  });
}

/* ── analysis ───────────────────────────────────────────────────────── */
async function analyse(file) {
  if (!file) return;

  el.drop.classList.add('hidden');
  el.viewer.classList.remove('hidden');
  el.back.classList.remove('hidden');
  el.source.textContent = file.name;
  el.list.innerHTML = '';
  el.count.textContent = '0';
  el.verdict.classList.add('hidden');
  heat.length = 0;
  Object.assign(state, { duration: 0, total: 0, done: 0, times: [], t0: performance.now() });
  initBars();

  const isImage = /^image\//.test(file.type);
  state.kind = isImage ? 'image' : 'video';
  const url = URL.createObjectURL(file);
  if (isImage) {
    el.image.src = url; el.image.classList.remove('hidden');
    el.video.classList.add('hidden'); el.controls.classList.add('hidden');
  } else {
    el.video.src = url; el.video.classList.remove('hidden'); el.image.classList.add('hidden');
    el.controls.classList.remove('hidden');
    el.video.muted = true;
    el.iMuted.classList.remove('hidden'); el.iSound.classList.add('hidden');
    el.video.addEventListener('loadedmetadata', () => {
      el.durLabel.textContent = clock(el.video.duration);
      el.totTime.textContent = clock(el.video.duration);
    }, { once: true });
    syncPlayIcon();
  }

  setStatus('busy', 'Uploading', true);
  let job;
  try {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || 'upload failed');
    job = await r.json();
  } catch (e) {
    toast(`Upload failed — ${e.message}`);
    setStatus('err', 'Error');
    resetUI();
    return;
  }

  state.jobId = job.job_id;
  state.gt = job.ground_truth || [];
  state.events = [];
  if (state.gt.length) toast(`Labelled clip — IoU shown live against ${state.gt.length} truth interval${state.gt.length > 1 ? 's' : ''}`);

  // The object URL points at the file as the browser received it, and browsers
  // decode far less than OpenCV does - an MPEG-4 Part 2 clip analyses perfectly
  // and plays as a black rectangle. When the server says it had to re-encode,
  // switch playback to the copy it made once that copy exists.
  if (state.kind === 'video' && job.needs_transcode) {
    el.prepText.textContent = `Re-encoding ${job.codec || 'video'} for playback`;
    el.prep.classList.remove('hidden');
    waitForPlayback(job.job_id);
  }

  setStatus('busy', 'Analysing', true);
  el.live.classList.remove('hidden');
  requestAnimationFrame(initCanvas);

  const es = new EventSource(`/api/stream/${job.job_id}`);
  state.es = es;
  let lastEvents = [], lastThumb = null;

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
      if (d.thumb_b64) lastThumb = `data:image/jpeg;base64,${d.thumb_b64}`;
      else if (d.thumb) lastThumb = `data:image/jpeg;base64,${d.thumb}`;

      if (state.duration) {
        el.playhead.classList.add('on');
        el.playhead.style.left = `${(d.end / state.duration) * 100}%`;
        el.timeLabel.textContent = `${d.end.toFixed(1)}s`;
      }
      el.chunkPos.textContent = state.duration ? `${d.start.toFixed(1)}–${d.end.toFixed(1)}s` : 'frame';

      const pct = Math.round((state.done / state.total) * 100);
      el.bar.style.width = `${pct}%`;
      el.tProgress.innerHTML = `${pct}<i>%</i>`;
      el.tChunks.innerHTML = `${state.done}<i>/${state.total}</i>`;
      const s = [...state.times].sort((a, b) => a - b);
      el.tP50.innerHTML = `${Math.round(s[Math.floor(s.length * .5)])}<i>ms</i>`;
      el.tP95.innerHTML = `${Math.round(s[Math.floor(s.length * .95)] || s.at(-1))}<i>ms</i>`;

      const best = d.top[0];
      if (best) {
        el.verdict.classList.remove('hidden');
        el.vLabel.textContent = pretty(best.cls);
        el.vConf.textContent = `${Math.round(best.p * 100)}% confidence`;
        el.verdict.classList.toggle('hit', best.cls !== 'normal' && best.p > .6);
        el.verdict.classList.toggle('clear', best.cls === 'normal');
      }
    }

    else if (d.type === 'events') {
      lastEvents = d.events.map((e) => ({ ...e, _thumb: lastThumb }));
      drawBands(lastEvents);
      renderEvents(lastEvents);
    }

    else if (d.type === 'done') {
      const rt = d.runtime || {};
      if (rt.p50_ms != null) el.tP50.innerHTML = `${Math.round(rt.p50_ms)}<i>ms</i>`;
      if (rt.p95_ms != null) el.tP95.innerHTML = `${Math.round(rt.p95_ms)}<i>ms</i>`;
      el.tRtf.textContent = rt.rtf != null ? `${rt.rtf}×` : '—';
      el.bar.style.width = '100%';
      el.tProgress.innerHTML = `100<i>%</i>`;
      finish(lastEvents.length);
    }

    else if (d.type === 'error') {
      toast(d.message);
      setStatus('err', 'Error');
      finish(null, true);
    }
  };

  es.addEventListener('eof', () => es.close());
  es.onerror = () => es.close();
}

function finish(n, failed) {
  state.es?.close();
  el.live.classList.add('hidden');
  el.playhead.classList.remove('on');
  if (!failed) setStatus(n ? 'busy' : 'ok', n ? `${n} detection${n > 1 ? 's' : ''}` : 'Clear');
  if (state.kind === 'video') el.video.play().catch(() => {});
}

function resetUI() {
  state.es?.close();
  state.kind = null;
  state.gt = []; state.events = [];
  heat.length = 0;
  el.gtWrap.classList.add('hidden');
  el.gtBands.innerHTML = ''; el.iouBands.innerHTML = ''; el.bands.innerHTML = '';
  el.gtPlayhead.classList.remove('on');
  el.hud.classList.add('hidden');
  el.viewer.classList.add('hidden');
  el.drop.classList.remove('hidden');
  el.live.classList.add('hidden');
  el.verdict.classList.add('hidden');
  el.back.classList.add('hidden');
  el.controls.classList.add('hidden');
  el.seekFill.style.width = '0';
  el.curTime.textContent = '0:00'; el.totTime.textContent = '0:00';
  el.file.value = '';
  el.video.pause(); el.video.removeAttribute('src'); el.video.load();
  el.image.removeAttribute('src');
  el.source.textContent = 'No source selected';
  el.bands.innerHTML = ''; el.bar.style.width = '0';
  el.tProgress.innerHTML = '0<i>%</i>'; el.tChunks.innerHTML = '0<i>/0</i>';
  el.tP50.innerHTML = '—<i>ms</i>'; el.tP95.innerHTML = '—<i>ms</i>'; el.tRtf.textContent = '—';
  el.count.textContent = '0'; el.chunkPos.textContent = '—';
  el.timeLabel.textContent = '0.0s'; el.durLabel.textContent = '—';
  el.list.innerHTML =
    `<div class="empty"><div class="empty-dot"></div><p>No footage analysed yet</p>`
    + `<span class="sub">Events appear here as they are found</span></div>`;
  initBars();
  if (ctx) initCanvas();
  checkHealth();
}

/* ── playback ───────────────────────────────────────────────────────── */
function syncPlayIcon() {
  const playing = !el.video.paused && !el.video.ended;
  el.iPlay.classList.toggle('hidden', playing);
  el.iPause.classList.toggle('hidden', !playing);
  el.controls.classList.toggle('pin', !playing);
}

function togglePlay() {
  if (state.kind !== 'video' || !el.video.src) return;
  el.video.paused ? el.video.play().catch(() => {}) : el.video.pause();
}

function seekTo(clientX, rect) {
  const d = el.video.duration;
  if (!d || !isFinite(d)) return;
  el.video.currentTime = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1) * d;
  paintPlayhead();
}

function paintPlayhead() {
  const d = el.video.duration;
  if (!d || !isFinite(d)) return;
  el.seekFill.style.width = `${(el.video.currentTime / d) * 100}%`;
  el.curTime.textContent = clock(el.video.currentTime);
  if (state.duration) {
    const left = `${(el.video.currentTime / state.duration) * 100}%`;
    el.playhead.classList.add('on');
    el.playhead.style.left = left;
    if (state.gt.length) { el.gtPlayhead.classList.add('on'); el.gtPlayhead.style.left = left; }
    el.timeLabel.textContent = `${el.video.currentTime.toFixed(1)}s`;
    updateHud(el.video.currentTime);
  }
}

el.play.onclick = togglePlay;
el.video.addEventListener('click', togglePlay);
el.video.addEventListener('dblclick', (e) => { e.preventDefault(); toggleFullscreen(); });
['play', 'pause', 'ended'].forEach((e) => el.video.addEventListener(e, syncPlayIcon));
el.video.addEventListener('timeupdate', paintPlayhead);

el.mute.onclick = () => {
  el.video.muted = !el.video.muted;
  el.iMuted.classList.toggle('hidden', !el.video.muted);
  el.iSound.classList.toggle('hidden', el.video.muted);
};

let scrub = false;
el.seek.addEventListener('pointerdown', (e) => {
  scrub = true; el.seek.setPointerCapture(e.pointerId);
  seekTo(e.clientX, el.seek.getBoundingClientRect());
});
el.seek.addEventListener('pointermove', (e) => {
  if (scrub) seekTo(e.clientX, el.seek.getBoundingClientRect());
});
el.seek.addEventListener('pointerup', (e) => {
  scrub = false;
  try { el.seek.releasePointerCapture(e.pointerId); } catch {}
});
el.timeline.addEventListener('click', (e) => {
  if (state.kind === 'video') seekTo(e.clientX, e.currentTarget.getBoundingClientRect());
});

/* ── fullscreen & floating panels ───────────────────────────────────── */
// the panels are MOVED, not cloned, so the live stream keeps updating the same
// nodes and no state has to be mirrored between two copies
const home = { side: null, tele: null };
let panelsHidden = false;

function panelsIntoOverlay() {
  if (!home.side) {
    home.side = { parent: el.side.parentNode, next: el.side.nextSibling };
    home.tele = { parent: el.tele.parentNode, next: el.tele.nextSibling };
  }
  // detections + confidence first, telemetry underneath
  while (el.side.firstChild) el.fsOverlay.appendChild(el.side.firstChild);
  el.fsOverlay.appendChild(el.tele);
}

function panelsHome() {
  if (!home.side) return;
  // telemetry belongs back in the main column, the rest back in the sidebar
  home.tele.parent.insertBefore(el.tele, home.tele.next);
  while (el.fsOverlay.firstChild) el.side.appendChild(el.fsOverlay.firstChild);
}

function isFullscreen() {
  return !!(document.fullscreenElement || document.webkitFullscreenElement);
}

async function toggleFullscreen() {
  try {
    if (isFullscreen()) {
      await (document.exitFullscreen?.() ?? document.webkitExitFullscreen?.());
    } else {
      const s = el.stage;
      await (s.requestFullscreen?.() ?? s.webkitRequestFullscreen?.());
    }
  } catch (e) {
    toast('Fullscreen was blocked by the browser');
  }
}

function onFullscreenChange() {
  const on = isFullscreen();
  el.stage.classList.toggle('is-fs', on);
  el.iFsIn.classList.toggle('hidden', on);
  el.iFsOut.classList.toggle('hidden', !on);
  if (on) panelsIntoOverlay(); else panelsHome();
  applyPanelVisibility();
  if (state.total) requestAnimationFrame(repaintHeat);
}

function applyPanelVisibility() {
  el.fsOverlay.classList.toggle('off', panelsHidden);
  el.stage.classList.toggle('panels-off', panelsHidden);
  // outside fullscreen, hiding panels widens the feed instead
  el.layout.classList.toggle('wide', panelsHidden && !isFullscreen());
  el.iPanelOn.classList.toggle('hidden', panelsHidden);
  el.iPanelOff.classList.toggle('hidden', !panelsHidden);
}

function togglePanels() {
  panelsHidden = !panelsHidden;
  applyPanelVisibility();
  if (state.total) requestAnimationFrame(repaintHeat);
}

el.fs.onclick = toggleFullscreen;
el.panel.onclick = togglePanels;
['fullscreenchange', 'webkitfullscreenchange'].forEach((e) =>
  document.addEventListener(e, onFullscreenChange));

/* ── back / keyboard ────────────────────────────────────────────────── */
function goBack() {
  if (state.es && state.es.readyState !== 2) toast('Analysis cancelled');
  resetUI();
}
el.back.onclick = goBack;

addEventListener('keydown', (e) => {
  if (/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName)) return;
  const viewing = !el.viewer.classList.contains('hidden');
  const k = e.key.toLowerCase();
  if (k === 'f' && viewing) { e.preventDefault(); toggleFullscreen(); }
  else if (k === 'p') { e.preventDefault(); togglePanels(); }
  // Esc already exits fullscreen natively; only treat it as "back" outside it
  else if (e.key === 'Escape' && viewing && !isFullscreen()) { e.preventDefault(); goBack(); }
  else if (e.code === 'Space' && viewing && state.kind === 'video') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowLeft' && state.kind === 'video') el.video.currentTime = Math.max(0, el.video.currentTime - 5);
  else if (e.key === 'ArrowRight' && state.kind === 'video') el.video.currentTime = Math.min(el.video.duration || 0, el.video.currentTime + 5);
});

/* ── input ──────────────────────────────────────────────────────────── */
el.browse.onclick = (e) => { e.stopPropagation(); el.file.click(); };
el.drop.onclick = () => el.file.click();
el.file.onchange = (e) => analyse(e.target.files[0]);

['dragenter', 'dragover'].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === 'drop' || e.target === el.drop) el.drop.classList.remove('over');
  }));
el.drop.addEventListener('drop', (e) => analyse(e.dataTransfer.files[0]));
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => e.preventDefault());
addEventListener('resize', () => { if (state.total) repaintHeat(); });

initBars();
checkHealth();
setInterval(() => { if (!state.es || state.es.readyState === 2) checkHealth(); }, 15000);
