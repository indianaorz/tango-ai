// selfplay_debug_ui/static/app.js

function fmtTs(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function fmtHz(x) {
  if (x === null || x === undefined) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function badgeForAge(ageS) {
  const a = Number(ageS || 0);
  if (!Number.isFinite(a)) return { cls: "warn", text: "stale" };
  if (a < 1.0) return { cls: "ok", text: "live" };
  if (a < 3.0) return { cls: "warn", text: "lag" };
  return { cls: "warn", text: "stale" };
}

function chipsEl(containerEl, list) {
  const xs = Array.isArray(list) ? list : [];
  containerEl.innerHTML = "";
  if (xs.length === 0) {
    const sp = document.createElement("span");
    sp.className = "small";
    sp.textContent = "—";
    containerEl.appendChild(sp);
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "rowchips";
  for (const x of xs) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = String(x);
    wrap.appendChild(c);
  }
  containerEl.appendChild(wrap);
}

// -----------------------------------------------------------------------------
// Timeline grid (viewer-style) for Mapped vs NG.
//
// Conventions:
// - "Mapped" (sent to game) => GT (green)
// - "NG" (model intent)     => PR (magenta)
// -----------------------------------------------------------------------------

const TIMELINE_SLOTS = ["UP","DOWN","LEFT","RIGHT","A","B","L","R","SELECT","START"];
const TIMELINE_T = 18;

function niceSlotLabel(slot) {
  if (slot === "SELECT") return "SELECT";
  if (slot === "START") return "START";
  return slot;
}

function slotsSetFromPressedButtons(pressedButtonsList) {
  const xs = Array.isArray(pressedButtonsList) ? pressedButtonsList : [];
  const set = new Set();
  for (const raw of xs) {
    const slot = btnNameToSlot(raw);
    if (slot) set.add(slot);
  }
  return set;
}

/**
 * Render a timeline grid into hostEl using entries.
 *
 * IMPORTANT UI behavior:
 *   Grid width is fixed to TIMELINE_T columns (max 18),
 *   even if entries length changes frame-to-frame.
 *
 * Each entry should have:
 *   - mapped_pressed_buttons: [...]
 *   - ng_pressed_buttons: [...]
 *
 * For history: newest-first.
 * For next buffer: earliest-next first.
 */
function renderMappedVsNgTimeline(hostEl, entries, titleText) {
  if (!hostEl) return;

  const xs = Array.isArray(entries) ? entries : [];

  // Fixed width: always 18 columns (prevents resizing/jitter).
  const T = TIMELINE_T;

  hostEl.innerHTML = "";

  // Title row
  const title = document.createElement("div");
  title.className = "timeline-title";
  title.textContent = titleText;
  hostEl.appendChild(title);

  const grid = document.createElement("div");
  grid.className = "timeline";
  grid.style.gridTemplateColumns = `60px repeat(${T}, 1fr)`;

  for (const slot of TIMELINE_SLOTS) {
    const lbl = document.createElement("div");
    lbl.className = "lane-label";
    lbl.textContent = niceSlotLabel(slot);
    grid.appendChild(lbl);

    for (let t = 0; t < T; t++) {
      const e = xs[t] || null;

      const mappedSet = e ? slotsSetFromPressedButtons(e.mapped_pressed_buttons) : new Set();
      const ngSet = e ? slotsSetFromPressedButtons(e.ng_pressed_buttons) : new Set();

      const gt = mappedSet.has(slot);
      const pr = ngSet.has(slot);

      const c = document.createElement("div");
      c.className = "cell";
      if (gt && pr) c.classList.add("both");
      else if (gt) c.classList.add("gt");
      else if (pr) c.classList.add("pr");
      // else: empty cell (keeps grid stable)

      grid.appendChild(c);
    }
  }

  hostEl.appendChild(grid);

  // Count line (helps debug quickly)
  const shown = Math.min(xs.length, T);
  const count = document.createElement("div");
  count.className = "small muted";
  count.style.marginTop = "6px";
  count.textContent = `steps filled: ${shown} / ${T} (src: ${xs.length})`;
  hostEl.appendChild(count);
}

/**
 * History is newest-first (your to_json does reversed(deque)).
 */
function renderHistoryTimeline(hostEl, history) {
  renderMappedVsNgTimeline(hostEl, history, "History (fixed 18): Mapped vs NG");
}

/**
 * Next actions buffer is already "future steps" (we encoded skipping t0 in DebugState),
 * and we want earliest-next first. So render as-is (index 0 = next step).
 */
function renderNextBufferTimeline(hostEl, nextActions) {
  renderMappedVsNgTimeline(hostEl, nextActions, "Next buffer (fixed 18): Mapped vs NG");
}

// -----------------------------------------------------------------------------
// Controller mapping (scope-safe, no global IDs)
// -----------------------------------------------------------------------------

function normBtnName(x) {
  return String(x ?? "")
    .trim()
    .toUpperCase()
    .replaceAll("-", "_")
    .replaceAll(" ", "_");
}

function btnNameToSlot(btnNameRaw) {
  const k = normBtnName(btnNameRaw);
  if (!k) return null;

  if (k === "DPAD_UP" || k === "UP") return "UP";
  if (k === "DPAD_DOWN" || k === "DOWN") return "DOWN";
  if (k === "DPAD_LEFT" || k === "LEFT") return "LEFT";
  if (k === "DPAD_RIGHT" || k === "RIGHT") return "RIGHT";

  if (k === "EAST" || k === "A" || k === "BTN_A" || k === "BUTTON_A") return "A";
  if (k === "SOUTH" || k === "B" || k === "BTN_B" || k === "BUTTON_B") return "B";

  if (k === "Z") return "A";
  if (k === "X") return "B";

  if (k === "LEFT_SHOULDER" || k === "L" || k === "LB" || k === "L1") return "L";
  if (k === "RIGHT_SHOULDER" || k === "R" || k === "RB" || k === "R1") return "R";

  if (k === "START") return "START";
  if (k === "BACK" || k === "SELECT") return "SELECT";

  return null;
}

function mountController(hostEl) {
  hostEl.innerHTML = `
    <div class="controller" data-controller="1">
      <div class="btn l" data-slot="L"></div>
      <div class="btn r" data-slot="R"></div>

      <div class="dpad">
        <div class="dpad-center"></div>
        <div class="btn up" data-slot="UP"></div>
        <div class="btn down" data-slot="DOWN"></div>
        <div class="btn left" data-slot="LEFT"></div>
        <div class="btn right" data-slot="RIGHT"></div>
      </div>

      <div class="btn b" data-slot="B">B</div>
      <div class="btn a" data-slot="A">A</div>

      <div class="btn select" data-slot="SELECT"></div>
      <div class="btn start" data-slot="START"></div>
    </div>
  `;

  const ctrlEl = hostEl.querySelector('[data-controller="1"]');
  const els = {};
  for (const slot of ["UP","DOWN","LEFT","RIGHT","A","B","L","R","START","SELECT"]) {
    els[slot] = ctrlEl.querySelector(`[data-slot="${slot}"]`);
  }

  return { ctrlEl, els };
}

function setActiveVisual(el, active) {
  if (!el) return;

  el.classList.toggle("active", !!active);

  if (!active) {
    el.style.background = "";
    el.style.boxShadow = "";
    el.style.borderColor = "";
    el.style.outline = "";
    return;
  }

  const slot = (el.getAttribute("data-slot") || "").toUpperCase();
  const isFace = slot === "A" || slot === "B";

  if (isFace) {
    el.style.background = "#ff3333";
    el.style.boxShadow = "0 0 15px #ff0000";
    el.style.borderColor = "#aa0000";
    el.style.outline = "2px solid rgba(255,0,0,.35)";
  } else {
    el.style.background = "#00ff00";
    el.style.boxShadow = "0 0 10px #00ff00";
    el.style.borderColor = "#00cc00";
    el.style.outline = "2px solid rgba(0,255,0,.25)";
  }
}

function clearController(ctrl) {
  if (!ctrl) return;
  for (const el of Object.values(ctrl.els)) setActiveVisual(el, false);
}

function setControllerPressed(ctrl, pressedButtonsList) {
  if (!ctrl) return;

  clearController(ctrl);

  const xs = Array.isArray(pressedButtonsList) ? pressedButtonsList : [];
  const slots = [];

  for (const raw of xs) {
    const slot = btnNameToSlot(raw);
    if (!slot) continue;
    slots.push(slot);
    const el = ctrl.els[slot];
    setActiveVisual(el, true);
  }

  return slots;
}

// -----------------------------------------------------------------------------
// Cards
// -----------------------------------------------------------------------------
const cardsByPort = new Map();

function createPortCard(port) {
  const card = document.createElement("div");
  card.className = "card";

  const h2 = document.createElement("h2");

  const left = document.createElement("span");
  left.textContent = `Port ${port}`;

  const badge = document.createElement("span");
  badge.className = "badge warn";
  badge.textContent = "stale";

  const tsEl = document.createElement("span");
  tsEl.className = "muted";
  tsEl.textContent = "—";

  h2.appendChild(left);
  h2.appendChild(badge);
  h2.appendChild(tsEl);

  const meta = document.createElement("div");
  meta.className = "meta";

  const infLine = document.createElement("div");
  infLine.innerHTML = `inferences/sec: <b class="kbd" data-role="hz">—</b> <span class="muted">(total <span class="kbd" data-role="total">0</span>)</span>`;

  const imgLine = document.createElement("div");
  imgLine.innerHTML = `image: <b data-role="dims">—</b>`;

  const grid2 = document.createElement("div");
  grid2.className = "grid2";

  const boxNg = document.createElement("div");
  boxNg.className = "box";
  boxNg.innerHTML = `
    <div class="label">NG wants to press (raw)</div>
    <div class="kbd" data-role="ngBin">—</div>
    <div class="padwrap" data-role="ngPad"></div>
    <div style="margin-top: 8px;" data-role="ngBtns"></div>
    <div class="small" style="margin-top:8px; display:none;" data-role="ngHint">(Waiting for decision.ng_key_bin)</div>
  `;

  const boxMapped = document.createElement("div");
  boxMapped.className = "box";
  boxMapped.innerHTML = `
    <div class="label">We interpret / send to game</div>
    <div class="kbd" data-role="mappedBin">—</div>
    <div class="padwrap" data-role="mappedPad"></div>
    <div style="margin-top: 8px;" data-role="mappedBtns"></div>
  `;

  grid2.appendChild(boxNg);
  grid2.appendChild(boxMapped);

  meta.appendChild(infLine);
  meta.appendChild(imgLine);
  meta.appendChild(grid2);

  // Keep history in meta (optional)
  const histWrap = document.createElement("div");
  histWrap.className = "timeline-wrap";
  histWrap.innerHTML = `<div data-role="histTimeline"></div>`;
  meta.appendChild(histWrap);

  const imgwrap = document.createElement("div");
  imgwrap.className = "imgwrap";

  const img = document.createElement("img");
  img.alt = `port ${port} image`;
  img.loading = "eager";
  img.decoding = "async";

  const mjpegUrl = `/api/mjpeg/${port}`;
  img.src = mjpegUrl;

  img.onerror = () => {
    img.style.opacity = "0.5";
    setTimeout(() => {
      img.src = `${mjpegUrl}?t=${Date.now()}`;
      img.style.opacity = "1.0";
    }, 2000);
  };

  imgwrap.appendChild(img);

  // Next buffer timeline goes *below* the image
  const nextWrap = document.createElement("div");
  nextWrap.className = "timeline-wrap";
  nextWrap.innerHTML = `<div data-role="nextTimeline"></div>`;
  imgwrap.appendChild(nextWrap);

  card.appendChild(h2);
  card.appendChild(meta);
  card.appendChild(imgwrap);

  const refs = {
    card,
    badge,
    tsEl,
    hzEl: card.querySelector('[data-role="hz"]'),
    totalEl: card.querySelector('[data-role="total"]'),
    dimsEl: card.querySelector('[data-role="dims"]'),

    ngBinEl: card.querySelector('[data-role="ngBin"]'),
    mappedBinEl: card.querySelector('[data-role="mappedBin"]'),
    ngBtnsEl: card.querySelector('[data-role="ngBtns"]'),
    mappedBtnsEl: card.querySelector('[data-role="mappedBtns"]'),
    ngHintEl: card.querySelector('[data-role="ngHint"]'),

    nextTimelineEl: card.querySelector('[data-role="nextTimeline"]'),
    histTimelineEl: card.querySelector('[data-role="histTimeline"]'),

    ngController: null,
    mappedController: null,
  };

  const ngPadHost = card.querySelector('[data-role="ngPad"]');
  const mappedPadHost = card.querySelector('[data-role="mappedPad"]');

  if (ngPadHost) refs.ngController = mountController(ngPadHost);
  if (mappedPadHost) refs.mappedController = mountController(mappedPadHost);

  return refs;
}

function ensureCardsForPorts(ports) {
  const root = document.getElementById("cards");
  for (const p of ports) {
    if (!cardsByPort.has(p)) {
      const refs = createPortCard(p);
      cardsByPort.set(p, refs);
      root.appendChild(refs.card);
    }
  }
}

function updateFromPayload(payload) {
  const meta = payload.meta || {};
  const portsObj = payload.ports || {};
  const ports = Object.keys(portsObj)
    .map((x) => parseInt(x, 10))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  document.getElementById("overallHz").textContent = fmtHz(meta.overall_infer_hz);

  const win = meta.rate_window_s;
  document.getElementById("windowInfo").textContent =
    Number.isFinite(Number(win)) ? `(≈${Number(win)}s window)` : "";

  const root = document.getElementById("cards");

  if (ports.length === 0) {
    if (root.dataset.emptyShown !== "1") {
      root.innerHTML = `<div class="card"><h2>No ports yet</h2><div class="meta muted">Waiting for first frames/actions…</div></div>`;
      root.dataset.emptyShown = "1";
      cardsByPort.clear();
    }
    return;
  }

  if (root.dataset.emptyShown === "1") {
    root.innerHTML = "";
    root.dataset.emptyShown = "0";
  }

  ensureCardsForPorts(ports);

  for (const p of ports) {
    const s = portsObj[String(p)] || {};
    const refs = cardsByPort.get(p);
    if (!refs) continue;

    const age = s.last_update_age_s ?? 0;
    const badge = badgeForAge(age);
    refs.badge.className = `badge ${badge.cls}`;
    refs.badge.textContent = badge.text;

    refs.tsEl.textContent = fmtTs(s.ts);

    refs.hzEl.textContent = fmtHz(s.infer_hz);
    refs.totalEl.textContent = String(s.infer_count_total ?? 0);

    const dims = s.img_w && s.img_h ? `${s.img_w}×${s.img_h}` : "—";
    refs.dimsEl.textContent = dims;

    const ngBin = s.ng_key_bin || "—";
    const mappedBin = s.mapped_key_bin || "—";
    refs.ngBinEl.textContent = ngBin;
    refs.mappedBinEl.textContent = mappedBin;

    const ngPressed = s.ng_pressed_buttons || [];
    const mappedPressed = s.mapped_pressed_buttons || [];

    setControllerPressed(refs.ngController, ngPressed);
    setControllerPressed(refs.mappedController, mappedPressed);

    // Next actions buffer (fixed width grid)
    renderNextBufferTimeline(refs.nextTimelineEl, s.next_actions || []);

    // History (fixed width grid)
    renderHistoryTimeline(refs.histTimelineEl, s.history || []);

    refs.ngHintEl.style.display = (ngBin === "—") ? "block" : "none";
  }
}

async function tick() {
  try {
    const r = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const payload = await r.json();
    updateFromPayload(payload);
    document.getElementById("status").textContent = "OK";
  } catch (e) {
    document.getElementById("status").textContent = `Error fetching state`;
    console.error(e);
  }
}

setInterval(tick, 100);
tick();
