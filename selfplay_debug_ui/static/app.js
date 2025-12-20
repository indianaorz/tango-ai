// selfplay_debug_ui/static/app.js
let timer = null;

function fmtTs(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function esc(s) {
  return (s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
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

function renderHistoryRows(hist, n) {
  const rows = Array.isArray(hist) ? hist.slice(0, n) : [];
  if (rows.length === 0) {
    return `<tr><td colspan="4" class="small">No history yet.</td></tr>`;
  }

  return rows.map(h => {
    const ngBtns = (h.ng_pressed_buttons || []).join(", ");
    const mappedBtns = (h.mapped_pressed_buttons || []).join(", ");

    const ngBin = h.ng_key_bin || "—";
    const mappedBin = h.mapped_key_bin || "—";

    return `
      <tr>
        <td class="mono">${esc(fmtTs(h.ts))}</td>
        <td class="mono">${esc(h.action_type || "—")}<div class="small">inside_window: ${h.inside_window ? "true" : "false"}</div></td>
        <td class="mono">
          <div>${esc(ngBin)}</div>
          <div class="small">${esc(ngBtns || "—")}</div>
        </td>
        <td class="mono">
          <div>${esc(mappedBin)}</div>
          <div class="small">${esc(mappedBtns || "—")}</div>
        </td>
      </tr>
    `;
  }).join("");
}

// -----------------------------------------------------------------------------
// Incremental DOM: create cards once, then update fields only.
// -----------------------------------------------------------------------------
const cardsByPort = new Map();

// Rate-limit history DOM churn so higher refresh Hz doesn’t tank FPS.
let lastHistoryRenderAt = 0;
const HISTORY_RENDER_MIN_MS = 500;

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
    <div style="margin-top: 8px;" data-role="ngBtns"></div>
    <div class="small" style="margin-top:8px; display:none;" data-role="ngHint">(Waiting for decision.ng_key_bin)</div>
  `;

  const boxMapped = document.createElement("div");
  boxMapped.className = "box";
  boxMapped.innerHTML = `
    <div class="label">We interpret / send to game</div>
    <div class="kbd" data-role="mappedBin">—</div>
    <div style="margin-top: 8px;" data-role="mappedBtns"></div>
  `;

  grid2.appendChild(boxNg);
  grid2.appendChild(boxMapped);

  meta.appendChild(infLine);
  meta.appendChild(imgLine);
  meta.appendChild(grid2);

  const imgwrap = document.createElement("div");
  imgwrap.className = "imgwrap";

  const img = document.createElement("img");
  img.alt = `port ${port} image`;
  img.loading = "eager";
  img.decoding = "async";
  
  // ---------------------------------------------------------
  // MJPEG Stream Setup with Reconnection Logic
  // ---------------------------------------------------------
  const mjpegUrl = `/api/mjpeg/${port}`;
  img.src = mjpegUrl;

  // If the server restarts or connection dies, retry every 2s
  img.onerror = () => {
    img.style.opacity = "0.5"; // visual cue that it's reconnecting
    setTimeout(() => {
        // Add timestamp to bust cache / force reconnect
        img.src = `${mjpegUrl}?t=${Date.now()}`;
        img.style.opacity = "1.0";
    }, 2000);
  };
  
  imgwrap.appendChild(img);

  const tablewrap = document.createElement("div");
  tablewrap.className = "tablewrap";
  tablewrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th style="width: 90px;">Time</th>
          <th style="width: 190px;">Context</th>
          <th>NG raw</th>
          <th>Mapped</th>
        </tr>
      </thead>
      <tbody data-role="histBody">
        <tr><td colspan="4" class="small">No history yet.</td></tr>
      </tbody>
    </table>
  `;

  card.appendChild(h2);
  card.appendChild(meta);
  card.appendChild(imgwrap);
  card.appendChild(tablewrap);

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
    histBodyEl: card.querySelector('[data-role="histBody"]'),
  };

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
  const ports = Object.keys(portsObj).map(x => parseInt(x, 10)).filter(Number.isFinite).sort((a, b) => a - b);

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

  const histN = Math.max(5, Math.min(400, parseInt(document.getElementById("histN").value || "40", 10)));
  const nowMs = performance.now();
  const doHistory = (nowMs - lastHistoryRenderAt) >= HISTORY_RENDER_MIN_MS;
  if (doHistory) lastHistoryRenderAt = nowMs;

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

    const dims = (s.img_w && s.img_h) ? `${s.img_w}×${s.img_h}` : "—";
    refs.dimsEl.textContent = dims;

    const ngBin = s.ng_key_bin || "—";
    const mappedBin = s.mapped_key_bin || "—";
    refs.ngBinEl.textContent = ngBin;
    refs.mappedBinEl.textContent = mappedBin;

    chipsEl(refs.ngBtnsEl, s.ng_pressed_buttons || []);
    chipsEl(refs.mappedBtnsEl, s.mapped_pressed_buttons || []);

    refs.ngHintEl.style.display = (ngBin === "—") ? "block" : "none";

    if (doHistory) {
      const hist = s.history || [];
      refs.histBodyEl.innerHTML = renderHistoryRows(hist, histN);
    }
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
    document.getElementById("status").textContent = "Error fetching state";
  }
}

// Start polling for stats (10 Hz = 100ms)
document.getElementById("histN").addEventListener("change", tick);
setInterval(tick, 100);
tick();