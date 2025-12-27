// ── Begin: selfplay_debug_ui/static/app.js ──
function fmtTs(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function fmtHz(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(1);
}

// 🚀 Helper: Cross IDs to Names
// 🚀 Updated Helper: Maps FORM_MAPPING indices to Names
function crossName(id) {
  const FORM_NAMES = [
    "Normal", "Fire", "Elec", "Slash", "Erase", "Charge",
    "Aqua", "Thawk", "Tengu", "Grnd", "Dust"
  ];
  return FORM_NAMES[id] || `ID:${id}`;
}

// 🚀 Helper: Code ID to Char
function codeChar(id) {
  const codes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*";
  if (id >= 0 && id < codes.length) return codes[id];
  return "?";
}

// -----------------------------------------------------------------------------
// Timeline & Controller Helpers
// -----------------------------------------------------------------------------
const TIMELINE_SLOTS = ["UP","DOWN","LEFT","RIGHT","A","B","L","R","SELECT","START"];
const TIMELINE_T = 18;

function btnNameToSlot(raw) {
  const k = String(raw).trim().toUpperCase();
  if (k === "UP" || k === "DPAD_UP") return "UP";
  if (k === "DOWN" || k === "DPAD_DOWN") return "DOWN";
  if (k === "LEFT" || k === "DPAD_LEFT") return "LEFT";
  if (k === "RIGHT" || k === "DPAD_RIGHT") return "RIGHT";
  if (k === "A" || k === "Z" || k === "EAST") return "A";
  if (k === "B" || k === "X" || k === "SOUTH") return "B";
  if (k === "L" || k === "LB") return "L";
  if (k === "R" || k === "RB") return "R";
  if (k === "START") return "START";
  if (k === "SELECT" || k === "BACK") return "SELECT";
  return null;
}

function renderTimeline(hostEl, entries) {
  if (!hostEl) return;
  hostEl.innerHTML = "";
  
  const grid = document.createElement("div");
  grid.className = "timeline";
  grid.style.gridTemplateColumns = `50px repeat(${TIMELINE_T}, 1fr)`;

  for (const slot of TIMELINE_SLOTS) {
    const lbl = document.createElement("div");
    lbl.className = "lane-label";
    lbl.textContent = slot;
    grid.appendChild(lbl);

    for (let t = 0; t < TIMELINE_T; t++) {
        const e = entries[t];
        const active = e && e.mapped_pressed_buttons && e.mapped_pressed_buttons.some(b => btnNameToSlot(b) === slot);
        const c = document.createElement("div");
        c.className = active ? "cell mapped" : "cell";
        grid.appendChild(c);
    }
  }
  hostEl.appendChild(grid);
}

function mountController(hostEl) {
  hostEl.innerHTML = `
    <div class="controller" data-controller="1">
      <div class="dpad">
        <div class="btn up" data-slot="UP"></div>
        <div class="btn down" data-slot="DOWN"></div>
        <div class="btn left" data-slot="LEFT"></div>
        <div class="btn right" data-slot="RIGHT"></div>
      </div>
      <div class="btn b" data-slot="B">B</div>
      <div class="btn a" data-slot="A">A</div>
      <div class="btn l" data-slot="L"></div>
      <div class="btn r" data-slot="R"></div>
      <div class="btn select" data-slot="SELECT"></div>
      <div class="btn start" data-slot="START"></div>
    </div>`;
  const ctrlEl = hostEl.querySelector('.controller');
  const els = {};
  TIMELINE_SLOTS.forEach(s => els[s] = ctrlEl.querySelector(`[data-slot="${s}"]`));
  return { els };
}

function setControllerPressed(ctrl, pressed) {
  if (!ctrl) return;
  Object.values(ctrl.els).forEach(e => e.classList.remove("active"));
  pressed.forEach(b => {
      const s = btnNameToSlot(b);
      if (s && ctrl.els[s]) ctrl.els[s].classList.add("active");
  });
}

// -----------------------------------------------------------------------------
// Cards
// -----------------------------------------------------------------------------
function createPortCard(port) {
  const card = document.createElement("div");
  card.className = "card";

  const h2 = document.createElement("h2");
  h2.innerHTML = `<span>Port ${port}</span> <span class="badge warn" data-role="badge">stale</span>`;
  
  const meta = document.createElement("div");
  meta.className = "meta";

  // Info Line
  const infLine = document.createElement("div");
  infLine.innerHTML = `Hz: <b class="kbd" data-role="hz">0</b>`;
  meta.appendChild(infLine);

  // 🚀 Game State Box
  const boxState = document.createElement("div");
  boxState.className = "box gamestate";
  boxState.innerHTML = `
    <div class="row-hp">
        <span class="hp-p" data-role="php">P: —</span>
        <span class="hp-e" data-role="ehp">E: —</span>
    </div>
    <div class="row-status" data-role="statusTags"></div>
    <div class="row-hist">
        <div class="label">Used Crosses</div>
        <div class="hist-list" data-role="usedCrosses">—</div>
    </div>
    <div class="row-chips" style="display:none;" data-role="chipWindow">
        <div class="label">Chip Window</div>
        <div class="chips-list" data-role="chipsList"></div>
    </div>
  `;
  meta.appendChild(boxState);

  // Controller Grid
  const grid2 = document.createElement("div");
  grid2.className = "grid2";
  const boxMap = document.createElement("div");
  boxMap.className = "box";
  boxMap.innerHTML = `<div class="label">Output</div><div class="padwrap" data-role="mappedPad"></div>`;
  grid2.appendChild(boxMap);
  meta.appendChild(grid2);

  // Image
  const imgwrap = document.createElement("div");
  imgwrap.className = "imgwrap";
  const img = document.createElement("img");
  img.src = `/api/mjpeg/${port}`;
  imgwrap.appendChild(img);
  
  // Future Timeline
  const timeWrap = document.createElement("div");
  timeWrap.className = "timeline-wrap";
  timeWrap.innerHTML = `<div data-role="nextTimeline"></div>`;
  imgwrap.appendChild(timeWrap);

  card.appendChild(h2);
  card.appendChild(meta);
  card.appendChild(imgwrap);

  return {
    card,
    badge: h2.querySelector('[data-role="badge"]'),
    hzEl: card.querySelector('[data-role="hz"]'),
    phpEl: card.querySelector('[data-role="php"]'),
    ehpEl: card.querySelector('[data-role="ehp"]'),
    statusEl: card.querySelector('[data-role="statusTags"]'),
    usedEl: card.querySelector('[data-role="usedCrosses"]'),
    winEl: card.querySelector('[data-role="chipWindow"]'),
    chipsEl: card.querySelector('[data-role="chipsList"]'),
    nextTimelineEl: card.querySelector('[data-role="nextTimeline"]'),
    mappedController: mountController(card.querySelector('[data-role="mappedPad"]')),
  };
}

function updateGameState(refs, s) {
    // Update HP
    refs.phpEl.textContent = `P: ${s.player_hp}`;
    refs.ehpEl.textContent = `E: ${s.enemy_hp}`;

    // Update Status Tags (Current Form, Beast, Sync)
    const formName = crossName(s.player_cross_id);
    let tags = [`<span class="tag form">${formName}</span>`];
    
    if (s.beast_mode) tags.push(`<span class="tag beast">BEAST</span>`);
    if (s.full_synchro) tags.push(`<span class="tag sync">FULL SYNC</span>`);
    refs.statusEl.innerHTML = tags.join(" ");

    // 🚀 Update Used Crosses History with correct names
    const p_used = s.player_used_crosses || [];
    if (p_used.length > 0) {
        refs.usedEl.innerHTML = p_used
            .map(id => `<span class="cross-dead">${crossName(id)}</span>`)
            .join(" ");
    } else {
        refs.usedEl.textContent = "None";
    }

    // Update Chip Window
    if (s.inside_window && s.chip_window && s.chip_window.length > 0) {
        refs.winEl.style.display = "block";
        refs.chipsEl.innerHTML = s.chip_window.map(c => 
            `<div class="chip-card">ID:${c.id}<br><b>${codeChar(c.code)}</b></div>`
        ).join("");
    } else {
        refs.winEl.style.display = "none";
    }
}

const cardsByPort = new Map();

function updateFromPayload(payload) {
  const ports = payload.ports || {};
  const root = document.getElementById("cards");

  for (const pid of Object.keys(ports)) {
      const p = parseInt(pid);
      if (!cardsByPort.has(p)) {
          const refs = createPortCard(p);
          cardsByPort.set(p, refs);
          root.appendChild(refs.card);
      }
  }

  for (const [pid, s] of Object.entries(ports)) {
      const p = parseInt(pid);
      const refs = cardsByPort.get(p);
      if (!refs) continue;

      refs.hzEl.textContent = fmtHz(s.infer_hz);
      setControllerPressed(refs.mappedController, s.mapped_pressed_buttons || []);
      renderTimeline(refs.nextTimelineEl, s.next_actions || []);
      updateGameState(refs, s);
      
      const age = (Date.now()/1000) - s.ts;
      refs.badge.className = age < 2 ? "badge ok" : "badge warn";
      refs.badge.textContent = age < 2 ? "LIVE" : "LAG";
  }
}

async function tick() {
  try {
    const r = await fetch(`/api/state?t=${Date.now()}`);
    const data = await r.json();
    updateFromPayload(data);
  } catch(e) { console.error(e); }
}
setInterval(tick, 100);
// ── End: selfplay_debug_ui/static/app.js ──