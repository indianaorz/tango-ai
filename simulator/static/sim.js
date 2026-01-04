// mmbn_sim/static/sim.js
"use strict";

function $(id) { return document.getElementById(id); }

function toast(msg) {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg || "";
  if (msg) setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 1400);
}

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (j && j.error) throw new Error(j.error);
  return j;
}

async function apiState() { return fetchJSON("/api/state"); }
async function apiReset() { return fetchJSON("/api/reset", { method: "POST" }); }
async function apiUi(actor, action) {
  return fetchJSON("/api/ui_action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor, action }),
  });
}
async function apiCommit() { return fetchJSON("/api/commit", { method: "POST" }); }

async function apiMctsRun(iterations, maxDepth) {
  return fetchJSON("/api/mcts/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ iterations, max_depth: maxDepth, seed: 0 }),
  });
}
async function apiMctsApply(who) {
  return fetchJSON("/api/mcts/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ who }),
  });
}

async function apiPlanRun(itersPerStep, lookaheadDepth) {
  return fetchJSON("/api/plan/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target_cust: 64,
      iters_per_step: itersPerStep,
      lookahead_depth: lookaheadDepth,
      seed: 0,
    }),
  });
}
async function apiPlanReplayReset() { return fetchJSON("/api/plan/replay_reset", { method: "POST" }); }
async function apiPlanReplayStep() { return fetchJSON("/api/plan/replay_step", { method: "POST" }); }

async function fetchTreeSubtree(depth) {
  const qs = new URLSearchParams({ node_id: "ROOT", depth: String(depth ?? 8) });
  return fetchJSON(`/api/tree/subtree?${qs.toString()}`);
}
async function setCurrentNode(nodeId) {
  return fetchJSON("/api/tree/set_current", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: nodeId }),
  });
}

// -------------------------------
// Client state
// -------------------------------
let state = null;     // {p1, p2, tree, mcts, plan}
let tree = null;      // subtree payload
let treeMcts = null;  // tree.mcts
const collapsed = new Set();

let replayTimer = null;

// -------------------------------
// Charge bar
// -------------------------------
function ensureChargeBar(el) {
  if (!el) return;
  if (el.childElementCount > 0) return;
  for (let i = 0; i < 5; i++) {
    const seg = document.createElement("div");
    seg.className = "charge-seg";
    el.appendChild(seg);
  }
}

function renderChargeBar(el, progress, fullAt, isLocked) {
  if (!el) return;
  ensureChargeBar(el);
  const segs = el.querySelectorAll(".charge-seg");
  const segCount = segs.length || 5;

  const fa = fullAt && fullAt > 0 ? fullAt : 5;
  const p = Math.max(0, Math.min(fa, progress | 0));
  const frac = p / fa;

  if (isLocked) el.classList.add("locked"); else el.classList.remove("locked");

  const filled = Math.round(frac * segCount);
  for (let i = 0; i < segCount; i++) {
    if (i < filled) segs[i].classList.add("on");
    else segs[i].classList.remove("on");
  }
}

// -------------------------------
// Button enable/disable based on server legal actions
// -------------------------------
function setBtnEnabled(id, enabled) {
  const el = $(id);
  if (!el) return;
  el.disabled = !enabled;
}

function validActionSet(view) {
  const arr = view?.valid_actions || [];
  return new Set(arr.map(String));
}

function renderActionButtons(prefix, view) {
  const valid = validActionSet(view);

  setBtnEnabled(`btn_${prefix}_hold_on`, valid.has("HOLD_ON"));
  setBtnEnabled(`btn_${prefix}_hold_off`, valid.has("HOLD_OFF"));

  setBtnEnabled(
    `btn_${prefix}_toggle`,
    valid.has("TOGGLE_HOLD") || valid.has("HOLD_ON") || valid.has("HOLD_OFF") || valid.has("NOOP")
  );

  setBtnEnabled(`btn_${prefix}_shoot`, valid.has("SHOOT"));
  setBtnEnabled(`btn_${prefix}_release`, valid.has("RELEASE_CHARGE"));
}

// -------------------------------
// Barrier visuals (lerped; works for any remaining HP)
// -------------------------------
const _barrierMem = new Map(); // key -> { max: number }

function _canonicalBarrierMaxFromHp(hp) {
  // MMBN-style tiers; keep sane behavior if hp exceeds 200.
  if (hp <= 0) return 0;
  if (hp <= 10) return 10;
  if (hp <= 100) return 100;
  if (hp <= 200) return 200;
  return hp;
}

function _barrierHueForMax(maxHp) {
  if (maxHp === 10) return 210;   // blue
  if (maxHp === 100) return 55;   // yellow
  if (maxHp === 200) return 315;  // pink
  return 120; // fallback (green-ish) if unknown
}

function applyBarrierVisual(entEl, barrierHp, memKey) {
  const hp = Number(barrierHp ?? 0) | 0;

  if (hp <= 0) {
    entEl.classList.remove("aura-barrier");
    entEl.style.removeProperty("--barrier-h");
    entEl.style.removeProperty("--barrier-a");
    _barrierMem.delete(memKey);
    return;
  }

  const prev = _barrierMem.get(memKey);
  let maxHp = prev?.max ?? 0;

  // If we don't know, or hp increased beyond the remembered max, treat as new barrier.
  // (This covers new barriers being applied mid-fight.)
  const canon = _canonicalBarrierMaxFromHp(hp);
  if (maxHp <= 0 || hp > maxHp) maxHp = canon;

  // Also, if we had a smaller remembered tier but hp clearly indicates a larger tier, upgrade it.
  if (canon > maxHp) maxHp = canon;

  _barrierMem.set(memKey, { max: maxHp });

  const hue = _barrierHueForMax(maxHp);

  // Lerp alpha based on remaining HP fraction.
  const frac = Math.max(0, Math.min(1, hp / Math.max(1, maxHp)));
  const alpha = 0.25 + 0.55 * Math.sqrt(frac); // 0.25 .. ~0.80 (nice “strong then fade”)

  entEl.classList.add("aura-barrier");
  entEl.style.setProperty("--barrier-h", String(hue));
  entEl.style.setProperty("--barrier-a", alpha.toFixed(3));
}


// -------------------------------
// HUD render
// -------------------------------
function renderMctsSummary() {
  const m = state?.mcts;
  if (!m) return;
  if ($("mcts_root_visits")) $("mcts_root_visits").textContent = String(m.root_visits ?? 0);
  if ($("mcts_p1_best")) $("mcts_p1_best").textContent = String(m.p1_maximin?.best ?? "?");
  if ($("mcts_p2_best")) $("mcts_p2_best").textContent = String(m.p2_minimax?.best ?? "?");
}

function renderPlanStatus() {
  const p = state?.plan;
  if (!p || !p.has_plan) {
    if ($("plan_status")) $("plan_status").textContent = "None";
    if ($("plan_replay")) $("plan_replay").textContent = "0/0";
    return;
  }
  const idx = p.replay_index ?? 0;
  const len = p.replay_len ?? 0;
  if ($("plan_status")) $("plan_status").textContent = `Ready (len ${len})`;
  if ($("plan_replay")) $("plan_replay").textContent = `${idx}/${len}`;
}

function renderSideHUD(prefix, v) {
  const lockEl = $(`${prefix}_lock`);
  if (lockEl) lockEl.textContent = v.is_locked ? `YES (${v.lock_remaining ?? 0})` : "NO";

  const fname = v.p_form_name ?? "?";
  const fid = v.p_form ?? "?";
  const formEl = $(`${prefix}_form`);
  if (formEl) formEl.textContent = `${fname} [${fid}]`;

  const posEl = $(`${prefix}_pos`);
  if (posEl) posEl.textContent = `r${v.p_rc?.[0] ?? "?"},c${v.p_rc?.[1] ?? "?"}`;

  const eposEl = $(`${prefix}_enemy_pos`);
  if (eposEl) eposEl.textContent = `r${v.e_rc?.[0] ?? "?"},c${v.e_rc?.[1] ?? "?"}`;

  const pendEl = $(`${prefix}_pending`);
  if (pendEl) pendEl.textContent = v.pending_action ? String(v.pending_action) : "None";

  const hpEl = $(`${prefix}_hp`);
  if (hpEl) hpEl.textContent = String(v.p_hp ?? 0);

  const ehpEl = $(`${prefix}_ehp`);
  if (ehpEl) ehpEl.textContent = String(v.e_hp ?? 0);

  const holdEl = $(`${prefix}_hold`);
  if (holdEl) holdEl.textContent = v.p_charge_hold ? "ON" : "OFF";

  // EXACT serializer fields:
  const barEl = $(`${prefix}_barrier`);
  if (barEl) barEl.textContent = String(v.p_barrier_hp ?? 0);

  const ebarEl = $(`${prefix}_ebarrier`);
  if (ebarEl) ebarEl.textContent = String(v.e_barrier_hp ?? 0);

  const fullAt = v.charge_full_at ?? 5;
  const progTxt = $(`${prefix}_prog_txt`);
  if (progTxt) progTxt.textContent = `${v.p_charge_progress ?? 0}/${fullAt}`;

  renderChargeBar($(`${prefix}_charge_bar`), v.p_charge_progress ?? 0, fullAt, !!v.is_locked);
}

function renderHUD() {
  if (!state) return;

  if ($("cust")) $("cust").textContent = String(state.p1?.cust ?? 0);
  if ($("tree_current")) $("tree_current").textContent = String(state.tree?.current_id ?? "?");

  renderPlanStatus();
  renderMctsSummary();

  if (state.p1) renderSideHUD("p1", state.p1);
  if (state.p2) renderSideHUD("p2", state.p2);

  if (state.p1) renderHand("p1", state.p1);
  if (state.p2) renderHand("p2", state.p2);

  renderActionButtons("p1", state.p1);
  renderActionButtons("p2", state.p2);

  const eventsEl = $("events");
  if (eventsEl) {
    const lines = (state.p1?.last_events || []);
    eventsEl.innerHTML = "";
    for (const line of lines.slice().reverse()) {
      const d = document.createElement("div");
      d.textContent = line;
      eventsEl.appendChild(d);
    }
  }
}

// -------------------------------
// Grid render + entities + shot overlay
// -------------------------------
function cellCenterInOverlay(gridEl, overlayEl, idx) {
  const cell = gridEl?.children?.[idx];
  if (!cell || !overlayEl) return { x: 0, y: 0 };

  const cellRect = cell.getBoundingClientRect();
  const overlayRect = overlayEl.getBoundingClientRect();

  return {
    x: cellRect.left - overlayRect.left + cellRect.width / 2,
    y: cellRect.top - overlayRect.top + cellRect.height / 2,
  };
}

function dirToOffset(dir, magnitudePx) {
  const m = magnitudePx | 0;
  switch (dir) {
    case "UP": return { dx: 0, dy: -m };
    case "DOWN": return { dx: 0, dy: +m };
    case "LEFT": return { dx: -m, dy: 0 };
    case "RIGHT": return { dx: +m, dy: 0 };
    default: return { dx: 0, dy: 0 };
  }
}

function applyEntityTransform(entEl, leanDir, entryDir, isEntering) {
  const LEAN_MAG = 10;
  const ENTRY_MAG = 6;

  const a = dirToOffset(leanDir, LEAN_MAG);
  const b = isEntering ? dirToOffset(entryDir, ENTRY_MAG) : { dx: 0, dy: 0 };

  entEl.style.transform = `translate(-50%, -50%) translate(${a.dx + b.dx}px, ${a.dy + b.dy}px)`;
}

function renderGrid(view, gridId, overlayId) {
  const gridEl = $(gridId);
  const overlayEl = $(overlayId);
  if (!gridEl || !overlayEl || !view) return;

  const owners = view.grid_owner_state || [];
  const tiles = view.grid_state || [];

  // idx -> "charge"|"buster"
  const hotKindByIdx = new Map();
  for (const hp of (view.hot_panels || [])) {
    if (!hp) continue;
    const idx = (hp.idx == null) ? null : (hp.idx | 0);
    const kind = String(hp.kind || "");
    if (idx == null) continue;
    if (kind === "charge" || kind === "buster") hotKindByIdx.set(idx, kind);
  }

  // Build 18 cells always
  gridEl.innerHTML = "";
  for (let i = 0; i < 18; i++) {
    const owner = owners[i] ?? 0;
    const tile = tiles[i] ?? 0;
    const kind = hotKindByIdx.get(i) || "";

    const d = document.createElement("div");
    d.className = `cell tile-${tile} p${owner}` + (kind ? ` hot-${kind}` : "");
    gridEl.appendChild(d);
  }

  const pIdx = view.p_idx ?? 0;
  const eIdx = view.e_idx ?? 0;

  const pCell = gridEl.children[pIdx];
  const eCell = gridEl.children[eIdx];

  if (pCell) {
    const ent = document.createElement("div");
    ent.className = "entity ent-p";

    const lvl = view.p_charge_level ?? 0;
    if (lvl === 1) ent.classList.add("aura-chg1");
    if (lvl === 2) ent.classList.add("aura-chg2");

    applyBarrierVisual(ent, view.p_barrier_hp ?? 0, `${gridId}:p`);


    applyEntityTransform(ent, view.p_lean_dir, view.p_entry_dir, !!view.p_is_entering);
    pCell.appendChild(ent);
  }

  if (eCell) {
    const ent = document.createElement("div");
    ent.className = "entity ent-e";

    const lvl = view.e_charge_level ?? 0;
    if (lvl === 1) ent.classList.add("aura-chg1");
    if (lvl === 2) ent.classList.add("aura-chg2");

    applyBarrierVisual(ent, view.e_barrier_hp ?? 0, `${gridId}:e`);


    applyEntityTransform(ent, view.e_lean_dir, view.e_entry_dir, !!view.e_is_entering);
    eCell.appendChild(ent);
  }

  // SVG overlay sizing
  const orect = overlayEl.getBoundingClientRect();
  const w = Math.max(1, Math.round(orect.width || 1));
  const h = Math.max(1, Math.round(orect.height || 1));
  overlayEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
  overlayEl.innerHTML = "";

  // Shots
  const shots = view.shot_lines || [];
  for (const sh of shots) {
    const a = cellCenterInOverlay(gridEl, overlayEl, sh.from_idx);
    const b = cellCenterInOverlay(gridEl, overlayEl, sh.to_idx);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(a.x));
    line.setAttribute("y1", String(a.y));
    line.setAttribute("x2", String(b.x));
    line.setAttribute("y2", String(b.y));
    line.setAttribute("stroke-width", "3");
    line.setAttribute("stroke-linecap", "round");

    const isCharged = sh.kind === "charge";
    line.setAttribute("stroke", isCharged ? "rgba(255, 80, 220, 0.95)" : "rgba(255, 235, 80, 0.95)");
    overlayEl.appendChild(line);
  }
}

function renderBoards() {
  if (!state) return;
  renderGrid(state.p1, "grid_p1", "overlay_p1");
  renderGrid(state.p2, "grid_p2", "overlay_p2");
}

// -------------------------------
// Chip / Hand rendering
// -------------------------------
function _firstDefined(obj, keys, fallback) {
  for (const k of keys) {
    if (obj && Object.prototype.hasOwnProperty.call(obj, k) && obj[k] != null) return obj[k];
  }
  return fallback;
}

function extractHand(view) {
  const h = _firstDefined(view, ["hand", "p_hand", "current_hand", "hand_chips"], null);
  if (Array.isArray(h)) return h;

  const names = _firstDefined(view, ["hand_names", "p_hand_names"], null);
  if (Array.isArray(names)) return names.map(n => ({ name: String(n) }));

  return [];
}

function chipRowLabel(ch) {
  if (ch == null) return { name: "?", meta: "" };
  if (typeof ch === "string") return { name: ch, meta: "" };

  const name =
    (ch.name != null ? String(ch.name) :
      ch.chip_name != null ? String(ch.chip_name) :
        ch.title != null ? String(ch.title) : "?");

  const code =
    (ch.code != null ? String(ch.code) :
      ch.letter != null ? String(ch.letter) :
        ch.chip_code != null ? String(ch.chip_code) : "");

  const dmg =
    (ch.dmg != null ? `dmg ${String(ch.dmg)}` :
      ch.damage != null ? `dmg ${String(ch.damage)}` : "");

  const extra =
    (ch.desc != null ? String(ch.desc) :
      ch.description != null ? String(ch.description) : "");

  const bits = [];
  if (code) bits.push(code);
  if (dmg) bits.push(dmg);
  if (extra) bits.push(extra);

  return { name, meta: bits.join(" · ") };
}

function extractSelectedChipIdx(view) {
  const v = _firstDefined(view, ["selected_chip_idx", "pending_chip_idx", "chip_selected_idx"], null);
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? (n | 0) : null;
}

function findChipActionForIndex(validSet, idx) {
  if (idx === 0 && validSet.has("USE_CHIP")) return "USE_CHIP";

  const candidates = [
    `CHIP_${idx}`,
    `USE_CHIP_${idx}`,
    `SELECT_CHIP_${idx}`,
    `CHIP_SELECT_${idx}`,
    `CHIP_USE_${idx}`,
  ];
  for (const c of candidates) {
    if (validSet.has(c)) return c;
  }

  for (const a of validSet) {
    const s = String(a);
    if (s === `CHIP:${idx}` || s === `USE_CHIP:${idx}` || s === `SELECT_CHIP:${idx}`) return s;
  }

  return null;
}

function renderHand(prefix, view) {
  const listEl = $(`${prefix}_hand_list`);
  const stateEl = $(`${prefix}_hand_state`);
  if (!listEl) return;

  const valid = validActionSet(view);
  const hand = extractHand(view);
  const selIdx = extractSelectedChipIdx(view);
  const inChip = !!_firstDefined(view, ["in_chip_window", "chip_window_open", "is_in_chip_select"], false);

  if (stateEl) stateEl.textContent = inChip ? "chip window" : "battle";

  listEl.innerHTML = "";

  if (!hand.length) {
    const empty = document.createElement("div");
    empty.style.fontSize = "11px";
    empty.style.color = "#777";
    empty.textContent = "(hand empty / not provided by server yet)";
    listEl.appendChild(empty);
    return;
  }

  hand.forEach((ch, idx) => {
    const { name, meta } = chipRowLabel(ch);
    const act = findChipActionForIndex(valid, idx);
    const enabled = !!act;

    const row = document.createElement("div");
    row.className = "chip-row" + (enabled ? "" : " disabled") + ((selIdx === idx) ? " selected" : "");

    const main = document.createElement("div");
    main.className = "chip-main";

    const nm = document.createElement("div");
    nm.className = "chip-name";
    nm.textContent = name;

    const mt = document.createElement("div");
    mt.className = "chip-meta";
    mt.textContent = meta || `slot ${idx}`;

    main.appendChild(nm);
    main.appendChild(mt);

    const actWrap = document.createElement("div");
    actWrap.className = "chip-act";

    const btn = document.createElement("button");
    btn.className = "chip-btn";
    btn.textContent = enabled ? "Use" : "N/A";
    btn.disabled = !enabled;

    const doUse = async (ev) => {
      if (ev) ev.stopPropagation();
      if (!enabled) return;
      try {
        const actor = prefix === "p1" ? "P1" : "P2";
        state = await apiUi(actor, act);
        stopAutoReplay();
        renderHUD();
        renderBoards();
      } catch (e) {
        toast(String(e));
      }
    };

    btn.onclick = doUse;
    actWrap.appendChild(btn);

    row.appendChild(main);
    row.appendChild(actWrap);
    row.onclick = doUse;

    listEl.appendChild(row);
  });
}

// -------------------------------
// Tree (unchanged)
// -------------------------------
function getChildren(nid) {
  if (!tree || !tree.edges) return [];
  const kids = tree.edges[nid] || {};
  return Object.entries(kids).sort((a, b) => a[0].localeCompare(b[0]));
}
function nodeHasAnyKnownChildren(nid) { return getChildren(nid).length > 0; }

function uncollapsePathToCurrent() {
  if (!tree || !tree.nodes) return;
  const nodes = tree.nodes;
  const rootId = tree.root_id;
  const currentId = tree.current_id;

  if (rootId) collapsed.delete(rootId);
  if (currentId) collapsed.delete(currentId);

  let nid = currentId;
  const guard = new Set();
  while (nid && nodes[nid] && !guard.has(nid)) {
    guard.add(nid);
    collapsed.delete(nid);
    const parent = nodes[nid].parent;
    if (!parent) break;
    collapsed.delete(parent);
    nid = parent;
  }
}

function fmtNodeSummary(n, nid) {
  const s = n.s || {};
  const ns = treeMcts?.node_stats?.[nid];
  const nvis = ns?.N ?? 0;
  const nq = ns?.Q ?? 0;
  const p1l = (s.p1_locked ? 1 : 0);
  const p2l = (s.p2_locked ? 1 : 0);
  return `cust${s.cust ?? "?"} hp ${s.p1_hp ?? "?"}/${s.p2_hp ?? "?"} L(${p1l},${p2l}) · N${nvis} Q${Number(nq).toFixed(3)}`;
}

function fmtEdgeStats(parentId, joint) {
  const es = treeMcts?.edge_stats?.[parentId]?.[joint];
  if (!es) return "";
  const n = es.N ?? 0;
  const q = es.Q ?? 0;
  return ` · eN${n} eQ${Number(q).toFixed(3)}`;
}

async function refreshTree() {
  const depth = parseInt($("tree_depth")?.value || "8", 10);
  tree = await fetchTreeSubtree(depth);
  treeMcts = tree?.mcts || null;

  const nodes = tree?.nodes || {};
  for (const nid of Object.keys(nodes)) {
    if (!collapsed.has(nid) && nid !== tree.root_id && nid !== tree.current_id) collapsed.add(nid);
  }
  uncollapsePathToCurrent();
  renderTreeView();
}

function renderTreeView() {
  const elTreeView = $("tree_view");
  if (!tree || !elTreeView) return;

  const nodes = tree.nodes || {};
  const rootId = tree.root_id;
  const currentId = tree.current_id;

  elTreeView.innerHTML = "";

  const walk = (nid, indent) => {
    const n = nodes[nid];
    if (!n) return;

    const row = document.createElement("div");
    row.className = "tree-node";
    row.style.marginLeft = `${indent * 16}px`;
    if (nid === currentId) row.classList.add("current");

    const hasKids = nodeHasAnyKnownChildren(nid);
    const isCol = collapsed.has(nid);

    const caret = document.createElement("span");
    caret.className = "tree-caret";
    if (!hasKids) {
      caret.textContent = " ";
      caret.style.cursor = "default";
    } else {
      caret.textContent = isCol ? "▶" : "▼";
      caret.onclick = (ev) => {
        ev.stopPropagation();
        if (collapsed.has(nid)) collapsed.delete(nid);
        else collapsed.add(nid);
        renderTreeView();
      };
    }

    const idSpan = document.createElement("span");
    idSpan.className = "tree-id";
    idSpan.textContent = nid;

    const label = document.createElement("span");
    label.className = "tree-label";
    label.textContent = n.action ? n.action : "(root)";

    const sum = document.createElement("span");
    sum.className = "tree-sum";
    sum.textContent = fmtNodeSummary(n, nid);

    if (n.parent && n.action) {
      const extra = fmtEdgeStats(n.parent, n.action);
      if (extra) sum.textContent += extra;
    }

    row.appendChild(caret);
    row.appendChild(idSpan);
    row.appendChild(label);
    row.appendChild(sum);

    row.onclick = async () => {
      try {
        state = await setCurrentNode(nid);
        stopAutoReplay();
        renderHUD();
        renderBoards();
        await refreshTree();
      } catch (e) {
        toast(`Load state failed: ${String(e)}`);
      }
    };

    elTreeView.appendChild(row);

    if (hasKids && !collapsed.has(nid)) {
      for (const [, cid] of getChildren(nid)) walk(cid, indent + 1);
    }
  };

  if (collapsed.size === 0) {
    for (const nid of Object.keys(nodes)) collapsed.add(nid);
    collapsed.delete(rootId);
    collapsed.delete(currentId);
    uncollapsePathToCurrent();
  }

  walk(rootId, 0);
}

// -------------------------------
// Inputs + replay + buttons + init
// -------------------------------
async function commit() {
  state = await apiCommit();
  renderHUD();
  renderBoards();
  await refreshTree();
}

function isActionValidForActor(actor, action) {
  if (!state) return true;
  const view = actor === "P1" ? state.p1 : state.p2;
  const valid = validActionSet(view);
  return valid.has(String(action));
}

async function onKeyDown(ev) {
  if (ev.repeat) return;
  const k = ev.key;

  if (k.startsWith("Arrow") || k === " " || k === "Space") ev.preventDefault();

  try {
    let actor = null;
    let action = null;

    if (k === "ArrowUp") { actor = "P1"; action = "MOVE_UP"; }
    else if (k === "ArrowDown") { actor = "P1"; action = "MOVE_DOWN"; }
    else if (k === "ArrowLeft") { actor = "P1"; action = "MOVE_LEFT"; }
    else if (k === "ArrowRight") { actor = "P1"; action = "MOVE_RIGHT"; }
    else if (k === "z" || k === "Z") { actor = "P1"; action = "SHOOT"; }
    else if (k === "c" || k === "C") { actor = "P1"; action = "RELEASE_CHARGE"; }
    else if (k === "x" || k === "X") { actor = "P1"; action = "TOGGLE_HOLD"; }

    else if (k === "w" || k === "W") { actor = "P2"; action = "MOVE_UP"; }
    else if (k === "s" || k === "S") { actor = "P2"; action = "MOVE_DOWN"; }
    else if (k === "a" || k === "A") { actor = "P2"; action = "MOVE_LEFT"; }
    else if (k === "d" || k === "D") { actor = "P2"; action = "MOVE_RIGHT"; }
    else if (k === "f" || k === "F") { actor = "P2"; action = "SHOOT"; }
    else if (k === "g" || k === "G") { actor = "P2"; action = "RELEASE_CHARGE"; }
    else if (k === "r" || k === "R") { actor = "P2"; action = "TOGGLE_HOLD"; }

    else if (k === " " || k === "Space") {
      await commit();
      return;
    } else {
      return;
    }

    if (actor && action && action !== "TOGGLE_HOLD") {
      if (!isActionValidForActor(actor, action)) {
        toast(`Invalid now: ${actor} ${action}`);
        return;
      }
    }

    state = await apiUi(actor, action);
    stopAutoReplay();
    renderHUD();
    renderBoards();
  } catch (e) {
    toast(String(e));
  }
}

function stopAutoReplay() {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
}

async function autoReplayTick() {
  try {
    state = await apiPlanReplayStep();
    renderHUD();
    renderBoards();
    await refreshTree();

    const p = state?.plan;
    if (!p || !p.has_plan) {
      stopAutoReplay();
      return;
    }
    const idx = p.replay_index ?? 0;
    const len = p.replay_len ?? 0;
    if (idx >= len) {
      stopAutoReplay();
      toast("Replay complete");
    }
  } catch (e) {
    stopAutoReplay();
    toast(String(e));
  }
}

function hookButtons() {
  const hook = (id, fn) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("click", async () => {
      try { await fn(); } catch (e) { toast(String(e)); }
    });
  };

  // P1
  hook("btn_p1_hold_on", async () => { state = await apiUi("P1", "HOLD_ON"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p1_hold_off", async () => { state = await apiUi("P1", "HOLD_OFF"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p1_toggle", async () => { state = await apiUi("P1", "TOGGLE_HOLD"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p1_shoot", async () => { state = await apiUi("P1", "SHOOT"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p1_release", async () => { state = await apiUi("P1", "RELEASE_CHARGE"); stopAutoReplay(); renderHUD(); renderBoards(); });

  // P2
  hook("btn_p2_hold_on", async () => { state = await apiUi("P2", "HOLD_ON"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p2_hold_off", async () => { state = await apiUi("P2", "HOLD_OFF"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p2_toggle", async () => { state = await apiUi("P2", "TOGGLE_HOLD"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p2_shoot", async () => { state = await apiUi("P2", "SHOOT"); stopAutoReplay(); renderHUD(); renderBoards(); });
  hook("btn_p2_release", async () => { state = await apiUi("P2", "RELEASE_CHARGE"); stopAutoReplay(); renderHUD(); renderBoards(); });

  // Commit/reset
  hook("btn_commit", async () => { stopAutoReplay(); await commit(); });
  hook("btn_reset", async () => { stopAutoReplay(); state = await apiReset(); renderHUD(); renderBoards(); await refreshTree(); });

  // Tree
  hook("btn_tree_refresh", async () => { await refreshTree(); });
  hook("btn_tree_collapse_all", async () => {
    if (!tree || !tree.nodes) return;
    collapsed.clear();
    for (const nid of Object.keys(tree.nodes)) collapsed.add(nid);
    collapsed.delete(tree.root_id);
    collapsed.delete(tree.current_id);
    uncollapsePathToCurrent();
    renderTreeView();
  });

  // MCTS
  hook("btn_mcts_run", async () => {
    const iters = parseInt($("mcts_iters")?.value || "600", 10);
    const depth = parseInt($("mcts_depth")?.value || "12", 10);
    state = await apiMctsRun(iters, depth);
    toast(`MCTS ran: ${iters} iters depth ${depth}`);
    stopAutoReplay();
    renderHUD();
    renderBoards();
    await refreshTree();
  });

  hook("btn_mcts_apply_p1", async () => {
    state = await apiMctsApply("P1");
    toast(`Applied P1: ${state?.mcts_applied?.p1 ?? "?"}`);
    stopAutoReplay();
    renderHUD(); renderBoards();
  });
  hook("btn_mcts_apply_p2", async () => {
    state = await apiMctsApply("P2");
    toast(`Applied P2: ${state?.mcts_applied?.p2 ?? "?"}`);
    stopAutoReplay();
    renderHUD(); renderBoards();
  });
  hook("btn_mcts_apply_both", async () => {
    state = await apiMctsApply("BOTH");
    toast(`Applied Both: P1=${state?.mcts_applied?.p1 ?? "?"} P2=${state?.mcts_applied?.p2 ?? "?"}`);
    stopAutoReplay();
    renderHUD(); renderBoards();
  });

  // Plan
  hook("btn_plan_run", async () => {
    const iters = parseInt($("plan_iters")?.value || "800", 10);
    const look = parseInt($("plan_look")?.value || "16", 10);
    stopAutoReplay();
    state = await apiPlanRun(iters, look);
    toast(`Planned to 64 (iters/step ${iters}, lookahead ${look})`);
    renderHUD();
    renderBoards();
    await refreshTree();
  });

  hook("btn_plan_replay_reset", async () => {
    stopAutoReplay();
    state = await apiPlanReplayReset();
    toast("Replay reset");
    renderHUD(); renderBoards();
    await refreshTree();
  });

  hook("btn_plan_replay_step", async () => {
    stopAutoReplay();
    state = await apiPlanReplayStep();
    renderHUD(); renderBoards();
    await refreshTree();
  });

  hook("btn_plan_replay_auto", async () => {
    stopAutoReplay();
    replayTimer = setInterval(autoReplayTick, 160);
    toast("Auto replay started");
  });

  hook("btn_plan_replay_stop", async () => {
    stopAutoReplay();
    toast("Stopped");
  });

  // Slider labels
  const depth = $("tree_depth");
  const depthVal = $("tree_depth_val");
  if (depth && depthVal) {
    depthVal.textContent = depth.value;
    depth.addEventListener("input", () => depthVal.textContent = depth.value);
    depth.addEventListener("change", async () => { await refreshTree(); });
  }

  const mi = $("mcts_iters");
  const miV = $("mcts_iters_val");
  if (mi && miV) {
    miV.textContent = mi.value;
    mi.addEventListener("input", () => miV.textContent = mi.value);
  }

  const md = $("mcts_depth");
  const mdV = $("mcts_depth_val");
  if (md && mdV) {
    mdV.textContent = md.value;
    md.addEventListener("input", () => mdV.textContent = md.value);
  }

  const pi = $("plan_iters");
  const piV = $("plan_iters_val");
  if (pi && piV) {
    piV.textContent = pi.value;
    pi.addEventListener("input", () => piV.textContent = pi.value);
  }

  const pl = $("plan_look");
  const plV = $("plan_look_val");
  if (pl && plV) {
    plV.textContent = pl.value;
    pl.addEventListener("input", () => plV.textContent = pl.value);
  }
}

async function init() {
  try {
    state = await apiState();
    renderHUD();
    renderBoards();
    await refreshTree();
  } catch (e) {
    toast(`Init failed: ${String(e)}`);
  }

  hookButtons();
  window.addEventListener("keydown", onKeyDown, { passive: false });
  window.addEventListener("resize", () => renderBoards());

  setTimeout(() => renderBoards(), 50);
}

init();
