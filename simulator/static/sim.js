// mmbn_sim/static/sim.js
"use strict";

function $(id) { return document.getElementById(id); }

function toast(msg) {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg || "";
  if (msg) setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 1200);
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
let state = null; // {p1, p2, tree}
let tree = null;
const collapsed = new Set();

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
    if (i < filled) segs[i].classList.add("on"); else segs[i].classList.remove("on");
  }
}

// -------------------------------
// HUD render
// -------------------------------
function renderHUD() {
  if (!state) return;

  $("cust").textContent = String(state.p1.cust ?? 0);
  $("tree_current").textContent = String(state.tree?.current_id ?? "?");

  renderSideHUD("p1", state.p1);
  renderSideHUD("p2", state.p2);

  const eventsEl = $("events");
  const lines = (state.p1.last_events || []);
  eventsEl.innerHTML = "";
  for (const line of lines.slice().reverse()) {
    const d = document.createElement("div");
    d.textContent = line;
    eventsEl.appendChild(d);
  }
}

function renderSideHUD(prefix, v) {
  $(`${prefix}_lock`).textContent = v.is_locked ? `YES (${v.lock_remaining ?? 0})` : "NO";
  $(`${prefix}_pos`).textContent = `r${v.p_rc[0]},c${v.p_rc[1]}`;
  $(`${prefix}_enemy_pos`).textContent = `r${v.e_rc[0]},c${v.e_rc[1]}`;
  $(`${prefix}_pending`).textContent = v.pending_action ? String(v.pending_action) : "None";
  $(`${prefix}_hp`).textContent = String(v.p_hp ?? 0);
  $(`${prefix}_ehp`).textContent = String(v.e_hp ?? 0);

  $(`${prefix}_hold`).textContent = v.p_charge_hold ? "ON" : "OFF";
  const fullAt = v.charge_full_at ?? 5;
  $(`${prefix}_prog_txt`).textContent = `${v.p_charge_progress ?? 0}/${fullAt}`;
  renderChargeBar($(`${prefix}_charge_bar`), v.p_charge_progress ?? 0, fullAt, !!v.is_locked);
}

// -------------------------------
// Grid render + entities + shot overlay
// -------------------------------
function cellCenterInOverlay(gridEl, idx) {
  const cell = gridEl?.children?.[idx];
  if (!cell) return { x: 0, y: 0 };
  const cellRect = cell.getBoundingClientRect();
  const gridRect = gridEl.getBoundingClientRect();
  return {
    x: cellRect.left - gridRect.left + cellRect.width / 2,
    y: cellRect.top - gridRect.top + cellRect.height / 2,
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
  // Tune these to taste
  const LEAN_MAG = 10;
  const ENTRY_MAG = 6;

  const a = dirToOffset(leanDir, LEAN_MAG);
  const b = isEntering ? dirToOffset(entryDir, ENTRY_MAG) : { dx: 0, dy: 0 };

  const dx = a.dx + b.dx;
  const dy = a.dy + b.dy;

  // Base center transform plus directional offsets.
  entEl.style.transform = `translate(-50%, -50%) translate(${dx}px, ${dy}px)`;
}


function renderGrid(view, gridId, overlayId) {
  const gridEl = $(gridId);
  const overlayEl = $(overlayId);
  if (!gridEl || !overlayEl) return;

  const owners = view.grid_owner_state || [];
  const tiles = view.grid_state || [];

  gridEl.innerHTML = "";
  for (let i = 0; i < owners.length; i++) {
    const d = document.createElement("div");
    d.className = `cell tile-${tiles[i] ?? 2} p${owners[i] ?? 0}`;
    gridEl.appendChild(d);
  }

  const pCell = gridEl.children[view.p_idx ?? 0];
  const eCell = gridEl.children[view.e_idx ?? 0];

  if (pCell) {
    const ent = document.createElement("div");
    ent.className = "entity ent-p";

    const lvl = view.p_charge_level ?? 0;
    if (lvl === 1) ent.classList.add("aura-chg1");
    if (lvl === 2) ent.classList.add("aura-chg2");

    applyEntityTransform(
      ent,
      view.p_lean_dir,
      view.p_entry_dir,
      !!view.p_is_entering
    );

    pCell.appendChild(ent);
  }


  if (eCell) {
    const ent = document.createElement("div");
    ent.className = "entity ent-e";

    const lvl = view.e_charge_level ?? 0;
    if (lvl === 1) ent.classList.add("aura-chg1");
    if (lvl === 2) ent.classList.add("aura-chg2");

    applyEntityTransform(
      ent,
      view.e_lean_dir,
      view.e_entry_dir,
      !!view.e_is_entering
    );

    eCell.appendChild(ent);
  }


  // Overlay shot lines (may be multiple)
  const rect = gridEl.getBoundingClientRect();
  overlayEl.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);
  overlayEl.innerHTML = "";

  const shots = view.shot_lines || [];
  for (const sh of shots) {
    const a = cellCenterInOverlay(gridEl, sh.from_idx);
    const b = cellCenterInOverlay(gridEl, sh.to_idx);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(a.x));
    line.setAttribute("y1", String(a.y));
    line.setAttribute("x2", String(b.x));
    line.setAttribute("y2", String(b.y));
    line.setAttribute("stroke-width", "3");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("opacity", "1");

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
// Tree
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

function fmtNodeSummary(n) {
  const s = n.s;
  return `cust${s.cust} hp ${s.p1_hp}/${s.p2_hp} L(${s.p1_locked ? 1 : 0},${s.p2_locked ? 1 : 0})`;
}

async function refreshTree() {
  const depth = parseInt($("tree_depth")?.value || "8", 10);
  tree = await fetchTreeSubtree(depth);

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
    const isCollapsed = collapsed.has(nid);

    const caret = document.createElement("span");
    caret.className = "tree-caret";
    if (!hasKids) {
      caret.textContent = " ";
      caret.style.cursor = "default";
    } else {
      caret.textContent = isCollapsed ? "▶" : "▼";
      caret.onclick = (ev) => {
        ev.stopPropagation();
        if (collapsed.has(nid)) collapsed.delete(nid); else collapsed.add(nid);
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
    sum.textContent = fmtNodeSummary(n);

    row.appendChild(caret);
    row.appendChild(idSpan);
    row.appendChild(label);
    row.appendChild(sum);

    row.onclick = async () => {
      try {
        state = await setCurrentNode(nid);
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
// Inputs
// -------------------------------
async function commit() {
  state = await apiCommit();
  renderHUD();
  renderBoards();
  await refreshTree();
}

async function onKeyDown(ev) {
  if (ev.repeat) return;

  const k = ev.key;

  // Prevent scrolling on these
  if (k.startsWith("Arrow") || k === " " || k === "Space") ev.preventDefault();

  try {
    // P1
    if (k === "ArrowUp") state = await apiUi("P1", "MOVE_UP");
    else if (k === "ArrowDown") state = await apiUi("P1", "MOVE_DOWN");
    else if (k === "ArrowLeft") state = await apiUi("P1", "MOVE_LEFT");
    else if (k === "ArrowRight") state = await apiUi("P1", "MOVE_RIGHT");
    else if (k === "z" || k === "Z") state = await apiUi("P1", "SHOOT");
    else if (k === "c" || k === "C") state = await apiUi("P1", "RELEASE_CHARGE");
    else if (k === "x" || k === "X") state = await apiUi("P1", "TOGGLE_HOLD");

    // P2
    else if (k === "w" || k === "W") state = await apiUi("P2", "MOVE_UP");
    else if (k === "s" || k === "S") state = await apiUi("P2", "MOVE_DOWN");
    else if (k === "a" || k === "A") state = await apiUi("P2", "MOVE_LEFT");
    else if (k === "d" || k === "D") state = await apiUi("P2", "MOVE_RIGHT");
    else if (k === "f" || k === "F") state = await apiUi("P2", "SHOOT");
    else if (k === "g" || k === "G") state = await apiUi("P2", "RELEASE_CHARGE");
    else if (k === "r" || k === "R") state = await apiUi("P2", "TOGGLE_HOLD");

    // Commit
    else if (k === " " || k === "Space") {
      await commit();
      return;
    } else {
      return;
    }

    renderHUD();
    renderBoards();
  } catch (e) {
    toast(String(e));
  }
}

// -------------------------------
// Buttons
// -------------------------------
function hookButtons() {
  const hook = (id, fn) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("click", async () => {
      try { await fn(); } catch (e) { toast(String(e)); }
    });
  };

  hook("btn_p1_hold_on", async () => { state = await apiUi("P1", "HOLD_ON"); renderHUD(); renderBoards(); });
  hook("btn_p1_hold_off", async () => { state = await apiUi("P1", "HOLD_OFF"); renderHUD(); renderBoards(); });
  hook("btn_p1_toggle", async () => { state = await apiUi("P1", "TOGGLE_HOLD"); renderHUD(); renderBoards(); });
  hook("btn_p1_shoot", async () => { state = await apiUi("P1", "SHOOT"); renderHUD(); renderBoards(); });
  hook("btn_p1_release", async () => { state = await apiUi("P1", "RELEASE_CHARGE"); renderHUD(); renderBoards(); });

  hook("btn_p2_hold_on", async () => { state = await apiUi("P2", "HOLD_ON"); renderHUD(); renderBoards(); });
  hook("btn_p2_hold_off", async () => { state = await apiUi("P2", "HOLD_OFF"); renderHUD(); renderBoards(); });
  hook("btn_p2_toggle", async () => { state = await apiUi("P2", "TOGGLE_HOLD"); renderHUD(); renderBoards(); });
  hook("btn_p2_shoot", async () => { state = await apiUi("P2", "SHOOT"); renderHUD(); renderBoards(); });
  hook("btn_p2_release", async () => { state = await apiUi("P2", "RELEASE_CHARGE"); renderHUD(); renderBoards(); });

  hook("btn_commit", async () => { await commit(); });
  hook("btn_reset", async () => { state = await apiReset(); renderHUD(); renderBoards(); await refreshTree(); });

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

  const depth = $("tree_depth");
  const depthVal = $("tree_depth_val");
  if (depth && depthVal) {
    depthVal.textContent = depth.value;
    depth.addEventListener("input", () => depthVal.textContent = depth.value);
    depth.addEventListener("change", async () => { await refreshTree(); });
  }
}

// -------------------------------
// Init
// -------------------------------
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
