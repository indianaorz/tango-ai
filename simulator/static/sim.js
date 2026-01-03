"use strict";

/**
 * static/sim.js — COMPLETE FILE
 *
 * Fixes (for real this time):
 *  - Clicking a node NEVER changes collapse state.
 *  - After selecting a node, we automatically un-collapse the path root -> current
 *    so it can’t “look like the tree collapsed” just because the selected node
 *    (or its parents) were in the collapsed set.
 *  - Nodes with ZERO known children show NO chevron.
 *
 * Notes:
 *  - This viewer is UI-only. It does NOT generate children on click.
 *
 * Endpoints:
 *   GET  /api/state
 *   POST /api/key_event   { key, is_down }
 *   POST /api/ui_action   { action }
 *
 * Tree endpoints:
 *   GET  /api/tree/subtree?node_id=ROOT&depth=8
 *   POST /api/tree/set_current { node_id }
 */

function $(id) {
  return document.getElementById(id);
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

// -------------------------------
// Elements
// -------------------------------
const elGrid = $("grid");
const elOverlaySvg = $("overlay");

const elToast = $("toast");

const elCust = $("cust");
const elLock = $("lock");
const elPPos = $("p_pos");
const elEPos = $("e_pos");
const elPending = $("pending");
const elLastAction = $("last_action");
const elEvents = $("events");

const elPHP = $("p_hp");
const elEHP = $("e_hp");

const elPChg = $("p_chg");
const elPHold = $("p_hold");
const elPBar = $("p_charge_bar");
const elPProgTxt = $("p_prog_txt");

const btnPToggle = $("btn_p_toggle");
const btnPHoldOn = $("btn_p_hold_on");
const btnPHoldOff = $("btn_p_hold_off");
const btnPShoot = $("btn_p_shoot");
const btnPRelease = $("btn_p_release");

// Tree UI
const elTreeView = $("tree_view");
const elTreeCurrent = $("tree_current");
const elTreeDepth = $("tree_depth");
const elTreeDepthVal = $("tree_depth_val");
const btnTreeCollapseAll = $("btn_tree_collapse_all");
const btnTreeRefresh = $("btn_tree_refresh");

// -------------------------------
// Client state
// -------------------------------
let state = null;
let tree = null;

// collapse state is purely client-side UI state
const collapsed = new Set();

// shot overlay line
let shotLine = null;

// -------------------------------
// Toast
// -------------------------------
function toast(msg) {
  if (!elToast) return;
  elToast.textContent = msg || "";
  if (msg) {
    setTimeout(() => {
      if (elToast.textContent === msg) elToast.textContent = "";
    }, 1200);
  }
}

// -------------------------------
// Fetch helpers
// -------------------------------
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (j && j.error) throw new Error(j.error);
  return j;
}

async function fetchState() {
  return fetchJSON("/api/state");
}

async function sendKeyEvent(key, isDown) {
  return fetchJSON("/api/key_event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: key, is_down: isDown }),
  });
}

async function sendUiAction(action) {
  return fetchJSON("/api/ui_action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
}

// Tree endpoints
async function fetchTreeSubtree(nodeId, depth) {
  const qs = new URLSearchParams({
    node_id: nodeId || "ROOT",
    depth: String(depth ?? 8),
  });
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

  if (isLocked) el.classList.add("locked");
  else el.classList.remove("locked");

  const filled = Math.round(frac * segCount);
  for (let i = 0; i < segCount; i++) {
    if (i < filled) segs[i].classList.add("on");
    else segs[i].classList.remove("on");
  }
}

// -------------------------------
// HUD render
// -------------------------------
function renderHUD() {
  if (!state) return;

  if (elCust) elCust.textContent = String(state.cust_gauge ?? 0);
  if (elLock)
    elLock.textContent = state.is_locked
      ? `YES (${state.lock_remaining ?? 0})`
      : "NO";

  const prc = state.p_rc || [0, 0];
  const erc = state.e_rc || [0, 0];
  if (elPPos) elPPos.textContent = `r${prc[0]},c${prc[1]}`;
  if (elEPos) elEPos.textContent = `r${erc[0]},c${erc[1]}`;

  if (elPending)
    elPending.textContent = state.pending_action
      ? String(state.pending_action)
      : "None";
  if (elLastAction)
    elLastAction.textContent = state.last_action_started
      ? String(state.last_action_started)
      : "None";

  if (elPHP) elPHP.textContent = String(state.p_hp ?? 0);
  if (elEHP) elEHP.textContent = String(state.e_hp ?? 0);

  if (elPChg) elPChg.textContent = String(state.p_charge_level ?? 0);
  if (elPHold) elPHold.textContent = state.p_charge_hold ? "ON" : "OFF";

  const fullAt = state.charge_full_at ?? 5;
  if (elPProgTxt)
    elPProgTxt.textContent = `${state.p_charge_progress ?? 0}/${fullAt}`;
  renderChargeBar(elPBar, state.p_charge_progress ?? 0, fullAt, !!state.is_locked);

  if (elEvents && Array.isArray(state.last_events)) {
    elEvents.innerHTML = "";
    for (const line of state.last_events.slice().reverse()) {
      const div = document.createElement("div");
      div.textContent = line;
      elEvents.appendChild(div);
    }
  }

  if (elTreeCurrent && state.tree && state.tree.current_id) {
    elTreeCurrent.textContent = String(state.tree.current_id);
  }
}

// -------------------------------
// Visual helpers for lean/enter
// -------------------------------
function dirToUnit(dir) {
  switch (dir) {
    case "UP":
      return { x: 0, y: -1 };
    case "DOWN":
      return { x: 0, y: 1 };
    case "LEFT":
      return { x: -1, y: 0 };
    case "RIGHT":
      return { x: 1, y: 0 };
    default:
      return { x: 0, y: 0 };
  }
}

function computeEntityOffset(actor /* "p"|"e" */) {
  const LEAN_PX = 8;
  const ENTER_PX = 6;

  const vis = actor === "p" ? state?.p_visual : state?.e_visual;
  if (vis && typeof vis === "object") {
    const phase = vis.phase || "idle";
    const dir = vis.dir || null;
    const t = clamp01(vis.t ?? 0);
    const u = dirToUnit(dir);

    if (phase === "leave") {
      return { dx: u.x * LEAN_PX * t, dy: u.y * LEAN_PX * t };
    }
    if (phase === "enter") {
      return { dx: -u.x * ENTER_PX * (1 - t), dy: -u.y * ENTER_PX * (1 - t) };
    }
    return { dx: 0, dy: 0 };
  }

  if (actor === "p" && typeof state?.pending_action === "string") {
    const m = state.pending_action.match(/^MOVE_(UP|DOWN|LEFT|RIGHT)$/);
    if (m) {
      const u = dirToUnit(m[1]);
      return { dx: u.x * (LEAN_PX * 0.6), dy: u.y * (LEAN_PX * 0.6) };
    }
  }

  return { dx: 0, dy: 0 };
}

// -------------------------------
// Grid render + entities
// -------------------------------
function renderGridCells() {
  if (!state || !elGrid) return;

  const owners = state.grid_owner_state || [];
  const tiles = state.grid_state || [];

  elGrid.innerHTML = "";

  for (let i = 0; i < owners.length; i++) {
    const d = document.createElement("div");
    d.className = `cell tile-${tiles[i] ?? 2} p${owners[i] ?? 0}`;
    elGrid.appendChild(d);
  }

  const pIdx = state.p_idx ?? 0;
  const eIdx = state.e_idx ?? 0;

  const pCell = elGrid.children[pIdx];
  const eCell = elGrid.children[eIdx];

  if (pCell) {
    const ent = document.createElement("div");
    ent.className = "entity ent-p";

    const pChg = state.p_charge_level ?? 0;
    if (pChg === 1) ent.classList.add("aura-chg1");
    if (pChg === 2) ent.classList.add("aura-chg2");

    const { dx, dy } = computeEntityOffset("p");
    ent.style.transform = `translate(-50%, -50%) translate(${dx}px, ${dy}px)`;
    pCell.appendChild(ent);
  }

  if (eCell) {
    const ent = document.createElement("div");
    ent.className = "entity ent-e";

    const eChg = state.e_charge_level ?? 0;
    if (eChg === 1) ent.classList.add("aura-chg1");
    if (eChg === 2) ent.classList.add("aura-chg2");

    const { dx, dy } = computeEntityOffset("e");
    ent.style.transform = `translate(-50%, -50%) translate(${dx}px, ${dy}px)`;
    eCell.appendChild(ent);
  }

  renderShotOverlay();
}

// -------------------------------
// Shot overlay (SVG line)
// -------------------------------
function ensureShotLine() {
  if (!elOverlaySvg) return;
  if (shotLine) return;

  elOverlaySvg.innerHTML = "";
  shotLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  shotLine.setAttribute("x1", "0");
  shotLine.setAttribute("y1", "0");
  shotLine.setAttribute("x2", "0");
  shotLine.setAttribute("y2", "0");
  shotLine.setAttribute("stroke-width", "3");
  shotLine.setAttribute("stroke-linecap", "round");
  shotLine.setAttribute("opacity", "0");
  elOverlaySvg.appendChild(shotLine);
}

function cellCenterInOverlay(idx) {
  const cell = elGrid?.children?.[idx];
  if (!cell) return { x: 0, y: 0 };

  const cellRect = cell.getBoundingClientRect();
  const gridRect = elGrid.getBoundingClientRect();

  return {
    x: cellRect.left - gridRect.left + cellRect.width / 2,
    y: cellRect.top - gridRect.top + cellRect.height / 2,
  };
}

function renderShotOverlay() {
  if (!elOverlaySvg || !elGrid) return;

  const gridRect = elGrid.getBoundingClientRect();
  elOverlaySvg.setAttribute("viewBox", `0 0 ${gridRect.width} ${gridRect.height}`);

  ensureShotLine();
  if (!shotLine) return;

  const shot = state?.shot_line || null;
  if (!shot || typeof shot !== "object") {
    shotLine.setAttribute("opacity", "0");
    return;
  }

  const fromIdx = Number.isFinite(shot.from_idx) ? shot.from_idx : null;
  const toIdx = Number.isFinite(shot.to_idx) ? shot.to_idx : null;
  if (fromIdx == null || toIdx == null) {
    shotLine.setAttribute("opacity", "0");
    return;
  }

  const a = cellCenterInOverlay(fromIdx);
  const b = cellCenterInOverlay(toIdx);

  shotLine.setAttribute("x1", String(a.x));
  shotLine.setAttribute("y1", String(a.y));
  shotLine.setAttribute("x2", String(b.x));
  shotLine.setAttribute("y2", String(b.y));

  const isCharged = shot.kind === "charge";
  shotLine.setAttribute(
    "stroke",
    isCharged ? "rgba(255, 80, 220, 0.95)" : "rgba(255, 235, 80, 0.95)"
  );
  shotLine.setAttribute("opacity", "1");
}

// -------------------------------
// Tree helpers
// -------------------------------
function fmtNodeSummary(n) {
  const s = n.s;
  const lock = s.locked ? `L${s.lock_rem}` : `L0`;
  const p = `P(${s.p_rc[0]},${s.p_rc[1]})`;
  const e = `E(${s.e_rc[0]},${s.e_rc[1]})`;
  const hp = `${s.p_hp}/${s.e_hp}`;
  const chg = `C${s.p_chg}${s.p_hold ? "H" : ""}`;
  return `cust${s.cust} HP ${hp} ${p} ${e} ${lock} ${chg}`;
}

function getChildren(nid) {
  if (!tree || !tree.edges) return [];
  const kids = tree.edges[nid] || {};
  return Object.entries(kids).sort((a, b) => a[0].localeCompare(b[0]));
}

function nodeHasAnyKnownChildren(nid) {
  return getChildren(nid).length > 0;
}

/**
 * Ensures the selection is visible:
 * uncollapse root -> ... -> current
 */
function uncollapsePathToCurrent() {
  if (!tree || !tree.nodes) return;

  const nodes = tree.nodes;
  const rootId = tree.root_id;
  const currentId = tree.current_id;

  // Always keep root & current expanded
  if (rootId) collapsed.delete(rootId);
  if (currentId) collapsed.delete(currentId);

  // Walk parents using `parent` field from server node summary
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

// -------------------------------
// Tree refresh + render
// -------------------------------
async function refreshTree() {
  if (!elTreeView) return;
  const depth = parseInt(elTreeDepth?.value || "8", 10);

  try {
    tree = await fetchTreeSubtree("ROOT", depth);

    // Only add *new* nodes to collapsed (never toggle existing ones here).
    const nodes = tree?.nodes || {};
    for (const nid of Object.keys(nodes)) {
      if (!collapsed.has(nid) && nid !== tree.root_id && nid !== tree.current_id) {
        collapsed.add(nid);
      }
    }

    // Critical: selection must not "collapse the tree"
    uncollapsePathToCurrent();

    renderTreeView();
  } catch (e) {
    toast(`Tree refresh failed: ${String(e)}`);
  }
}

function renderTreeView() {
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

    // Caret: ONLY if known children exist
    const caret = document.createElement("span");
    caret.className = "tree-caret";

    if (!hasKids) {
      caret.textContent = " ";
      caret.style.cursor = "default";
    } else {
      caret.textContent = isCollapsed ? "▶" : "▼";
      caret.style.cursor = "pointer";
      caret.onclick = (ev) => {
        ev.stopPropagation(); // caret click never selects
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
    sum.textContent = fmtNodeSummary(n);

    row.appendChild(caret);
    row.appendChild(idSpan);
    row.appendChild(label);
    row.appendChild(sum);

    // Selecting a node NEVER mutates `collapsed`.
    row.onclick = async () => {
      try {
        state = await setCurrentNode(nid);
        renderHUD();
        renderGridCells();
        await refreshTree();
      } catch (e) {
        toast(`Load state failed: ${String(e)}`);
      }
    };

    elTreeView.appendChild(row);

    if (hasKids && !collapsed.has(nid)) {
      const kids = getChildren(nid);
      for (const [, cid] of kids) {
        if (!nodes[cid]) continue;
        walk(cid, indent + 1);
      }
    }
  };

  // first render fallback (collapse everything except root/current path)
  if (collapsed.size === 0) {
    for (const nid of Object.keys(nodes)) collapsed.add(nid);
    collapsed.delete(rootId);
    collapsed.delete(currentId);
    uncollapsePathToCurrent();
  }

  walk(rootId, 0);
}

// -------------------------------
// Input forwarding
// -------------------------------
async function onKeyDown(ev) {
  if (ev.repeat) return;

  if (ev.key.startsWith("Arrow") || ev.key === " ") ev.preventDefault();

  try {
    state = await sendKeyEvent(ev.key, true);
    renderHUD();
    renderGridCells();

    if (ev.key === " " || ev.key === "Space") {
      await refreshTree();
    }
  } catch (e) {
    toast(String(e));
  }
}

async function onKeyUp(ev) {
  try {
    state = await sendKeyEvent(ev.key, false);
    renderHUD();
    renderGridCells();
  } catch (e) {
    toast(String(e));
  }
}

// -------------------------------
// Buttons
// -------------------------------
function hookButtons() {
  const hook = (el, action) => {
    if (!el) return;
    el.addEventListener("click", async () => {
      try {
        state = await sendUiAction(action);
        renderHUD();
        renderGridCells();
        await refreshTree();
      } catch (e) {
        toast(String(e));
      }
    });
  };

  hook(btnPToggle, "P_TOGGLE_CHARGE");
  hook(btnPHoldOn, "P_HOLD_ON");
  hook(btnPHoldOff, "P_HOLD_OFF");
  hook(btnPShoot, "P_SHOOT");
  hook(btnPRelease, "P_RELEASE_CHARGE");

  if (btnTreeCollapseAll) {
    btnTreeCollapseAll.addEventListener("click", () => {
      if (!tree || !tree.nodes) return;
      collapsed.clear();
      for (const nid of Object.keys(tree.nodes)) collapsed.add(nid);
      collapsed.delete(tree.root_id);
      collapsed.delete(tree.current_id);
      uncollapsePathToCurrent();
      renderTreeView();
    });
  }

  if (btnTreeRefresh) {
    btnTreeRefresh.addEventListener("click", async () => {
      await refreshTree();
    });
  }

  if (elTreeDepth && elTreeDepthVal) {
    elTreeDepthVal.textContent = elTreeDepth.value;
    elTreeDepth.addEventListener("input", () => {
      elTreeDepthVal.textContent = elTreeDepth.value;
    });
    elTreeDepth.addEventListener("change", async () => {
      await refreshTree();
    });
  }
}

// -------------------------------
// Init
// -------------------------------
async function init() {
  try {
    state = await fetchState();
    renderHUD();
    renderGridCells();
    await refreshTree();
  } catch (e) {
    toast(`Init failed: ${String(e)}`);
  }

  hookButtons();

  window.addEventListener("keydown", onKeyDown, { passive: false });
  window.addEventListener("keyup", onKeyUp);

  window.addEventListener("resize", () => {
    if (!state) return;
    renderGridCells();
  });

  setTimeout(() => {
    if (!state) return;
    renderGridCells();
  }, 50);
}

init();
