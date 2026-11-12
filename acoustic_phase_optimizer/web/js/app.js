// Entry point: top bar, start-up, keyboard shortcuts.

import { api } from './api.js';
import {
  canRedo, canUndo, commit, duplicateSpeaker, editSpeaker, flushSession, forgetSaved, loadProject, markSaved, mutate, on, redo, removeMic,
  removeSpeaker, select, setUI, speakerById, state, themeChanged, undo,
} from './store.js';
import { clamp, clear, debounce, download, h, icon, toast } from './util.js';
import { ask, currentDialog, openDialog } from './dialog.js';
import { isDark, toggleTheme } from './theme.js';
import { createStage } from './stage.js';
import { nudge as nudgeStep, nudgeLabel } from './units.js';
import { createStageBar } from './stagebar.js';
import { createChart } from './chart.js';
import { createLeftPanel } from './panels.js';
import { createResultsPanel } from './results.js';
import { enableDrop } from './drop.js';

const $ = (id) => document.getElementById(id);

// -------------------------------------------------------------------- menus
let openMenu = null;
function closeMenu() {
  if (!openMenu) return;
  openMenu.menu.remove();
  openMenu.button.setAttribute('aria-expanded', 'false');
  openMenu = null;
}
document.addEventListener('pointerdown', (e) => {
  if (openMenu && !openMenu.wrap.contains(e.target)) closeMenu();
});

/** A button that opens a menu whose items are built fresh (possibly async) each time. */
function menuButton(content, build, { title, align = 'left', cls = 'btn ghost' } = {}) {
  const wrap = h('div', { class: 'menu-wrap' });
  const button = h('button', { class: cls, title, attrs: { 'aria-haspopup': 'menu', 'aria-expanded': 'false' } }, content);
  button.addEventListener('click', async () => {
    const wasOpen = openMenu?.wrap === wrap;
    closeMenu();
    if (wasOpen) return;
    const menu = h('div', { class: 'menu' + (align === 'right' ? ' align-end' : ''), attrs: { role: 'menu' } });
    wrap.append(menu);
    openMenu = { wrap, menu, button };
    button.setAttribute('aria-expanded', 'true');
    menu.append(h('div', { class: 'empty', text: 'Loading…' }));
    try {
      const items = await build();
      if (openMenu?.menu !== menu) return;
      clear(menu).append(...items.filter(Boolean));
      menu.querySelector('.menu-item')?.focus();
    } catch (e) {
      clear(menu).append(h('div', { class: 'empty', text: e.message }));
    }
  });
  wrap.append(button);
  return wrap;
}

const item = (label, run, { hint = null, ico = null, danger = false } = {}) =>
  h('button', { class: 'menu-item', attrs: { role: 'menuitem' }, style: danger ? { color: 'var(--down)' } : null,
    onclick: () => { closeMenu(); run(); } },
  ico ? icon(ico) : null, h('span', { class: 'grow', text: label }), hint ? h('small', { text: hint }) : null);

// --------------------------------------------------------------- file actions
async function confirmDiscard() {
  return !state.dirty || ask('This venue has changes that are not saved. If you continue they will be lost.',
    { title: 'Discard changes?', yes: 'Discard changes', danger: true });
}

async function openVenue(id) {
  if (!(await confirmDiscard())) return;
  try {
    loadProject(await api.venue(id));
    toast(id === 'blank' ? 'Started an empty room.' : `Loaded the ${state.project.name} example.`);
  } catch (e) { toast(`Could not load the venue: ${e.message}`, { error: true }); }
}

async function openSaved(key) {
  if (!(await confirmDiscard())) return;
  try {
    loadProject(await api.loadProject(key), { saved: true });
    toast(`Opened ${state.project.name}.`);
  } catch (e) { toast(`Could not open it: ${e.message}`, { error: true }); }
}

async function saveVenue() {
  commit();
  const name = state.project.name.trim() || 'Untitled venue';
  if (name !== state.savedName) {
    try {
      const existing = await api.projects();
      const clash = existing.find((p) => p.name.toLowerCase() === name.toLowerCase());
      if (clash && !(await ask(`A saved venue called "${clash.name}" already exists. Saving will replace it.`, { title: 'Replace saved venue?', yes: 'Replace' }))) return;
    } catch (e) { /* listing is only a courtesy check */ }
  }
  try {
    const res = await api.saveProject(state.project);
    markSaved();
    toast(`Saved to ${res.path}`);
  } catch (e) { toast(`Save failed: ${e.message}`, { error: true }); }
}

async function deleteSaved(p) {
  if (!(await ask(`"${p.name}" will be deleted from this computer. This cannot be undone.`, { title: 'Delete saved venue?', yes: 'Delete', danger: true }))) return;
  try {
    await api.deleteProject(p.key);
    if (state.savedName && state.savedName.toLowerCase() === p.name.toLowerCase()) forgetSaved();
    toast(`Deleted ${p.name}.`);
  } catch (e) { toast(`Delete failed: ${e.message}`, { error: true }); }
}

async function exportDsp(format) {
  commit();
  try {
    const { blob, name } = await api.exportDsp(state.project, format);
    download(name || `dsp-settings.${format}`, blob);
  } catch (e) { toast(`Export failed: ${e.message}`, { error: true }); }
}

// -------------------------------------------------------------------- topbar
function createTopbar(root) {
  const logo = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  logo.setAttribute('viewBox', '0 0 32 32');
  logo.setAttribute('aria-hidden', 'true');
  logo.innerHTML = '<rect width="32" height="32" rx="7" fill="var(--surface-3)"/>'
    + '<path d="M5 16c3-9 6-9 9 0s6 9 9 0" fill="none" stroke="#3987e5" stroke-width="2.6" stroke-linecap="round"/>'
    + '<path d="M9 16c3 9 6 9 9 0s6-9 9 0" fill="none" stroke="#d95926" stroke-width="2.6" stroke-linecap="round" opacity=".9"/>';

  const name = h('input', { type: 'text', class: 'venue-name', maxLength: 80, attrs: { 'aria-label': 'Venue name' },
    oninput: () => mutate((p) => { p.name = name.value; }, { commit: false, source: 'topbar' }),
    onchange: () => mutate((p) => { p.name = name.value.trim() || 'Untitled venue'; }, { source: 'topbar' }),
    onkeydown: (e) => { if (e.key === 'Enter' || e.key === 'Escape') name.blur(); } });
  const saveState = h('span', { class: 'save-state', attrs: { 'aria-live': 'polite' } });
  const undoBtn = h('button', { class: 'icon-btn', title: 'Undo (Ctrl+Z)', attrs: { 'aria-label': 'Undo' }, onclick: undo }, icon('undo'));
  const redoBtn = h('button', { class: 'icon-btn', title: 'Redo (Ctrl+Y)', attrs: { 'aria-label': 'Redo' }, onclick: redo }, icon('redo'));
  const themeBtn = h('button', { class: 'icon-btn', attrs: { 'aria-label': 'Switch light or dark theme' },
    onclick: () => { toggleTheme(); renderTheme(); themeChanged(); } });

  const venues = menuButton([icon('plus'), h('span', { text: 'New' })], async () => [
    h('h4', { text: 'Start from an example' }),
    ...state.meta.venues.map((v) => item(v.name, () => openVenue(v.id))),
    h('hr'),
    item('Empty room', () => openVenue('blank'), { ico: 'plus' }),
  ], { title: 'Start a new venue' });

  const projects = menuButton([icon('folder'), h('span', { text: 'Open' })], async () => {
    const list = await api.projects();
    const fromFile = [h('hr'), item('Open a .trky archive…', () => import('./trky.js').then((m) => m.chooseTrky()), { ico: 'folder', hint: '.trky' })];
    if (!list.length) return [h('div', { class: 'empty', text: 'Nothing saved yet. Press Save to keep this venue.' }), ...fromFile];
    return [h('h4', { text: 'Saved venues' }), ...list.map((p) => h('div', { style: { display: 'flex', alignItems: 'center' } },
      item(p.name, () => openSaved(p.key), { hint: `${p.speakers} spk · ${p.modified.slice(0, 10)}` }),
      h('button', { class: 'icon-btn small', title: `Delete ${p.name}`, attrs: { 'aria-label': `Delete ${p.name}` },
        onclick: () => { closeMenu(); deleteSaved(p); } }, icon('trash')))),
    ...fromFile, h('hr'), h('div', { class: 'empty', text: `Stored in ${state.meta.data_dir}` })];
  }, { title: 'Open a saved venue' });

  const saveBtn = h('button', { class: 'btn ghost', title: 'Save (Ctrl+S)', onclick: saveVenue }, icon('save'), h('span', { text: 'Save' }));

  const exportMenu = menuButton([icon('export'), h('span', { text: 'Export' })], async () => [
    h('h4', { text: 'Processor settings' }),
    item('Spreadsheet (CSV)', () => exportDsp('csv'), { hint: '.csv' }),
    item('Plain text sheet', () => exportDsp('txt'), { hint: '.txt' }),
    item('JSON', () => exportDsp('json'), { hint: '.json' }),
    h('h4', { text: 'For a processor' }),
    item('dbx DriveRack VENU360 sheet', () => exportDsp('venu360'), { hint: '.txt' }),
    item('REW / Equalizer APO filters', () => exportDsp('rew'), { hint: '.zip' }),
    item('miniDSP biquads', () => exportDsp('minidsp'), { hint: '.zip' }),
    h('hr'),
    item('Client report…', () => import('./report-ui.js').then((m) => m.reportDialog()), { hint: 'PDF' }),
    h('hr'),
    h('h4', { text: 'Share the venue' }),
    item('Project archive, with the scan', () => import('./trky.js').then((m) => m.saveTrky()), { hint: '.trky' }),
  ], { title: 'Export settings, a report or the whole venue', align: 'right' });

  root.append(h('div', { class: 'brand' }, logo, h('span', { text: 'Acoustic Phase Optimizer' })),
    h('span', { class: 'sep' }), name, saveState, h('span', { class: 'spacer' }),
    undoBtn, redoBtn, h('span', { class: 'sep' }), venues, projects, saveBtn, exportMenu, h('span', { class: 'sep' }), themeBtn);

  function renderName() { if (document.activeElement !== name) name.value = state.project?.name ?? ''; }
  function renderSaved() {
    saveState.textContent = !state.savedName ? 'Not saved' : state.dirty ? 'Edited' : 'Saved';
    saveState.title = state.savedName ? `Saved as "${state.savedName}"` : 'Autosaved for next time, but not kept as a named venue yet';
  }
  function renderHistory() { undoBtn.disabled = !canUndo(); redoBtn.disabled = !canRedo(); }
  function renderTheme() {
    clear(themeBtn).append(icon(isDark() ? 'sun' : 'moon'));
    themeBtn.title = isDark() ? 'Light theme' : 'Dark theme';
  }
  on('project', () => { renderName(); renderHistory(); });
  on('saved', renderSaved);
  on('history', renderHistory);
  renderName(); renderSaved(); renderHistory(); renderTheme();
}

// ------------------------------------------------------------------ keyboard
function editingText(e) {
  const t = e.target;
  return t instanceof HTMLElement && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName));
}

document.addEventListener('keydown', (e) => {
  if (!state.project || currentDialog()) return;
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  if (mod && k === 's') { e.preventDefault(); saveVenue(); return; }
  if (e.key === 'Escape' && openMenu) { closeMenu(); return; }
  if (editingText(e)) return;
  if (mod && k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); } else if (mod && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); } else if ((e.key === 'Delete' || e.key === 'Backspace') && state.selection) {
    e.preventDefault();
    if (state.selection.kind === 'speaker') removeSpeaker(state.selection.id); else removeMic(state.selection.id);
  } else if (e.key === 'Escape' && state.selection) select(null);
  else if (mod && k === 'd' && state.selection?.kind === 'speaker') { e.preventDefault(); duplicateSpeaker(state.selection.id); }
  else if (!mod && e.key.startsWith('Arrow') && state.selection) { e.preventDefault(); nudge(e.key, nudgeStep(e.shiftKey)); }
  else if (!mod && k === 'r' && state.selection?.kind === 'speaker') { e.preventDefault(); turn(e.shiftKey ? -5 : 5); }
  else if (!mod && k === 'm' && state.selection?.kind === 'speaker') {
    mutate((p) => { const s = p.speakers.find((x) => x.id === state.selection.id); if (s) s.mute = !s.mute; }, { structural: true });
  } else if (!mod && k === '1') setUI({ view: 'plan' });
  else if (!mod && k === '2') setUI({ view: '3d' });
  else if (!mod && k === '3') setUI({ view: 'section' });
  else if (!mod && (e.key === '?' || (k === '/' && e.shiftKey))) { e.preventDefault(); showShortcuts(); }
  else if (!mod && e.key === 'Tab' && !e.altKey && document.activeElement === document.body) cycleSelection(e);
});

// Nudges and turns land as one undo step per burst of key presses.
const settle = debounce(() => { commit(); mutate(() => {}, { source: 'keys' }); }, 600);
function nudge(key, step) {
  const sel = state.selection;
  const r = state.project.room;
  // The plan draws distance-from-stage left to right and across-the-room top to bottom.
  const [dx, dy] = { ArrowRight: [0, step], ArrowLeft: [0, -step], ArrowDown: [step, 0], ArrowUp: [-step, 0] }[key];
  const live = { commit: false, interacting: true, source: 'keys' };
  const fix = (v, hi) => Math.round(clamp(v, 0.1, hi - 0.1) * 1000) / 1000;
  if (sel.kind === 'speaker') {
    const s = speakerById(sel.id);
    if (s) editSpeaker(sel.id, { x: fix(s.x + dx, r.width), y: fix(s.y + dy, r.length) }, live);
  } else {
    mutate((p) => { const m = p.mics.find((o) => o.id === sel.id); if (m) { m.x = fix(m.x + dx, r.width); m.y = fix(m.y + dy, r.length); } }, live);
  }
  settle();
}
function turn(deg) {
  const s = speakerById(state.selection.id);
  if (!s) return;
  let yaw = s.yaw + deg;
  if (yaw > 180) yaw -= 360;
  if (yaw < -180) yaw += 360;
  editSpeaker(s.id, { yaw }, { commit: false, interacting: true, source: 'keys' });
  settle();
}
function cycleSelection(e) {
  const all = [...state.project.speakers.map((s) => ['speaker', s.id]), ...state.project.mics.map((m) => ['mic', m.id])];
  if (!all.length) return;
  e.preventDefault();
  const i = all.findIndex(([, id]) => id === state.selection?.id);
  const next = all[(i + (e.shiftKey ? all.length - 1 : 1)) % all.length];
  select(next[0], next[1]);
}

const SHORTCUTS = [
  ['Ctrl+Z / Ctrl+Y', 'Undo / redo'], ['Ctrl+S', 'Save the venue'], ['Ctrl+D', 'Duplicate the selected speaker'],
  ['Delete', 'Delete the selection'], ['Arrow keys', nudgeLabel], ['R / Shift+R', 'Turn 5° right / left'],
  ['M', 'Mute or unmute the selected speaker'], ['Tab / Shift+Tab', 'Select the next / previous speaker or mic'],
  ['Esc', 'Clear the selection'], ['1 / 2 / 3', 'Plan / 3D / section view'], ['Scroll, pinch', 'Zoom the plan'],
  ['Right-drag, Space-drag', 'Pan the plan'], ['0', 'Fit the plan to the room'], ['Drag a field label', 'Scrub a number (Shift faster, Alt finer)'],
  ['Drag in 3D', 'Move a speaker or mic; orbit on empty space'],
];
function showShortcuts() {
  openDialog({ title: 'Keyboard and mouse', body: h('dl', { class: 'shortcuts' }, SHORTCUTS.flatMap(([k, v]) => [h('dt', null, k.split(' / ').map((x, i) => [i ? ' / ' : '', h('kbd', { text: x })])), h('dd', { text: typeof v === 'function' ? v() : v })])),
    actions: [{ label: 'Close', kind: 'primary' }] });
}

// ------------------------------------------------------------------- startup
async function boot() {
  const bootEl = $('boot');
  try {
    const [meta, lib] = await Promise.all([api.meta(), api.speakers().catch(() => null)]);
    state.meta = meta;
    if (lib) state.meta.speaker_types = [...meta.speaker_types, ...lib.library, ...lib.user];
    let session = null;
    try { session = await api.session(); } catch (e) { /* first run */ }
    const fresh = session?.project ? null : await api.venue(state.meta.venues[0]?.id || 'blank');
    // Views first, so they all hear the first load.
    createTopbar($('topbar'));
    createLeftPanel($('left'));
    createStage($('stage'), $('stage-foot'));
    createStageBar($('stage-toolbar'), $('stage'));
    createChart($('chart-toolbar'), $('chart-legend'), $('chart'));
    createResultsPanel($('right'));
    enableDrop();
    if (session?.project) {
      const { probe, dirty, savedName, lastAction, compare, ...rest } = session.ui || {};
      loadProject(session.project, { baseline: session.baseline || null, lastAction: lastAction || null, ui: rest,
        probe: probe || null, savedName: savedName || null, dirty: dirty !== false });
    } else loadProject(fresh);
    bootEl.classList.add('hide');
    setTimeout(() => bootEl.remove(), 400);
  } catch (e) {
    clear(bootEl.firstElementChild).append(h('span', { text: `Could not start: ${e.message}` }),
      h('button', { class: 'btn small', text: 'Retry', onclick: () => location.reload() }));
    console.error(e);
  }
}

// The engine stops a few seconds after the last window closes; a reload pings again in time.
const ping = () => api.ping().catch(() => {});
setInterval(ping, 15000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) ping(); });
window.addEventListener('pagehide', () => { flushSession(); api.bye(); });
ping();

boot();
