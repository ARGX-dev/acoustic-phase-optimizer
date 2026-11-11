// Entry point: top bar, start-up, keyboard shortcuts.

import { api } from './api.js';
import {
  canRedo, canUndo, commit, flushSession, forgetSaved, loadProject, markSaved, mutate, on, redo, removeMic,
  removeSpeaker, select, state, themeChanged, undo,
} from './store.js';
import { clear, download, h, icon, toast } from './util.js';
import { isDark, toggleTheme } from './theme.js';
import { createStage } from './stage.js';
import { createStageBar } from './stagebar.js';
import { createChart } from './chart.js';
import { createLeftPanel } from './panels.js';
import { createResultsPanel } from './results.js';

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
function confirmDiscard() {
  return !state.dirty || window.confirm('This venue has changes that are not saved. Continue and lose them?');
}

async function openVenue(id) {
  if (!confirmDiscard()) return;
  try {
    loadProject(await api.venue(id));
    toast(id === 'blank' ? 'Started an empty room.' : `Loaded the ${state.project.name} example.`);
  } catch (e) { toast(`Could not load the venue: ${e.message}`, { error: true }); }
}

async function openSaved(key) {
  if (!confirmDiscard()) return;
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
      if (clash && !window.confirm(`A saved venue called "${clash.name}" already exists. Replace it?`)) return;
    } catch (e) { /* listing is only a courtesy check */ }
  }
  try {
    const res = await api.saveProject(state.project);
    markSaved();
    toast(`Saved to ${res.path}`);
  } catch (e) { toast(`Save failed: ${e.message}`, { error: true }); }
}

async function deleteSaved(p) {
  if (!window.confirm(`Delete the saved venue "${p.name}"? This cannot be undone.`)) return;
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

async function openReport() {
  commit();
  const t = toast('Building the report…', { ms: 20000 });
  try {
    const html = await api.report(state.project, state.baseline);
    const url = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
    const win = window.open(url, '_blank');
    if (!win) download(`${state.project.name || 'venue'} report.html`, html, 'text/html');
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    t.remove();
    toast(win ? 'Report opened. Use Print to save it as a PDF.' : 'Report downloaded.');
  } catch (e) {
    t.remove();
    toast(`Report failed: ${e.message}`, { error: true });
  }
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
    if (!list.length) return [h('div', { class: 'empty', text: 'Nothing saved yet. Press Save to keep this venue.' })];
    return [h('h4', { text: 'Saved venues' }), ...list.map((p) => h('div', { style: { display: 'flex', alignItems: 'center' } },
      item(p.name, () => openSaved(p.key), { hint: `${p.speakers} spk · ${p.modified.slice(0, 10)}` }),
      h('button', { class: 'icon-btn small', title: `Delete ${p.name}`, attrs: { 'aria-label': `Delete ${p.name}` },
        onclick: () => { closeMenu(); deleteSaved(p); } }, icon('trash')))),
    h('hr'), h('div', { class: 'empty', text: `Stored in ${state.meta.data_dir}` })];
  }, { title: 'Open a saved venue' });

  const saveBtn = h('button', { class: 'btn ghost', title: 'Save (Ctrl+S)', onclick: saveVenue }, icon('save'), h('span', { text: 'Save' }));

  const exportMenu = menuButton([icon('export'), h('span', { text: 'Export' })], async () => [
    h('h4', { text: 'Processor settings' }),
    item('Spreadsheet (CSV)', () => exportDsp('csv'), { hint: '.csv' }),
    item('Plain text sheet', () => exportDsp('txt'), { hint: '.txt' }),
    item('JSON', () => exportDsp('json'), { hint: '.json' }),
    h('hr'),
    item('Printable report', openReport, { hint: 'HTML / PDF' }),
  ], { title: 'Export settings or a report', align: 'right' });

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
  if (!state.project) return;
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  if (mod && k === 's') { e.preventDefault(); saveVenue(); return; }
  if (e.key === 'Escape' && openMenu) { closeMenu(); return; }
  if (editingText(e)) return;
  if (mod && k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); } else if (mod && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); } else if ((e.key === 'Delete' || e.key === 'Backspace') && state.selection) {
    e.preventDefault();
    if (state.selection.kind === 'speaker') removeSpeaker(state.selection.id); else removeMic(state.selection.id);
  } else if (e.key === 'Escape' && state.selection) select(null);
});

// ------------------------------------------------------------------- startup
async function boot() {
  const bootEl = $('boot');
  try {
    state.meta = await api.meta();
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
