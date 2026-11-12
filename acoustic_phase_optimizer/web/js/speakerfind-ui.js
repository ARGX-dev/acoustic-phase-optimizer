// "Found N possible speakers": look through the scan for speakers (MVR fixtures, named objects, speaker-shaped
// boxes), show them on a plan with a tick box and a model per row, and add only the ones the user accepts.

import { state, addFoundSpeakers } from './store.js';
import { h, toast } from './util.js';
import { openDialog, currentDialog } from './dialog.js';
import { cssVar, seriesColor } from './theme.js';
import { len } from './units.js';
import { findSpeakers } from './speakerfind.js';

const P = () => state.project;
const GENERIC = { sub: 'sub_18', array: 'line_array_hang', front_fill: 'frontfill_5', fill: 'compact_8', delay: 'generic_12_2way', main: 'generic_12_2way' };
const SOURCE = { fixture: 'MVR fixture', name: 'Name', shape: 'Shape' };

let busy = false;

/** Look for speakers in the current scan. auto: called right after an import, so stay quiet when nothing is found. */
export async function findSpeakersInScan({ auto = false } = {}) {
  const ref = P().scan;
  if (!ref || busy) return;
  busy = true;
  let res, skippedShapes = false;
  try {
    const [{ scanObject }, { extractFound }] = await Promise.all([import('./scan.js'), import('./speakerfind-three.js')]);
    const found = extractFound(await scanObject(ref));
    skippedShapes = found.skippedShapes;
    res = findSpeakers(found, { room: P().room, audience: P().audience, types: state.meta.speaker_types });
  } catch (e) {
    if (!auto) toast(`Could not look for speakers: ${e.message}`, { error: true });
    return;
  } finally { busy = false; }
  const { candidates, skippedFixtures } = res;
  if (auto) {
    // Interrupt the import only for strong evidence (a rigged fixture, a speaker's name). A box that merely has a
    // speaker's shape may be furniture, so it is only mentioned.
    const strong = candidates.some((c) => c.checked && c.source !== 'shape');
    if (!strong) {
      const n = candidates.filter((c) => c.checked).length;
      if (n) toast(`${n} speaker-shaped object${n === 1 ? '' : 's'} in the scan. To add ${n === 1 ? 'it' : 'them'}: Room > Find speakers in the scan…`);
      return;
    }
  }
  if (!candidates.length) {
    toast(`No speakers found in the scan${skippedFixtures ? ` (${skippedFixtures} MVR fixtures look like lights)` : ''}. `
      + 'Speakers that are part of one scanned surface cannot be told apart; add them by hand.');
    return;
  }
  showFound(candidates, { skippedFixtures, skippedShapes });
}

function defaultType(c) {
  if (c.typeId) return c.typeId;
  if (c.role === 'sub') return GENERIC.sub;
  if (c.boxes > 1) return GENERIC.array;
  return GENERIC[c.role] || GENERIC.main;
}

function typeSelect(value, onchange) {
  const types = state.meta.speaker_types;
  const groups = new Map();
  for (const t of types) {
    const g = t.group === 'Generic' || !t.brand ? (t.group || 'Generic') : t.brand;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(t);
  }
  const sel = h('select', { attrs: { 'aria-label': 'Loudspeaker model' }, onchange: () => onchange(sel.value) },
    [...groups].map(([g, list]) => h('optgroup', { attrs: { label: g } }, list.map((t) => h('option', { value: t.id, text: t.label })))));
  sel.value = types.some((t) => t.id === value) ? value : types[0]?.id;
  return sel;
}

function showFound(candidates, { skippedFixtures, skippedShapes }) {
  const rows = candidates.map((c) => ({ c, on: c.checked, typeId: defaultType(c) }));
  let hover = null;
  const plan = h('canvas', { class: 'scan-preview find-plan', width: 520, height: 260,
    attrs: { role: 'img', 'aria-label': 'Plan view of the room from above, stage on the left, with the speakers found. Ticked ones are filled.' } });

  function draw() {
    const ctx = plan.getContext('2d');
    const W = plan.width, H = plan.height, r = P().room, a = P().audience;
    ctx.fillStyle = cssVar('--page');
    ctx.fillRect(0, 0, W, H);
    const pad = 22, sc = Math.min((W - 2 * pad) / r.length, (H - 2 * pad) / r.width);
    const ox = (W - r.length * sc) / 2, oy = (H - r.width * sc) / 2;
    const X = (x, y) => [ox + y * sc, oy + x * sc]; // room y runs left to right, x top to bottom
    ctx.strokeStyle = cssVar('--line');
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ox, oy, r.length * sc, r.width * sc);
    ctx.fillStyle = cssVar('--accent');
    ctx.fillRect(ox - 5, oy, 5, r.width * sc); // the stage wall
    ctx.globalAlpha = 0.12;
    ctx.fillRect(...X(a.x0, a.y0), (a.y1 - a.y0) * sc, (a.x1 - a.x0) * sc);
    ctx.globalAlpha = 1;
    for (const s of P().speakers) { // the speakers already in the project, small and grey
      const [px, py] = X(s.x, s.y);
      ctx.fillStyle = cssVar('--ink-2');
      ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.font = '600 10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    rows.forEach((row, i) => {
      const { c } = row;
      const [px, py] = X(c.x, c.y);
      const col = seriesColor(i);
      const yaw = (c.yaw * Math.PI) / 180; // yaw 0 faces +y (right on the plan), positive turns toward +x (down)
      ctx.strokeStyle = col;
      ctx.lineWidth = hover === i ? 3 : 1.5;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + Math.cos(yaw) * 18, py + Math.sin(yaw) * 18); ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, hover === i ? 10 : 8, 0, 2 * Math.PI);
      ctx.fillStyle = row.on ? col : cssVar('--page');
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = row.on ? '#fff' : col;
      ctx.fillText(String(i + 1), px, py + 0.5);
    });
  }

  const addBtnLabel = 'Add the ticked speakers';
  const sync = () => {
    const n = rows.filter((r) => r.on).length;
    const btn = currentDialog()?.buttons.find((b) => b.textContent.startsWith('Add'));
    if (btn) { btn.disabled = !n; btn.title = `${n} speaker${n === 1 ? '' : 's'}`; }
    count.textContent = `${n} of ${rows.length} ticked`;
    draw();
  };
  const count = h('span', { class: 'hint' });
  const list = h('div', { class: 'find-list', attrs: { role: 'list' } }, rows.map((row, i) => {
    const { c } = row;
    const tick = h('input', { type: 'checkbox', checked: row.on, attrs: { 'aria-label': `Add speaker ${i + 1}` },
      onchange: () => { row.on = tick.checked; sync(); } });
    const title = c.name || (c.role === 'sub' ? 'Sub-shaped box' : c.boxes > 1 ? `Column of ${c.boxes} boxes` : 'Speaker-shaped box');
    const where = `${len(c.z)} high${c.boxes > 1 ? `, ${c.boxes} boxes` : ''}`;
    const el = h('div', { class: 'find-row', attrs: { role: 'listitem', 'data-testid': 'found-speaker' },
      onpointerenter: () => { hover = i; draw(); }, onpointerleave: () => { hover = null; draw(); } },
      h('span', { class: 'find-num', style: { background: seriesColor(i) }, text: String(i + 1) }),
      tick,
      h('div', { class: 'find-what' }, h('strong', { text: title }),
        h('small', { text: `${SOURCE[c.source]}: ${c.why}. ${where}. ${Math.round(c.confidence * 100)}% sure` })),
      typeSelect(row.typeId, (v) => { row.typeId = v; }));
    return el;
  }));

  const notes = [];
  if (skippedFixtures) notes.push(`${skippedFixtures} MVR fixture${skippedFixtures === 1 ? '' : 's'} looked like lights and were left out.`);
  if (skippedShapes) notes.push('The model is too big to search for speaker shapes; only names and fixtures were used.');
  const body = h('div', { class: 'find-body' },
    h('p', { class: 'hint', text: 'Found in the scan by fixture, name or shape. Check each one: nothing is added until you say so. '
      + 'Each is placed where it was found and aimed at the seats; you can change it after.' }),
    plan,
    h('div', { class: 'find-tools' }, count,
      h('button', { class: 'btn small ghost', onclick: () => { rows.forEach((r) => { r.on = true; }); list.querySelectorAll('input[type=checkbox]').forEach((x) => { x.checked = true; }); sync(); } }, 'Tick all'),
      h('button', { class: 'btn small ghost', onclick: () => { rows.forEach((r) => { r.on = false; }); list.querySelectorAll('input[type=checkbox]').forEach((x) => { x.checked = false; }); sync(); } }, 'Untick all')),
    list,
    notes.length ? h('p', { class: 'hint', text: notes.join(' ') }) : null);

  openDialog({ title: `Found ${rows.length} possible speaker${rows.length === 1 ? '' : 's'}`, body, wide: true, actions: [
    { label: 'Not now' },
    { label: addBtnLabel, kind: 'primary', run: () => {
      const take = rows.filter((r) => r.on);
      if (!take.length) return false;
      const ids = addFoundSpeakers(take.map(({ c, typeId }) => ({ name: c.name ? c.name.slice(0, 40) : null, typeId,
        x: c.x, y: c.y, z: c.z, yaw: c.yaw, pitch: c.pitch, boxes: c.boxes })));
      toast(`Added ${ids.length} speaker${ids.length === 1 ? '' : 's'} from the scan. Undo takes them all back out.`);
      return true;
    } },
  ] });
  sync();
}
