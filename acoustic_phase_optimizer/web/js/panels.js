// Left panel: the venue editor (room, speakers, microphones, setup).

import {
  addMic, addSpeaker, DEFAULT_WEIGHTS, duplicateSpeaker, editSpeaker, mirrorSpeaker, mutate, on, partnerOf,
  hangFields, loadSpeakerLibrary, removeMic, removeSpeaker, select, setUI, speakerById, speakerType, state, typeFields,
} from './store.js';
import { clamp, clear, fmt, fmtHz, h, icon, toast, uid } from './util.js';
import { pickSpeakerType } from './picker.js';
import { openDialog } from './dialog.js';
import { api } from './api.js';
import { matchOffset, parseResponse } from './measure.js';
import { seriesColor } from './theme.js';
import { aimPatch } from './aim.js';
import { area, dispStep, fromDisp, len, lenUnit, speed, toDisp, volume } from './units.js';

const TABS = [['room', 'Room'], ['speakers', 'Speakers'], ['mics', 'Mics'], ['setup', 'Setup']];
const ROLES = [['main', 'Main'], ['fill', 'Fill'], ['front_fill', 'Front fill'], ['delay', 'Delay'], ['sub', 'Subwoofer']];
const ROLE_NAME = Object.fromEntries(ROLES);
const SURFACES = [['floor', 'Floor'], ['ceiling', 'Ceiling'], ['front', 'Stage wall'], ['back', 'Back wall'],
  ['left', 'Side wall, top of plan'], ['right', 'Side wall, bottom of plan']];
const FILTERS = [['bell', 'Bell'], ['low_shelf', 'Low shelf'], ['high_shelf', 'High shelf'], ['highpass', 'High-pass'], ['lowpass', 'Low-pass']];
const PRIORITIES = [['coverage', 'Even coverage'], ['interference', 'Summation'], ['clarity', 'Clarity'], ['timing', 'Tight timing'], ['tonal', 'Tonal balance']];
const LIVE = { commit: false, interacting: true, source: 'panel' };
const DONE = { commit: true, source: 'panel' };

export function createLeftPanel(root) {
  const tabBar = h('div', { class: 'tabs', attrs: { role: 'tablist' } });
  const body = h('div', { class: 'tab-body', attrs: { role: 'tabpanel' } });
  root.append(tabBar, body);
  let bound = []; // inputs that mirror model values: { el, get, nd }
  let listBox = null;
  let statsBox = null;

  // ---------------------------------------------------------------- fields
  const show = (v, nd) => (v == null || Number.isNaN(v) ? '' : String(Number(v.toFixed(nd))));

  function num(label, getM, setM, { min, max, step = 0.1, unit = '', nd = 2, wide = false, title = null, bare = false } = {}) {
    let get = getM, set = setM;
    if (unit === 'm') {
      // Lengths: the model keeps metres; the field speaks the chosen unit.
      get = () => toDisp(getM());
      set = (v, o) => setM(v == null ? v : Math.round(fromDisp(v) * 10000) / 10000, o);
      min = min == null ? min : toDisp(min);
      max = max == null ? max : toDisp(max);
      step = dispStep(step);
      unit = lenUnit();
    }
    // Plain text, not type=number: the browser would format decimals with the Windows locale ("4,6") while
    // every readout uses a point. Either separator is accepted when typing.
    const input = h('input', { type: 'text', value: show(get(), nd), class: 'num' + (bare ? ' bare' : ''), title,
      attrs: { 'aria-label': label, inputmode: 'decimal', autocomplete: 'off', spellcheck: 'false', role: 'spinbutton',
        'aria-valuemin': min, 'aria-valuemax': max } });
    const parse = (t) => parseFloat(String(t).trim().replace(',', '.'));
    input.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      e.preventDefault();
      const cur = parse(input.value);
      const d = step * (e.shiftKey ? 10 : e.altKey ? 0.1 : 1) * (e.key === 'ArrowUp' ? 1 : -1);
      const v = clamp(Math.round(((Number.isFinite(cur) ? cur : get() ?? 0) + d) / (step * (e.altKey ? 0.1 : 1))) * step * (e.altKey ? 0.1 : 1), min ?? -Infinity, max ?? Infinity);
      input.value = show(v, nd);
      set(v, DONE);
    });
    const read = () => {
      const v = parse(input.value);
      return Number.isFinite(v) && (min == null || v >= min) && (max == null || v <= max) ? v : null;
    };
    input.addEventListener('input', () => {
      const v = read();
      input.classList.toggle('invalid', v == null);
      if (v != null && v !== get()) set(v, LIVE);
    });
    input.addEventListener('change', () => {
      let v = parse(input.value);
      v = Number.isFinite(v) ? clamp(v, min ?? -Infinity, max ?? Infinity) : get();
      input.classList.remove('invalid');
      set(v, DONE);
      input.value = show(v, nd);
    });
    bound.push({ el: input, get, nd });
    if (bare) return input;
    // Drag the label sideways to scrub the value: Shift for big steps, Alt for fine ones.
    const caption = h('span', { text: label, class: 'scrub', title: title || 'Drag sideways to adjust (Shift: faster, Alt: finer)' });
    caption.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const start = get() ?? 0, x0 = e.clientX;
      let moved = false;
      caption.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const steps = Math.round((ev.clientX - x0) / 4);
        if (!steps && !moved) return;
        moved = true;
        const mult = ev.shiftKey ? 10 : ev.altKey ? 0.1 : 1;
        const v = clamp(Math.round((start + steps * step * mult) / (step * (ev.altKey ? 0.1 : 1))) * step * (ev.altKey ? 0.1 : 1), min ?? -Infinity, max ?? Infinity);
        input.value = show(v, nd);
        if (v !== get()) set(v, LIVE);
      };
      const up = () => {
        caption.removeEventListener('pointermove', move);
        caption.removeEventListener('pointerup', up);
        caption.removeEventListener('pointercancel', up);
        if (moved) set(get(), DONE); else input.focus();
      };
      caption.addEventListener('pointermove', move);
      caption.addEventListener('pointerup', up);
      caption.addEventListener('pointercancel', up);
    });
    return h('label', { class: 'field' + (wide ? ' wide' : ''), title }, caption,
      h('span', { class: 'input-wrap' }, input, unit ? h('em', { text: unit }) : null));
  }

  function choice(label, options, get, set, { wide = false, bare = false, title = null } = {}) {
    const el = h('select', { title, attrs: { 'aria-label': label }, onchange: () => set(el.value, { ...DONE, structural: true }) },
      options.map(([v, t]) => h('option', { value: v, text: t })));
    el.value = get();
    bound.push({ el, get, raw: true });
    return bare ? el : h('label', { class: 'field' + (wide ? ' wide' : ''), title }, h('span', { text: label }), el);
  }

  function textField(label, get, set, { wide = true, placeholder = '' } = {}) {
    const el = h('input', { type: 'text', value: get() ?? '', placeholder, maxLength: 60, attrs: { 'aria-label': label },
      oninput: () => set(el.value, LIVE), onchange: () => set(el.value.trim() || get() || 'Untitled', DONE) });
    bound.push({ el, get, raw: true });
    return h('label', { class: 'field' + (wide ? ' wide' : '') }, h('span', { text: label }), el);
  }

  function check(label, get, set, hint = null) {
    const el = h('input', { type: 'checkbox', checked: !!get(), onchange: () => set(el.checked, { ...DONE, structural: true }) });
    bound.push({ el, get, checkbox: true });
    return h('label', { class: 'check' }, el, h('span', null, label, hint ? h('small', { text: hint }) : null));
  }

  function sync() {
    for (const b of bound) {
      if (document.activeElement === b.el) continue;
      let v;
      try { v = b.get(); } catch (e) { continue; }
      if (b.checkbox) b.el.checked = !!v;
      else if (b.raw) { if (b.el.value !== String(v ?? '')) b.el.value = v ?? ''; } else {
        const s = show(v, b.nd);
        if (b.el.value !== s) b.el.value = s;
        b.el.classList.remove('invalid');
      }
    }
  }

  const group = (title, ...children) => h('div', { class: 'group' }, title ? h('h3', null, title) : null, children);
  const P = () => state.project;
  const setIn = (path) => (v, opts) => mutate((p) => { let o = p; for (const k of path.slice(0, -1)) o = o[k]; o[path.at(-1)] = v; }, opts);

  // ------------------------------------------------------------------ room
  function roomTab() {
    statsBox = h('div', { class: 'stat-grid' });
    renderStats();
    const mats = [...state.meta.materials.map((m) => [m.id, m.name]), ...(P().custom_materials || []).map((m) => [m.id, `★ ${m.name}`])];
    const kind = P().room.kind || 'indoor';
    const openSurface = (k) => (kind !== 'indoor' && k === 'ceiling') || (kind === 'outdoor' && k !== 'floor');
    const altitude = () => Math.round(44330.8 * (1 - (P().environment.pressure_kpa / 101.325) ** 0.190263));
    const setAltitude = (v, o) => setIn(['environment', 'pressure_kpa'])(Math.round(101.325 * (1 - 2.25577e-5 * clamp(v, -400, 5000)) ** 5.25588 * 100) / 100, o);
    const occOut = h('output', { text: `${Math.round((P().audience.occupancy || 0) * 100)}%` });
    const occ = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: P().audience.occupancy || 0, attrs: { 'aria-label': 'Audience occupancy' },
      oninput: () => { occOut.textContent = `${Math.round(occ.value * 100)}%`; setIn(['audience', 'occupancy'])(Number(occ.value), LIVE); },
      onchange: () => setIn(['audience', 'occupancy'])(Number(occ.value), DONE) });
    const KINDS = [['indoor', 'Indoor room', 'Walls, floor and ceiling all reflect'], ['stadium', 'Stadium', 'Open roof: walls and stands reflect'],
      ['outdoor', 'Open air', 'Only the ground reflects']];
    const kindSeg = h('div', { class: 'seg block', attrs: { role: 'radiogroup', 'aria-label': 'Venue type' } }, KINDS.map(([id, label, tip]) =>
      h('button', { class: kind === id ? 'on' : '', title: tip, attrs: { role: 'radio', 'aria-checked': String(kind === id) },
        onclick: () => setIn(['room', 'kind'])(id, { ...DONE, structural: true }) }, label)));
    return [
      scanGroup(),
      group('Venue type', kindSeg,
        h('p', { class: 'hint', text: KINDS.find((k) => k[0] === kind)[2] + '. The box still sets the size of the area you are covering.' })),
      group(kind === 'outdoor' ? 'Area' : 'Room size',
        h('div', { class: 'grid3' },
          num('Stage to back', () => P().room.length, setIn(['room', 'length']), { min: 3, max: 300, step: 0.5, unit: 'm' }),
          num('Width', () => P().room.width, setIn(['room', 'width']), { min: 3, max: 200, step: 0.5, unit: 'm' }),
          num('Height', () => P().room.height, setIn(['room', 'height']), { min: 2, max: 60, step: 0.1, unit: 'm' }))),
      group('Surfaces', h('div', { class: 'grid2' }, SURFACES.map(([k, label]) => (openSurface(k)
        ? h('div', { class: 'field' }, h('span', { text: label }), h('div', { class: 'open-surface', text: kind === 'outdoor' && k !== 'ceiling' ? 'Open side' : 'Open sky' }))
        : choice(label, mats, () => P().room.materials[k], setIn(['room', 'materials', k]))))),
      h('p', { class: 'hint', text: 'Each surface sets how strongly reflections come back, band by band, and the reverb time. ★ marks your own materials.' })),
      customMaterialsGroup(),
      group('Audience area',
        h('div', { class: 'grid2' },
          num('Starts, from stage', () => P().audience.y0, setIn(['audience', 'y0']), { min: 0, step: 0.5, unit: 'm' }),
          num('Ends, from stage', () => P().audience.y1, setIn(['audience', 'y1']), { min: 0, step: 0.5, unit: 'm' }),
          num('Starts, across', () => P().audience.x0, setIn(['audience', 'x0']), { min: 0, step: 0.5, unit: 'm' }),
          num('Ends, across', () => P().audience.x1, setIn(['audience', 'x1']), { min: 0, step: 0.5, unit: 'm' }),
          num('Ear height', () => P().audience.ear_height, setIn(['audience', 'ear_height']), { min: 0.3, max: 20, step: 0.1, unit: 'm',
            title: `At the front row: ${len(1.2)} seated, ${len(1.7)} standing` }),
          num('Rake: back row higher by', () => P().audience.rake || 0, setIn(['audience', 'rake']), { min: 0, max: Math.max(0, P().room.height - P().audience.ear_height - 0.1), step: 0.1, unit: 'm',
            title: 'Raked seating or a balcony slope: the ears rise evenly from the front row to the back row. 0 for a flat floor.' })),
        h('div', { class: 'slider-row', style: { marginTop: '10px' } }, h('span', { text: 'Seats filled' }), occ, occOut),
        h('p', { class: 'hint', text: 'People absorb sound. Soundcheck in an empty room, then set the expected crowd to see the show.' })),
      group('Air',
        h('div', { class: 'grid2' },
          num('Temperature', () => P().environment.temperature_c, setIn(['environment', 'temperature_c']), { min: -20, max: 50, step: 1, unit: '°C' }),
          num('Humidity', () => P().environment.humidity_pct, setIn(['environment', 'humidity_pct']), { min: 0, max: 100, step: 5, unit: '%' }),
          num('Pressure', () => P().environment.pressure_kpa, setIn(['environment', 'pressure_kpa']), { min: 60, max: 110, step: 0.5, unit: 'kPa' }),
          num('Altitude', altitude, setAltitude, { min: -400, max: 5000, step: 50, unit: 'm', nd: 0, title: 'Sets the pressure of a standard atmosphere at this height' })),
        h('p', { class: 'hint', text: 'Temperature, humidity and pressure set the speed of sound (every delay time) and how much treble the air absorbs on every path (ISO 9613-1).' })),
      group('Predicted room acoustics', statsBox),
      group('Notes', (() => {
        const el = h('textarea', { value: P().notes || '', placeholder: 'Anything worth remembering about this venue…', attrs: { 'aria-label': 'Notes' },
          oninput: () => setIn(['notes'])(el.value, { commit: false, source: 'panel' }), onchange: () => setIn(['notes'])(el.value, DONE) });
        bound.push({ el, get: () => P().notes, raw: true });
        return el;
      })()),
    ];
  }

  function scanGroup() {
    const sc = P().scan;
    const pick = () => import('./scan.js').then((m) => m.chooseScanFile()).catch((e) => toast(`Scan import is unavailable: ${e.message}`, { error: true }));
    if (!sc) {
      return group('3D scan',
        h('button', { class: 'btn block', onclick: pick }, icon('cube'), 'Import a 3D scan…'),
        h('button', { class: 'btn small ghost', style: { marginTop: '6px' }, title: 'For an .mvr that was unzipped into a folder of .glb, .gdtf and GeneralSceneDescription.xml files',
          onclick: () => import('./scan.js').then((m) => m.chooseMvrFolder()) }, 'Import an unpacked MVR folder…'),
        h('p', { class: 'hint', text: 'A scan (GLB, OBJ or PLY from Polycam, Scaniverse, 3d Scanner App or a laser scanner) or a venue model (MVR, 3DS, FBX, COLLADA or STL, e.g. from Vectorworks: File > Export). The room size is measured from it.' }));
    }
    const setScan = (k) => (v, o) => mutate((p) => { if (p.scan) p.scan[k] = v; }, o);
    const out = h('output', { text: `${Math.round(sc.opacity * 100)}%` });
    const opacity = h('input', { type: 'range', min: 0.1, max: 1, step: 0.05, value: sc.opacity, attrs: { 'aria-label': 'Scan opacity' },
      oninput: () => { out.textContent = `${Math.round(Number(opacity.value) * 100)}%`; setScan('opacity')(Number(opacity.value), LIVE); },
      onchange: () => setScan('opacity')(Number(opacity.value), DONE) });
    return group('3D scan',
      h('div', { class: 'scan-card' }, icon('cube'), h('span', { class: 'name', text: sc.name }),
        h('button', { class: 'btn small ghost', onclick: () => { setUI({ view: '3d' }); } }, 'View in 3D')),
      check('Show the scan in 3D', () => P().scan?.visible !== false, setScan('visible')),
      h('div', { class: 'slider-row' }, h('span', { text: 'Opacity' }), opacity, out),
      h('div', { style: { display: 'flex', gap: '6px', marginTop: '8px' } },
        h('button', { class: 'btn small', onclick: pick }, 'Replace…'),
        h('span', { style: { flex: 1 } }),
        h('button', { class: 'btn small danger', onclick: () => mutate((p) => { p.scan = null; }, { structural: true }) }, icon('trash'), 'Remove scan')),
      h('p', { class: 'hint', text: 'The prediction uses the box below; adjust it if the scan was measured oddly.' }));
  }

  function customMaterialsGroup() {
    const list = P().custom_materials || [];
    const BANDS = ['125', '250', '500', '1k', '2k', '4k'];
    const setMat = (id, fn) => mutate((p) => { const m = (p.custom_materials || []).find((x) => x.id === id); if (m) fn(m); }, { ...DONE, structural: true });
    const rows = list.map((m) => h('div', { class: 'mat-row' },
      h('input', { type: 'text', value: m.name, maxLength: 60, attrs: { 'aria-label': 'Material name' },
        onchange: (e) => setMat(m.id, (x) => { x.name = e.target.value.trim() || 'My material'; }) }),
      h('button', { class: 'icon-btn small', title: 'Delete material', attrs: { 'aria-label': `Delete ${m.name}` },
        onclick: () => mutate((p) => {
          p.custom_materials = p.custom_materials.filter((x) => x.id !== m.id);
          for (const k of Object.keys(p.room.materials)) if (p.room.materials[k] === m.id) p.room.materials[k] = 'concrete';
        }, { structural: true }) }, icon('trash')),
      h('div', { class: 'mat-alpha' }, m.alpha.map((a, i) => h('label', null, h('span', { text: BANDS[i] }),
        h('input', { type: 'text', value: String(a), class: 'num bare', attrs: { 'aria-label': `${m.name} absorption at ${BANDS[i]} Hz`, inputmode: 'decimal' },
          onchange: (e) => setMat(m.id, (x) => { x.alpha = x.alpha.map((v, j) => (j === i ? clamp(parseFloat(e.target.value.replace(',', '.')) || 0, 0, 1) : v)); }) }))))));
    return group('My materials', rows,
      h('button', { class: 'btn small', onclick: () => mutate((p) => {
        p.custom_materials = [...(p.custom_materials || []), { id: `custom_${uid()}`, name: 'My material', alpha: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] }];
      }, { structural: true }) }, icon('plus'), 'Add a material'),
      h('p', { class: 'hint', text: 'Absorption coefficients per octave band (0 reflects everything, 1 absorbs everything), from the product datasheet.' }));
  }

  function renderStats() {
    if (!statsBox) return;
    const r = state.sim?.room;
    const cell = (label, value, title) => h('div', { title }, h('span', { text: label }), h('b', { text: value }));
    const room = P().room;
    if (room.kind === 'outdoor') {
      // Free field: no room to reverberate, so reverb time, critical distance and room modes do not apply.
      clear(statsBox).append(
        cell('Reverberation', 'None (open air)', 'Nothing to reflect from except the ground, which the model includes'),
        cell('Area', area(room.width * room.length)),
        cell('Speed of sound', r ? speed(r.speed_of_sound) : '–'),
        cell('Air loss at 8 kHz', r && r.air_loss_8k_30m_db != null ? `${fmt(r.air_loss_8k_30m_db, 1)} dB / ${len(30, 0)}` : '–', `Extra treble the air absorbs over ${len(30, 0)}: this is what limits long throws outdoors`),
        h('p', { class: 'hint', style: { gridColumn: '1 / -1' }, text: 'Level falls 6 dB per doubling of distance, plus the air loss above. Wind and temperature gradients bend sound outdoors and are not modelled.' }));
      return;
    }
    clear(statsBox).append(
      cell('Reverb time (mid)', r ? `${fmt(r.rt60_mid, 2)} s` : '–', room.kind === 'stadium' ? 'Eyring, with the roof open: walls and stands still reflect' : 'Eyring, average of 500 Hz and 1 kHz'),
      cell('Critical distance', r ? len(r.critical_distance_m[3]) : '–', 'Where direct and reverberant sound are equal, for an omni source at 1 kHz'),
      cell('Volume', r ? volume(r.volume_m3) : '–'),
      cell('Speed of sound', r ? speed(r.speed_of_sound) : '–'),
      cell('Schroeder frequency', r ? `${Math.round(r.schroeder_hz)} Hz` : '–', 'Below this, room modes dominate'),
      cell('Air loss at 8 kHz', r && r.air_loss_8k_30m_db != null ? `${fmt(r.air_loss_8k_30m_db, 1)} dB / ${len(30, 0)}` : '–', `Extra treble the air absorbs over ${len(30, 0)}`));
    if (r) {
      const max = Math.max(...r.rt60_eyring, 0.1);
      statsBox.append(h('div', { class: 'rt-bands', attrs: { role: 'img', 'aria-label': 'Reverb time per octave band: ' + r.bands.map((b, i) => `${fmtHz(b)} ${fmt(r.rt60_eyring[i], 2)} s`).join(', ') } },
        r.bands.map((b, i) => h('div', { class: 'rt-band', title: `${fmtHz(b)}: ${fmt(r.rt60_eyring[i], 2)} s` },
          h('b', { text: fmt(r.rt60_eyring[i], 1) }),
          h('i', { style: { height: `${Math.max(4, (r.rt60_eyring[i] / max) * 56)}px` } }),
          h('span', { text: fmtHz(b).replace(' kHz', 'k').replace(' Hz', '') })))));
    }
  }

  // -------------------------------------------------------------- speakers
  function speakerRow(s) {
    const sel = state.selection?.kind === 'speaker' && state.selection.id === s.id;
    return h('div', { class: 'row-item' + (sel ? ' selected' : '') + (s.mute ? ' muted' : ''), tabIndex: 0,
      attrs: { role: 'button', 'aria-pressed': String(sel) },
      onclick: () => select('speaker', s.id), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select('speaker', s.id); } } },
    s.role === 'sub' ? h('span', { class: 'swatch', style: { background: seriesColor(s.color), borderRadius: '2px' } })
      : h('span', { class: 'swatch', style: { background: seriesColor(s.color) } }),
    h('span', { class: 'name', text: s.name }),
    h('span', { class: 'meta', title: s.array_delay_ms || s.array_invert ? 'Includes the sub array pattern offset' : null,
      text: `${fmt(s.delay_ms + (s.array_delay_ms || 0), 1)} ms · ${fmt(s.gain_db, 1, true)} dB${s.invert !== !!s.array_invert ? ' · ø' : ''}` }),
    h('button', { class: 'icon-btn small' + (s.mute ? ' on' : ''), title: s.mute ? 'Unmute' : 'Mute',
      attrs: { 'aria-label': (s.mute ? 'Unmute ' : 'Mute ') + s.name, 'aria-pressed': String(!!s.mute) },
      onclick: (e) => { e.stopPropagation(); mutate((p) => { const t = p.speakers.find((x) => x.id === s.id); t.mute = !t.mute; }, { structural: true }); } },
    icon(s.mute ? 'mute' : 'sound')));
  }

  function renderList() {
    if (!listBox) return;
    clear(listBox);
    if (state.ui.tab === 'speakers') {
      if (!P().speakers.length) listBox.append(h('p', { class: 'hint', text: 'No speakers yet. Pick a type above and press Add.' }));
      listBox.append(...P().speakers.map(speakerRow));
    } else if (state.ui.tab === 'mics') {
      if (!P().mics.length) listBox.append(h('p', { class: 'hint', text: 'No microphone positions yet.' }));
      listBox.append(...P().mics.map((m) => {
        const sel = state.selection?.kind === 'mic' && state.selection.id === m.id;
        return h('div', { class: 'row-item' + (sel ? ' selected' : ''), tabIndex: 0, attrs: { role: 'button', 'aria-pressed': String(sel) },
          onclick: () => select('mic', m.id), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select('mic', m.id); } } },
        h('span', { class: 'name', text: m.name }),
        h('span', { class: 'meta', text: `${len(m.y)} from stage` }));
      }));
    }
  }

  function eqEditor(s) {
    const set = (peq, opts) => editSpeaker(s.id, { peq }, opts);
    const cur = () => speakerById(s.id)?.peq || [];
    const patch = (i, key) => (v, opts) => set(cur().map((b, k) => (k === i ? { ...b, [key]: v } : b)), opts);
    const rows = cur().map((b, i) => {
      const flat = b.type === 'highpass' || b.type === 'lowpass';
      const enabled = h('input', { type: 'checkbox', checked: b.enabled, title: 'Band on/off', attrs: { 'aria-label': `Band ${i + 1} enabled` },
        onchange: () => patch(i, 'enabled')(enabled.checked, DONE) });
      return h('div', { class: 'eq-row' }, enabled,
        choice(`Band ${i + 1} type`, FILTERS, () => cur()[i]?.type, patch(i, 'type'), { bare: true }),
        num(`Band ${i + 1} frequency`, () => cur()[i]?.freq, patch(i, 'freq'), { min: 10, max: 24000, step: 1, nd: 1, bare: true }),
        flat ? h('span') : num(`Band ${i + 1} gain`, () => cur()[i]?.gain_db, patch(i, 'gain_db'), { min: -24, max: 24, step: 0.5, nd: 1, bare: true }),
        num(`Band ${i + 1} Q`, () => cur()[i]?.q, patch(i, 'q'), { min: 0.1, max: 30, step: 0.1, nd: 2, bare: true }),
        h('button', { class: 'icon-btn small', title: 'Remove band', attrs: { 'aria-label': `Remove band ${i + 1}` },
          onclick: () => set(cur().filter((_, k) => k !== i), { ...DONE, structural: true }) }, icon('x')));
    });
    return group(h('span', null, 'Parametric EQ'),
      rows.length ? h('div', { class: 'eq-row eq-head' }, h('span'), h('span', { text: 'Type' }), h('span', { text: 'Hz' }), h('span', { text: 'dB' }), h('span', { text: 'Q' }), h('span')) : null,
      rows,
      h('button', { class: 'btn small', onclick: () => set([...cur(), { id: uid(), type: 'bell', freq: 1000, gain_db: 0, q: 1.4, enabled: true }], { ...DONE, structural: true }) },
        icon('plus'), 'Add band'),
      cur().some((b) => b.id.startsWith('auto')) ? h('p', { class: 'hint', text: 'Bands fitted by Auto EQ are replaced the next time it runs. Bands you add by hand are kept.' }) : null);
  }

  function speakerEditor(s) {
    const g = (k) => () => speakerById(s.id)?.[k];
    const set = (k) => (v, opts) => editSpeaker(s.id, { [k]: v }, opts);
    const partner = partnerOf(s.id);
    const applyType = (id, opts) => {
      const t = speakerType(id);
      if (!t) return;
      const { role, ...acoustic } = typeFields(t); // keep the role the user gave it (a delay stays a delay)
      editSpeaker(s.id, { model: t.id, ...acoustic, ...(t.role === 'sub' || s.role === 'sub' ? { role } : {}) }, { ...opts, structural: true });
    };
    const type = speakerType(s.model);
    const r = P().room;
    const arrayGroup = type?.spec?.kind === 'element' && s.array_boxes > 0 ? group('Line array hang',
      h('div', { class: 'grid2' },
        num('Boxes', g('array_boxes'), (v, o) => editSpeaker(s.id, hangFields(type, clamp(v, 1, 32), speakerById(s.id).array_splay), o),
          { min: 1, max: 32, step: 1, nd: 0 }),
        type.spec.constant_curvature
          ? h('div', { class: 'field' }, h('span', { text: 'Splay' }), h('div', { class: 'open-surface', text: `${type.spec.splay_max}° fixed per box` }))
          : num('Splay per box', g('array_splay'), (v, o) => editSpeaker(s.id, hangFields(type, speakerById(s.id).array_boxes, clamp(v, 0, type.spec.splay_max)), o),
            { min: 0, max: type.spec.splay_max, step: 0.5, unit: '°', nd: 1, title: `0° to ${type.spec.splay_max}° between boxes` })),
      h('p', { class: 'hint', text: `${s.array_boxes} × ${type.label.replace(/, hang of \d+$/, '')}, ${len(s.array_boxes * (s.array_box_h || 0.3))} long, fanned over ${fmt(s.array_boxes * s.array_splay, 0)}°. Each box (${type.spec.box_spl} dB peak at 1 m) is computed on its own, so throw, the narrowing beam and the low-frequency build-up come from how the boxes add up.` }),
      s.array_boxes * (s.array_box_h || 0.3) / 2 > s.z
        ? h('p', { class: 'hint warn', text: `The hang is longer than twice its height above the floor: its lower boxes would be below the floor. Raise it to at least ${len(s.array_boxes * (s.array_box_h || 0.3) / 2 + 0.2)}.` })
        : s.z + s.array_boxes * (s.array_box_h || 0.3) / 2 > r.height
          ? h('p', { class: 'hint warn', text: `The top of the hang would be above the ceiling. Lower it to at most ${len(r.height - s.array_boxes * (s.array_box_h || 0.3) / 2 - 0.1)}, or use fewer boxes.` }) : null) : null;
    const mates = s.array_group ? P().speakers.filter((q) => q.array_group === s.array_group) : [];
    const subArray = mates.length > 1 ? group('Sub array',
      h('p', { class: 'hint', text: `One of ${mates.length} boxes in an array. Processor output: ${fmt(s.delay_ms + s.array_delay_ms, 2)} ms (alignment ${fmt(s.delay_ms, 2)} + pattern ${fmt(s.array_delay_ms, 2)})${s.invert !== s.array_invert ? ', polarity inverted' : ''}. The optimizer moves the whole array together and never changes the pattern. The pattern was worked out for the boxes where they were built: if you move one, rebuild the array.` }),
      h('button', { class: 'btn small', title: 'Fold each box\'s offset into its own delay and polarity, and let the optimizer treat them separately',
        onclick: () => mutate((p) => {
          for (const q of p.speakers) {
            if (q.array_group !== s.array_group) continue;
            q.delay_ms = Math.round((q.delay_ms + q.array_delay_ms) * 100) / 100;
            q.invert = q.invert !== q.array_invert;
            q.array_group = null; q.array_delay_ms = 0; q.array_invert = false;
          }
        }, { structural: true }) }, 'Break up the array')) : null;
    return h('div', { class: 'editor' },
      partner ? h('div', { class: 'linked', title: 'Mirror-image pairs share every setting. Turn this off under Setup.' }, icon('link'),
        h('span', { text: `Linked with ${partner.name}: edits apply to both.` })) : null,
      group(null, h('div', { class: 'grid2' },
        textField('Name', g('name'), set('name')),
        h('div', { class: 'field wide' }, h('span', { text: 'Model' }),
          h('button', { class: 'btn type-btn', attrs: { 'aria-label': 'Change loudspeaker model' }, onclick: () => pickSpeakerType(s.model, (id) => applyType(id, DONE)) },
            h('span', { text: type?.label || 'Custom' }), icon('chevron'))),
        choice('Role', ROLES, g('role'), set('role'), { wide: true, title: 'Mains are the time reference. Fills, delays and subs are aligned to them.' }))),
      group('Position and aim', h('div', { class: 'grid3' },
        num('From stage', g('y'), set('y'), { min: 0.1, max: r.length - 0.1, step: 0.1, unit: 'm' }),
        num('Across', g('x'), set('x'), { min: 0.1, max: r.width - 0.1, step: 0.1, unit: 'm' }),
        num('Height', g('z'), set('z'), { min: 0.1, max: r.height - 0.05, step: 0.1, unit: 'm' }),
        num('Turn', g('yaw'), set('yaw'), { min: -180, max: 180, step: 1, unit: '°', nd: 0, title: '0° faces the audience. Positive turns toward the bottom of the plan.' }),
        num('Tilt', g('pitch'), set('pitch'), { min: -90, max: 90, step: 1, unit: '°', nd: 0, title: 'Negative tilts down toward the seats.' })),
        s.role === 'sub' || s.coverage_h >= 300 ? null : h('button', { class: 'btn small', style: { marginTop: '8px' },
          title: 'Point it at the seats it serves: the back third for a single box; for a hang, the top box on the last row and the bottom box on the first',
          onclick: () => {
            const { patch, note } = aimPatch(speakerById(s.id), P(), type);
            if (patch) editSpeaker(s.id, patch, { structural: true });
            toast(note, { ms: 7000 });
          } }, icon('target'), 'Aim at the audience')),
      arrayGroup,
      subArray,
      group('Processor output', h('div', { class: 'grid2' },
        num('Delay', g('delay_ms'), set('delay_ms'), { min: 0, max: 500, step: 0.1, unit: 'ms' }),
        num('Gain', g('gain_db'), set('gain_db'), { min: -40, max: 20, step: 0.5, unit: 'dB', nd: 1 })),
      h('div', { class: 'grid2', style: { marginTop: '8px' } },
        check('Invert polarity', g('invert'), set('invert')),
        check('Mute', g('mute'), (v, o) => mutate((p) => { p.speakers.find((x) => x.id === s.id).mute = v; }, o)),
        check('Lock delay', g('lock_delay'), set('lock_delay')),
        check('Lock gain', g('lock_gain'), set('lock_gain'))),
      h('p', { class: 'hint', text: 'Locked values are never changed by the optimizer.' })),
      eqEditor(s),
      type?.spec ? specCard(type) : null,
      h('details', { class: 'group' }, h('summary', { class: 'hint', style: { cursor: 'pointer', marginBottom: '8px' }, text: 'Loudspeaker model' }),
        h('div', { class: 'grid2' },
          num('Coverage, horizontal', g('coverage_h'), set('coverage_h'), { min: 5, max: 360, step: 5, unit: '°', nd: 0 }),
          num('Coverage, vertical', g('coverage_v'), set('coverage_v'), { min: 5, max: 360, step: 5, unit: '°', nd: 0 }),
          num('Max SPL at 1 m', g('level_db'), set('level_db'), { min: 60, max: 160, step: 1, unit: 'dB', nd: 0 }),
          num('Pattern control from', g('directivity_hz'), set('directivity_hz'), { min: 50, max: 10000, step: 50, unit: 'Hz', nd: 0,
            title: 'Below this frequency the box gradually becomes omnidirectional.' }),
          num('High-pass', g('hp_hz'), set('hp_hz'), { min: 0, max: 2000, step: 5, unit: 'Hz', nd: 0, title: '0 = none' }),
          num('Low-pass', g('lp_hz'), set('lp_hz'), { min: 0, max: 24000, step: 5, unit: 'Hz', nd: 0, title: '0 = none' }))),
      h('div', { style: { display: 'flex', gap: '6px', flexWrap: 'wrap' } },
        h('button', { class: 'btn small', title: 'Add a mirror-image partner on the other side of the room', onclick: () => mirrorSpeaker(s.id) }, icon('mirror'), 'Mirror'),
        h('button', { class: 'btn small', onclick: () => duplicateSpeaker(s.id) }, icon('copy'), 'Duplicate'),
        h('button', { class: 'btn small', title: 'Keep these coverage and level figures as one of your models', onclick: () => saveAsModel(speakerById(s.id)) }, icon('save'), 'Save as my model'),
        h('span', { style: { flex: 1 } }),
        h('button', { class: 'btn small danger', onclick: () => removeSpeaker(s.id) }, icon('trash'), 'Delete')));
  }

  function specCard(t) {
    const sp = t.spec;
    const low = sp.low_limit ? `${sp.low_limit} Hz${sp.low_limit_db ? ` (${sp.low_limit_db} dB)` : ''}` : '–';
    return group('Datasheet',
      h('dl', { class: 'spec' },
        h('dt', { text: 'Coverage' }), h('dd', { text: sp.coverage }),
        h('dt', { text: 'Max SPL' }), h('dd', { text: `${sp.max_spl} dB at 1 m${sp.kind === 'element' ? ', one box' : ''}` }),
        h('dt', { text: 'Low limit' }), h('dd', { text: low })),
      sp.note ? h('p', { class: 'hint', text: sp.note }) : null,
      h('p', { class: 'hint', text: `Estimated for the simulation: ${sp.estimated.join('; ')}.` }),
      h('a', { class: 'hint link', href: sp.source, target: '_blank', rel: 'noopener', text: 'Manufacturer source ↗' }));
  }

  async function saveAsModel(s) {
    const name = h('input', { type: 'text', value: s.name, maxLength: 60, autofocus: true, attrs: { 'aria-label': 'Model name' } });
    const brand = h('input', { type: 'text', value: 'My speakers', maxLength: 60, attrs: { 'aria-label': 'Brand' } });
    const r = await openDialog({ title: 'Save as my model', wide: false,
      body: h('div', { class: 'grid2' }, h('label', { class: 'field' }, h('span', { text: 'Brand' }), brand), h('label', { class: 'field' }, h('span', { text: 'Model' }), name),
        h('p', { class: 'hint', style: { gridColumn: '1 / -1' }, text: `Saves ${fmt(s.coverage_h, 0)}° × ${fmt(s.coverage_v, 0)}°, ${fmt(s.level_db, 0)} dB, high-pass ${fmt(s.hp_hz, 0)} Hz and pattern control from ${fmt(s.directivity_hz, 0)} Hz.` })),
      actions: [{ label: 'Cancel' }, { label: 'Save', kind: 'primary' }] });
    if (r !== 'Save') return;
    const id = `${brand.value} ${name.value}`.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 60) || `my_${uid()}`;
    try {
      await api.saveUserSpeaker({ id: `my_${id}`.slice(0, 60), brand: brand.value.trim() || 'My speakers', model: name.value.trim() || s.name,
        kind: s.role === 'sub' ? 'sub' : 'point', coverage_h: s.coverage_h, coverage_v: s.coverage_v, max_spl: s.level_db, spl_basis: 'peak',
        lf_hz: Math.max(10, s.hp_hz || 40), lf_db: -3, hf_hz: s.lp_hz || null, pattern_hz: s.directivity_hz, note: 'Saved from a speaker in the app.' });
      await loadSpeakerLibrary(api);
      toast(`Saved "${name.value}" to your speakers.`);
      render();
    } catch (e) { toast(`Could not save: ${e.message}`, { error: true }); }
  }

  function importSpeakers() {
    const input = h('input', { type: 'file', accept: '.csv,.json,text/csv,application/json', style: { display: 'none' },
      onchange: async () => {
        const f = input.files?.[0];
        input.remove();
        if (!f) return;
        try {
          const res = await api.importSpeakers(await f.text());
          await loadSpeakerLibrary(api);
          render();
          const bad = res.error_count ? ` ${res.error_count} row(s) skipped: ${res.errors.slice(0, 3).map((e) => `row ${e.row}: ${e.error}`).join('; ')}` : '';
          toast(`Imported ${res.added} speaker(s).${bad}`, { error: !res.added, ms: bad ? 12000 : 5000 });
        } catch (e) { toast(e.message, { error: true }); }
      } });
    document.body.append(input);
    input.click();
  }

  function speakersTab() {
    const libCount = state.meta.speaker_types.filter((t) => t.spec && !t.spec.user).length;
    const mine = state.meta.speaker_types.filter((t) => t.spec?.user).length;
    const openAdd = () => pickSpeakerType(null, (id) => addSpeaker(id), { title: 'Add a speaker',
      footer: h('p', { class: 'hint', style: { marginTop: '10px' }, text: `${libCount} checked library models, ${mine} of your own, plus generic types. Enter adds the highlighted one.` }) });
    const pick = h('div', { class: 'add-row' },
      h('button', { class: 'btn primary', onclick: openAdd, title: 'Search the loudspeaker library' }, icon('plus'), 'Add a speaker…'),
      h('button', { class: 'btn', onclick: () => import('./subarray.js').then((m) => m.openSubArrayBuilder()), title: 'Cardioid, end-fire or arc-delayed sub arrays' }, 'Sub array…'));
    listBox = h('div', { class: 'row-list' });
    renderList();
    const sel = state.selection?.kind === 'speaker' ? speakerById(state.selection.id) : null;
    return [
      group('Add a speaker', pick,
        h('div', { class: 'lib-actions' },
          h('span', { class: 'hint', text: `${libCount} library models · ${mine} of yours` }),
          h('span', { style: { flex: 1 } }),
          h('a', { class: 'btn small ghost', href: '/api/speakers/template.csv', download: 'speakers-template.csv', title: 'A CSV with the columns the import expects' }, 'Template'),
          h('button', { class: 'btn small', onclick: importSpeakers, title: 'Add your own loudspeakers from a CSV or JSON file' }, 'Import list…'))),
      group(`Speakers (${P().speakers.length})`, listBox,
        P().speakers.some((q) => q.role !== 'sub') ? h('button', { class: 'btn small', style: { marginTop: '8px' },
          title: 'Aim every speaker except subs at the seats it serves, in one step you can undo',
          onclick: () => {
            const notes = [];
            mutate((p) => {
              for (const q of p.speakers) {
                const { patch } = aimPatch(q, p, speakerType(q.model));
                if (patch) { Object.assign(q, patch); notes.push(q.name); }
              }
            }, { structural: true });
            toast(notes.length ? `Aimed ${notes.length} speaker${notes.length > 1 ? 's' : ''} at the audience. Ctrl+Z undoes it.` : 'Nothing to aim.', { ms: 6000 });
          } }, 'Aim every speaker at the audience') : null),
      sel ? speakerEditor(sel) : (P().speakers.length ? h('p', { class: 'hint', text: 'Select a speaker here or on the plan to edit it.' }) : null),
    ];
  }

  // ------------------------------------------------------------------ measurements
  function predictedCurve() {
    if (state.ui.source === 'average') return state.sim?.average ? { f: state.sim.freqs, v: state.sim.average.spl } : null;
    return state.point ? { f: state.point.freqs, v: state.point.spl_smooth } : null;
  }

  function importMeasurement() {
    const input = h('input', { type: 'file', accept: '.txt,.csv,.tsv,.frd,.dat,text/plain,text/csv', style: { display: 'none' },
      onchange: async () => {
        const f = input.files?.[0];
        input.remove();
        if (!f) return;
        if (f.size > 20_000_000) { toast('That file is larger than 20 MB.', { error: true }); return; }
        try {
          const m = parseResponse(await f.text());
          const used = new Set([...(P().measurements || []).map((x) => x.color), ...P().speakers.map((x) => x.color)]);
          const color = [5, 6, 7, 3, 2, 1, 0, 4].find((c) => !used.has(c)) ?? 5;
          const pred = predictedCurve();
          const offset = pred ? matchOffset(m, pred.f, pred.v) : 0;
          const name = f.name.replace(/\.[^.]+$/, '').slice(0, 80) || 'Measurement';
          if ((P().measurements || []).length >= 8) { toast('Up to 8 measurements per venue. Delete one first.', { error: true }); return; }
          mutate((p) => { p.measurements = [...(p.measurements || []), { id: `m${uid()}`, name, ...m, offset_db: offset, color }]; }, { structural: true });
          toast(`Imported "${name}": ${m.freqs.length} points${m.phase ? ' with phase' : ''}, ${fmtHzRange(m.freqs)}.${offset ? ` Level offset ${fmt(offset, 1, true)} dB to line it up with the prediction (200 Hz–2 kHz).` : ''}`, { ms: 8000 });
        } catch (e) { toast(e.message, { error: true }); }
      } });
    document.body.append(input);
    input.click();
  }
  const fmtHzRange = (f) => `${Math.round(f[0])} Hz–${f.at(-1) >= 1000 ? `${fmt(f.at(-1) / 1000, 1)} kHz` : `${Math.round(f.at(-1))} Hz`}`;

  function measurementsGroup() {
    const list = P().measurements || [];
    const setM = (id, fn) => mutate((p) => { const m = p.measurements.find((x) => x.id === id); if (m) fn(m); });
    return group('Measurements',
      list.length ? h('div', {}, list.map((m) => h('div', { class: 'meas-row' },
        h('span', { class: 'swatch', style: { background: seriesColor(m.color) } }),
        h('input', { type: 'text', class: 'bare name', value: m.name, maxLength: 80, attrs: { 'aria-label': 'Measurement name' }, onchange: (e) => setM(m.id, (x) => { x.name = e.target.value.trim() || 'Measurement'; }) }),
        num('Level offset', () => (P().measurements || []).find((x) => x.id === m.id)?.offset_db ?? 0, (v, o) => setM(m.id, (x) => { x.offset_db = v; }),
          { min: -80, max: 80, step: 0.5, nd: 1, bare: true, title: 'Level offset in dB added to the measured curve' }),
        h('button', { class: 'btn small ghost', title: 'Line the measured curve up with the prediction between 200 Hz and 2 kHz', onclick: () => {
          const pred = predictedCurve();
          if (!pred) { toast('No prediction to match yet.', { error: true }); return; }
          setM(m.id, (x) => { x.offset_db = matchOffset(x, pred.f, pred.v); });
        } }, 'Match'),
        h('button', { class: 'icon-btn small', attrs: { 'aria-label': `Delete ${m.name}` }, onclick: () => mutate((p) => { p.measurements = p.measurements.filter((x) => x.id !== m.id); }, { structural: true }) }, icon('trash'))))) : null,
      h('button', { class: 'btn small', style: { marginTop: '8px' }, onclick: importMeasurement }, icon('plus'), 'Import a measurement…'),
      h('p', { class: 'hint', text: 'Overlay a measured response on the chart: REW "Export measurement as text", a Smaart ASCII export, or any CSV of frequency, dB and phase. Shown dashed, smoothed to 1/6 octave like the prediction. Measurements are rarely calibrated, so the level is matched to the prediction on import.' }));
  }

  // ------------------------------------------------------------------ mics
  function micsTab() {
    listBox = h('div', { class: 'row-list' });
    renderList();
    const m = state.selection?.kind === 'mic' ? P().mics.find((x) => x.id === state.selection.id) : null;
    const g = (k) => () => P().mics.find((x) => x.id === m.id)?.[k];
    const set = (k) => (v, opts) => mutate((p) => { const t = p.mics.find((x) => x.id === m.id); if (t) t[k] = v; }, opts);
    const r = P().room;
    return [
      group('Measurement positions', listBox,
        h('button', { class: 'btn small', style: { marginTop: '8px' }, onclick: addMic }, icon('plus'), 'Add at the probe'),
        h('p', { class: 'hint', text: 'Mark where you would stand a measurement mic. Pick any of them in the response chart to see what it would read.' })),
      measurementsGroup(),
      m ? h('div', { class: 'editor' },
        group(null, h('div', { class: 'grid2' }, textField('Name', g('name'), set('name')))),
        group('Position', h('div', { class: 'grid3' },
          num('From stage', g('y'), set('y'), { min: 0, max: r.length, step: 0.1, unit: 'm' }),
          num('Across', g('x'), set('x'), { min: 0, max: r.width, step: 0.1, unit: 'm' }),
          num('Height', g('z'), set('z'), { min: 0.1, max: r.height, step: 0.1, unit: 'm' }))),
        h('div', { style: { display: 'flex', gap: '6px' } },
          h('button', { class: 'btn small', onclick: () => setUI({ source: m.id }) }, 'Show its response'),
          h('span', { style: { flex: 1 } }),
          h('button', { class: 'btn small danger', onclick: () => removeMic(m.id) }, icon('trash'), 'Delete'))) : null,
    ];
  }

  // ----------------------------------------------------------------- setup
  function setupTab() {
    const st = (k) => () => P().settings[k];
    const slider = ([key, label]) => {
      const out = h('output', { text: `${fmt(state.ui.weights[key], 2)}×` });
      const el = h('input', { type: 'range', min: 0, max: 3, step: 0.25, value: state.ui.weights[key], attrs: { 'aria-label': `${label} priority` },
        oninput: () => { out.textContent = `${fmt(Number(el.value), 2)}×`; },
        onchange: () => setUI({ weights: { ...state.ui.weights, [key]: Number(el.value) } }) });
      return h('div', { class: 'slider-row' }, h('span', { text: label }), el, out);
    };
    return [
      group('Alignment',
        h('div', { class: 'grid2' },
          num('Target tilt', st('target_tilt_db_per_oct'), setIn(['settings', 'target_tilt_db_per_oct']), { min: -6, max: 3, step: 0.25, unit: 'dB/oct',
            title: 'Slope of the target response. −1 dB/oct is a gentle, natural-sounding tilt; 0 is flat.' }),
          num('Fill precedence', st('precedence_ms'), setIn(['settings', 'precedence_ms']), { min: 0, max: 30, step: 1, unit: 'ms', nd: 0,
            title: 'Extra delay on fills and delays so the sound still seems to come from the stage.' })),
        check('Keep mirror pairs matched', st('link_symmetric'), setIn(['settings', 'link_symmetric']), 'Left/right twins always get identical settings.')),
      group('Optimizer priorities', PRIORITIES.map(slider),
        h('div', { class: 'grid2', style: { marginTop: '8px' } },
          (() => { const f = num('Search time', () => state.ui.budget, (v, o) => { if (o.commit) setUI({ budget: v }); }, { min: 3, max: 120, step: 1, unit: 's', nd: 0 }); return f; })(),
          num('Auto EQ bands, max', () => state.ui.eqBands, (v, o) => { if (o.commit) setUI({ eqBands: Math.round(v) }); }, { min: 1, max: 10, step: 1, nd: 0 })),
        h('button', { class: 'btn small', style: { marginTop: '8px' }, onclick: () => { setUI({ weights: { ...DEFAULT_WEIGHTS } }); render(); } }, 'Reset priorities')),
      group('Units',
        choice('Lengths and distances', [['metric', 'Metric (m)'], ['imperial', 'Imperial (ft)']], () => state.ui.units || 'metric', (v) => setUI({ units: v })),
        h('p', { class: 'hint', text: 'Only what is shown changes. Venues are saved in metres, so a file opens the same either way.' })),
      group('Model detail',
        h('div', { class: 'grid2' },
          choice('Reflections', [['0', 'None (free field)'], ['1', '1st order'], ['2', '2nd order'], ['3', '3rd order (slower)'], ['4', '4th order (slow)']],
            () => String(P().settings.reflection_order), (v, o) => setIn(['settings', 'reflection_order'])(Number(v), o)),
          num('Map resolution', st('grid_resolution'), setIn(['settings', 'grid_resolution']), { min: 0.2, max: 3, step: 0.1, unit: 'm' })),
        h('p', { class: 'hint', text: 'This is a prediction from a box-shaped room and generic loudspeaker patterns. Use it to plan and to get close; confirm delays and EQ with a measurement on site.' })),
    ];
  }

  // ---------------------------------------------------------------- render
  function render() {
    bound = [];
    listBox = null;
    statsBox = null;
    clear(tabBar).append(...TABS.map(([id, label]) => h('button', { class: 'tab', text: label, attrs: { role: 'tab', 'aria-selected': String(state.ui.tab === id) },
      onclick: () => setUI({ tab: id }) })));
    const scroll = body.scrollTop;
    clear(body);
    if (!state.project || !state.meta) return;
    const content = { room: roomTab, speakers: speakersTab, mics: micsTab, setup: setupTab }[state.ui.tab] || roomTab;
    body.append(...content().filter(Boolean));
    body.scrollTop = scroll;
  }

  let lastTab = null, lastDisp = null;
  let lastUnits = '';
  on('project', (d) => {
    if (d.structural && !(d.source === 'panel' && document.activeElement && body.contains(document.activeElement) && document.activeElement.tagName === 'INPUT' && document.activeElement.type !== 'checkbox')) render();
    else { sync(); renderList(); }
  });
  on('selection', render);
  on('ui', () => {
    // Fields capture their unit when built, so a change of units (including a restored session) rebuilds them.
    if (state.ui.tab !== lastTab || state.ui.units !== lastDisp) { if (state.ui.tab !== lastTab) body.scrollTop = 0; lastTab = state.ui.tab; lastDisp = state.ui.units; render(); } else renderList();
  });
  on('sim', () => {
    renderStats();
    const units = JSON.stringify(state.sim?.units || []);
    if (units !== lastUnits) { lastUnits = units; if (state.ui.tab === 'speakers' && !body.contains(document.activeElement)) render(); }
  });
  on('theme', render);
  lastTab = state.ui.tab;
  lastDisp = state.ui.units;
  render();
  return { render };
}
