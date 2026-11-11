// Left panel: the venue editor (room, speakers, microphones, setup).

import {
  addMic, addSpeaker, DEFAULT_WEIGHTS, duplicateSpeaker, editSpeaker, mirrorSpeaker, mutate, on, partnerOf,
  removeMic, removeSpeaker, select, setUI, speakerById, state,
} from './store.js';
import { clamp, clear, fmt, h, icon, toast, uid } from './util.js';
import { seriesColor } from './theme.js';

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

  function num(label, get, set, { min, max, step = 0.1, unit = '', nd = 2, wide = false, title = null, bare = false } = {}) {
    const input = h('input', { type: 'number', min, max, step, value: show(get(), nd), class: bare ? 'bare' : '', title,
      attrs: { 'aria-label': label, inputmode: 'decimal' } });
    const read = () => {
      const v = parseFloat(input.value);
      return Number.isFinite(v) && (min == null || v >= min) && (max == null || v <= max) ? v : null;
    };
    input.addEventListener('input', () => {
      const v = read();
      input.classList.toggle('invalid', v == null);
      if (v != null && v !== get()) set(v, LIVE);
    });
    input.addEventListener('change', () => {
      let v = parseFloat(input.value);
      v = Number.isFinite(v) ? clamp(v, min ?? -Infinity, max ?? Infinity) : get();
      input.classList.remove('invalid');
      set(v, DONE);
      input.value = show(v, nd);
    });
    bound.push({ el: input, get, nd });
    if (bare) return input;
    return h('label', { class: 'field' + (wide ? ' wide' : ''), title }, h('span', { text: label }),
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
    const mats = state.meta.materials.map((m) => [m.id, m.name]);
    return [
      scanGroup(),
      group('Room size',
        h('div', { class: 'grid3' },
          num('Stage to back', () => P().room.length, setIn(['room', 'length']), { min: 3, max: 300, step: 0.5, unit: 'm' }),
          num('Width', () => P().room.width, setIn(['room', 'width']), { min: 3, max: 200, step: 0.5, unit: 'm' }),
          num('Height', () => P().room.height, setIn(['room', 'height']), { min: 2, max: 60, step: 0.1, unit: 'm' }))),
      group('Surfaces', h('div', { class: 'grid2' }, SURFACES.map(([k, label]) =>
        choice(label, mats, () => P().room.materials[k], setIn(['room', 'materials', k])))),
      h('p', { class: 'hint', text: 'Empty rooms are live. Set the floor to "Audience, seated" to see the room with people in it.' })),
      group('Audience area',
        h('div', { class: 'grid2' },
          num('Starts, from stage', () => P().audience.y0, setIn(['audience', 'y0']), { min: 0, step: 0.5, unit: 'm' }),
          num('Ends, from stage', () => P().audience.y1, setIn(['audience', 'y1']), { min: 0, step: 0.5, unit: 'm' }),
          num('Starts, across', () => P().audience.x0, setIn(['audience', 'x0']), { min: 0, step: 0.5, unit: 'm' }),
          num('Ends, across', () => P().audience.x1, setIn(['audience', 'x1']), { min: 0, step: 0.5, unit: 'm' }),
          num('Ear height', () => P().audience.ear_height, setIn(['audience', 'ear_height']), { min: 0.3, max: 20, step: 0.1, unit: 'm',
            title: '1.2 m seated, 1.7 m standing' }))),
      group('Air',
        h('div', { class: 'grid2' },
          num('Temperature', () => P().environment.temperature_c, setIn(['environment', 'temperature_c']), { min: -20, max: 50, step: 1, unit: '°C' }),
          num('Humidity', () => P().environment.humidity_pct, setIn(['environment', 'humidity_pct']), { min: 0, max: 100, step: 5, unit: '%' })),
        h('p', { class: 'hint', text: 'Temperature changes the speed of sound, and with it every delay time.' })),
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
        h('p', { class: 'hint', text: 'GLB, OBJ or PLY from a LiDAR app (Polycam, Scaniverse, 3d Scanner App) or a laser scanner. The room size is measured from it.' }));
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

  function renderStats() {
    if (!statsBox) return;
    const r = state.sim?.room;
    const cell = (label, value) => h('div', null, h('span', { text: label }), h('b', { text: value }));
    clear(statsBox).append(
      cell('Reverb time (mid)', r ? `${fmt(r.rt60_mid, 2)} s` : '–'),
      cell('Critical distance', r ? `${fmt(r.critical_distance_m[3], 1)} m` : '–'),
      cell('Volume', r ? `${Math.round(r.volume_m3).toLocaleString()} m³` : '–'),
      cell('Speed of sound', r ? `${fmt(r.speed_of_sound, 1)} m/s` : '–'));
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
    h('span', { class: 'meta', text: `${fmt(s.delay_ms, 1)} ms · ${fmt(s.gain_db, 1, true)} dB${s.invert ? ' · ø' : ''}` }),
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
        h('span', { class: 'meta', text: `${fmt(m.y, 1)} m from stage` }));
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
    const types = state.meta.speaker_types;
    const applyType = (id, opts) => {
      const t = types.find((x) => x.id === id);
      if (!t) return;
      const { label, id: model, ...acoustic } = t;
      editSpeaker(s.id, { model, ...acoustic }, opts);
    };
    const r = P().room;
    return h('div', { class: 'editor' },
      partner ? h('div', { class: 'linked', title: 'Mirror-image pairs share every setting. Turn this off under Setup.' }, icon('link'),
        h('span', { text: `Linked with ${partner.name}: edits apply to both.` })) : null,
      group(null, h('div', { class: 'grid2' },
        textField('Name', g('name'), set('name')),
        choice('Type', [...types.map((t) => [t.id, t.label]), ...(types.some((t) => t.id === s.model) ? [] : [[s.model, 'Custom']])], g('model'), applyType, { wide: true }),
        choice('Role', ROLES, g('role'), set('role'), { wide: true, title: 'Mains are the time reference. Fills, delays and subs are aligned to them.' }))),
      group('Position and aim', h('div', { class: 'grid3' },
        num('From stage', g('y'), set('y'), { min: 0.1, max: r.length - 0.1, step: 0.1, unit: 'm' }),
        num('Across', g('x'), set('x'), { min: 0.1, max: r.width - 0.1, step: 0.1, unit: 'm' }),
        num('Height', g('z'), set('z'), { min: 0.1, max: r.height - 0.05, step: 0.1, unit: 'm' }),
        num('Turn', g('yaw'), set('yaw'), { min: -180, max: 180, step: 1, unit: '°', nd: 0, title: '0° faces the audience. Positive turns toward the bottom of the plan.' }),
        num('Tilt', g('pitch'), set('pitch'), { min: -90, max: 90, step: 1, unit: '°', nd: 0, title: 'Negative tilts down toward the seats.' }))),
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
        h('span', { style: { flex: 1 } }),
        h('button', { class: 'btn small danger', onclick: () => removeSpeaker(s.id) }, icon('trash'), 'Delete')));
  }

  function speakersTab() {
    const types = state.meta.speaker_types;
    const pick = h('select', { attrs: { 'aria-label': 'Speaker type to add' } }, types.map((t) => h('option', { value: t.id, text: t.label })));
    listBox = h('div', { class: 'row-list' });
    renderList();
    const sel = state.selection?.kind === 'speaker' ? speakerById(state.selection.id) : null;
    return [
      group('Add a speaker', h('div', { style: { display: 'flex', gap: '6px' } }, pick,
        h('button', { class: 'btn primary', onclick: () => addSpeaker(pick.value) }, icon('plus'), 'Add'))),
      group(`Speakers (${P().speakers.length})`, listBox),
      sel ? speakerEditor(sel) : (P().speakers.length ? h('p', { class: 'hint', text: 'Select a speaker here or on the plan to edit it.' }) : null),
    ];
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

  let lastTab = null;
  let lastUnits = '';
  on('project', (d) => {
    if (d.structural && !(d.source === 'panel' && document.activeElement && body.contains(document.activeElement) && document.activeElement.tagName === 'INPUT' && document.activeElement.type !== 'checkbox')) render();
    else { sync(); renderList(); }
  });
  on('selection', render);
  on('ui', () => { if (state.ui.tab !== lastTab) { lastTab = state.ui.tab; body.scrollTop = 0; render(); } else renderList(); });
  on('sim', () => {
    renderStats();
    const units = JSON.stringify(state.sim?.units || []);
    if (units !== lastUnits) { lastUnits = units; if (state.ui.tab === 'speakers' && !body.contains(document.activeElement)) render(); }
  });
  on('theme', render);
  lastTab = state.ui.tab;
  render();
  return { render };
}
