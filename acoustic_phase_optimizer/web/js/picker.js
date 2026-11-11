// Searchable loudspeaker picker: generic types first, then real models grouped by brand.

import { state } from './store.js';
import { h, icon } from './util.js';
import { currentDialog, openDialog } from './dialog.js';

const KIND = { point: 'Point source', element: 'Line array', column: 'Column', sub: 'Subwoofer' };
const ROLE_KIND = { sub: 'Subwoofer', front_fill: 'Front fill', fill: 'Fill' };

function kindOf(t) {
  if (t.spec) return KIND[t.spec.kind];
  if (t.id.startsWith('line_array')) return 'Line array';
  if (t.id === 'column') return 'Column';
  return ROLE_KIND[t.role] || 'Point source';
}

function detail(t) {
  const cov = t.role === 'sub' || t.coverage_h >= 360 ? 'omni' : `${t.coverage_h}°×${t.coverage_v}°`;
  const spl = t.spec ? t.spec.max_spl : t.level_db;
  return `${kindOf(t)} · ${cov} · ${spl} dB`;
}

const LIMIT = 150;
const norm = (s) => s.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, ' ');

/**
 * A search box with a grouped, keyboard-navigable list.
 * onPick(id) runs on click or Enter. current: the id to highlight.
 */
export function speakerPicker({ onPick, current = null, autofocus = false, tall = false }) {
  const types = state.meta.speaker_types;
  const kinds = ['All', 'Point source', 'Line array', 'Subwoofer', 'Column'];
  let kind = 'All';
  let active = 0;
  let shown = [];
  const input = h('input', { type: 'search', placeholder: 'Search brand, model or size…', autofocus,
    attrs: { 'aria-label': 'Search loudspeakers', role: 'combobox', 'aria-expanded': 'true', 'aria-controls': 'spk-list', autocomplete: 'off' },
    oninput: () => { active = 0; render(); }, onkeydown: key });
  const list = h('div', { class: 'picker-list' + (tall ? ' tall' : ''), id: 'spk-list', attrs: { role: 'listbox' } });
  const chips = h('div', { class: 'chips' }, kinds.map((k) => h('button', {
    class: 'chip' + (k === kind ? ' on' : ''), text: k, attrs: { 'aria-pressed': String(k === kind) },
    onclick: (e) => { kind = k; active = 0; chips.querySelectorAll('.chip').forEach((c) => { c.classList.toggle('on', c === e.currentTarget); c.setAttribute('aria-pressed', String(c === e.currentTarget)); }); render(); },
  })));

  function render() {
    const q = norm(input.value).trim().split(' ').filter(Boolean);
    shown = types.filter((t) => {
      if (kind !== 'All' && kindOf(t) !== kind && !(kind === 'Point source' && /fill/i.test(kindOf(t)))) return false;
      const hay = norm(`${t.label} ${t.group || ''} ${kindOf(t)} ${t.role}`);
      return q.every((w) => hay.includes(w));
    });
    list.replaceChildren();
    if (!shown.length) { list.append(h('p', { class: 'hint empty', text: 'No loudspeaker matches. Try a brand or a size, like “15”.' })); return; }
    const total = shown.length;
    shown = shown.slice(0, LIMIT); // keep the list quick however big the library gets
    active = Math.min(active, shown.length - 1);
    let group = null;
    shown.forEach((t, i) => {
      if (t.group !== group) { group = t.group; list.append(h('div', { class: 'picker-group', text: group, attrs: { role: 'presentation' } })); }
      const label = t.spec ? t.label.replace(`${t.brand} `, '') : t.label.replace(/\s*\(.*\)$/, '');
      list.append(h('div', {
        class: 'picker-item' + (i === active ? ' active' : '') + (t.id === current ? ' current' : ''),
        id: `spk-${t.id}`, attrs: { role: 'option', 'aria-selected': String(i === active) },
        onpointerdown: (e) => e.preventDefault(), onclick: () => onPick(t.id), onpointermove: () => { if (active !== i) { active = i; mark(); } },
      }, h('span', { class: 'name', text: label }),
      t.spec?.approximate ? h('span', { class: 'tag', text: 'approx.', title: 'Some figures could not be confirmed on the manufacturer datasheet' })
        : t.spec?.user ? h('span', { class: 'tag mine', text: 'yours' }) : null,
      h('span', { class: 'meta', text: detail(t) })));
    });
    if (total > LIMIT) list.append(h('p', { class: 'hint empty', text: `${total - LIMIT} more. Type to narrow the list.` }));
    mark();
  }

  function mark() {
    list.querySelectorAll('.picker-item').forEach((el, i) => { el.classList.toggle('active', i === active); el.setAttribute('aria-selected', String(i === active)); });
    const el = list.querySelectorAll('.picker-item')[active];
    if (el) { input.setAttribute('aria-activedescendant', el.id); el.scrollIntoView({ block: 'nearest' }); }
  }

  function key(e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(shown.length - 1, active + 1); mark(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(0, active - 1); mark(); }
    if (e.key === 'Enter' && shown[active]) { e.preventDefault(); onPick(shown[active].id); }
  }

  render();
  return h('div', { class: 'picker' }, h('div', { class: 'search' }, icon('search'), input), chips, list);
}

/** Choose a model in a dialog. */
export function pickSpeakerType(current, onPick, { title = 'Loudspeaker model', footer = null } = {}) {
  const picker = speakerPicker({ current, autofocus: true, tall: true, onPick: (id) => { currentDialog()?.close(id); onPick(id); } });
  const p = openDialog({ title, body: footer ? h('div', {}, picker, footer) : picker, actions: [{ label: 'Cancel' }] });
  picker.querySelector('input').focus();
  return p;
}
