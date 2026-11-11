// Toolbar above the coverage view: which map, which band, plan or 3D.

import { mutate, on, setUI, state } from './store.js';
import { clear, h, toast } from './util.js';
import { MAPS } from './stage.js';

const BANDS = [['all', 'All'], ['63', '63'], ['125', '125'], ['250', '250'], ['500', '500'], ['1000', '1k'], ['2000', '2k'], ['4000', '4k'], ['8000', '8k']];

export function createStageBar(bar, stageRoot) {
  let view3d = null;
  let loading = null;
  let section = null;
  let sectionLoading = null;

  async function applyView() {
    const want3d = state.ui.view === '3d';
    const wantSection = state.ui.view === 'section';
    stageRoot.classList.toggle('is-3d', want3d || wantSection);
    if (!wantSection) section?.hide();
    if (wantSection) {
      view3d?.hide();
      sectionLoading ??= import('./section.js').then((m) => m.createSection(stageRoot));
      section = await sectionLoading;
      if (state.ui.view === 'section') section.show();
      return;
    }
    if (!want3d) { view3d?.hide(); return; }
    if (!view3d) {
      loading ??= import('./view3d.js').then((m) => m.create3D(stageRoot)).catch((e) => {
        console.error(e);
        toast(`The 3D view could not start: ${e.message}`, { error: true });
        setUI({ view: 'plan' });
        return null;
      });
      view3d = await loading;
      if (!view3d) return;
    }
    if (state.ui.view === '3d') view3d.show();
  }

  function render() {
    clear(bar);
    const def = MAPS[state.ui.map] || MAPS.direct;
    const mapSel = h('select', { attrs: { 'aria-label': 'Map' }, title: def.help, onchange: (e) => setUI({ map: e.target.value }) },
      Object.entries(MAPS).map(([id, m]) => h('option', { value: id, text: m.label })));
    mapSel.value = state.ui.map;
    const band = def.bands ? h('label', { class: 'band' }, h('span', { text: 'Band, Hz' }),
      h('select', { attrs: { 'aria-label': 'Frequency band' }, onchange: (e) => setUI({ band: e.target.value }) },
        BANDS.map(([v, t]) => h('option', { value: v, text: t, selected: state.ui.band === v })))) : null;
    const views = h('div', { class: 'seg', attrs: { role: 'group', 'aria-label': 'View' } },
      [['plan', 'Plan', 'Floor plan (1)'], ['section', 'Section', 'Vertical slice through the probe (3)'], ['3d', '3D', 'Orbit the venue (2)']].map(([id, label, tip]) =>
        h('button', { text: label, title: tip, attrs: { 'aria-pressed': String((state.ui.view || 'plan') === id) }, onclick: () => setUI({ view: id }) })));
    const inSection = state.ui.view === 'section';
    if (inSection) {
      mapSel.disabled = true;
      mapSel.value = 'direct';
      mapSel.title = 'The section shows direct level; pick a band to look at one octave.';
    }
    const bandSel = inSection ? h('label', { class: 'band' }, h('span', { text: 'Band, Hz' }),
      h('select', { attrs: { 'aria-label': 'Frequency band' }, onchange: (e) => setUI({ band: e.target.value }) },
        BANDS.map(([v, t]) => h('option', { value: v, text: t, selected: state.ui.band === v })))) : band;
    const levelMap = inSection || state.ui.map === 'direct' || state.ui.map === 'total';
    const wsel = levelMap ? h('label', { class: 'band', title: 'Frequency weighting of broadband levels: Z is flat, A follows the ear at moderate levels (and most noise limits), C keeps the low end and suits loud music.' },
      h('span', { text: 'Weighting' }),
      h('select', { attrs: { 'aria-label': 'Frequency weighting' }, onchange: (e) => mutate((p) => { p.settings.weighting = e.target.value; }) },
        ['Z', 'A', 'C'].map((w) => h('option', { value: w, text: w === 'Z' ? 'Z (flat)' : `${w}`, selected: (state.project?.settings?.weighting || 'Z') === w })))) : null;
    bar.append(...[h('span', { class: 'title', text: 'Coverage' }), mapSel, bandSel, wsel, h('span', { class: 'spacer' }), views].filter(Boolean));
  }

  on('project', () => {
    const sel = bar.querySelector("select[aria-label='Frequency weighting']");
    if (sel && sel.value !== (state.project?.settings?.weighting || 'Z')) render();
  });
  on('ui', (patch) => {
    if (!patch || 'map' in patch || 'band' in patch || 'view' in patch || Object.keys(patch).length === 0) render();
    if (!patch || 'view' in patch || Object.keys(patch).length === 0) applyView();
  });
  render();
  applyView();
  return { render };
}
