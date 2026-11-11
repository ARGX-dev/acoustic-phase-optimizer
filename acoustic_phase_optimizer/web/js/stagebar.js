// Toolbar above the coverage view: which map, which band, plan or 3D.

import { on, setUI, state } from './store.js';
import { clear, h, toast } from './util.js';
import { MAPS } from './stage.js';

const BANDS = [['all', 'All'], ['63', '63'], ['125', '125'], ['250', '250'], ['500', '500'], ['1000', '1k'], ['2000', '2k'], ['4000', '4k'], ['8000', '8k']];

export function createStageBar(bar, stageRoot) {
  let view3d = null;
  let loading = null;

  async function applyView() {
    const want3d = state.ui.view === '3d';
    stageRoot.classList.toggle('is-3d', want3d);
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
      [['plan', 'Plan'], ['3d', '3D']].map(([id, label]) =>
        h('button', { text: label, attrs: { 'aria-pressed': String((state.ui.view || 'plan') === id) }, onclick: () => setUI({ view: id }) })));
    bar.append(h('span', { class: 'title', text: 'Coverage' }), mapSel, band, h('span', { class: 'spacer' }), views);
  }

  on('ui', (patch) => {
    if (!patch || 'map' in patch || 'band' in patch || 'view' in patch || Object.keys(patch).length === 0) render();
    if (!patch || 'view' in patch || Object.keys(patch).length === 0) applyView();
  });
  render();
  applyView();
  return { render };
}
