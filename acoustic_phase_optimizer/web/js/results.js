// Right panel: score, metrics, the optimizer and what it changed.

import { autoEq, cancelOptimize, clearComparison, deepOptimize, on, quickAlign, revertLastAction, setUI, state } from './store.js';
import { clear, fmt, h, icon, statusBadge } from './util.js';
import { seriesColor } from './theme.js';

const METRICS = [
  { key: 'coverage', label: 'Coverage evenness', show: (v) => `±${fmt(v, 1)} dB`, better: 'low',
    help: 'How much the direct sound level varies from seat to seat, per octave band. ±2 dB or less is very even.' },
  { key: 'clarity', label: 'Clarity', show: (v) => `${fmt(v, 1, true)} dB`, better: 'high',
    help: 'Early sound relative to late reverberation, averaged over the seats. Above +6 dB speech is easy to follow; below 0 it is a struggle.' },
  { key: 'echo_pct', label: 'Seats with late arrivals', show: (v) => `${fmt(v, 0)}%`, better: 'low',
    help: 'Seats where two audible speakers arrive more than 30 ms apart, which is heard as an echo. Alignment fixes this.' },
  { key: 'summation', label: 'Summation', show: (v) => `${fmt(v, 2, true)} dB`, better: 'high',
    help: 'Average level gained (+) or lost (−) by the speakers interacting, compared with simply adding their power.' },
  { key: 'cancel_pct', label: 'Cancellation', show: (v) => `${fmt(v, 1)}%`, better: 'low',
    help: 'Share of seat-and-frequency combinations where the speakers lose more than 3 dB against each other.' },
  { key: 'tonal', label: 'Off the target curve', show: (v) => `${fmt(v, 1)} dB`, better: 'low',
    help: 'RMS deviation of the audience-average response from the target tilt. Auto EQ reduces it.' },
  { key: 'mean_spl', label: 'Average level', show: (v) => {
    const w = state.project?.settings?.weighting;
    return `${fmt(v, 1)} ${w && w !== 'Z' ? `dB(${w})` : 'dB SPL'}`;
  }, better: null,
    help: 'Average level over the audience with every speaker at full output.' },
];
const LOST = [['coverage', 'Coverage', 0], ['clarity', 'Clarity', 1], ['timing', 'Timing', 2], ['interference', 'Interference', 3], ['tonal', 'Tone', 4]];
const FIELD = { delay_ms: ['Delay', 'ms', 2], gain_db: ['Gain', 'dB', 1] };

function grade(score) {
  if (score >= 85) return ['good', 'Excellent'];
  if (score >= 70) return ['good', 'Good'];
  if (score >= 55) return ['warning', 'Fair'];
  if (score >= 40) return ['serious', 'Poor'];
  return ['critical', 'Needs work'];
}

function delta(before, after, better, nd = 1) {
  if (before == null || after == null || better == null) return null;
  const d = after - before;
  if (Math.abs(d) < 0.5 * 10 ** -nd) return h('span', { class: 'delta flat', text: 'no change' });
  const good = (d < 0) === (better === 'low');
  return h('span', { class: 'delta ' + (good ? 'up' : 'down'), text: `${d > 0 ? '▲' : '▼'} ${fmt(Math.abs(d), nd)} ${good ? 'better' : 'worse'}` });
}

export function createResultsPanel(root) {
  const scoreBox = h('section');
  const actionBox = h('section');
  const changeBox = h('section');
  const findingBox = h('section');
  root.append(scoreBox, actionBox, changeBox, findingBox);

  function renderScore() {
    clear(scoreBox);
    const showingBefore = state.ui.compare === 'before' && state.simBaseline?.metrics;
    const m = showingBefore ? state.simBaseline.metrics : state.sim?.metrics;
    const ref = showingBefore ? null : (state.simBaseline?.metrics || state.lastAction?.before || null);
    scoreBox.append(h('div', { class: 'panel-title', text: showingBefore ? 'System score - before' : 'System score' }));
    if (!m) {
      scoreBox.append(h('p', { class: 'hint', text: state.project?.speakers.some((s) => !s.mute) ? 'Calculating…' : 'Add or unmute a speaker to get a score.' }));
      return;
    }
    const [level, label] = grade(m.score);
    scoreBox.append(h('div', { class: 'score' },
      h('div', { class: 'num', text: String(Math.round(m.score)) }),
      h('div', { class: 'score-side' }, statusBadge(level, label),
        ref ? delta(ref.score, m.score, 'high', 0) : h('span', { class: 'of', text: 'out of 100' }))));

    const lost = m.lost || {};
    const parts = LOST.map(([k, name, slot]) => ({ name, slot, v: Math.max(0, lost[k] || 0) })).filter((p) => p.v >= 0.05);
    if (parts.length) {
      const bar = h('div', { class: 'lost', attrs: { role: 'img', 'aria-label': 'Points lost: ' + parts.map((p) => `${p.name} ${fmt(p.v, 0)}`).join(', ') } },
        parts.map((p) => h('div', { style: { flex: `${p.v} 0 0`, background: seriesColor(p.slot), borderRadius: '4px' }, title: `${p.name}: ${fmt(p.v, 1)} points lost` })));
      scoreBox.append(bar, h('div', { class: 'lost-legend' }, h('span', { text: 'Points lost:' }),
        parts.map((p) => h('span', null, h('i', { style: { background: seriesColor(p.slot) } }), `${p.name} ${fmt(p.v, 0)}`))));
    }
    scoreBox.append(h('div', { class: 'metrics', style: { marginTop: '10px' } }, METRICS.map((d) => {
      const nd = d.key === 'summation' ? 2 : d.key === 'echo_pct' ? 0 : 1;
      const dl = ref ? delta(ref[d.key], m[d.key], d.better, nd) : null;
      return h('div', { class: 'metric', title: d.help }, h('span', { class: 'l', text: d.label }), h('span', { class: 'v', text: m[d.key] == null ? '–' : d.show(m[d.key]) }),
        dl ? h('div', { class: 'd' }, h('span', { text: `was ${d.show(ref[d.key])}` }), dl) : null);
    })));
  }

  function renderActions() {
    clear(actionBox);
    actionBox.append(h('div', { class: 'panel-title', text: 'Optimize' }));
    const none = !state.project?.speakers.some((s) => !s.mute);
    if (state.job) {
      const pr = state.job.progress || {};
      const pct = Math.round((pr.fraction || 0) * 100);
      actionBox.append(h('div', { class: 'job', attrs: { role: 'status' } },
        h('div', { class: 'line' }, h('b', { text: state.job.cancelling ? 'Stopping…' : 'Searching for better settings…' }),
          h('span', { class: 'delta flat', text: pr.generation ? `round ${pr.generation}` : 'aligning' })),
        h('div', { class: 'progress', attrs: { role: 'progressbar', 'aria-valuenow': String(pct), 'aria-valuemin': '0', 'aria-valuemax': '100' } },
          h('div', { style: { width: `${Math.max(3, pct)}%` } })),
        h('div', { class: 'line' },
          h('span', { class: 'hint', style: { margin: 0 }, text: pr.score != null ? `Best so far: ${fmt(pr.score, 1)}  (started at ${fmt(pr.start_score, 1)})` : 'Preparing…' }),
          h('button', { class: 'btn small', disabled: state.job.cancelling, onclick: cancelOptimize }, 'Cancel'))));
      return;
    }
    const busy = !!state.busy;
    const btn = (cls, ico, label, running, fn) => h('button', { class: `btn block ${cls}`, disabled: busy || none, onclick: fn },
      state.busy === running ? h('span', { class: 'spinner', style: { width: '14px', height: '14px' } }) : icon(ico), state.busy === running ? 'Working…' : label);
    actionBox.append(h('div', { class: 'actions' },
      btn('primary', 'bolt', 'Quick align', 'align', quickAlign),
      h('p', { class: 'action-hint', text: 'Sets delays, levels and sub polarity the way a system tech would. Under a second.' }),
      btn('', 'target', 'Deep optimize', 'deep', deepOptimize),
      h('p', { class: 'action-hint', text: `Searches around the alignment for a better overall score. About ${state.ui.budget} s.` }),
      btn('', 'sliders', 'Auto EQ', 'eq', autoEq),
      h('p', { class: 'action-hint', text: `Fits up to ${state.ui.eqBands} EQ bands per speaker toward the target curve.` })));
  }

  function renderChanges() {
    clear(changeBox);
    const a = state.lastAction;
    if (!a) return;
    const names = new Map(state.project.speakers.map((s) => [s.id, s]));
    const rows = a.changes.map((c) => {
      const spk = names.get(c.id);
      let what;
      if (c.field === 'invert') what = c.after ? 'Polarity inverted' : 'Polarity back to normal';
      else if (c.field === 'peq') what = `EQ: ${c.after} band${c.after === 1 ? '' : 's'}`;
      else { const [label, unit, nd] = FIELD[c.field] || [c.field, '', 1]; what = h('span', null, `${label} `, h('small', { text: `${fmt(c.before, nd)} → ` }), `${fmt(c.after, nd)} ${unit}`); }
      return h('tr', null, h('td', null, h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '6px' } },
        h('span', { class: 'swatch', style: { background: seriesColor(spk?.color) } }), c.name)), h('td', null, what));
    });
    changeBox.append(h('div', { class: 'changes' },
      h('h4', null, h('span', { text: a.title }), h('span', { class: 'grow' }),
        a.before?.score != null ? h('span', { class: 'delta flat', text: `${fmt(a.before.score, 0)} → ${fmt(a.after.score, 0)}` }) : null),
      a.notes.length ? h('ul', null, a.notes.map((n) => h('li', { text: n }))) : null,
      rows.length ? h('table', null, rows) : h('p', { class: 'hint', text: 'No settings needed to change.' }),
      h('div', { style: { display: 'flex', gap: '6px', marginTop: '10px', flexWrap: 'wrap' } },
        state.simBaseline ? h('div', { class: 'seg', attrs: { role: 'group', 'aria-label': 'Compare' } },
          ['before', 'after'].map((v) => h('button', { text: v === 'before' ? 'Before' : 'After', attrs: { 'aria-pressed': String(state.ui.compare === v) }, onclick: () => setUI({ compare: v }) }))) : null,
        h('span', { style: { flex: 1 } }),
        rows.length ? h('button', { class: 'btn small', title: 'Put back the settings from before', onclick: revertLastAction }, icon('undo'), 'Revert') : null,
        h('button', { class: 'btn small ghost', title: 'Keep the settings and hide this comparison', onclick: clearComparison }, 'Dismiss'))));
  }

  function renderFindings() {
    clear(findingBox);
    const advice = state.sim?.advice;
    if (!advice?.length) return;
    findingBox.append(h('div', { class: 'panel-title', text: 'Findings' }),
      ...advice.map((f) => h('div', { class: 'finding' }, statusBadge(f.level), h('b', { text: f.title }), h('p', { text: f.text }))));
  }

  on('sim ui theme', () => { renderScore(); renderFindings(); renderChanges(); });
  on('busy job ui project', (d) => { if (!d?.interacting) renderActions(); });
  on('action', () => { renderChanges(); renderScore(); });
  renderScore(); renderActions(); renderChanges(); renderFindings();
  return {};
}
