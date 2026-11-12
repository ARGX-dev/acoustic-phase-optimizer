// The client report: the company's branding (kept app-wide), the job's client and reference (kept in the
// project), then the report opened in its own window or sent straight to Print, where "Save as PDF" makes the file.

import { api } from './api.js';
import { openDialog } from './dialog.js';
import { commit, mutate, state } from './store.js';
import { download, h, toast } from './util.js';

const LOGO_W = 600, LOGO_H = 240, LOGO_MAX = 290_000; // the server takes up to 300 000 characters

/** A logo file as a PNG or JPEG data URL no bigger than 600 x 240, small enough to send. */
export async function logoDataUrl(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = () => reject(new Error('That file is not an image this browser can read.'));
      i.src = url;
    });
    const k = Math.min(1, LOGO_W / img.naturalWidth, LOGO_H / img.naturalHeight);
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(img.naturalWidth * k));
    c.height = Math.max(1, Math.round(img.naturalHeight * k));
    const g = c.getContext('2d');
    if (file.type === 'image/jpeg') { g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height); }
    g.drawImage(img, 0, 0, c.width, c.height);
    let out = file.type === 'image/jpeg' ? c.toDataURL('image/jpeg', 0.88) : c.toDataURL('image/png');
    if (out.length > LOGO_MAX) { // a busy PNG: a JPEG on white is far smaller
      g.globalCompositeOperation = 'destination-over';
      g.fillStyle = '#fff';
      g.fillRect(0, 0, c.width, c.height);
      out = c.toDataURL('image/jpeg', 0.85);
    }
    if (out.length > LOGO_MAX) throw new Error('That logo is too detailed to embed. Try a simpler or smaller file.');
    return out;
  } finally { URL.revokeObjectURL(url); }
}

/** Call once the report has loaded in win (navigating replaces its Window, so no load listener survives). */
function whenLoaded(win, url, fn) {
  const t0 = Date.now();
  const tick = setInterval(() => {
    let ready = false;
    try { ready = win.location.href === url && win.document.readyState === 'complete'; } catch { /* not there yet */ }
    if (ready || win.closed || Date.now() - t0 > 30000) {
      clearInterval(tick);
      if (ready) fn();
    }
  }, 150);
}

/** Show the report in win (a window opened during the click, so it is not blocked); with print, open Print there. */
export async function showReport(branding, { print = false, win = window.open('', '_blank') } = {}) {
  const t = toast('Building the report…', { ms: 30000 });
  try {
    commit();
    const html = await api.report(state.project, state.baseline, branding);
    if (!win) {
      download(`${state.project.name || 'venue'} report.html`, html, 'text/html');
      toast('Pop-ups are blocked, so the report was downloaded. Open it and print to save a PDF.');
      return;
    }
    const url = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
    win.location.href = url;
    if (print) whenLoaded(win, url, () => setTimeout(() => win.print(), 200));
    setTimeout(() => URL.revokeObjectURL(url), 120000);
    toast(print ? 'In the Print window, choose "Save as PDF".' : 'Report opened. Print it to save a PDF.');
  } catch (e) {
    win?.close();
    toast(`Report failed: ${e.message}`, { error: true });
  } finally { t.remove(); }
}

export async function reportDialog() {
  let brand;
  try { brand = await api.branding(); } catch { brand = { company: '', engineer: '', contact: '', accent: '#256abf', logo: null }; }
  const info = { client: '', reference: '', ...(state.project.report || {}) };
  const field = (label, el, title) => h('label', { class: 'field', title }, h('span', { text: label }), el);
  const text = (value, max, label, placeholder = '') => h('input', { type: 'text', value: value || '', maxLength: max, placeholder, attrs: { 'aria-label': label } });

  const company = text(brand.company, 120, 'Company');
  const engineer = text(brand.engineer, 120, 'Engineer');
  const contact = text(brand.contact, 200, 'Contact', 'Phone, email or website');
  const accent = h('input', { type: 'color', value: brand.accent || '#256abf', attrs: { 'aria-label': 'Accent colour' } });
  const client = text(info.client, 120, 'Client');
  const reference = text(info.reference, 60, 'Reference', 'Job or quote number');
  let logo = brand.logo || null;
  const preview = h('div', { class: 'logo-preview' });
  const showLogo = () => {
    preview.replaceChildren(logo ? h('img', { src: logo, alt: 'Logo' }) : h('span', { class: 'hint', text: 'No logo' }),
      logo ? h('button', { class: 'btn ghost small', text: 'Remove', onclick: (e) => { e.preventDefault(); logo = null; showLogo(); } }) : null);
  };
  showLogo();
  const file = h('input', { type: 'file', accept: 'image/png,image/jpeg,image/webp,image/gif', attrs: { 'aria-label': 'Logo file' },
    onchange: async () => {
      const f = file.files[0];
      file.value = '';
      if (!f) return;
      try { logo = await logoDataUrl(f); showLogo(); } catch (e) { toast(e.message, { error: true }); }
    } });

  const current = () => ({ company: company.value.trim(), engineer: engineer.value.trim(), contact: contact.value.trim(), accent: accent.value, logo });
  const save = async () => {
    const r = { client: client.value.trim(), reference: reference.value.trim() };
    if (r.client !== info.client || r.reference !== info.reference) mutate((p) => { p.report = r; }, { source: 'report' });
    const b = current();
    try { await api.saveBranding(b); } catch (e) { toast(`Could not keep the branding: ${e.message}`, { error: true }); }
    return b;
  };

  const body = h('div', { class: 'report-dlg' },
    h('h4', { text: 'This job' }),
    h('div', { class: 'grid2' }, field('Client', client, 'Who the report is for'), field('Reference', reference)),
    h('h4', { text: 'Your company (kept for every report)' }),
    h('div', { class: 'grid2' }, field('Company', company), field('Engineer', engineer), field('Contact', contact),
      field('Accent colour', accent, 'Used for headings and the cover')),
    h('div', { class: 'field' }, h('span', { text: 'Logo' }), h('div', { class: 'logo-row' }, preview, file)),
    h('p', { class: 'hint', text: 'The report opens in a new window. To make a PDF, press Save as PDF here, or print the report and choose "Save as PDF" as the printer.' }));

  await openDialog({
    title: 'Client report', body, wide: true,
    actions: [
      { label: 'Cancel' },
      { label: 'Open report', run: async () => { const win = window.open('', '_blank'); await showReport(await save(), { win }); } },
      { label: 'Save as PDF', kind: 'primary', run: async () => { const win = window.open('', '_blank'); await showReport(await save(), { print: true, win }); } },
    ],
  });
}
