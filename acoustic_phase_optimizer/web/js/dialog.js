// Modal dialogs that match the app (instead of the browser's plain confirm()).

import { h, icon } from './util.js';

let open = null;

/**
 * Show a dialog. actions: [{ label, kind: 'primary' | 'danger' | '', run?: () => (false to keep open) }].
 * Escape or the close button resolves with null; an action resolves with its label.
 */
export function openDialog({ title, body = null, actions = [{ label: 'OK', kind: 'primary' }], wide = false, onClose = null }) {
  open?.close(null);
  const previous = document.activeElement;
  return new Promise((resolve) => {
    const close = (result) => {
      if (!backdrop.isConnected) return;
      backdrop.remove();
      document.removeEventListener('keydown', onKey, true);
      open = null;
      onClose?.(result);
      previous?.focus?.({ preventScroll: true });
      resolve(result);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(null); }
      if (e.key === 'Tab') { // keep focus inside the dialog
        const f = [...box.querySelectorAll('button, input, select, textarea, [tabindex]:not([tabindex="-1"])')].filter((x) => !x.disabled && x.offsetParent);
        if (!f.length) return;
        const i = f.indexOf(document.activeElement);
        if (e.shiftKey && i <= 0) { e.preventDefault(); f.at(-1).focus(); } else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
      }
    };
    const buttons = actions.map((a) => h('button', {
      class: 'btn' + (a.kind === 'primary' ? ' primary' : a.kind === 'danger' ? ' danger-solid' : ''),
      text: a.label, disabled: !!a.disabled,
      onclick: async () => {
        if (a.run && (await a.run()) === false) return;
        close(a.label);
      },
    }));
    const box = h('div', { class: 'dialog' + (wide ? ' wide' : ''), attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-label': title } },
      h('div', { class: 'dialog-head' }, h('h2', { text: title }),
        h('button', { class: 'icon-btn small', attrs: { 'aria-label': 'Close' }, onclick: () => close(null) }, icon('x'))),
      body ? h('div', { class: 'dialog-body' }, body) : null,
      h('div', { class: 'dialog-actions' }, buttons));
    const backdrop = h('div', { class: 'backdrop', onpointerdown: (e) => { if (e.target === backdrop) close(null); } }, box);
    document.body.append(backdrop);
    document.addEventListener('keydown', onKey, true);
    open = { close, box, buttons };
    (box.querySelector('[autofocus]') || buttons.find((b) => b.classList.contains('primary') || b.classList.contains('danger-solid')) || buttons[0])?.focus();
  });
}

/** Ask a yes/no question. Resolves true for yes. */
export async function ask(message, { title = 'Are you sure?', yes = 'Continue', no = 'Cancel', danger = false } = {}) {
  const r = await openDialog({ title, body: h('p', { text: message }), actions: [{ label: no }, { label: yes, kind: danger ? 'danger' : 'primary' }] });
  return r === yes;
}

export const currentDialog = () => open;
