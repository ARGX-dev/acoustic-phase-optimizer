// Thin client for the local engine API.

function describe(detail) {
  if (Array.isArray(detail)) {
    return detail.slice(0, 3).map((d) => {
      const where = (d.loc || []).filter((p) => p !== 'body' && p !== 'project').join(' › ');
      return (where ? where + ': ' : '') + (d.msg || 'invalid value');
    }).join('; ');
  }
  return typeof detail === 'string' ? detail : 'Request failed';
}

async function request(method, path, body, { signal, as = 'json', keepalive = false } = {}) {
  let res;
  try {
    res = await fetch(path, {
      method, signal, keepalive,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new Error('The engine is not responding. Is the app still running?');
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = describe((await res.json()).detail); } catch (e) { /* not JSON */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  if (as === 'text') return res.text();
  if (as === 'blob') return { blob: await res.blob(), name: /filename="([^"]+)"/.exec(res.headers.get('content-disposition') || '')?.[1] };
  return res.json();
}

// Measured overlays are display-only and can be large; computations never need them.
const lean = (project) => (project?.measurements?.length ? { ...project, measurements: [] } : project);

export const api = {
  meta: () => request('GET', '/api/meta'),
  venue: (id) => request('GET', `/api/venues/${encodeURIComponent(id)}`),
  simulate: (project, preview = false, signal) => request('POST', '/api/simulate', { project: lean(project), preview }, { signal }),
  section: (project, axis, at, band, signal, coarse = false) => request('POST', '/api/section', { project: lean(project), axis, at, band, coarse }, { signal }),
  subarray: (project, opts) => request('POST', '/api/subarray', { project, ...opts }),
  point: (project, x, y, z) => request('POST', '/api/point', { project: lean(project), x, y, z }),
  align: (project) => request('POST', '/api/align', { project }),
  autoEq: (project, max_bands) => request('POST', '/api/autoeq', { project, max_bands }),
  optimize: (project, weights, time_budget_s) => request('POST', '/api/optimize', { project, weights, time_budget_s }),
  job: (id) => request('GET', `/api/jobs/${id}`),
  cancelJob: (id) => request('DELETE', `/api/jobs/${id}`),
  exportDsp: (project, format) => request('POST', '/api/export', { project, format }, { as: 'blob' }),
  report: (project, baseline) => request('POST', '/api/report', { project, baseline }, { as: 'text' }),
  projects: () => request('GET', '/api/projects'),
  saveProject: (project) => request('POST', '/api/projects', { project }),
  checkProject: (project) => request('POST', '/api/projects/check', { project }),
  loadProject: (key) => request('GET', `/api/projects/${encodeURIComponent(key)}`),
  deleteProject: (key) => request('DELETE', `/api/projects/${encodeURIComponent(key)}`),
  session: () => request('GET', '/api/session'),
  saveSession: (payload, keepalive = false) => request('PUT', '/api/session', payload, { keepalive }),
  ping: () => request('POST', '/api/ping'),
  speakers: () => request('GET', '/api/speakers'),
  saveUserSpeaker: (row) => request('POST', '/api/speakers/user', row),
  deleteUserSpeaker: (id) => request('DELETE', `/api/speakers/user/${encodeURIComponent(id)}`),
  importSpeakers: async (text) => {
    const res = await fetch('/api/speakers/import', { method: 'POST', body: text, headers: { 'Content-Type': 'text/plain' } });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Import failed');
    return body;
  },
  scanFit: (points, normals) => request('POST', '/api/scan-fit', { points, normals }),
  bye: () => navigator.sendBeacon('/api/bye'),
};
