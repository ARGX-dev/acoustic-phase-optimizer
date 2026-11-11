// 3D view of the venue: room with cut-away walls, the coverage map at ear height, speakers with their
// coverage, mics, the probe and (when imported) the 3D scan. Loaded only when the 3D view is first opened.
//
// Room coordinates are x across, y away from the stage, z up. three.js is y-up, so a room point (x, y, z)
// is drawn at (y, z, x): a cyclic swap, so nothing is mirrored and the 3D view matches the plan.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { commit, earAt, editSpeaker, mutate, on, select, setProbe, state } from './store.js';
import { clamp, clear, fmt, h } from './util.js';
import { cssVar, isDark, seriesColor } from './theme.js';
import { currentScale, heatImage, mapKey, MAPS, shownSim, unitOf, valueAt } from './stage.js';
import { scanObject as loadScan } from './scan.js';
import { toast } from './util.js';
import { len } from './units.js';

const P = (x, y, z) => new THREE.Vector3(y, z, x);
const toRoom = (v) => ({ x: v.z, y: v.x, z: v.y });

/** Aim direction in three.js space; same convention as engine/field.py orientation(). */
function forward(yaw, pitch) {
  const yw = (yaw * Math.PI) / 180, pt = (pitch * Math.PI) / 180;
  return P(Math.sin(yw) * Math.cos(pt), Math.cos(yw) * Math.cos(pt), Math.sin(pt));
}

const SURFACES = [ // BoxGeometry face order: +x, -x, +y, -y, +z, -z
  ['back', 'Back wall'], ['front', 'Stage wall'], ['ceiling', 'Ceiling'], ['floor', 'Floor'],
  ['right', 'Side wall'], ['left', 'Side wall'],
];

export function create3D(root) {
  const host = h('div', { class: 'view3d' });
  const labels = h('div', { class: 'labels3d', attrs: { 'aria-hidden': 'true' } });
  const tip = h('div', { class: 'tip', hidden: true });
  const views = h('div', { class: 'view-buttons seg', attrs: { role: 'group', 'aria-label': 'Camera' } });
  host.append(labels, tip, views);
  root.append(host);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  } catch (e) {
    host.append(h('div', { class: 'empty-state', text: '3D needs WebGL, which this browser has turned off. The plan view still works.' }));
    return { show() {}, hide() {}, setScan() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping; // keeps the map and speaker colours true
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.localClippingEnabled = true; // cut-away of the scan's near walls and ceiling
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.prepend(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', '3D view of the venue. Drag to orbit, scroll to zoom, right-drag to pan. Click the map to move the probe.');
  renderer.domElement.tabIndex = 0;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.screenSpacePanning = true;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.1);
  const fill = new THREE.AmbientLight(0xffffff, 1.0);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0004;
  scene.add(hemi, fill, sun, sun.target);

  const groups = { room: new THREE.Group(), heat: new THREE.Group(), speakers: new THREE.Group(), marks: new THREE.Group(), scan: new THREE.Group() };
  Object.values(groups).forEach((g) => scene.add(g));

  let visible = false, dirty = true, W = 0, H = 0;
  let roomSig = '', fitted = false, anim = null;
  let heatMesh = null, heatTex = null;
  const pickables = []; // speaker and mic meshes
  const labelEls = [];
  let scanObject = null;
  // Planes that slice away whichever scanned walls (and ceiling) stand between the camera and the room.
  const OFF = 1e6;
  const clip = {
    back: new THREE.Plane(new THREE.Vector3(-1, 0, 0), OFF), front: new THREE.Plane(new THREE.Vector3(1, 0, 0), OFF),
    right: new THREE.Plane(new THREE.Vector3(0, 0, -1), OFF), left: new THREE.Plane(new THREE.Vector3(0, 0, 1), OFF),
    ceiling: new THREE.Plane(new THREE.Vector3(0, -1, 0), OFF),
  };
  const clipList = Object.values(clip);
  function updateClip() {
    const r = state.project.room, c = camera.position, inset = 0.45;
    clip.back.constant = c.x > r.length ? r.length - inset : OFF;
    clip.front.constant = c.x < 0 ? -inset : OFF;
    clip.right.constant = c.z > r.width ? r.width - inset : OFF;
    clip.left.constant = c.z < 0 ? -inset : OFF;
    clip.ceiling.constant = c.y > r.height - 0.3 ? r.height - 0.3 : OFF;
  }
  let scanSig = '';

  const requestRender = () => { dirty = true; };
  controls.addEventListener('change', requestRender);

  // ---------------------------------------------------------------- colours
  function palette() {
    const dark = isDark();
    return {
      bg: new THREE.Color(cssVar('--page')),
      wall: new THREE.Color(dark ? '#5d5b56' : '#ecebe5'),
      stage: new THREE.Color(dark ? '#6e6a62' : '#d6d1c4'),
      floor: new THREE.Color(dark ? '#3d3b37' : '#dedad0'),
      edge: new THREE.Color(cssVar('--muted')),
      ink: new THREE.Color(cssVar('--ink')),
      accent: new THREE.Color(cssVar('--accent')),
      box: new THREE.Color(dark ? '#1c1c1b' : '#2b2b29'),
    };
  }

  function disposeGroup(g) {
    g.traverse((o) => {
      o.geometry?.dispose?.();
      (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { m?.map?.dispose?.(); m?.dispose?.(); });
    });
    g.clear();
  }

  // ------------------------------------------------------------------ room
  function buildRoom() {
    const r = state.project.room;
    const c = palette();
    scene.background = c.bg;
    disposeGroup(groups.room);
    const showScan = !!(scanObject && state.project.scan?.visible !== false);
    const geo = new THREE.BoxGeometry(r.length, r.height, r.width);
    const mats = SURFACES.map(([key]) => {
      const base = key === 'front' ? c.stage : key === 'floor' ? c.floor : c.wall;
      // more absorptive surfaces read a little darker, like soft furnishings
      const a = (state.meta?.materials.find((m) => m.id === r.materials[key]) || state.project.custom_materials?.find((m) => m.id === r.materials[key]))?.alpha || [];
      const mid = a.length ? (a[2] + a[3] + a[4]) / 3 : 0.1;
      const col = base.clone().lerp(new THREE.Color(isDark() ? '#2b2a27' : '#b9b3a5'), clamp(mid, 0, 1) * 0.5);
      const open = (r.kind !== 'indoor' && r.kind && key === 'ceiling') || (r.kind === 'outdoor' && key !== 'floor');
      return new THREE.MeshStandardMaterial({ color: col, side: THREE.BackSide, roughness: 0.92, metalness: 0,
        transparent: showScan || open, opacity: open ? 0 : showScan ? 0.18 : 1, depthWrite: !showScan && !open, visible: !open });
    });
    const box = new THREE.Mesh(geo, mats);
    box.position.copy(P(r.width / 2, r.length / 2, r.height / 2));
    box.receiveShadow = true;
    groups.room.add(box);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: c.edge, transparent: true, opacity: 0.55 }));
    edges.position.copy(box.position);
    groups.room.add(edges);

    // floor grid, 1 m (5 m major), and the audience area outline
    const grid = new THREE.Group();
    const minor = [], major = [];
    for (let m = 0; m <= r.length + 1e-6; m += 1) (m % 5 === 0 ? major : minor).push(P(0, m, 0.005), P(r.width, m, 0.005));
    for (let m = 0; m <= r.width + 1e-6; m += 1) (m % 5 === 0 ? major : minor).push(P(m, 0, 0.005), P(m, r.length, 0.005));
    grid.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(minor), new THREE.LineBasicMaterial({ color: c.edge, transparent: true, opacity: 0.12 })));
    grid.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(major), new THREE.LineBasicMaterial({ color: c.edge, transparent: true, opacity: 0.3 })));
    groups.room.add(grid);

    const a = state.project.audience;
    const e0 = earAt(a, a.y0), e1 = earAt(a, a.y1);
    const outline = [P(a.x0, a.y0, e0), P(a.x1, a.y0, e0), P(a.x1, a.y1, e1), P(a.x0, a.y1, e1), P(a.x0, a.y0, e0)];
    if (a.rake > 0.05) {
      // the seating slope itself, ear height below the ear plane
      const fl = (y) => Math.max(0, earAt(a, y) - a.ear_height);
      const g = new THREE.BufferGeometry().setFromPoints([P(a.x0, a.y0, fl(a.y0)), P(a.x1, a.y0, fl(a.y0)), P(a.x1, a.y1, fl(a.y1)),
        P(a.x0, a.y0, fl(a.y0)), P(a.x1, a.y1, fl(a.y1)), P(a.x0, a.y1, fl(a.y1))]);
      g.computeVertexNormals();
      groups.room.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: c.box, roughness: 0.8, side: THREE.DoubleSide, transparent: true, opacity: 0.85 })));
    }
    groups.room.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(outline), new THREE.LineBasicMaterial({ color: c.edge })));

    // light from above the audience, shadows inside the room only
    const span = Math.max(r.length, r.width);
    sun.position.copy(P(r.width * 0.3, r.length * 0.65, r.height + span * 0.8));
    sun.target.position.copy(P(r.width / 2, r.length / 2, 0));
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -span * 0.8;
    sc.right = sc.top = span * 0.8;
    sc.near = 0.5;
    sc.far = span * 3 + r.height;
    sc.updateProjectionMatrix();
    hemi.color.set(isDark() ? '#dfe6f2' : '#ffffff');
    hemi.groundColor.set(isDark() ? '#2a2620' : '#b8b0a0');
    hemi.intensity = isDark() ? 2.2 : 2.0;
    fill.intensity = isDark() ? 1.6 : 1.4;
    sun.intensity = isDark() ? 1.8 : 2.0;
  }

  // ------------------------------------------------------------- heat map
  function buildHeat() {
    const sim = shownSim();
    const key = mapKey();
    if (!sim?.maps?.[key]) {
      if (heatMesh) heatMesh.visible = false;
      return;
    }
    const g = sim.grid, a = state.project.audience;
    const img = heatImage(sim, key, currentScale());
    const cx = (g.x1 - g.x0) / Math.max(g.nx - 1, 1), cy = (g.y1 - g.y0) / Math.max(g.ny - 1, 1);
    // the grid's cells, clipped to the audience area (as on the plan)
    const x0 = Math.max(g.x0 - cx / 2, a.x0), x1 = Math.min(g.x1 + cx / 2, a.x1);
    const y0 = Math.max(g.y0 - cy / 2, a.y0), y1 = Math.min(g.y1 + cy / 2, a.y1);
    if (!heatTex || heatTex.image !== img) {
      heatTex?.dispose();
      heatTex = new THREE.CanvasTexture(img);
      heatTex.colorSpace = THREE.SRGBColorSpace;
      heatTex.minFilter = THREE.LinearFilter;
      heatTex.magFilter = THREE.LinearFilter;
      heatTex.generateMipmaps = false;
    }
    // map the clipped rectangle onto the full grid texture
    const gx0 = g.x0 - cx / 2, gx1 = g.x1 + cx / 2, gy0 = g.y0 - cy / 2, gy1 = g.y1 + cy / 2;
    heatTex.repeat.set((y1 - y0) / (gy1 - gy0), (x1 - x0) / (gx1 - gx0));
    heatTex.offset.set((y0 - gy0) / (gy1 - gy0), 1 - (x1 - gx0) / (gx1 - gx0));
    if (!heatMesh) {
      heatMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.94, side: THREE.DoubleSide, toneMapped: false }));
      heatMesh.rotation.x = -Math.PI / 2;
      heatMesh.renderOrder = 2;
      groups.heat.add(heatMesh);
    }
    heatMesh.material.map = heatTex;
    heatMesh.material.needsUpdate = true;
    heatTex.needsUpdate = true;
    // On raked seating the map is a slope: tilt the plane by the rise over the audience depth.
    const rise = earAt(a, y1) - earAt(a, y0);
    heatMesh.rotation.set(-Math.PI / 2, 0, 0);
    heatMesh.scale.set(Math.hypot(y1 - y0, rise), x1 - x0, 1);
    heatMesh.position.copy(P((x0 + x1) / 2, (y0 + y1) / 2, (earAt(a, y0) + earAt(a, y1)) / 2));
    heatMesh.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), Math.atan2(rise, y1 - y0)); // room y is three.js x, up is y
    heatMesh.userData.bounds = { x0, x1, y0, y1 };
    heatMesh.visible = true;
  }

  // ------------------------------------------------------- speakers & marks
  function coverageShape(s, color, selected) {
    const group = new THREE.Group();
    if (s.coverage_h >= 300) return group;
    const r = state.project.room;
    const drop = s.z - earAt(state.project.audience, s.y + 10);
    const reach = s.pitch < -1 && drop > 0 ? drop / Math.tan((-s.pitch * Math.PI) / 180) : 12;
    const L = clamp(reach * 1.15, 3, Math.max(r.length, r.width) * 0.9);
    const th = Math.tan((Math.min(s.coverage_h, 160) * Math.PI) / 360) * L;
    const tv = Math.tan((Math.min(s.coverage_v, 160) * Math.PI) / 360) * L;
    const apex = new THREE.Vector3(0, 0, 0);
    const c = [new THREE.Vector3(-th, -tv, L), new THREE.Vector3(th, -tv, L), new THREE.Vector3(th, tv, L), new THREE.Vector3(-th, tv, L)];
    const pos = [];
    for (let i = 0; i < 4; i++) pos.push(apex, c[i], c[(i + 1) % 4]);
    const geo = new THREE.BufferGeometry().setFromPoints(pos);
    group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: selected ? 0.13 : 0.035,
      side: THREE.DoubleSide, depthWrite: false })));
    const lines = [];
    for (let i = 0; i < 4; i++) lines.push(apex, c[i], c[i], c[(i + 1) % 4]);
    group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lines),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: selected ? 0.75 : 0.1 })));
    return group;
  }

  function cabinet(s) {
    if (s.role === 'sub') return new THREE.BoxGeometry(0.6, 0.55, 0.7);
    if (s.array_boxes >= 2) return new THREE.BoxGeometry(1.0, s.array_boxes * (s.array_box_h || 0.3), 0.55); // a hang, its real length
    if (s.coverage_h >= 140 && s.coverage_v <= 30) return new THREE.BoxGeometry(0.22, 1.1, 0.25); // column
    if (s.coverage_v <= 25) return new THREE.BoxGeometry(1.1, 1.6, 0.55); // a hang of line-array boxes
    const size = s.level_db >= 128 ? 0.62 : s.level_db >= 118 ? 0.45 : 0.3;
    return new THREE.BoxGeometry(size * 0.75, size, size * 0.65);
  }

  function buildSpeakers() {
    disposeGroup(groups.speakers);
    disposeGroup(groups.marks);
    pickables.length = 0;
    const c = palette();
    const selId = state.selection?.kind === 'speaker' ? state.selection.id : null;
    for (const s of state.project.speakers) {
      const color = new THREE.Color(seriesColor(s.color));
      const sel = s.id === selId;
      const holder = new THREE.Group();
      holder.position.copy(P(s.x, s.y, s.z));
      const fwd = forward(s.yaw, s.pitch);
      holder.lookAt(holder.position.clone().add(fwd));
      const body = new THREE.Mesh(cabinet(s), new THREE.MeshStandardMaterial({ color: c.box, roughness: 0.55, metalness: 0.1,
        transparent: s.mute, opacity: s.mute ? 0.35 : 1 }));
      body.castShadow = true;
      // coloured front baffle so the speaker's identity matches the plan and chart
      const bb = new THREE.Box3().setFromObject(body);
      const face = new THREE.Mesh(new THREE.PlaneGeometry((bb.max.x - bb.min.x) * 0.86, (bb.max.y - bb.min.y) * 0.86),
        new THREE.MeshStandardMaterial({ color, roughness: 0.4, emissive: color, emissiveIntensity: sel ? 0.55 : 0.25,
          transparent: s.mute, opacity: s.mute ? 0.35 : 1 }));
      face.position.z = bb.max.z + 0.003;
      holder.add(body, face);
      if (!s.mute && sel) holder.add(coverageShape(s, color, sel));
      if (sel) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.75, 0.025, 8, 48), new THREE.MeshBasicMaterial({ color: c.accent }));
        ring.position.z = 0.02;
        holder.add(ring);
      }
      body.userData.pick = { kind: 'speaker', id: s.id };
      face.userData.pick = body.userData.pick;
      pickables.push(body, face);
      groups.speakers.add(holder);
      // a thin line down to the floor helps read height
      const drop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([P(s.x, s.y, s.z), P(s.x, s.y, 0)]),
        new THREE.LineDashedMaterial({ color: c.edge, dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: 0.6 }));
      drop.computeLineDistances();
      groups.marks.add(drop);
    }
    const micSel = state.selection?.kind === 'mic' ? state.selection.id : null;
    for (const m of state.project.mics) {
      const active = m.id === micSel || state.ui.source === m.id;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 14),
        new THREE.MeshStandardMaterial({ color: active ? c.accent : c.ink, roughness: 0.35 }));
      mesh.position.copy(P(m.x, m.y, m.z));
      mesh.castShadow = true;
      mesh.userData.pick = { kind: 'mic', id: m.id };
      pickables.push(mesh);
      const stand = new THREE.Line(new THREE.BufferGeometry().setFromPoints([P(m.x, m.y, m.z), P(m.x, m.y, 0)]),
        new THREE.LineBasicMaterial({ color: c.edge }));
      groups.marks.add(mesh, stand);
    }
    if (state.probe) {
      const z = earAt(state.project.audience, state.probe.y);
      const on = state.ui.source === 'probe';
      const pin = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.5, 20), new THREE.MeshStandardMaterial({ color: on ? c.accent : c.edge }));
      pin.rotation.x = Math.PI;
      pin.position.copy(P(state.probe.x, state.probe.y, z + 0.27));
      groups.marks.add(pin);
    }
    buildLabels();
  }

  function buildLabels() {
    clear(labels);
    labelEls.length = 0;
    const add = (text, pos, cls = '') => {
      const el = h('span', { class: 'label3d ' + cls, text });
      labels.append(el);
      labelEls.push({ el, pos });
    };
    const selId = state.selection?.id;
    for (const s of state.project.speakers) add(s.name, P(s.x, s.y, s.z + 0.6), s.id === selId ? 'sel' : s.mute ? 'muted' : '');
    for (const m of state.project.mics) add(m.name, P(m.x, m.y, m.z + 0.35), 'mic');
    const r = state.project.room;
    add('STAGE', P(r.width / 2, 0, r.height + 0.4), 'stage');
    requestRender();
  }

  // Labels are placed in priority order (selected, speakers, mics, stage); one that would overlap a label
  // already shown is hidden. Hovering an object still names it in the tooltip.
  function placeLabels() {
    const v = new THREE.Vector3();
    const shown = [];
    const order = [...labelEls].sort((a, b) => (b.el.classList.contains('sel') ? 1 : 0) - (a.el.classList.contains('sel') ? 1 : 0));
    for (const item of order) {
      const { el, pos } = item;
      v.copy(pos).project(camera);
      let hide = v.z > 1 || v.x < -1.1 || v.x > 1.1 || v.y < -1.1 || v.y > 1.1;
      const x = ((v.x + 1) / 2) * W, y = ((1 - v.y) / 2) * H;
      if (!hide) {
        if (!item.w) { el.style.display = ''; item.w = el.offsetWidth || 60; item.h = el.offsetHeight || 18; }
        const box = { l: x - item.w / 2 - 2, r: x + item.w / 2 + 2, t: y - item.h - 1, b: y + 1 };
        hide = shown.some((o) => box.l < o.r && box.r > o.l && box.t < o.b && box.b > o.t);
        if (!hide) shown.push(box);
      }
      el.style.display = hide ? 'none' : '';
      if (!hide) el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
    }
  }

  // ---------------------------------------------------------------- camera
  // Each view is a direction to look from; the distance is fitted so the whole room is in frame.
  const VIEWS = {
    orbit: ['3D', (r) => ({ dir: P(r.width * 0.55, r.length * 0.75, Math.max(r.length, r.width) * 0.62), target: P(r.width / 2, r.length * 0.5, r.height * 0.2) })],
    top: ['Top', (r) => ({ dir: P(0.0001, 0, 1), target: P(r.width / 2, r.length / 2, 0) })],
    stage: ['From stage', (r) => ({ pos: P(r.width / 2, 0.3, Math.min(r.height * 0.75, r.height - 0.3)), target: P(r.width / 2, r.length * 0.7, state.project.audience.ear_height) })],
    back: ['From the back', (r) => ({ dir: P(0, 1, 0.42), target: P(r.width / 2, r.length * 0.45, r.height * 0.3) })],
  };

  function placement(view) {
    const r = state.project.room;
    const v = VIEWS[view][1](r);
    if (v.pos) return v;
    const radius = 0.5 * Math.hypot(r.length, r.width, r.height);
    const vfov = (camera.fov * Math.PI) / 180;
    const fit = 2 * Math.atan(Math.tan(vfov / 2) * Math.min(camera.aspect || 1, 1));
    const dist = (radius * 0.98) / Math.sin(fit / 2);
    return { pos: v.target.clone().add(v.dir.clone().normalize().multiplyScalar(dist)), target: v.target };
  }

  function flyTo(view, instant = false) {
    const { pos, target } = placement(view);
    if (instant) {
      anim = null;
      camera.position.copy(pos);
      controls.target.copy(target);
      controls.update();
      requestRender();
      renderViews(view);
      return;
    }
    const from = { pos: camera.position.clone(), target: controls.target.clone() };
    anim = { from, to: { pos, target }, t0: performance.now(), ms: 650 };
    renderViews(view);
  }

  function renderViews(current) {
    clear(views).append(...Object.entries(VIEWS).map(([id, [label]]) =>
      h('button', { text: label, attrs: { 'aria-pressed': String(current === id) }, onclick: () => flyTo(id) })));
  }

  // ------------------------------------------------------------- picking
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  function pick(e) {
    const b = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const objHit = ray.intersectObjects(pickables, false)[0];
    if (objHit) return { kind: 'object', objKind: objHit.object.userData.pick.kind, id: objHit.object.userData.pick.id };
    if (heatMesh?.visible) {
      const hit = ray.intersectObject(heatMesh, false)[0];
      if (hit) return { kind: 'map', ...toRoom(hit.point), sx: e.clientX - b.left, sy: e.clientY - b.top };
    }
    return null;
  }

  // Dragging an object moves it across the room at its own height (orbiting stays on empty space).
  let drag3d = null;
  const dragPlane = new THREE.Plane();
  const hitPoint = new THREE.Vector3();
  host.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || e.target !== renderer.domElement) return;
    const p = pick(e);
    if (p?.kind !== 'object') return;
    const obj = p.objKind === 'speaker' ? state.project.speakers.find((q) => q.id === p.id) : state.project.mics.find((q) => q.id === p.id);
    if (!obj) return;
    e.stopPropagation(); // keep OrbitControls out of it
    select(p.objKind, p.id);
    dragPlane.set(new THREE.Vector3(0, 1, 0), -obj.z); // room z is three.js y
    drag3d = { kind: p.objKind, id: p.id, moved: false, x: e.clientX, y: e.clientY };
    renderer.domElement.setPointerCapture(e.pointerId);
    renderer.domElement.style.cursor = 'grabbing';
  }, true);
  host.addEventListener('pointermove', (e) => {
    if (!drag3d) return;
    e.stopPropagation();
    if (!drag3d.moved && Math.hypot(e.clientX - drag3d.x, e.clientY - drag3d.y) < 3) return;
    const b = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    if (!ray.ray.intersectPlane(dragPlane, hitPoint)) return;
    drag3d.moved = true;
    const r = state.project.room;
    const q = toRoom(hitPoint);
    const x = Math.round(clamp(q.x, 0.1, r.width - 0.1) * 10) / 10, y = Math.round(clamp(q.y, 0.1, r.length - 0.1) * 10) / 10;
    const live = { commit: false, interacting: true, source: 'view3d' };
    if (drag3d.kind === 'speaker') editSpeaker(drag3d.id, { x, y }, live);
    else mutate((pr) => { const m = pr.mics.find((o) => o.id === drag3d.id); if (m) { m.x = x; m.y = y; } }, live);
  }, true);
  const endDrag3d = (e) => {
    if (!drag3d) return;
    e.stopPropagation();
    const moved = drag3d.moved;
    drag3d = null;
    downAt = null;
    try { renderer.domElement.releasePointerCapture(e.pointerId); } catch (err) { /* released */ }
    renderer.domElement.style.cursor = 'pointer';
    if (moved) { commit(); mutate(() => {}, { source: 'view3d' }); }
  };
  host.addEventListener('pointerup', endDrag3d, true);
  host.addEventListener('pointercancel', endDrag3d, true);

  let downAt = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY }; anim = null; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4 || e.button !== 0) { downAt = null; return; }
    downAt = null;
    const p = pick(e);
    if (p?.kind === 'object') select(p.objKind, p.id);
    else if (p?.kind === 'map') setProbe(p.x, p.y);
    else select(null);
  });
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (e.buttons) { tip.hidden = true; return; }
    const p = pick(e);
    renderer.domElement.style.cursor = p?.kind === 'object' ? 'grab' : p ? 'crosshair' : 'grab';
    if (!p) { tip.hidden = true; return; }
    const b = renderer.domElement.getBoundingClientRect();
    const sx = e.clientX - b.left, sy = e.clientY - b.top;
    const rows = [];
    if (p.kind === 'object') {
      const s = state.project.speakers.find((q) => q.id === p.id);
      const m = state.project.mics.find((q) => q.id === p.id);
      if (s) rows.push(h('div', { class: 'row' }, h('span', { class: 'swatch', style: { background: seriesColor(s.color) } }), h('span', { class: 'v', style: { textAlign: 'left' }, text: s.name })),
        h('div', { class: 'l', text: `${len(s.y)} from stage · ${len(s.z)} high · aim ${fmt(s.yaw, 0)}° / ${fmt(s.pitch, 0)}°` }),
        h('div', { class: 'l', text: `Delay ${fmt(s.delay_ms, 2)} ms · Gain ${fmt(s.gain_db, 1, true)} dB` }));
      if (m) rows.push(h('div', { class: 'v', text: m.name }), h('div', { class: 'l', text: `${len(m.y)} from stage, ${len(m.z)} high` }));
    } else {
      const cell = valueAt(p.x, p.y);
      const def = MAPS[state.ui.map];
      if (cell?.value != null) {
        rows.push(h('div', { class: 'v', text: `${fmt(cell.value, def.nd ?? (def.unit === 'ms' ? 0 : 1), def.kind === 'diverging')} ${unitOf(def)}` }),
          h('div', { class: 'l', text: def.label }));
        if (cell.dominant) rows.push(h('div', { class: 'row' }, h('span', { class: 'swatch', style: { background: seriesColor(cell.dominant.color) } }), h('span', { class: 'l', text: `Loudest here: ${cell.dominant.name}` })));
      }
      rows.push(h('div', { class: 'l', text: `${len(p.y)} from stage, ${len(p.x)} across · click to probe` }));
    }
    clear(tip).append(...rows);
    tip.hidden = false;
    tip.style.left = `${sx}px`;
    tip.style.top = `${sy}px`;
    tip.style.transform = `translate(${sx > W - 260 ? 'calc(-100% - 12px)' : '12px'}, ${sy > H - 110 ? 'calc(-100% - 12px)' : '12px'})`;
  });
  renderer.domElement.addEventListener('pointerleave', () => { tip.hidden = true; });

  // ---------------------------------------------------------------- loop
  function frame(now) {
    if (!visible) return;
    requestAnimationFrame(frame);
    if (anim) {
      const t = clamp((now - anim.t0) / anim.ms, 0, 1);
      const k = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
      camera.position.lerpVectors(anim.from.pos, anim.to.pos, k);
      controls.target.lerpVectors(anim.from.target, anim.to.target, k);
      dirty = true;
      if (t >= 1) anim = null;
    }
    if (controls.update()) dirty = true;
    if (!dirty || W < 10 || H < 10) return;
    dirty = false;
    if (scanObject) updateClip();
    renderer.render(scene, camera);
    placeLabels();
  }

  function resize() {
    const b = host.getBoundingClientRect();
    W = Math.floor(b.width);
    H = Math.floor(b.height);
    if (W < 10 || H < 10) return;
    renderer.setSize(W, H, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    requestRender();
  }
  new ResizeObserver(resize).observe(host);

  // ------------------------------------------------------------- updates
  function refreshAll() {
    if (!state.project || !visible) return;
    const r = state.project.room;
    const sig = JSON.stringify([r, state.project.custom_materials, state.project.audience, isDark(), !!(scanObject && state.project.scan?.visible !== false), state.meta?.materials?.length]);
    if (sig !== roomSig) {
      roomSig = sig;
      buildRoom();
      if (!fitted && W > 10) { flyTo('orbit', true); fitted = true; }
    }
    buildHeat();
    buildSpeakers();
    syncScan();
    requestRender();
  }

  function syncScan() {
    const ref = state.project.scan;
    const sig = JSON.stringify(ref || null);
    if (sig === scanSig) return;
    scanSig = sig;
    if (!ref) { if (scanObject) api3d.setScan(null); return; }
    loadScan(ref).then((obj) => {
      if (JSON.stringify(state.project.scan || null) !== sig) return; // changed while loading
      obj.traverse((o) => {
        if (!o.material) return;
        (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => {
          if (m.clippingPlanes !== clipList) { m.clippingPlanes = clipList; m.needsUpdate = true; }
        });
      });
      if (obj !== scanObject) api3d.setScan(obj);
      else { roomSig = ''; schedule(); }
      if (!obj.visible && scanObject) roomSig = '';
      requestRender();
    }).catch((e) => toast(`The 3D scan could not be shown: ${e.message}`, { error: true }));
  }

  let pending = false;
  const schedule = () => {
    if (!visible || pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; refreshAll(); });
  };
  on('project sim selection probe ui theme', schedule);
  on('project', (d) => { if (d?.source === 'load') { fitted = false; roomSig = ''; } });

  const api3d = {
    show() {
      if (visible) return;
      visible = true;
      host.hidden = false;
      resize();
      refreshAll();
      requestAnimationFrame(frame);
    },
    hide() {
      visible = false;
      host.hidden = true;
      tip.hidden = true;
    },
    /** Show an imported scan (a THREE.Object3D already placed in three.js space), or null to remove it. */
    setScan(obj) {
      groups.scan.clear(); // detach only: scans are cached by file so undo/redo can bring them back instantly
      scanObject = obj;
      if (obj) groups.scan.add(obj);
      roomSig = '';
      schedule();
    },
    flyTo,
  };
  return api3d;
}
