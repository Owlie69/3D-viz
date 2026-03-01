/**
 * Three.js renderer for 3D Gaussian splats.
 *
 * Mouse interaction is handled entirely on the GPU: the vertex shader
 * computes each splat's NDC position, measures distance to the cursor,
 * and displaces the splat in view space with a hard repulsion core +
 * an expanding ripple wave. No per-frame CPU physics loop is needed –
 * only three uniforms are updated each frame.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Shaders ──────────────────────────────────────────────────────────────────

const VERT = /* glsl */ `
  attribute vec3  aPos;    // splat world-space centre
  attribute vec4  aColor;  // RGBA (linear)
  attribute float aScale;  // billboard radius in world units

  uniform vec2  uMouse;       // cursor in NDC  (-1..1 each axis)
  uniform float uInteracting; // 0 = idle, 1 = cursor over canvas (smooth)
  uniform float uTime;        // elapsed seconds

  varying vec2  vUV;
  varying vec4  vColor;
  varying float vHighlight;   // 0..1 proximity glow passed to fragment

  void main() {
    // 1. Transform splat centre to view space
    vec4 viewCenter = modelViewMatrix * vec4(aPos, 1.0);

    // 2. Project centre to clip/NDC space so we can measure cursor distance
    vec4 clipC = projectionMatrix * viewCenter;
    vec2 ndc   = clipC.xy / clipC.w;

    // 3. Mouse-space distance (NDC is aspect-corrected in the uniform)
    vec2  diff = ndc - uMouse;
    float dist = length(diff);
    vec2  dir  = dist > 0.001 ? diff / dist : vec2(0.7071, 0.7071);

    vHighlight = 0.0;

    if (uInteracting > 0.01) {
      float t = uInteracting;

      // Hard repulsion core (push-away bubble)
      float hard = pow(max(0.0, 1.0 - dist / 0.13), 2.0);

      // Ripple wave that propagates outward from core
      float ripple = max(0.0, 1.0 - dist / 0.46)
                   * sin(dist * 28.0 - uTime * 9.5)
                   * 0.28;

      float push = (hard * 0.13 + ripple * 0.025) * t;

      // Displace in view space (consistent world-unit push regardless of NDC)
      viewCenter.xy += dir * push;

      vHighlight = hard * t;
    }

    // 4. Screen-aligned billboard quad offset
    viewCenter.xy += position.xy * aScale;

    vUV    = position.xy;
    vColor = aColor;
    gl_Position = projectionMatrix * viewCenter;
  }
`;

const FRAG = /* glsl */ `
  precision mediump float;

  varying vec2  vUV;
  varying vec4  vColor;
  varying float vHighlight;

  void main() {
    // Crisp 2D Gaussian (tighter falloff for small, refined dots)
    float r2    = dot(vUV, vUV);
    float gauss = exp(-r2 * 11.0);
    if (gauss < 0.008) discard;

    // Subtle cyan proximity glow
    vec3 col = vColor.rgb + vec3(0.05, 0.18, 0.30) * vHighlight * 0.55;
    gl_FragColor = vec4(col, vColor.a * gauss);
  }
`;

// ── Constants ────────────────────────────────────────────────────────────────

const SORT_INTERVAL = 8;
const BG_COLOR      = 0x05050c;

// ── SplatRenderer ────────────────────────────────────────────────────────────

export class SplatRenderer {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this._canvas     = canvas;
    this._frameCount = 0;
    this._splatMesh  = null;

    // Raw (unsorted) splat data – kept for re-sorting
    this._rawPositions = null;
    this._rawColors    = null;
    this._rawScales    = null;
    this._splatCount   = 0;

    // Sorted work arrays reused every sort pass
    this._sortedPos  = null;
    this._sortedCol  = null;
    this._sortedScl  = null;
    this._sortedIdx  = null;
    this._distances  = null;

    // Mouse interaction state
    this._mouseNDC   = new THREE.Vector2(0, 0);
    this._mouseOver  = false;
    this._interacting = 0.0;   // smoothly transitions 0 → 1

    this._clock = new THREE.Clock();

    this._initThree();
    this._initControls();
    this._initMouse();
    this._bindResize();
    this._animate();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  loadSplats({ positions, colors, scales, count }) {
    if (this._splatMesh) {
      this._scene.remove(this._splatMesh);
      this._splatMesh.geometry.dispose();
      this._splatMesh.material.dispose();
      this._splatMesh = null;
    }

    this._rawPositions = positions.slice();
    this._rawColors    = colors.slice();
    this._rawScales    = scales.slice();
    this._splatCount   = count;

    this._sortedPos  = new Float32Array(count * 3);
    this._sortedCol  = new Float32Array(count * 4);
    this._sortedScl  = new Float32Array(count);
    this._sortedIdx  = new Int32Array(count);
    this._distances  = new Float32Array(count);
    for (let i = 0; i < count; i++) this._sortedIdx[i] = i;

    this._sortSplats();
    this._splatMesh = this._buildMesh(
      this._sortedPos.slice(0, count * 3),
      this._sortedCol.slice(0, count * 4),
      this._sortedScl.slice(0, count),
      count
    );
    this._scene.add(this._splatMesh);
  }

  resetCamera() {
    this._camera.position.set(0, 0, 2.2);
    this._camera.lookAt(0, 0, 0);
    this._controls.target.set(0, 0, 0);
    this._controls.update();
  }

  // ── Three.js init ──────────────────────────────────────────────────────────

  _initThree() {
    const r = new THREE.WebGLRenderer({ canvas: this._canvas, antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.setSize(window.innerWidth, window.innerHeight);
    r.setClearColor(BG_COLOR, 1);
    this._renderer = r;

    this._scene = new THREE.Scene();
    this._scene.fog = new THREE.FogExp2(BG_COLOR, 0.15);

    this._camera = new THREE.PerspectiveCamera(
      55, window.innerWidth / window.innerHeight, 0.005, 80
    );
    this._camera.position.set(0, 0, 2.2);
  }

  _initControls() {
    const ctrl = new OrbitControls(this._camera, this._canvas);
    ctrl.enableDamping     = true;
    ctrl.dampingFactor     = 0.06;
    ctrl.minDistance       = 0.05;
    ctrl.maxDistance       = 30;
    ctrl.enablePan         = true;
    ctrl.screenSpacePanning = true;
    this._controls = ctrl;
  }

  _initMouse() {
    const el = this._canvas;

    el.addEventListener('mousemove', e => {
      const r = el.getBoundingClientRect();
      this._mouseNDC.x =  ((e.clientX - r.left) / r.width)  * 2 - 1;
      this._mouseNDC.y = -((e.clientY - r.top)  / r.height) * 2 + 1;
      this._mouseOver  = true;
    });

    el.addEventListener('mouseleave', () => { this._mouseOver = false; });
    el.addEventListener('mouseenter', () => { this._mouseOver = true;  });

    // Touch support
    el.addEventListener('touchmove', e => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const t = e.touches[0];
      this._mouseNDC.x =  ((t.clientX - r.left) / r.width)  * 2 - 1;
      this._mouseNDC.y = -((t.clientY - r.top)  / r.height) * 2 + 1;
      this._mouseOver  = true;
    }, { passive: false });

    el.addEventListener('touchend', () => { this._mouseOver = false; });
  }

  _bindResize() {
    window.addEventListener('resize', () => {
      const w = window.innerWidth, h = window.innerHeight;
      this._camera.aspect = w / h;
      this._camera.updateProjectionMatrix();
      this._renderer.setSize(w, h);
    });
  }

  // ── GPU mesh ───────────────────────────────────────────────────────────────

  _buildMesh(positions, colors, scales, count) {
    const baseGeom = new THREE.PlaneGeometry(1, 1);
    const geom     = new THREE.InstancedBufferGeometry();

    geom.index                   = baseGeom.index;
    geom.attributes.position     = baseGeom.attributes.position;
    geom.setAttribute('aPos',   new THREE.InstancedBufferAttribute(positions, 3));
    geom.setAttribute('aColor', new THREE.InstancedBufferAttribute(colors,    4));
    geom.setAttribute('aScale', new THREE.InstancedBufferAttribute(scales,    1));
    geom.instanceCount = count;

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uMouse:       { value: new THREE.Vector2(0, 0) },
        uInteracting: { value: 0.0 },
        uTime:        { value: 0.0 },
      },
      vertexShader:   VERT,
      fragmentShader: FRAG,
      transparent:    true,
      depthWrite:     false,
      depthTest:      true,
      blending:       THREE.NormalBlending,
    });

    baseGeom.dispose();
    return new THREE.Mesh(geom, mat);
  }

  // ── Depth sort ─────────────────────────────────────────────────────────────

  _sortSplats() {
    const n   = this._splatCount;
    const pos = this._rawPositions;
    const cam = this._camera.position;

    for (let i = 0; i < n; i++) {
      const dx = pos[i * 3]     - cam.x;
      const dy = pos[i * 3 + 1] - cam.y;
      const dz = pos[i * 3 + 2] - cam.z;
      this._distances[i] = dx * dx + dy * dy + dz * dz;
    }

    const dist = this._distances;
    this._sortedIdx.sort((a, b) => dist[b] - dist[a]);

    const sp = this._sortedPos, sc = this._sortedCol, ss = this._sortedScl;
    const rp = this._rawPositions, rc = this._rawColors, rs = this._rawScales;
    const idx = this._sortedIdx;

    for (let i = 0; i < n; i++) {
      const s = idx[i];
      sp[i * 3]     = rp[s * 3];
      sp[i * 3 + 1] = rp[s * 3 + 1];
      sp[i * 3 + 2] = rp[s * 3 + 2];
      sc[i * 4]     = rc[s * 4];
      sc[i * 4 + 1] = rc[s * 4 + 1];
      sc[i * 4 + 2] = rc[s * 4 + 2];
      sc[i * 4 + 3] = rc[s * 4 + 3];
      ss[i]         = rs[s];
    }
  }

  _pushSortedToGPU() {
    if (!this._splatMesh) return;
    const g = this._splatMesh.geometry;
    const n = this._splatCount;
    const pa = g.attributes.aPos;
    const ca = g.attributes.aColor;
    const sa = g.attributes.aScale;
    pa.array.set(this._sortedPos.subarray(0, n * 3));
    ca.array.set(this._sortedCol.subarray(0, n * 4));
    sa.array.set(this._sortedScl.subarray(0, n));
    pa.needsUpdate = ca.needsUpdate = sa.needsUpdate = true;
  }

  // ── Render loop ────────────────────────────────────────────────────────────

  _animate() {
    requestAnimationFrame(() => this._animate());

    this._controls.update();
    this._frameCount++;

    // Smooth transition for interaction intensity
    if (this._mouseOver) {
      this._interacting = Math.min(1.0, this._interacting + 0.09);
    } else {
      this._interacting = Math.max(0.0, this._interacting - 0.055);
    }

    // Update interaction uniforms every frame (cheap)
    if (this._splatMesh) {
      const u = this._splatMesh.material.uniforms;
      u.uMouse.value.copy(this._mouseNDC);
      u.uInteracting.value = this._interacting;
      u.uTime.value        = this._clock.getElapsedTime();
    }

    // Depth re-sort every N frames
    if (this._splatMesh && this._frameCount % SORT_INTERVAL === 0) {
      this._sortSplats();
      this._pushSortedToGPU();
    }

    this._renderer.render(this._scene, this._camera);
  }
}
