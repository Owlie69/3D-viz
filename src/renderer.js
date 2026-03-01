/**
 * Three.js renderer for 3D Gaussian splats.
 *
 * Each splat is a screen-aligned billboard quad drawn with a custom
 * shader that applies a 2D Gaussian falloff in the fragment stage.
 * Instances are rendered via InstancedBufferGeometry for a single draw call.
 *
 * Depth sorting (back-to-front) runs every SORT_INTERVAL frames so that
 * alpha-blended splats composite correctly as the camera moves.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Shaders ──────────────────────────────────────────────────────────────────

const VERT = /* glsl */ `
  // Per-instance data
  attribute vec3  aPos;    // splat world-space centre
  attribute vec4  aColor;  // RGBA (linear)
  attribute float aScale;  // billboard radius in world units

  varying vec2 vUV;
  varying vec4 vColor;

  void main() {
    // Transform instance centre to view space
    vec4 viewCenter = modelViewMatrix * vec4(aPos, 1.0);

    // Add screen-aligned quad offset (billboard) in view space
    // position.xy comes from PlaneGeometry(1,1): range -0.5 .. 0.5
    viewCenter.xy += position.xy * aScale;

    vUV    = position.xy;   // -0.5 .. 0.5 used for Gaussian eval
    vColor = aColor;

    gl_Position = projectionMatrix * viewCenter;
  }
`;

const FRAG = /* glsl */ `
  precision mediump float;

  varying vec2 vUV;
  varying vec4 vColor;

  void main() {
    // 2D Gaussian centered on the quad, falling off toward the edges
    // dot(vUV,vUV) = 0 at centre, 0.25 at axis-edge, 0.5 at corner
    float r2    = dot(vUV, vUV);
    float gauss = exp(-r2 * 9.0);
    if (gauss < 0.008) discard;

    gl_FragColor = vec4(vColor.rgb, vColor.a * gauss);
  }
`;

// ── Constants ────────────────────────────────────────────────────────────────

const SORT_INTERVAL = 8;  // re-sort every N frames
const BG_COLOR      = 0x05050c;

// ── SplatRenderer ────────────────────────────────────────────────────────────

export class SplatRenderer {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this._canvas      = canvas;
    this._frameCount  = 0;
    this._splatMesh   = null;
    this._rawPositions = null;  // unsorted Float32Arrays kept for re-sorting
    this._rawColors    = null;
    this._rawScales    = null;
    this._splatCount   = 0;

    // Sorted (temp) arrays reused each sort pass
    this._sortedPos    = null;
    this._sortedCol    = null;
    this._sortedScl    = null;
    this._sortedIdx    = null;
    this._distances    = null;

    this._initThree();
    this._initControls();
    this._bindResize();
    this._animate();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Upload splat data to the GPU.
   * @param {{ positions:Float32Array, colors:Float32Array, scales:Float32Array, count:number }} splats
   */
  loadSplats(splats) {
    const { positions, colors, scales, count } = splats;

    // Dispose previous mesh
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

    // Pre-allocate sorted work arrays
    this._sortedPos  = new Float32Array(count * 3);
    this._sortedCol  = new Float32Array(count * 4);
    this._sortedScl  = new Float32Array(count);
    this._sortedIdx  = new Int32Array(count);
    this._distances  = new Float32Array(count);

    for (let i = 0; i < count; i++) this._sortedIdx[i] = i;

    // Initial sort
    this._sortSplats();

    // Build geometry with sorted data
    this._splatMesh = this._buildMesh(
      this._sortedPos.slice(0, count * 3),
      this._sortedCol.slice(0, count * 4),
      this._sortedScl.slice(0, count),
      count
    );
    this._scene.add(this._splatMesh);
  }

  /** Reposition camera to sensibly frame the loaded splats. */
  resetCamera() {
    this._camera.position.set(0, 0, 2.2);
    this._camera.lookAt(0, 0, 0);
    this._controls.target.set(0, 0, 0);
    this._controls.update();
  }

  get renderer() { return this._renderer; }

  // ── Three.js init ──────────────────────────────────────────────────────────

  _initThree() {
    const r = new THREE.WebGLRenderer({
      canvas:    this._canvas,
      antialias: true,
      alpha:     false,
    });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.setSize(window.innerWidth, window.innerHeight);
    r.setClearColor(BG_COLOR, 1);
    this._renderer = r;

    this._scene = new THREE.Scene();

    // Subtle depth-fog so far splats fade to background
    this._scene.fog = new THREE.FogExp2(BG_COLOR, 0.18);

    this._camera = new THREE.PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.005,
      80
    );
    this._camera.position.set(0, 0, 2.2);
  }

  _initControls() {
    const ctrl = new OrbitControls(this._camera, this._canvas);
    ctrl.enableDamping  = true;
    ctrl.dampingFactor  = 0.06;
    ctrl.minDistance    = 0.05;
    ctrl.maxDistance    = 30;
    ctrl.enablePan      = true;
    ctrl.screenSpacePanning = true;
    this._controls = ctrl;
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

    geom.index      = baseGeom.index;
    geom.attributes.position = baseGeom.attributes.position;

    geom.setAttribute('aPos',
      new THREE.InstancedBufferAttribute(positions, 3));
    geom.setAttribute('aColor',
      new THREE.InstancedBufferAttribute(colors, 4));
    geom.setAttribute('aScale',
      new THREE.InstancedBufferAttribute(scales, 1));

    geom.instanceCount = count;

    const mat = new THREE.ShaderMaterial({
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

    // Compute squared distance to camera for every splat
    for (let i = 0; i < n; i++) {
      const dx = pos[i * 3]     - cam.x;
      const dy = pos[i * 3 + 1] - cam.y;
      const dz = pos[i * 3 + 2] - cam.z;
      this._distances[i] = dx * dx + dy * dy + dz * dz;
    }

    // Sort index array far-to-near (back-to-front for correct alpha blend)
    const dist = this._distances;
    this._sortedIdx.sort((a, b) => dist[b] - dist[a]);

    // Write sorted data into reusable arrays
    const sp = this._sortedPos;
    const sc = this._sortedCol;
    const ss = this._sortedScl;
    const raw = this._rawPositions;
    const rcol = this._rawColors;
    const rscl = this._rawScales;
    const idx  = this._sortedIdx;

    for (let i = 0; i < n; i++) {
      const s = idx[i];
      sp[i * 3]     = raw[s * 3];
      sp[i * 3 + 1] = raw[s * 3 + 1];
      sp[i * 3 + 2] = raw[s * 3 + 2];
      sc[i * 4]     = rcol[s * 4];
      sc[i * 4 + 1] = rcol[s * 4 + 1];
      sc[i * 4 + 2] = rcol[s * 4 + 2];
      sc[i * 4 + 3] = rcol[s * 4 + 3];
      ss[i]         = rscl[s];
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

    pa.needsUpdate = true;
    ca.needsUpdate = true;
    sa.needsUpdate = true;
  }

  // ── Render loop ────────────────────────────────────────────────────────────

  _animate() {
    requestAnimationFrame(() => this._animate());

    this._controls.update();
    this._frameCount++;

    if (this._splatMesh && this._frameCount % SORT_INTERVAL === 0) {
      this._sortSplats();
      this._pushSortedToGPU();
    }

    this._renderer.render(this._scene, this._camera);
  }
}
