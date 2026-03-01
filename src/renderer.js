/**
 * Three.js renderer for 3D Gaussian splats – additive blending edition.
 *
 * Mouse interaction: pure positional wave ripple only — no scale change,
 * no brightness/colour change, no glow. Just points gently undulating.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Shaders ──────────────────────────────────────────────────────────────────

const VERT = /* glsl */ `
  attribute vec3  aPos;
  attribute vec4  aColor;
  attribute float aScale;

  uniform vec2  uMouse;        // cursor NDC  −1..1
  uniform float uInteracting;  // 0 idle → 1 active (smoothly animated)
  uniform float uTime;

  varying vec2 vUV;
  varying vec4 vColor;

  void main() {
    vec4 viewCenter = modelViewMatrix * vec4(aPos, 1.0);

    // Project centre to NDC to measure cursor proximity
    vec4 clipC = projectionMatrix * viewCenter;
    vec2 ndc   = clipC.xy / clipC.w;
    vec2 diff  = ndc - uMouse;
    float dist = length(diff);
    vec2  dir  = dist > 0.001 ? diff / dist : vec2(0.7071, 0.7071);
    vec2  perp = vec2(-dir.y, dir.x);

    // Scale and colour are always unchanged – no visual effect near cursor
    vColor = aColor;

    if (uInteracting > 0.01) {
      // Smooth radial falloff: full effect at cursor, zero at radius 0.35 NDC
      float t    = uInteracting;
      float core = pow(max(0.0, 1.0 - dist / 0.35), 2.2) * t;

      // Traveling wave: appears to radiate outward from the cursor position
      float wave = sin(uTime * 3.5 - dist * 24.0);

      // Small transverse component for organic feel
      float sway = cos(uTime * 2.1 - dist * 17.0) * 0.35;

      // Pure positional nudge – amplitude ~0.011 view-space units (invisible at scale)
      viewCenter.xy += (dir * wave + perp * sway) * core * 0.011;
    }

    viewCenter.xy += position.xy * aScale;

    vUV = position.xy;
    gl_Position = projectionMatrix * viewCenter;
  }
`;

const FRAG = /* glsl */ `
  precision mediump float;

  varying vec2 vUV;
  varying vec4 vColor;

  void main() {
    float r2    = dot(vUV, vUV);
    float gauss = exp(-r2 * 11.5);
    if (gauss < 0.007) discard;
    gl_FragColor = vec4(vColor.rgb, vColor.a * gauss);
  }
`;

// ── Constants ────────────────────────────────────────────────────────────────

const BG_COLOR = 0x05050c;

// ── SplatRenderer ────────────────────────────────────────────────────────────

export class SplatRenderer {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this._canvas      = canvas;
    this._splatMesh   = null;
    this._mouseNDC    = new THREE.Vector2(0, 0);
    this._mouseOver   = false;
    this._interacting = 0.0;
    this._clock       = new THREE.Clock();

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

    // Build the GPU mesh once – additive blending means no per-frame sort
    this._splatMesh = this._buildMesh(positions, colors, scales, count);
    this._scene.add(this._splatMesh);
  }

  resetCamera() {
    this._camera.position.set(0, 0, 2.4);
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

    this._camera = new THREE.PerspectiveCamera(
      52, window.innerWidth / window.innerHeight, 0.005, 80
    );
    this._camera.position.set(0, 0, 2.4);
  }

  _initControls() {
    const ctrl = new OrbitControls(this._camera, this._canvas);
    ctrl.enableDamping      = true;
    ctrl.dampingFactor      = 0.06;
    ctrl.minDistance        = 0.05;
    ctrl.maxDistance        = 30;
    ctrl.enablePan          = true;
    ctrl.screenSpacePanning = true;
    this._controls = ctrl;
  }

  _initMouse() {
    const el = this._canvas;

    const setMouse = (cx, cy) => {
      const r = el.getBoundingClientRect();
      this._mouseNDC.x =  ((cx - r.left) / r.width)  * 2 - 1;
      this._mouseNDC.y = -((cy - r.top)  / r.height) * 2 + 1;
    };

    el.addEventListener('mousemove',  e => { setMouse(e.clientX, e.clientY); this._mouseOver = true;  });
    el.addEventListener('mouseleave', ()  => { this._mouseOver = false; });
    el.addEventListener('mouseenter', ()  => { this._mouseOver = true;  });

    el.addEventListener('touchmove', e => {
      e.preventDefault();
      const t = e.touches[0];
      setMouse(t.clientX, t.clientY);
      this._mouseOver = true;
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

    geom.index               = baseGeom.index;
    geom.attributes.position = baseGeom.attributes.position;
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
      depthTest:      false,
      // Additive blending: dst += src×alpha  – no sort needed, no dark halos
      blending:       THREE.AdditiveBlending,
    });

    baseGeom.dispose();
    return new THREE.Mesh(geom, mat);
  }

  // ── Render loop ────────────────────────────────────────────────────────────

  _animate() {
    requestAnimationFrame(() => this._animate());

    this._controls.update();

    // Smooth hover fade
    this._interacting = this._mouseOver
      ? Math.min(1.0, this._interacting + 0.09)
      : Math.max(0.0, this._interacting - 0.055);

    if (this._splatMesh) {
      const u = this._splatMesh.material.uniforms;
      u.uMouse.value.copy(this._mouseNDC);
      u.uInteracting.value = this._interacting;
      u.uTime.value        = this._clock.getElapsedTime();
    }

    this._renderer.render(this._scene, this._camera);
  }
}
