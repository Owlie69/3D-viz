import './style.css';
import { DepthEstimator } from './depth.js';
import { SplatGenerator }  from './splats.js';
import { SplatRenderer }   from './renderer.js';

// ── Elements ─────────────────────────────────────────────────────────────────

const uploadScreen  = document.getElementById('upload-screen');
const processingScreen = document.getElementById('processing-screen');
const viewerScreen  = document.getElementById('viewer-screen');
const canvas        = document.getElementById('canvas');
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const demoBtn       = document.getElementById('demo-btn');
const progressBar   = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const splatCountEl  = document.getElementById('splat-count');
const backBtn       = document.getElementById('back-btn');

// ── Global state ─────────────────────────────────────────────────────────────

const depthEstimator = new DepthEstimator();
const splatGenerator = new SplatGenerator();
let   renderer       = null;

// ── Pipeline ──────────────────────────────────────────────────────────────────

async function processImage(imageData) {
  showScreen(processingScreen);
  await tick();

  setProgress(5, 'Analysing image…');
  await tick();

  // Depth estimation runs on the main thread; yield so the UI can repaint
  setProgress(15, 'Estimating depth…');
  await tick();
  const depthMap = await runInChunks(() => depthEstimator.estimate(imageData));

  setProgress(60, 'Generating Gaussian splats…');
  await tick();
  const splats = splatGenerator.generate(imageData, depthMap, { step: 1 });

  setProgress(80, `Uploading ${splats.count.toLocaleString()} splats to GPU…`);
  await tick();

  if (!renderer) {
    renderer = new SplatRenderer(canvas);
  }
  renderer.loadSplats(splats);
  renderer.resetCamera();

  splatCountEl.textContent = splats.count.toLocaleString() + ' splats';

  setProgress(100, 'Done');
  await tick();

  showScreen(viewerScreen);
}

/** Wraps a synchronous computation in a zero-timeout so the browser repaints. */
function runInChunks(fn) {
  return new Promise(resolve => setTimeout(() => resolve(fn()), 0));
}

function setProgress(pct, label) {
  progressBar.style.width  = pct + '%';
  progressLabel.textContent = label;
}

function showScreen(el) {
  for (const s of [uploadScreen, processingScreen, viewerScreen]) {
    s.classList.toggle('active', s === el);
  }
}

async function tick() {
  return new Promise(r => requestAnimationFrame(r));
}

// ── Image loading helpers ─────────────────────────────────────────────────────

function fileToImageData(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const MAX = 380;
      const scale = Math.min(1, MAX / Math.max(img.width, img.height));
      const w = Math.round(img.width  * scale);
      const h = Math.round(img.height * scale);
      const offscreen = document.createElement('canvas');
      offscreen.width  = w;
      offscreen.height = h;
      const ctx = offscreen.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      resolve(ctx.getImageData(0, 0, w, h));
      URL.revokeObjectURL(url);
    };
    img.onerror = reject;
    img.src = url;
  });
}

// ── Demo mode – procedurally generated test image ────────────────────────────

function buildDemoImage() {
  const SIZE = 300;
  const off  = document.createElement('canvas');
  off.width  = SIZE;
  off.height = SIZE;
  const ctx = off.getContext('2d');

  // Sky gradient
  const sky = ctx.createLinearGradient(0, 0, 0, SIZE);
  sky.addColorStop(0,   '#0d1b6e');
  sky.addColorStop(0.5, '#1a3a8a');
  sky.addColorStop(1,   '#e08040');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Sun glow
  const sun = ctx.createRadialGradient(SIZE * 0.5, SIZE * 0.52, 4, SIZE * 0.5, SIZE * 0.52, 90);
  sun.addColorStop(0,   'rgba(255,240,80,0.95)');
  sun.addColorStop(0.3, 'rgba(255,160,20,0.6)');
  sun.addColorStop(1,   'rgba(255,100,0,0)');
  ctx.fillStyle = sun;
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Horizon ground
  const gnd = ctx.createLinearGradient(0, SIZE * 0.52, 0, SIZE);
  gnd.addColorStop(0, '#4a3020');
  gnd.addColorStop(1, '#1a0a05');
  ctx.fillStyle = gnd;
  ctx.fillRect(0, SIZE * 0.52, SIZE, SIZE * 0.48);

  // Stars
  ctx.fillStyle = 'white';
  for (let i = 0; i < 80; i++) {
    const sx = Math.random() * SIZE;
    const sy = Math.random() * SIZE * 0.45;
    const sr = Math.random() * 1.2 + 0.3;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fill();
  }

  // Silhouette mountains
  ctx.fillStyle = '#0d0a18';
  ctx.beginPath();
  ctx.moveTo(0, SIZE * 0.62);
  for (let x = 0; x <= SIZE; x += SIZE / 12) {
    const h = SIZE * 0.52 - Math.sin(x * 0.07) * 40 - Math.random() * 20;
    ctx.lineTo(x, h);
  }
  ctx.lineTo(SIZE, SIZE);
  ctx.lineTo(0, SIZE);
  ctx.closePath();
  ctx.fill();

  return ctx.getImageData(0, 0, SIZE, SIZE);
}

// ── Events ────────────────────────────────────────────────────────────────────

fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  processImage(await fileToImageData(file));
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', async e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    processImage(await fileToImageData(file));
  }
});

dropZone.addEventListener('click', () => fileInput.click());

demoBtn.addEventListener('click', () => {
  processImage(buildDemoImage());
});

backBtn.addEventListener('click', () => {
  showScreen(uploadScreen);
});

// ── Init ──────────────────────────────────────────────────────────────────────

showScreen(uploadScreen);
