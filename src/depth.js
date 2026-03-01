/**
 * Monocular depth estimation.
 *
 * PRIMARY  – Depth Anything Small (via @xenova/transformers).
 *            State-of-the-art monocular depth; works on any photo.
 *            Weights (~98 MB) download on first use and are cached in
 *            the browser's Cache API for subsequent visits.
 *
 * FALLBACK – Multi-cue heuristic used when the model is unavailable
 *            (offline, CSP blocked, etc.).  Notably worse on
 *            structurally complex images but instant and dependency-free.
 */

const MODEL_ID = 'Xenova/depth-anything-small-hf';

export class DepthEstimator {
  constructor() {
    this._pipe  = null;
    this._ready = false;
  }

  /**
   * Download and warm up the ML model.
   * Safe to call multiple times – only downloads once.
   *
   * @param {(fraction:number, filename:string)=>void} onProgress
   * @returns {Promise<boolean>} true = ML ready, false = fell back to heuristic
   */
  async loadModel(onProgress) {
    if (this._ready) return true;
    try {
      // Dynamic import keeps @xenova/transformers out of the critical path.
      const { pipeline, env } = await import('@xenova/transformers');

      // Single-threaded WASM – avoids SharedArrayBuffer / COEP requirements
      // that GitHub Pages (and most static hosts) do not send.
      env.backends.onnx.wasm.numThreads = 1;

      this._pipe = await pipeline('depth-estimation', MODEL_ID, {
        progress_callback: (info) => {
          if (info.status === 'progress' && onProgress) {
            onProgress((info.progress ?? 0) / 100, info.file ?? '');
          }
        },
      });
      this._ready = true;
      return true;
    } catch (err) {
      console.warn('[DepthEstimator] ML model unavailable – using heuristic.', err.message);
      return false;
    }
  }

  /**
   * Estimate a depth map for imageData.
   * @param {ImageData} imageData
   * @returns {Promise<Float32Array>}  length = width×height, 1 = close, 0 = far
   */
  async estimate(imageData) {
    if (this._ready && this._pipe) {
      try { return await this._mlEstimate(imageData); }
      catch (err) {
        console.warn('[DepthEstimator] Inference failed – using heuristic.', err.message);
      }
    }
    return this._heuristicEstimate(imageData);
  }

  // ── ML path ──────────────────────────────────────────────────────────────

  async _mlEstimate(imageData) {
    const { RawImage } = await import('@xenova/transformers');

    // Wrap the raw pixel buffer so the pipeline can read it directly.
    const input = new RawImage(
      new Uint8ClampedArray(imageData.data),
      imageData.width,
      imageData.height,
      4 // RGBA
    );

    const result = await this._pipe(input);

    // Use the raw float tensor for maximum precision.
    // predicted_depth shape: [1, H, W]  or  [H, W]
    if (result.predicted_depth) {
      return this._tensorToDepth(
        result.predicted_depth,
        imageData.width,
        imageData.height
      );
    }

    // Fallback: use the 8-bit visualisation RawImage.
    return this._rawImageToDepth(result.depth);
  }

  /**
   * Convert a float32 depth tensor to a normalised closeness Float32Array.
   * Depth Anything outputs inverse-depth (disparity): larger value = closer.
   */
  _tensorToDepth(tensor, targetW, targetH) {
    const raw   = tensor.data;                      // Float32Array
    const dims  = tensor.dims;
    const tH    = dims[dims.length - 2];
    const tW    = dims[dims.length - 1];
    const n     = tH * tW;

    // Percentile stretch (5th–95th) to ignore depth outliers at image borders.
    const sorted = Float32Array.from(raw).sort();
    const lo     = sorted[Math.floor(n * 0.05)];
    const hi     = sorted[Math.floor(n * 0.95)];
    const range  = hi - lo || 1;

    const out    = new Float32Array(targetW * targetH);
    const scaleX = tW / targetW;
    const scaleY = tH / targetH;

    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        const tx  = Math.min(tW - 1, Math.round(x * scaleX));
        const ty  = Math.min(tH - 1, Math.round(y * scaleY));
        // Clamp, then map: disparity higher = closer → closeness = direct value
        out[y * targetW + x] = Math.max(0, Math.min(1,
          (raw[ty * tW + tx] - lo) / range
        ));
      }
    }
    return out;
  }

  /** Fallback when predicted_depth is absent – use the 8-bit RawImage. */
  _rawImageToDepth(img) {
    const { data, width, height, channels } = img;
    const n   = width * height;
    const out = new Float32Array(n);
    let min = 255, max = 0;
    for (let i = 0; i < n; i++) {
      const v = data[i * channels];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    for (let i = 0; i < n; i++) {
      out[i] = (data[i * channels] - min) / range; // disparity: higher = closer
    }
    return out;
  }

  // ── Heuristic fallback ────────────────────────────────────────────────────

  _heuristicEstimate(imageData) {
    const { width, height, data } = imageData;
    const n = width * height;

    const lum  = new Float32Array(n);
    const warm = new Float32Array(n);
    const sat  = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const r = data[i * 4]     / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      lum[i]  = 0.299 * r + 0.587 * g + 0.114 * b;
      warm[i] = Math.max(0, Math.min(1, r * 0.65 + g * 0.35 - b * 0.55 + 0.5));
      sat[i]  = Math.max(r, g, b) - Math.min(r, g, b);
    }

    const sharp = this._laplacianEnergy(lum, width, height);
    const raw   = new Float32Array(n);

    for (let y = 0; y < height; y++) {
      const vertClose = y / (height - 1);
      for (let x = 0; x < width; x++) {
        const i  = y * width + x;
        const cx = 2 * (x / (width  - 1)) - 1;
        const cy = 2 * (y / (height - 1)) - 1;
        const cBias = Math.max(0, 1 - Math.sqrt(cx * cx + cy * cy) * 0.75);
        raw[i] =
          sharp[i]   * 0.32 +
          sat[i]     * 0.18 +
          lum[i]     * 0.14 +
          warm[i]    * 0.14 +
          vertClose  * 0.14 +
          cBias      * 0.08;
      }
    }

    return this._normalize(
      this._gaussianBlur(this._contrastStretch(raw), width, height, 4)
    );
  }

  // ── Shared utilities ──────────────────────────────────────────────────────

  _laplacianEnergy(lum, w, h) {
    const out = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        out[i] = Math.abs(
          -lum[(y-1)*w+x] - lum[(y+1)*w+x]
          -lum[y*w+(x-1)] - lum[y*w+(x+1)]
          + 4 * lum[i]
        );
      }
    }
    return this._normalize(this._gaussianBlur(out, w, h, 10));
  }

  _contrastStretch(arr) {
    const s  = Float32Array.from(arr).sort();
    const lo = s[Math.floor(s.length * 0.05)];
    const hi = s[Math.floor(s.length * 0.95)];
    const r  = hi - lo || 1;
    const o  = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) o[i] = Math.max(0, Math.min(1, (arr[i] - lo) / r));
    return o;
  }

  _gaussianBlur(data, w, h, radius) {
    const k = this._gaussianKernel(radius);
    return this._conv1D(this._conv1D(data, w, h, k, true), w, h, k, false);
  }

  _gaussianKernel(r) {
    const size = r * 2 + 1, sig = r / 2.5;
    const k = new Float32Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) { k[i] = Math.exp(-((i-r)**2) / (2*sig*sig)); sum += k[i]; }
    for (let i = 0; i < size; i++) k[i] /= sum;
    return k;
  }

  _conv1D(data, w, h, k, horiz) {
    const r = (k.length - 1) / 2;
    const o = new Float32Array(data.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let d = -r; d <= r; d++) {
          const sx = horiz ? Math.max(0, Math.min(w-1, x+d)) : x;
          const sy = horiz ? y : Math.max(0, Math.min(h-1, y+d));
          s += data[sy*w+sx] * k[d+r];
        }
        o[y*w+x] = s;
      }
    }
    return o;
  }

  _normalize(arr) {
    let min = Infinity, max = -Infinity;
    for (const v of arr) { if (v < min) min = v; if (v > max) max = v; }
    const r = max - min || 1;
    const o = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) o[i] = (arr[i] - min) / r;
    return o;
  }
}
