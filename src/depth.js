/**
 * Monocular depth estimation from a single 2D image.
 *
 * Uses a multi-cue heuristic that combines:
 *   - Laplacian sharpness  (sharp detail = in focus = close)
 *   - Luminance            (bright subjects tend to be foreground)
 *   - Color warmth         (atmospheric perspective: warm = near, cool = far)
 *   - Vertical position    (bottom = near, top = far for typical photos)
 *
 * Returns a normalized Float32Array where 0 = far, 1 = close.
 */
export class DepthEstimator {
  estimate(imageData) {
    const { width, height, data } = imageData;
    const n = width * height;

    const lum = new Float32Array(n);
    const warm = new Float32Array(n);

    // Pass 1: per-pixel luminance and color warmth
    for (let i = 0; i < n; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      // Warm bias: (r+g) vs b, clamped 0-1
      warm[i] = Math.max(0, Math.min(1, (r * 0.7 + g * 0.3 - b * 0.5 + 0.5)));
    }

    // Pass 2: Laplacian sharpness (local edge energy = in-focus = close)
    const sharp = this._laplacianEnergy(lum, width, height);

    // Pass 3: combine cues into a raw closeness map
    const raw = new Float32Array(n);
    for (let y = 0; y < height; y++) {
      const vert = 1 - y / (height - 1); // 0 at bottom, 1 at top → invert: close at bottom
      const vertClose = 1 - vert;         // 1 at bottom (close), 0 at top (far)
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        raw[i] =
          lum[i]    * 0.20 +
          warm[i]   * 0.20 +
          sharp[i]  * 0.35 +
          vertClose * 0.25;
      }
    }

    // Pass 4: smooth the map so depth transitions are continuous
    const smooth = this._gaussianBlur(raw, width, height, 6);

    // Pass 5: normalize 0-1
    return this._normalize(smooth);
  }

  // ── Internal helpers ────────────────────────────────────────────────────────

  /** Laplacian edge energy, normalised 0-1. */
  _laplacianEnergy(lum, width, height) {
    const n = width * height;
    const out = new Float32Array(n);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        const lap =
          -lum[(y - 1) * width + x] +
          -lum[(y + 1) * width + x] +
          -lum[y * width + (x - 1)] +
          -lum[y * width + (x + 1)] +
          4 * lum[i];
        out[i] = Math.abs(lap);
      }
    }
    // Blur the sharpness map so a sharp region influences its neighbourhood
    const blurred = this._gaussianBlur(out, width, height, 8);
    return this._normalize(blurred);
  }

  /** Separable Gaussian blur with a given pixel radius. */
  _gaussianBlur(data, width, height, radius) {
    const kernel = this._gaussianKernel(radius);
    const tmp = this._convolve1D(data, width, height, kernel, true);
    return this._convolve1D(tmp, width, height, kernel, false);
  }

  _gaussianKernel(radius) {
    const size = radius * 2 + 1;
    const sigma = radius / 2.5;
    const k = new Float32Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const x = i - radius;
      k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += k[i];
    }
    for (let i = 0; i < size; i++) k[i] /= sum;
    return k;
  }

  _convolve1D(data, width, height, kernel, horizontal) {
    const radius = (kernel.length - 1) / 2;
    const out = new Float32Array(data.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const sx = horizontal ? Math.max(0, Math.min(width - 1, x + k)) : x;
          const sy = horizontal ? y : Math.max(0, Math.min(height - 1, y + k));
          sum += data[sy * width + sx] * kernel[k + radius];
        }
        out[y * width + x] = sum;
      }
    }
    return out;
  }

  _normalize(arr) {
    let min = Infinity, max = -Infinity;
    for (const v of arr) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = (arr[i] - min) / range;
    return out;
  }
}
