/**
 * Monocular depth estimation from a single 2D image.
 *
 * Combines six complementary cues – each captures a different physical
 * property of how photographers and optics create depth:
 *
 *  1. Laplacian sharpness  — sharp detail = in-focus = close
 *  2. Luminance            — bright subjects tend to be foreground
 *  3. Colour warmth        — atmospheric perspective: warm=near, cool=far
 *  4. Saturation           — vivid, saturated colours = foreground subject
 *  5. Vertical position    — horizon/sky at top = far, ground at bottom = near
 *  6. Centre bias          — subjects are usually framed centrally
 *
 * The map is post-processed with edge-aware contrast stretching so the
 * full 0–1 range is always used, then lightly smoothed.
 */
export class DepthEstimator {
  estimate(imageData) {
    const { width, height, data } = imageData;
    const n = width * height;

    // ── Per-pixel cue arrays ───────────────────────────────────────────────
    const lum  = new Float32Array(n);
    const warm = new Float32Array(n);
    const sat  = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const r = data[i * 4]     / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;

      lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;

      // Warm bias: (r+g mix) vs blue, clamped 0-1
      warm[i] = Math.max(0, Math.min(1, (r * 0.65 + g * 0.35 - b * 0.55 + 0.5)));

      // Saturation: max channel − min channel
      const cmax = Math.max(r, g, b), cmin = Math.min(r, g, b);
      sat[i] = cmax - cmin;
    }

    // ── Sharpness (Laplacian energy) ───────────────────────────────────────
    const sharp = this._laplacianEnergy(lum, width, height);

    // ── Combine into a single closeness map ───────────────────────────────
    const raw = new Float32Array(n);

    for (let y = 0; y < height; y++) {
      // Vertical: bottom = close (1), top = far (0) – works for landscapes/portraits
      const vertClose = y / (height - 1);

      for (let x = 0; x < width; x++) {
        const i = y * width + x;

        // Centre bias: subjects are usually photographed near the frame centre
        const cx = 2 * (x / (width  - 1)) - 1;   // −1..1
        const cy = 2 * (y / (height - 1)) - 1;
        const centreBias = Math.max(0, 1 - Math.sqrt(cx * cx + cy * cy) * 0.75);

        raw[i] =
          sharp[i]    * 0.32 +
          sat[i]      * 0.18 +
          lum[i]      * 0.14 +
          warm[i]     * 0.14 +
          vertClose   * 0.14 +
          centreBias  * 0.08;
      }
    }

    // ── Contrast stretch to always use the full 0-1 range ─────────────────
    const stretched = this._contrastStretch(raw);

    // ── Light smoothing (preserve edges, remove pixel noise) ──────────────
    const smooth = this._gaussianBlur(stretched, width, height, 4);

    return this._normalize(smooth);
  }

  // ── Internal helpers ────────────────────────────────────────────────────────

  /** Laplacian edge energy, normalised 0-1.  Large radius captures broad focus zones. */
  _laplacianEnergy(lum, width, height) {
    const n   = width * height;
    const out = new Float32Array(n);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i   = y * width + x;
        const lap =
          -lum[(y - 1) * width + x] +
          -lum[(y + 1) * width + x] +
          -lum[y * width + (x - 1)] +
          -lum[y * width + (x + 1)] +
           4 * lum[i];
        out[i] = Math.abs(lap);
      }
    }
    // Blur with larger radius so a sharp region pulls its neighbours close too
    const blurred = this._gaussianBlur(out, width, height, 10);
    return this._normalize(blurred);
  }

  /** Spread values so 5th–95th percentile fills 0-1 (robust to outliers). */
  _contrastStretch(arr) {
    const sorted = Float32Array.from(arr).sort();
    const lo = sorted[Math.floor(sorted.length * 0.05)];
    const hi = sorted[Math.floor(sorted.length * 0.95)];
    const range = hi - lo || 1;
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      out[i] = Math.max(0, Math.min(1, (arr[i] - lo) / range));
    }
    return out;
  }

  _gaussianBlur(data, width, height, radius) {
    const kernel = this._gaussianKernel(radius);
    const tmp    = this._convolve1D(data, width, height, kernel, true);
    return this._convolve1D(tmp,  width, height, kernel, false);
  }

  _gaussianKernel(radius) {
    const size  = radius * 2 + 1;
    const sigma = radius / 2.5;
    const k     = new Float32Array(size);
    let   sum   = 0;
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
    const out    = new Float32Array(data.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const sx = horizontal ? Math.max(0, Math.min(width  - 1, x + k)) : x;
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
    const out   = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = (arr[i] - min) / range;
    return out;
  }
}
