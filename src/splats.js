/**
 * Converts a 2D ImageData + a depth map (Float32Array, 0=far 1=close)
 * into a set of 3D Gaussian splat primitives ready for the GPU.
 *
 * Each splat is represented by:
 *   position  (x, y, z)  – world-space centre
 *   color     (r, g, b, a)
 *   scale     (uniform float)  – billboard radius in world units
 *
 * The image occupies the 2×2 XY plane centred at origin.
 * Close pixels sit near z = 0; far pixels extend back to z = -depthRange.
 */
export class SplatGenerator {
  /**
   * @param {ImageData}    imageData
   * @param {Float32Array} depthMap  – same length as width*height, 0=far 1=close
   * @param {object}       opts
   * @returns {{ positions, colors, scales, count }}
   */
  generate(imageData, depthMap, opts = {}) {
    const { width, height, data } = imageData;
    const {
      step       = 2,     // sample every N pixels
      oversample = 1,     // splats placed per pixel (sub-pixel jitter for density)
      depthRange = 3.5,   // total z extent of the scene (deeper = more parallax)
      baseScale  = 0.010, // splat radius at step=1; multiplied by step
    } = opts;

    const cols = Math.ceil(width  / step);
    const rows = Math.ceil(height / step);
    const maxCount = cols * rows * oversample;

    const positions = new Float32Array(maxCount * 3);
    const colors    = new Float32Array(maxCount * 4);
    const scales    = new Float32Array(maxCount);

    // Per-splat alpha is divided by oversample so total additive brightness stays correct
    const alphaScale = 0.90 / oversample;

    let idx = 0;

    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const pi = (y * width + x) * 4;
        const a  = data[pi + 3] / 255;
        if (a < 0.05) continue;           // skip fully transparent pixels

        const r = data[pi]     / 255;
        const g = data[pi + 1] / 255;
        const b = data[pi + 2] / 255;

        const closeness = depthMap[y * width + x];
        const scale = baseScale * step * (1.0 + (1 - closeness) * 0.6);

        for (let s = 0; s < oversample; s++) {
          // Sub-pixel jitter keeps each splat within the pixel footprint
          const jx = oversample > 1 ? (Math.random() - 0.5) * step : 0;
          const jy = oversample > 1 ? (Math.random() - 0.5) * step : 0;
          const jz = oversample > 1 ? (Math.random() - 0.5) * depthRange * 0.03 : 0;

          // depth: 0 = far, 1 = close  →  z: close ≈ 0, far ≈ -depthRange
          const z = -(1 - closeness) * depthRange + jz;

          // image pixel → [-1,1] NDC, Y flipped
          const wx = ((x + jx) / (width  - 1) - 0.5) * 2.0;
          const wy = -((y + jy) / (height - 1) - 0.5) * 2.0;

          positions[idx * 3]     = wx;
          positions[idx * 3 + 1] = wy;
          positions[idx * 3 + 2] = z;

          colors[idx * 4]     = r;
          colors[idx * 4 + 1] = g;
          colors[idx * 4 + 2] = b;
          colors[idx * 4 + 3] = a * alphaScale;

          scales[idx] = scale;
          idx++;
        }
      }
    }

    // Return tight slices so we don't send unused buffer to the GPU
    return {
      positions: positions.slice(0, idx * 3),
      colors:    colors.slice(0, idx * 4),
      scales:    scales.slice(0, idx),
      count:     idx,
    };
  }
}
