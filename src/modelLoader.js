/**
 * 3D Model → Gaussian Splat converter.
 *
 * Supports .glb / .gltf (via Three.js GLTFLoader) and
 * .obj (via OBJLoader).  Walks the loaded scene, samples
 * points uniformly on every triangle's surface (area-weighted),
 * reads colour from vertex-colours → texture map → material colour,
 * normalises all positions into [-1, 1]³, and returns a splat
 * buffer compatible with SplatRenderer.loadSplats().
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader  } from 'three/addons/loaders/OBJLoader.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Bake a texture (THREE.Texture) into a flat Uint8Array via an offscreen canvas. */
function bakeTexture(texture) {
  const img = texture.image;
  if (!img || (!img.width && !img.naturalWidth)) return null;
  const w = img.naturalWidth  || img.width;
  const h = img.naturalHeight || img.height;
  const off = document.createElement('canvas');
  off.width  = w;
  off.height = h;
  const ctx = off.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return { pixels: ctx.getImageData(0, 0, w, h).data, w, h };
}

/** Sample a baked texture at UV [0..1]² → [r,g,b] 0-255. */
function sampleTexture(baked, u, v) {
  const x = Math.min(baked.w - 1, Math.max(0, Math.round(((u % 1) + 1) % 1 * (baked.w - 1))));
  const y = Math.min(baked.h - 1, Math.max(0, Math.round((1 - ((v % 1) + 1) % 1) * (baked.h - 1))));
  const p = (y * baked.w + x) * 4;
  return [baked.pixels[p], baked.pixels[p + 1], baked.pixels[p + 2]];
}

/** Signed-area of a triangle (only used for winding; abs gives real area). */
function triangleArea(ax, ay, az, bx, by, bz, cx, cy, cz) {
  const ex = bx - ax, ey = by - ay, ez = bz - az;
  const fx = cx - ax, fy = cy - ay, fz = cz - az;
  const nx = ey * fz - ez * fy;
  const ny = ez * fx - ex * fz;
  const nz = ex * fy - ey * fx;
  return 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
}

// ── ModelLoader ───────────────────────────────────────────────────────────────

export class ModelLoader {
  /**
   * Load a model file and return splat buffers.
   *
   * @param {File} file
   * @param {(pct:number, label:string)=>void} onProgress
   * @returns {Promise<{positions:Float32Array, colors:Float32Array, scales:Float32Array, count:number}>}
   */
  async load(file, onProgress) {
    onProgress(5, 'Loading 3D model…');
    const scene = await this._loadScene(file);

    onProgress(40, 'Sampling surface points…');
    await tick();
    const splats = await this._sampleScene(scene);

    onProgress(90, 'Uploading to GPU…');
    return splats;
  }

  // ── Scene loading ───────────────────────────────────────────────────────────

  async _loadScene(file) {
    const ext  = file.name.split('.').pop().toLowerCase();
    const url  = URL.createObjectURL(file);
    try {
      if (ext === 'obj') {
        return await this._loadOBJ(url);
      }
      return await this._loadGLTF(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  _loadGLTF(url) {
    return new Promise((resolve, reject) => {
      new GLTFLoader().load(url, gltf => resolve(gltf.scene), undefined, reject);
    });
  }

  _loadOBJ(url) {
    return new Promise((resolve, reject) => {
      new OBJLoader().load(url, resolve, undefined, reject);
    });
  }

  // ── Surface sampling ────────────────────────────────────────────────────────

  async _sampleScene(scene) {
    // Collect all triangles from the scene
    const triangles = [];

    scene.traverse(node => {
      if (!node.isMesh) return;
      const geo  = node.geometry;
      if (!geo)  return;

      // Ensure the geometry has an index (convert if needed)
      const indexed = geo.index
        ? geo
        : this._toIndexed(geo);

      const pos   = indexed.attributes.position;
      const uv    = indexed.attributes.uv  || null;
      const col   = indexed.attributes.color || null;
      const index = indexed.index.array;
      const mat   = Array.isArray(node.material) ? node.material[0] : node.material;

      // Bake texture once per mesh
      let baked = null;
      if (mat?.map) {
        try { baked = bakeTexture(mat.map); } catch { /* skip */ }
      }

      // Flat material colour fallback
      const flatCol = mat?.color
        ? [Math.round(mat.color.r * 255), Math.round(mat.color.g * 255), Math.round(mat.color.b * 255)]
        : [180, 180, 180];

      const worldMat = node.matrixWorld;

      for (let i = 0; i < index.length; i += 3) {
        const ia = index[i], ib = index[i + 1], ic = index[i + 2];

        const va = new THREE.Vector3().fromBufferAttribute(pos, ia).applyMatrix4(worldMat);
        const vb = new THREE.Vector3().fromBufferAttribute(pos, ib).applyMatrix4(worldMat);
        const vc = new THREE.Vector3().fromBufferAttribute(pos, ic).applyMatrix4(worldMat);

        const area = triangleArea(va.x, va.y, va.z, vb.x, vb.y, vb.z, vc.x, vc.y, vc.z);
        if (area < 1e-12) continue;

        triangles.push({ va, vb, vc, area, uv, ia, ib, ic, col, baked, flatCol });
      }
    });

    if (triangles.length === 0) {
      throw new Error('No triangles found in 3D model.');
    }

    // Build area-weighted CDF for uniform surface sampling
    const areas = triangles.map(t => t.area);
    const totalArea = areas.reduce((s, a) => s + a, 0);
    const cdf = new Float64Array(triangles.length);
    let acc = 0;
    for (let i = 0; i < triangles.length; i++) {
      acc += areas[i] / totalArea;
      cdf[i] = acc;
    }

    // Target splat count (similar density to image pipeline)
    const TARGET = Math.min(300_000, Math.max(50_000, triangles.length * 20));

    const positions = new Float32Array(TARGET * 3);
    const colors    = new Float32Array(TARGET * 4);
    const scales    = new Float32Array(TARGET);

    // Bounding box for normalisation
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    // Temp storage for un-normalised positions
    const rawPos = new Float32Array(TARGET * 3);

    for (let s = 0; s < TARGET; s++) {
      // Pick a random triangle proportional to area (binary search on CDF)
      const r = Math.random();
      let lo = 0, hi = triangles.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cdf[mid] < r) lo = mid + 1; else hi = mid;
      }
      const tri = triangles[lo];

      // Uniform point on triangle (Osada et al. method)
      const r1 = Math.random(), r2 = Math.random();
      const sq = Math.sqrt(r1);
      const u  = 1 - sq;
      const v  = sq * (1 - r2);
      const w  = sq * r2;

      const px = u * tri.va.x + v * tri.vb.x + w * tri.vc.x;
      const py = u * tri.va.y + v * tri.vb.y + w * tri.vc.y;
      const pz = u * tri.va.z + v * tri.vb.z + w * tri.vc.z;

      rawPos[s * 3    ] = px;
      rawPos[s * 3 + 1] = py;
      rawPos[s * 3 + 2] = pz;

      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
      if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;

      // Colour: vertex colour → texture → flat material
      let cr = tri.flatCol[0], cg = tri.flatCol[1], cb = tri.flatCol[2];

      if (tri.col) {
        // Interpolate vertex colours
        cr = Math.round((u * tri.col.getX(tri.ia) + v * tri.col.getX(tri.ib) + w * tri.col.getX(tri.ic)) * 255);
        cg = Math.round((u * tri.col.getY(tri.ia) + v * tri.col.getY(tri.ib) + w * tri.col.getY(tri.ic)) * 255);
        cb = Math.round((u * tri.col.getZ(tri.ia) + v * tri.col.getZ(tri.ib) + w * tri.col.getZ(tri.ic)) * 255);
      } else if (tri.baked && tri.uv) {
        // Interpolate UVs
        const uvU = u * tri.uv.getX(tri.ia) + v * tri.uv.getX(tri.ib) + w * tri.uv.getX(tri.ic);
        const uvV = u * tri.uv.getY(tri.ia) + v * tri.uv.getY(tri.ib) + w * tri.uv.getY(tri.ic);
        [cr, cg, cb] = sampleTexture(tri.baked, uvU, uvV);
      }

      colors[s * 4    ] = cr / 255;
      colors[s * 4 + 1] = cg / 255;
      colors[s * 4 + 2] = cb / 255;
      colors[s * 4 + 3] = 0.82;
    }

    // Normalise positions to [-1, 1]³ keeping aspect ratio
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
    const invSpan = 2.0 / span;

    for (let s = 0; s < TARGET; s++) {
      positions[s * 3    ] = (rawPos[s * 3    ] - cx) * invSpan;
      positions[s * 3 + 1] = (rawPos[s * 3 + 1] - cy) * invSpan;
      positions[s * 3 + 2] = (rawPos[s * 3 + 2] - cz) * invSpan;
    }

    // Scale: uniform based on target density
    const splatScale = (2.0 / Math.sqrt(TARGET)) * 1.1;
    scales.fill(splatScale);

    return { positions, colors, scales, count: TARGET };
  }

  /** Convert non-indexed BufferGeometry to indexed (needed for triangle loop). */
  _toIndexed(geo) {
    // Three.js toNonIndexed exists; for the inverse we build a simple 0,1,2,3,… index
    const count = geo.attributes.position.count;
    const idx   = new Uint32Array(count);
    for (let i = 0; i < count; i++) idx[i] = i;
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    return geo;
  }
}

function tick() {
  return new Promise(r => requestAnimationFrame(r));
}
