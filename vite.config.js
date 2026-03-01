import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'docs',
    assetsInlineLimit: 0,
  },
  // @xenova/transformers loads ONNX WASM at runtime from CDN;
  // excluding it from pre-bundling avoids Vite choking on the
  // Node-specific onnxruntime-node sub-package.
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },
  worker: {
    format: 'es',
  },
});
