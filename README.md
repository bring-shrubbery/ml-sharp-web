# ml-sharp-web

A browser-based Gaussian splat generator built on top of [Apple SHARP](https://github.com/apple/ml-sharp).

This project lets you:
- upload one image
- generate Gaussian splats in the browser
- preview the result
- download a `.ply` file

## Links

- Project repo: [bring-shrubbery/ml-sharp-web](https://github.com/bring-shrubbery/ml-sharp-web)
- Upstream SHARP repo (Apple): [apple/ml-sharp](https://github.com/apple/ml-sharp)
- SHARP project page: [apple.github.io/ml-sharp](https://apple.github.io/ml-sharp/)
- SHARP paper: [arXiv:2512.10685](https://arxiv.org/abs/2512.10685)

## Before you start (important license note)

Apple's SHARP repository has separate licenses for code and model weights.

- SHARP code license: [LICENSE](https://github.com/apple/ml-sharp/blob/main/LICENSE)
- SHARP model license: [LICENSE_MODEL](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL)

If you use Apple's released SHARP checkpoint/weights, you must follow `LICENSE_MODEL` (research-use restrictions apply).

## What you need

- [Bun](https://bun.sh/) installed
- A modern desktop browser (Chrome or Edge recommended)
- Enough disk space and RAM for the SHARP model (the exported ONNX sidecar is large, ~2.4 GB)

## Quick start (run the app)

### 1. Install dependencies

```bash
bun install
```

This also copies ONNX Runtime Web WASM assets into `public/ort/` automatically.

### 2. Start the app

```bash
bun dev
```

Open the URL shown by Vite (usually `http://localhost:5173`).

### 3. Use the app

1. Upload an image.
2. Click `Generate Splat`.
3. Preview the result and download the `.ply` file.

## Important model file note (`.onnx` + `.onnx.data`)

SHARP exports usually produce **two files**:

- `sharp_web_predictor.onnx`
- `sharp_web_predictor.onnx.data`

Both files must be served together from the same folder (for example `public/models/`).

Why this matters:
- The `.onnx` file is only the graph and metadata.
- The `.onnx.data` file contains most of the model weights.

For that reason, the app uses the hosted model by default.
Uploading only the `.onnx` file directly in the browser usually will not work because the `.onnx.data` sidecar is separate.

## Export the SHARP model to ONNX (beginner-friendly steps)

Everything runs in the browser, but you still need an exported SHARP ONNX model.

### 1. Clone Apple's SHARP repo (reference code)

```bash
git clone https://github.com/apple/ml-sharp /tmp/ml-sharp-upstream
```

### 2. Prepare a Python environment for export

You need Python + SHARP dependencies + ONNX export dependencies.

The easiest route is to follow the upstream SHARP setup first, then run this exporter script from this repo.

### 3. Export the browser predictor ONNX

From this repo:

```bash
python3 scripts/export_sharp_onnx.py \
  --sharp-repo /tmp/ml-sharp-upstream \
  --output public/models/sharp_web_predictor.onnx
```

If the model is large (it is), the script will also write:

```text
public/models/sharp_web_predictor.onnx.data
```

### Optional export flags

- `--checkpoint /path/to/sharp_2572gikvuh.pt` to use a manually downloaded checkpoint
- `--device cuda` to export on GPU (if your environment supports it)
- `--opset 20` to change ONNX opset (default is `20`)

## Static build (optional)

If you want a static build instead of running `bun dev`:

```bash
bun run build
bun run preview
```

Notes:
- `bun run build` copies `public/` into `dist/`, including the model files.
- If `sharp_web_predictor.onnx.data` is present, the build output will be very large.

## How it works (high level)

- React + TypeScript UI (`src/`)
- ONNX Runtime Web worker for inference (`src/workers/sharpWorker.ts`)
- Browser-side SHARP postprocessing (NDC -> metric gaussian conversion)
- Browser-side PLY writer
- In-page preview with [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D)

## Troubleshooting

### "expected magic word ... found 3c 21 64 6f" (WASM error)

This means a WASM file request returned HTML instead.

Try:
- run the app with `bun dev` (not `file://...`)
- restart the dev server after `bun install`
- verify these load in your browser:
  - `/ort/ort-wasm-simd-threaded.asyncify.mjs`
  - `/ort/ort-wasm-simd-threaded.asyncify.wasm`

### "Failed to load external data file ... sharp_web_predictor.onnx.data"

This means the ONNX sidecar file is missing or not served correctly.

Check:
- `public/models/sharp_web_predictor.onnx`
- `public/models/sharp_web_predictor.onnx.data`
- The app can reach the hosted model files in your deployment/browser environment

### The app runs, but generation is very slow or crashes

SHARP is large and browser inference is heavy.

Try:
- Chrome or Edge (desktop)
- smaller `Max gaussians` in the UI
- closing other memory-heavy tabs/apps
- waiting longer on first run (model + runtime initialization can take time)

## Tech stack

- [Bun](https://bun.sh/)
- [React](https://react.dev/)
- [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vite.dev/)
- [ONNX Runtime Web](https://onnxruntime.ai/)
- [GaussianSplats3D viewer](https://github.com/mkkellogg/GaussianSplats3D)

## Project status

Working prototype / experimental.

The app runs end-to-end in the browser, but performance and compatibility depend heavily on browser WebGPU/WASM support and your machine's available memory.
