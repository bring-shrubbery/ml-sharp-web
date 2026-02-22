# ml-sharp-web

Browser-first (client-side) web UI for generating and previewing Gaussian splats from a single image using an exported ONNX wrapper around Apple SHARP.

## Status

Experimental.

This project implements:
- single-page web UI (React + TypeScript + Bun/Vite)
- image upload
- in-browser ONNX inference worker (no backend inference server)
- Gaussian splat preview in the page
- `.ply` download

## Important license note

Apple's `ml-sharp` repository has separate licenses for code and model weights.

- Code (`LICENSE`): allows modification/redistribution with conditions
- Model weights (`LICENSE_MODEL`): research-purpose restrictions apply

If you use Apple's released SHARP checkpoint/weights, your usage must comply with `LICENSE_MODEL`.

## Architecture

The web app expects an ONNX wrapper model with these inputs and outputs:

Inputs:
- `image` (`float32[1,3,1536,1536]`) RGB in `[0,1]`
- `disparity_factor` (`float32[1]`) = `f_px / image_width`
- `f_px` (`float32[1]`)
- `orig_width` (`float32[1]`)
- `orig_height` (`float32[1]`)

Outputs:
- `mean_vectors`
- `singular_values`
- `quaternions`
- `colors`
- `opacities`

The UI then filters/caps Gaussians for browser performance, serializes them to binary `.ply`, previews them with a browser splat viewer, and exposes a download link.

## Local development

### 1. Install dependencies

```bash
bun install
```

### 2. Start the app

```bash
bun dev
```

Open the local Vite URL and upload an image.

## Exporting the SHARP model to ONNX

You need a local clone of Apple SHARP and a Python environment with its dependencies (plus ONNX export support).

### 1. Clone upstream SHARP (reference code)

```bash
git clone https://github.com/apple/ml-sharp /tmp/ml-sharp-upstream
```

### 2. Prepare a Python env for SHARP + ONNX export

Use the upstream SHARP setup instructions first, then make sure PyTorch/ONNX export dependencies are installed. The exact package set can vary by platform.

### 3. Export the browser wrapper ONNX

```bash
python3 scripts/export_sharp_onnx.py \
  --sharp-repo /tmp/ml-sharp-upstream \
  --output public/models/sharp_web_wrapper.onnx
```

Optional:
- `--checkpoint /path/to/sharp_2572gikvuh.pt` to avoid auto-download
- `--device cuda` if export on GPU is more stable/faster in your environment
- `--opset 20` (default)

## Runtime caveats

- SHARP is a large model. Browser memory usage is substantial.
- ONNX Runtime Web operator support can vary by browser and backend (WebGPU vs WASM).
- The app defaults to filtering/capping splats (`opacityThreshold`, `maxGaussians`) to keep preview/export practical in-browser.
- Focal length estimation from EXIF is approximate when the image lacks 35mm-equivalent EXIF data. The UI exposes a manual focal length override because SHARP quality depends on it.

## Stack

- Bun
- React + TypeScript
- Vite
- ONNX Runtime Web
- `@mkkellogg/gaussian-splats-3d` for preview rendering

## Next steps (recommended)

- verify the ONNX export on a real SHARP checkpoint and pin the required Python/torch/onnx versions
- benchmark browser compatibility (Chrome/Edge/Safari) with WebGPU
- add an ONNX graph validation step and sample model metadata manifest
- optionally add a quantized/fp16 export path to reduce model size and memory pressure
