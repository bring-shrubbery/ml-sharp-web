import { startTransition, useEffect, useRef, useState } from 'react'

import './App.css'
import { SplatPreview } from './components/SplatPreview'
import { estimateFocalLengthFromFile, type FocalEstimate } from './lib/focal'
import { imageFileToSharpTensor, readImageInfo } from './lib/image'
import { DEFAULT_MAX_GAUSSIANS, DEFAULT_OPACITY_THRESHOLD, DEFAULT_WEB_MODEL_URL } from './lib/sharpConstants'
import { SharpWorkerClient } from './lib/sharpWorkerClient'
import type { WorkerStatusMessage } from './workers/messages'

interface SelectedImage {
  file: File
  previewUrl: string
  width: number
  height: number
  focalEstimate: FocalEstimate
}

interface GenerationResult {
  plyUrl: string
  downloadName: string
  selectedGaussians: number
  totalGaussians: number
  fileSizeBytes: number
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

function sourceLabel(source: FocalEstimate['source']): string {
  switch (source) {
    case 'exif-35mm':
      return 'EXIF 35mm-equivalent'
    case 'exif-mm-approx':
      return 'EXIF mm (approx. normalized to 35mm)'
    case 'default-30mm':
      return 'Default 30mm estimate'
    default:
      return source
  }
}

function toOutputName(fileName: string): string {
  const dot = fileName.lastIndexOf('.')
  const stem = dot > 0 ? fileName.slice(0, dot) : fileName
  return `${stem}.ply`
}

function App() {
  const workerRef = useRef<SharpWorkerClient | null>(null)
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null)
  const [manualFocalPx, setManualFocalPx] = useState<number | null>(null)

  const [modelUrlInput, setModelUrlInput] = useState(DEFAULT_WEB_MODEL_URL)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [modelFileUrl, setModelFileUrl] = useState<string | null>(null)

  const [opacityThreshold, setOpacityThreshold] = useState(DEFAULT_OPACITY_THRESHOLD)
  const [maxGaussians, setMaxGaussians] = useState(DEFAULT_MAX_GAUSSIANS)

  const [isBusy, setIsBusy] = useState(false)
  const [statusText, setStatusText] = useState<string>('Upload an image to begin.')
  const [workerStage, setWorkerStage] = useState<WorkerStatusMessage['stage']>('idle')
  const [errorText, setErrorText] = useState<string | null>(null)

  const [result, setResult] = useState<GenerationResult | null>(null)
  const [generationKey, setGenerationKey] = useState(0)

  useEffect(() => {
    const worker = new SharpWorkerClient((message) => {
      setWorkerStage(message.stage)
      setStatusText(message.message)
    })
    workerRef.current = worker

    return () => {
      workerRef.current = null
      worker.dispose()
    }
  }, [])

  useEffect(() => {
    if (!modelFile) {
      setModelFileUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous)
        return null
      })
      return
    }

    const url = URL.createObjectURL(modelFile)
    setModelFileUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous)
      return url
    })

    return () => {
      URL.revokeObjectURL(url)
    }
  }, [modelFile])

  useEffect(() => {
    return () => {
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage.previewUrl)
      }
      if (result) {
        URL.revokeObjectURL(result.plyUrl)
      }
    }
  }, [selectedImage, result])

  const effectiveModelUrl = modelFileUrl ?? modelUrlInput.trim()
  const focalPx = manualFocalPx ?? selectedImage?.focalEstimate.focalPx ?? 0

  const canGenerate = Boolean(selectedImage && effectiveModelUrl && focalPx > 0 && !isBusy)

  const resultRatio =
    result && result.totalGaussians > 0 ? (100 * result.selectedGaussians) / result.totalGaussians : 0
  const resultSummary = result
    ? `${result.selectedGaussians.toLocaleString()} / ${result.totalGaussians.toLocaleString()} gaussians (${resultRatio.toFixed(1)}%) • ${formatBytes(result.fileSizeBytes)}`
    : null

  const handleImageSelection = async (file: File | null) => {
    if (!file) {
      setSelectedImage((previous) => {
        if (previous) URL.revokeObjectURL(previous.previewUrl)
        return null
      })
      setManualFocalPx(null)
      setResult((previous) => {
        if (previous) URL.revokeObjectURL(previous.plyUrl)
        return null
      })
      setStatusText('Upload an image to begin.')
      setErrorText(null)
      return
    }

    setErrorText(null)
    setStatusText('Reading image metadata…')

    let previewUrl: string | null = null

    try {
      previewUrl = URL.createObjectURL(file)
      const info = await readImageInfo(file)
      const focalEstimate = await estimateFocalLengthFromFile(file, info.width, info.height)

      setSelectedImage((previous) => {
        if (previous) URL.revokeObjectURL(previous.previewUrl)
        const nextPreviewUrl = previewUrl as string
        return {
          file,
          previewUrl: nextPreviewUrl,
          width: info.width,
          height: info.height,
          focalEstimate,
        }
      })
      setManualFocalPx(focalEstimate.focalPx)
      setStatusText('Image ready. Configure settings and generate the splat.')
      setResult((previous) => {
        if (previous) URL.revokeObjectURL(previous.plyUrl)
        return null
      })
      setGenerationKey((key) => key + 1)
    } catch (error) {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
      setErrorText(error instanceof Error ? error.message : String(error))
      setStatusText('Failed to read image.')
    }
  }

  const runGeneration = async () => {
    if (!selectedImage || !workerRef.current) {
      return
    }

    if (!effectiveModelUrl) {
      setErrorText('Provide an ONNX model URL or upload an ONNX predictor file.')
      return
    }

    if (!Number.isFinite(focalPx) || focalPx <= 0) {
      setErrorText('Focal length must be a positive number.')
      return
    }

    setErrorText(null)
    setIsBusy(true)
    setStatusText('Preparing image tensor…')

    const startTime = performance.now()

    try {
      const { tensor, width, height } = await imageFileToSharpTensor(selectedImage.file)

      await workerRef.current.loadModel({ modelUrl: effectiveModelUrl })
      const inference = await workerRef.current.runInference({
        modelUrl: effectiveModelUrl,
        imageTensor: tensor.buffer,
        imageWidth: width,
        imageHeight: height,
        focalPx,
        disparityFactor: focalPx / width,
        opacityThreshold,
        maxGaussians,
      })

      const blob = new Blob([inference.plyBuffer as ArrayBuffer], {
        type: 'application/octet-stream',
      })
      const plyUrl = URL.createObjectURL(blob)
      const elapsedMs = performance.now() - startTime

      startTransition(() => {
        setResult((previous) => {
          if (previous) URL.revokeObjectURL(previous.plyUrl)
          return {
            plyUrl,
            downloadName: inference.outputName ?? toOutputName(selectedImage.file.name),
            selectedGaussians: inference.selectedGaussians,
            totalGaussians: inference.totalGaussians,
            fileSizeBytes: blob.size,
          }
        })
        setGenerationKey((key) => key + 1)
        setStatusText(`Done in ${(elapsedMs / 1000).toFixed(2)}s. Preview and download are ready.`)
        setWorkerStage('idle')
      })
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error))
      setStatusText('Generation failed.')
      setWorkerStage('idle')
    } finally {
      setIsBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">SHARP in the Browser (experimental)</p>
          <h1>Single-image to Gaussian splats, fully client-side</h1>
          <p className="hero-copy">
            Upload an image, run an exported SHARP ONNX predictor in the browser, preview the generated splat,
            and download a `.ply` file. Everything runs in the browser.
          </p>
        </div>
        <div className="license-callout">
          <strong>License reminder</strong>
          <p>
            Apple&apos;s released SHARP model weights are licensed for research purposes only. This UI is open
            source, but your model/weights usage must comply with the upstream `LICENSE_MODEL`.
          </p>
        </div>
      </header>

      <main className="grid">
        <section className="panel controls-panel">
          <h2>Inputs</h2>

          <label className="field">
            <span>Image</span>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                const file = event.currentTarget.files?.[0] ?? null
                void handleImageSelection(file)
              }}
            />
          </label>

          <label className="field">
            <span>Model URL (ONNX predictor)</span>
            <input
              type="url"
              value={modelUrlInput}
              onChange={(event) => setModelUrlInput(event.currentTarget.value)}
              placeholder="/models/sharp_web_predictor.onnx"
              disabled={Boolean(modelFile)}
            />
            <small>Used when no uploaded ONNX file is selected.</small>
          </label>

          <label className="field">
            <span>OR upload ONNX file</span>
            <input
              type="file"
              accept=".onnx,application/octet-stream"
              onChange={(event) => setModelFile(event.currentTarget.files?.[0] ?? null)}
            />
            <small>
              {modelFile
                ? `Using uploaded model: ${modelFile.name}`
                : 'Optional. Note: SHARP exports usually include a companion `.onnx.data` file, so URL mode is the reliable option.'}
            </small>
          </label>

          <div className="field-grid two-col">
            <label className="field compact">
              <span>Opacity threshold</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={opacityThreshold}
                onChange={(event) => setOpacityThreshold(Number(event.currentTarget.value))}
              />
              <small>Prunes low-alpha splats before preview/export.</small>
            </label>

            <label className="field compact">
              <span>Max gaussians</span>
              <input
                type="number"
                min={1000}
                step={1000}
                value={maxGaussians}
                onChange={(event) => setMaxGaussians(Math.max(1000, Math.floor(Number(event.currentTarget.value) || 1000)))}
              />
              <small>Caps output for browser memory/perf.</small>
            </label>
          </div>

          <div className="field-grid two-col">
            <label className="field compact">
              <span>Focal length (px)</span>
              <input
                type="number"
                min={1}
                step={1}
                value={Number.isFinite(focalPx) ? Math.round(focalPx) : ''}
                onChange={(event) => {
                  const next = Number(event.currentTarget.value)
                  setManualFocalPx(Number.isFinite(next) && next > 0 ? next : null)
                }}
                disabled={!selectedImage}
              />
            </label>

            <div className="field compact">
              <span>Focal source</span>
              <div className="meta-card">
                {selectedImage ? sourceLabel(selectedImage.focalEstimate.source) : 'No image selected'}
              </div>
              <small>SHARP quality depends heavily on focal accuracy.</small>
            </div>
          </div>

          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={() => void runGeneration()} disabled={!canGenerate}>
              {isBusy ? 'Generating…' : 'Generate Splat'}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => selectedImage && setManualFocalPx(selectedImage.focalEstimate.focalPx)}
              disabled={!selectedImage || isBusy}
            >
              Reset Focal to EXIF/Default
            </button>
            <a
              className={`btn ${result ? '' : 'btn-disabled'}`}
              href={result?.plyUrl ?? undefined}
              download={result?.downloadName ?? 'sharp-output.ply'}
              aria-disabled={!result}
              onClick={(event) => {
                if (!result) event.preventDefault()
              }}
            >
              Download `.ply`
            </a>
          </div>

          <div className="status-card" data-stage={workerStage}>
            <div className="status-row">
              <span className="status-dot" />
              <span>{statusText}</span>
            </div>
            {errorText ? <p className="error-text">{errorText}</p> : null}
            {resultSummary ? <p className="result-text">{resultSummary}</p> : null}
          </div>
        </section>

        <section className="panel image-panel">
          <div className="panel-header">
            <h2>Input Image</h2>
            {selectedImage ? (
              <span className="dim-label">
                {selectedImage.width} × {selectedImage.height}
              </span>
            ) : null}
          </div>
          <div className="image-frame">
            {selectedImage ? (
              <img src={selectedImage.previewUrl} alt="Selected input" />
            ) : (
              <div className="empty-state">Select an image to see the preview.</div>
            )}
          </div>
        </section>

        <SplatPreview plyUrl={result?.plyUrl ?? null} generationKey={generationKey} />
      </main>

      <footer className="footer-note">
        <p>
          This app expects an exported SHARP predictor ONNX with outputs `mean_vectors_ndc`,
          `singular_values_ndc`, `quaternions_ndc`, `colors`, and `opacities`. Use
          `scripts/export_sharp_onnx.py` to generate it from the upstream SHARP checkpoint. If a
          `.onnx.data` sidecar is produced, keep it next to the `.onnx` file under `/models/`.
        </p>
      </footer>
    </div>
  )
}

export default App
