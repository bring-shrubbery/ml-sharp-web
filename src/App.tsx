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

  const effectiveModelUrl = modelFileUrl ?? DEFAULT_WEB_MODEL_URL
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
      setErrorText('No in-browser model source is available. Upload an ONNX predictor file or configure the hosted model URL.')
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
        <nav className="top-links" aria-label="External links">
          <a
            className="top-link"
            href="https://x.com/bringshrubberyy"
            target="_blank"
            rel="noreferrer"
            aria-label="X (Twitter)"
            title="X (Twitter)"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
            </svg>
          </a>
          <a
            className="top-link"
            href="https://github.com/bring-shrubbery/ml-sharp-web"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            title="GitHub repository"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.27-.01-1.16-.02-2.11-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.27-1.69-1.27-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.72-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.9-.39.99 0 1.98.13 2.9.39 2.21-1.49 3.18-1.18 3.18-1.18.62 1.59.23 2.76.11 3.05.74.81 1.18 1.84 1.18 3.1 0 4.42-2.7 5.39-5.27 5.68.41.36.78 1.06.78 2.14 0 1.55-.01 2.79-.01 3.17 0 .31.21.68.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.73 18.27.5 12 .5z" />
            </svg>
          </a>
        </nav>
        <div>
          <p className="eyebrow">SHARP in the Browser (experimental)</p>
          <h1>Single-image to Gaussian splats, fully client-side</h1>
          <p className="hero-copy">
            Upload an image, run an exported SHARP ONNX predictor in the browser, preview the generated splat,
            and download a `.ply` file. Everything runs in the browser.
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
            <span>Optional: upload ONNX file</span>
            <input
              type="file"
              accept=".onnx,application/octet-stream"
              onChange={(event) => setModelFile(event.currentTarget.files?.[0] ?? null)}
            />
            <small>
              {modelFile
                ? `Using uploaded model: ${modelFile.name}`
                : 'Optional override for testing. The app uses the hosted model by default. Note: SHARP exports usually include a companion `.onnx.data` file, so uploaded `.onnx` files alone often will not work.'}
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
          License note: Apple&apos;s released SHARP model weights are subject to upstream research-use
          restrictions. Please review{' '}
          <a
            href="https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL"
            target="_blank"
            rel="noreferrer"
          >
            LICENSE_MODEL
          </a>{' '}
          (and{' '}
          <a href="https://github.com/apple/ml-sharp/blob/main/LICENSE" target="_blank" rel="noreferrer">
            LICENSE
          </a>
          ) before using the model files.
        </p>
      </footer>
    </div>
  )
}

export default App
