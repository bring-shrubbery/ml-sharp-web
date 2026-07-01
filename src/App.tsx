import { startTransition, useCallback, useEffect, useRef, useState } from 'react'
import { DialRoot, useDialKitController } from 'dialkit'

import './App.css'
import { AholoSplatPreview } from './components/AholoSplatPreview'
import { SplatPreview, type CameraSnapshot } from './components/SplatPreview'
import { estimateFocalLengthFromFile, type FocalEstimate } from './lib/focal'
import { EMBED_SNIPPET } from './lib/embed'
import { imageFileToSharpTensor, readImageInfo } from './lib/image'
import { readSharpViewerMeta, writeSharpViewerMeta, type SharpViewerMeta } from '@ml-sharp-web/ply-metadata'
import { DEFAULT_MAX_GAUSSIANS, DEFAULT_OPACITY_THRESHOLD, DEFAULT_WEB_MODEL_URL } from './lib/sharpConstants'
import { SharpWorkerClient } from './lib/sharpWorkerClient'
import type { WorkerStatusMessage } from './workers/messages'

const DEFAULT_BG_COLOR = '#101014'
const DEFAULT_FOV = 60
const DEFAULT_AUTO_ROTATE = false
const DEFAULT_MAX_SCREEN_SIZE = 2048
const DEFAULT_SPLAT_POSITION: [number, number, number] = [0, 0, 0]
// SHARP's raw output is mirrored and back-to-front, so the model-correct
// orientation is: flip X, flip Y, then 180° about the vertical (Y) axis.
// Transforms compose scale-then-rotate, so this is flip [X,Y] + rotate Y 180°.
const DEFAULT_SPLAT_ROTATION: [number, number, number] = [0, 180, 0]
const DEFAULT_SPLAT_FLIP: [boolean, boolean, boolean] = [true, true, false]

type RendererChoice = 'mkkellogg' | 'aholo'

interface SelectedImage {
  file: File
  previewUrl: string
  width: number
  height: number
  focalEstimate: FocalEstimate
}

interface GenerationResult {
  previewPlyUrl: string
  downloadPlyUrl: string
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
  const imageInputRef = useRef<HTMLInputElement | null>(null)
  const modelInputRef = useRef<HTMLInputElement | null>(null)
  const actionsRef = useRef<Record<string, () => void>>({})

  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null)

  const [modelFile, setModelFile] = useState<File | null>(null)
  const [modelFileUrl, setModelFileUrl] = useState<string | null>(null)

  const [isBusy, setIsBusy] = useState(false)
  const [statusText, setStatusText] = useState<string>('Load the model to begin.')
  const [workerStage, setWorkerStage] = useState<WorkerStatusMessage['stage']>('idle')
  const [workerProgress, setWorkerProgress] = useState<number | undefined>(undefined)
  const [errorText, setErrorText] = useState<string | null>(null)

  const [result, setResult] = useState<GenerationResult | null>(null)
  const [plyBytes, setPlyBytes] = useState<Uint8Array | null>(null)
  const [generationKey, setGenerationKey] = useState(0)

  const [modelLoadState, setModelLoadState] = useState<'idle' | 'loading' | 'loaded' | 'error'>('idle')
  const [loadElapsedSec, setLoadElapsedSec] = useState(0)
  const loadedModelUrlRef = useRef<string | null>(null)

  const [bakedMeta, setBakedMeta] = useState<SharpViewerMeta | null>(null)
  const [saveStatus, setSaveStatus] = useState<string | undefined>(undefined)
  const cameraSnapshotRef = useRef<CameraSnapshot | null>(null)

  // Position and rotation are no longer user-editable — they hold the fixed
  // SHARP orientation (or whatever a loaded PLY baked in). Flip stays editable,
  // toggled by the Flip X/Y/Z action buttons.
  const [splatPosition, setSplatPosition] = useState<[number, number, number]>(DEFAULT_SPLAT_POSITION)
  const [splatRotation, setSplatRotation] = useState<[number, number, number]>(DEFAULT_SPLAT_ROTATION)
  const [splatFlip, setSplatFlip] = useState<[boolean, boolean, boolean]>(DEFAULT_SPLAT_FLIP)

  const runAction = useCallback((action: string) => {
    actionsRef.current[action]?.()
  }, [])

  const effectiveModelUrl = modelFileUrl ?? DEFAULT_WEB_MODEL_URL
  const modelLoaded = modelLoadState === 'loaded' && loadedModelUrlRef.current === effectiveModelUrl

  // Once the model is loaded, the load/upload buttons are replaced by a static
  // "loaded" indicator so it can't be loaded twice. Typed as a plain record so
  // the branch difference doesn't collapse the config's inferred value types.
  const modelSection: Record<string, { type: 'action'; label: string }> = modelLoaded
    ? {
        loaded: { type: 'action', label: '✓ Model loaded' },
        resetModel: { type: 'action', label: 'Reset model' },
      }
    : {
        loadModel: { type: 'action', label: 'Load model' },
        uploadModel: { type: 'action', label: 'Upload .onnx' },
        resetModel: { type: 'action', label: 'Reset model' },
      }

  // A single panel whose controls are grouped into labelled folders that follow
  // the workflow: Model → Image → Generation → Export → Viewer. The viewer
  // folder starts collapsed (`_collapsed`); DialKit only supports collapsing
  // nested folders, not sibling panels, which is why everything lives under one
  // controller. Folder nesting means control/action paths are dot-prefixed
  // (e.g. `image.focal`, `generation.generate`).
  const controls = useDialKitController(
    'SHARP',
    {
      model: modelSection,
      image: {
        uploadImage: { type: 'action', label: 'Upload image' },
        focal: [0, 0, 8000, 1],
        resetFocal: { type: 'action', label: 'Reset focal to EXIF' },
      },
      generation: {
        opacity: [DEFAULT_OPACITY_THRESHOLD, 0, 1, 0.01],
        maxGaussians: [DEFAULT_MAX_GAUSSIANS, 1000, 10_000_000, 1000],
        generate: { type: 'action', label: 'Generate splat' },
      },
      export: {
        download: { type: 'action', label: 'Download .ply' },
        copyEmbed: { type: 'action', label: 'Copy embed' },
      },
      viewer: {
        _collapsed: true,
        renderer: { type: 'select', options: ['mkkellogg', 'aholo'], default: 'mkkellogg' },
        bg: { type: 'color', default: DEFAULT_BG_COLOR },
        fov: [DEFAULT_FOV, 20, 120, 1],
        maxScreenSize: [DEFAULT_MAX_SCREEN_SIZE, 256, 4096, 1],
        autoRotate: DEFAULT_AUTO_ROTATE as boolean,
        flipX: { type: 'action', label: 'Flip X' },
        flipY: { type: 'action', label: 'Flip Y' },
        flipZ: { type: 'action', label: 'Flip Z' },
        resetFlip: { type: 'action', label: 'Reset flip' },
        saveDefaults: { type: 'action', label: 'Save as defaults' },
      },
    },
    { onAction: runAction },
  )

  const renderer = controls.values.viewer.renderer as RendererChoice
  const focalPx = controls.values.image.focal

  useEffect(() => {
    const worker = new SharpWorkerClient((message) => {
      setWorkerStage(message.stage)
      setStatusText(message.message)
      setWorkerProgress(message.progress)
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
      if (selectedImage) URL.revokeObjectURL(selectedImage.previewUrl)
      if (result) {
        URL.revokeObjectURL(result.previewPlyUrl)
        if (result.downloadPlyUrl !== result.previewPlyUrl) {
          URL.revokeObjectURL(result.downloadPlyUrl)
        }
      }
    }
  }, [selectedImage, result])

  useEffect(() => {
    if (loadedModelUrlRef.current && loadedModelUrlRef.current !== effectiveModelUrl) {
      loadedModelUrlRef.current = null
      setModelLoadState('idle')
      setStatusText('Model URL changed — load the model again to proceed.')
    }
  }, [effectiveModelUrl])

  useEffect(() => {
    if (modelLoadState !== 'loading') {
      setLoadElapsedSec(0)
      return
    }
    const startedAt = Date.now()
    setLoadElapsedSec(0)
    const interval = window.setInterval(() => {
      setLoadElapsedSec(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)
    return () => window.clearInterval(interval)
  }, [modelLoadState])

  // DialKit action buttons expose no per-button hook, so tag the primary
  // (Generate) and the "model loaded" indicator by label and re-tag whenever
  // DialKit re-renders the panel, letting CSS style them distinctly.
  useEffect(() => {
    const tag = () => {
      for (const el of document.querySelectorAll<HTMLButtonElement>('.dialkit-button')) {
        const text = el.textContent?.trim() ?? ''
        el.classList.toggle('dialkit-button--primary', text === 'Generate splat')
        el.classList.toggle('dialkit-button--ok', text.startsWith('✓'))
      }
    }
    tag()
    const observer = new MutationObserver(tag)
    observer.observe(document.body, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [])

  const handleLoadModel = async () => {
    if (!workerRef.current || !effectiveModelUrl || modelLoadState === 'loading') return
    setModelLoadState('loading')
    setErrorText(null)
    setStatusText('Starting model download…')
    setWorkerStage('loading-model')
    try {
      await workerRef.current.loadModel({ modelUrl: effectiveModelUrl })
      loadedModelUrlRef.current = effectiveModelUrl
      setModelLoadState('loaded')
      setStatusText('Model loaded. Upload an image and generate.')
      setWorkerStage('idle')
    } catch (error) {
      setModelLoadState('error')
      const message = error instanceof Error ? error.message : String(error)
      setErrorText(`Model load failed: ${message}`)
      setStatusText('Model load failed.')
      setWorkerStage('idle')
    }
  }

  const handleResetModel = () => {
    const message =
      modelLoadState === 'loading'
        ? 'Cancel the model load? The current download will be aborted.'
        : 'Reset the model? This will clear the loaded session and any uploaded file.'
    if (!window.confirm(message)) return
    workerRef.current?.reset()
    setModelFile(null)
    loadedModelUrlRef.current = null
    setModelLoadState('idle')
    setStatusText('Load the model to begin.')
    setWorkerStage('idle')
    setWorkerProgress(undefined)
    setErrorText(null)
  }

  const handleCameraChange = useCallback((snap: CameraSnapshot) => {
    cameraSnapshotRef.current = snap
  }, [])

  const handleImageSelection = async (file: File | null) => {
    if (!file) return
    setErrorText(null)
    setStatusText('Reading image metadata…')

    let previewUrl: string | null = null
    try {
      previewUrl = URL.createObjectURL(file)
      const info = await readImageInfo(file)
      const focalEstimate = await estimateFocalLengthFromFile(file, info.width, info.height)

      const nextImage: SelectedImage = {
        file,
        previewUrl: previewUrl as string,
        width: info.width,
        height: info.height,
        focalEstimate,
      }
      setSelectedImage((previous) => {
        if (previous) URL.revokeObjectURL(previous.previewUrl)
        return nextImage
      })
      controls.setValue('image.focal', focalEstimate.focalPx)
      setResult((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous.previewPlyUrl)
          if (previous.downloadPlyUrl !== previous.previewPlyUrl) URL.revokeObjectURL(previous.downloadPlyUrl)
        }
        return null
      })
      setGenerationKey((key) => key + 1)

      // Auto-generate as soon as an image is ready, provided the model is loaded.
      if (modelLoaded) {
        void runGeneration(nextImage, focalEstimate.focalPx)
      } else {
        setStatusText('Image ready. Load the model to generate.')
      }
    } catch (error) {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setErrorText(error instanceof Error ? error.message : String(error))
      setStatusText('Failed to read image.')
    }
  }

  const runGeneration = async (imageArg?: SelectedImage, focalArg?: number) => {
    const image = imageArg ?? selectedImage
    const focal = focalArg ?? focalPx
    if (!image) {
      setErrorText('Upload an image before generating.')
      return
    }
    if (!workerRef.current || !effectiveModelUrl) {
      setErrorText('No in-browser model source is available. Upload an ONNX predictor or load the hosted model.')
      return
    }
    if (!modelLoaded) {
      setErrorText('Load the model before generating.')
      return
    }
    if (!Number.isFinite(focal) || focal <= 0) {
      setErrorText('Focal length must be a positive number.')
      return
    }

    setErrorText(null)
    setIsBusy(true)
    setStatusText('Preparing image tensor…')
    const startTime = performance.now()

    try {
      const { tensor, width, height } = await imageFileToSharpTensor(image.file)

      await workerRef.current.loadModel({ modelUrl: effectiveModelUrl })
      const inference = await workerRef.current.runInference({
        modelUrl: effectiveModelUrl,
        imageTensor: tensor.buffer,
        imageWidth: width,
        imageHeight: height,
        focalPx: focal,
        disparityFactor: focal / width,
        opacityThreshold: controls.values.generation.opacity,
        maxGaussians: controls.values.generation.maxGaussians,
      })

      const bytes = new Uint8Array(inference.plyBuffer as ArrayBuffer)
      const blob = new Blob([bytes as BlobPart], { type: 'application/octet-stream' })
      const plyUrl = URL.createObjectURL(blob)
      const elapsedMs = performance.now() - startTime
      const meta = readSharpViewerMeta(bytes)

      controls.setValues({
        viewer: {
          bg: meta?.bgColor ?? DEFAULT_BG_COLOR,
          fov: meta?.fov ?? DEFAULT_FOV,
          maxScreenSize: meta?.maxScreenSize ?? DEFAULT_MAX_SCREEN_SIZE,
          autoRotate: meta?.autoRotate ?? DEFAULT_AUTO_ROTATE,
        },
      })

      startTransition(() => {
        setPlyBytes(bytes)
        setBakedMeta(meta)
        setSplatPosition(meta?.splatPosition ?? DEFAULT_SPLAT_POSITION)
        setSplatRotation(meta?.splatRotation ?? DEFAULT_SPLAT_ROTATION)
        setSplatFlip(meta?.splatFlip ?? DEFAULT_SPLAT_FLIP)
        cameraSnapshotRef.current = null
        setSaveStatus(undefined)
        setResult((previous) => {
          if (previous) {
            URL.revokeObjectURL(previous.previewPlyUrl)
            if (previous.downloadPlyUrl !== previous.previewPlyUrl) URL.revokeObjectURL(previous.downloadPlyUrl)
          }
          return {
            previewPlyUrl: plyUrl,
            downloadPlyUrl: plyUrl,
            downloadName: inference.outputName ?? toOutputName(image.file.name),
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

  const handleSaveDefaults = () => {
    if (!plyBytes) {
      setSaveStatus('Generate a splat first.')
      return
    }
    const camera = cameraSnapshotRef.current
    const meta: SharpViewerMeta = {
      ...(camera ? { cameraPosition: camera.position, cameraTarget: camera.target, cameraUp: camera.up } : {}),
      bgColor: controls.values.viewer.bg,
      fov: controls.values.viewer.fov,
      autoRotate: controls.values.viewer.autoRotate,
      maxScreenSize: controls.values.viewer.maxScreenSize,
      ...(splatPosition.some((n) => n !== 0) ? { splatPosition } : {}),
      ...(splatRotation.some((n) => n !== 0) ? { splatRotation } : {}),
      ...(splatFlip.some(Boolean) ? { splatFlip } : {}),
    }
    let nextBytes: Uint8Array
    try {
      nextBytes = writeSharpViewerMeta(plyBytes, meta)
    } catch (error) {
      setSaveStatus(`Save failed: ${error instanceof Error ? error.message : String(error)}`)
      return
    }
    setPlyBytes(nextBytes)
    setBakedMeta(meta)
    const blob = new Blob([nextBytes as BlobPart], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    setResult((previous) => {
      if (!previous) return previous
      if (previous.downloadPlyUrl !== previous.previewPlyUrl) URL.revokeObjectURL(previous.downloadPlyUrl)
      return { ...previous, downloadPlyUrl: url, fileSizeBytes: blob.size }
    })
    setSaveStatus('Baked into download')
    window.setTimeout(() => setSaveStatus(undefined), 2000)
  }

  const handleResetTransform = () => {
    setSplatPosition(DEFAULT_SPLAT_POSITION)
    setSplatRotation(DEFAULT_SPLAT_ROTATION)
    setSplatFlip(DEFAULT_SPLAT_FLIP)
  }

  const triggerDownload = () => {
    if (!result) {
      setErrorText('Generate a splat before downloading.')
      return
    }
    const anchor = document.createElement('a')
    anchor.href = result.downloadPlyUrl
    anchor.download = result.downloadName
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
  }

  const copyEmbed = async () => {
    try {
      await navigator.clipboard.writeText(EMBED_SNIPPET)
      setSaveStatus('Embed snippet copied')
      window.setTimeout(() => setSaveStatus(undefined), 2000)
    } catch {
      setSaveStatus('Copy failed — clipboard blocked')
    }
  }

  // Reassigned each render so the DialKit action dispatcher always calls the
  // latest handlers (which close over current state).
  actionsRef.current = {
    'model.loadModel': () => void handleLoadModel(),
    'model.uploadModel': () => modelInputRef.current?.click(),
    'model.resetModel': handleResetModel,
    'image.uploadImage': () => imageInputRef.current?.click(),
    'image.resetFocal': () => selectedImage && controls.setValue('image.focal', selectedImage.focalEstimate.focalPx),
    'generation.generate': () => void runGeneration(),
    'export.download': triggerDownload,
    'export.copyEmbed': () => void copyEmbed(),
    'viewer.flipX': () => setSplatFlip((f) => [!f[0], f[1], f[2]]),
    'viewer.flipY': () => setSplatFlip((f) => [f[0], !f[1], f[2]]),
    'viewer.flipZ': () => setSplatFlip((f) => [f[0], f[1], !f[2]]),
    'viewer.resetFlip': handleResetTransform,
    'viewer.saveDefaults': handleSaveDefaults,
  }

  const resultRatio =
    result && result.totalGaussians > 0 ? (100 * result.selectedGaussians) / result.totalGaussians : 0
  const resultSummary = result
    ? `${result.selectedGaussians.toLocaleString()} / ${result.totalGaussians.toLocaleString()} gaussians (${resultRatio.toFixed(1)}%) • ${formatBytes(result.fileSizeBytes)}`
    : null

  const previewProps = {
    plyUrl: result?.previewPlyUrl ?? null,
    generationKey,
    initialCameraPosition: bakedMeta?.cameraPosition,
    initialCameraTarget: bakedMeta?.cameraTarget,
    initialCameraUp: bakedMeta?.cameraUp,
    bgColor: controls.values.viewer.bg,
    fov: controls.values.viewer.fov,
    autoRotate: controls.values.viewer.autoRotate,
    splatPosition,
    splatRotation,
    splatFlip,
    onCameraChange: handleCameraChange,
  }

  const modelLoading = modelLoadState === 'loading'
  const showProgress = workerProgress !== undefined && workerProgress < 1

  return (
    <div className="stage">
      {renderer === 'aholo' ? (
        <AholoSplatPreview {...previewProps} />
      ) : (
        <SplatPreview {...previewProps} maxScreenSize={controls.values.viewer.maxScreenSize} />
      )}

      <input
        ref={imageInputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => {
          const file = event.currentTarget.files?.[0] ?? null
          event.currentTarget.value = ''
          void handleImageSelection(file)
        }}
      />
      <input
        ref={modelInputRef}
        type="file"
        accept=".onnx,application/octet-stream"
        hidden
        onChange={(event) => {
          setModelFile(event.currentTarget.files?.[0] ?? null)
          event.currentTarget.value = ''
        }}
      />

      <div className="overlays">
        <div className="status-overlay" data-stage={workerStage}>
          <div className="status-overlay-row">
            <span className={`status-dot${isBusy || modelLoading ? ' is-busy' : ''}`} />
            <span className="status-overlay-text">
              {modelLoading
                ? `Loading model ${String(Math.floor(loadElapsedSec / 60)).padStart(2, '0')}:${String(loadElapsedSec % 60).padStart(2, '0')}`
                : statusText}
            </span>
          </div>
          {showProgress ? (
            <div
              className="progress-bar"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round((workerProgress ?? 0) * 100)}
            >
              <div className="progress-bar-fill" style={{ width: `${((workerProgress ?? 0) * 100).toFixed(1)}%` }} />
            </div>
          ) : null}
          {errorText ? <p className="status-overlay-error">{errorText}</p> : null}
          {resultSummary ? <p className="status-overlay-summary">{resultSummary}</p> : null}
          {saveStatus ? <p className="status-overlay-save">{saveStatus}</p> : null}
        </div>

        {selectedImage ? (
          <figure className="thumb-overlay">
            <img src={selectedImage.previewUrl} alt="Input" />
            <figcaption>
              {selectedImage.width}×{selectedImage.height} · {sourceLabel(selectedImage.focalEstimate.source)}
            </figcaption>
          </figure>
        ) : null}

        <div className="links-overlay">
          <a href="https://x.com/bringshrubberyy" target="_blank" rel="noreferrer" aria-label="X (Twitter)" title="X (Twitter)">
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
            </svg>
          </a>
          <a href="https://github.com/bring-shrubbery/ml-sharp-web" target="_blank" rel="noreferrer" aria-label="GitHub repository" title="GitHub repository">
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.27-.01-1.16-.02-2.11-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.27-1.69-1.27-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.72-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.9-.39.99 0 1.98.13 2.9.39 2.21-1.49 3.18-1.18 3.18-1.18.62 1.59.23 2.76.11 3.05.74.81 1.18 1.84 1.18 3.1 0 4.42-2.7 5.39-5.27 5.68.41.36.78 1.06.78 2.14 0 1.55-.01 2.79-.01 3.17 0 .31.21.68.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.73 18.27.5 12 .5z" />
            </svg>
          </a>
          <a
            className="links-overlay-license"
            href="https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL"
            target="_blank"
            rel="noreferrer"
          >
            Model license
          </a>
        </div>
      </div>

      <DialRoot mode="popover" theme="dark" position="top-right" productionEnabled defaultOpen />
    </div>
  )
}

export default App
