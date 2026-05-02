/**
 * `<sharp-splat>` Web Component — drop-in viewer for SHARP-generated Gaussian
 * splat .ply files. Reads any baked-in default camera and render settings from
 * the file's PLY header (see ./plyMetadata) and lets HTML attributes override
 * them.
 *
 * Usage:
 *   <script type="module"
 *     src="https://cdn.jsdelivr.net/npm/@bring-shrubbery/sharp-splat-viewer/dist/sharp-splat-viewer.iife.js">
 *   </script>
 *   <sharp-splat src="my.ply" style="width:600px;height:400px"></sharp-splat>
 */

import { readSharpViewerMeta } from '@ml-sharp-web/ply-metadata'

interface SharpViewerMeta {
  cameraPosition?: [number, number, number]
  cameraTarget?: [number, number, number]
  cameraUp?: [number, number, number]
  fov?: number
  bgColor?: string
  maxScreenSize?: number
  autoRotate?: boolean
}

const OBSERVED_ATTRS = [
  'src',
  'camera-position',
  'camera-target',
  'camera-up',
  'bg-color',
  'fov',
  'max-screen-size',
  'auto-rotate',
] as const

interface ResolvedSettings {
  cameraPosition?: [number, number, number]
  cameraTarget?: [number, number, number]
  cameraUp?: [number, number, number]
  bgColor: string
  fov: number
  maxScreenSize: number
  autoRotate: boolean
}

const DEFAULTS = {
  bgColor: '#101014',
  fov: 60,
  maxScreenSize: 2048,
  autoRotate: false,
} as const

const RECREATE_ATTRS = new Set<string>(['src', 'max-screen-size', 'camera-position', 'camera-target', 'camera-up'])

interface ViewerLike {
  start: () => void
  dispose: () => Promise<void>
  addSplatScene: (url: string, options: Record<string, unknown>) => Promise<unknown>
  camera?: { fov?: number; updateProjectionMatrix?: () => void }
  renderer?: { setClearColor?: (c: string) => void }
  controls?: { autoRotate?: boolean; autoRotateSpeed?: number }
}

interface SplatLib {
  Viewer: new (options: Record<string, unknown>) => ViewerLike
  SceneFormat: { Ply: unknown }
}

let libPromise: Promise<SplatLib> | null = null
function loadSplatLib(): Promise<SplatLib> {
  if (!libPromise) {
    libPromise = import('@mkkellogg/gaussian-splats-3d') as unknown as Promise<SplatLib>
  }
  return libPromise
}

export class SharpSplatElement extends HTMLElement {
  static get observedAttributes(): readonly string[] {
    return OBSERVED_ATTRS
  }

  private hostEl: HTMLDivElement
  private statusEl: HTMLDivElement
  private viewer: ViewerLike | null = null
  private currentSrc: string | null = null
  private currentBytes: Uint8Array | null = null
  private inflightToken = 0

  constructor() {
    super()
    const root = this.attachShadow({ mode: 'open' })
    const style = document.createElement('style')
    style.textContent = `
      :host { display: block; position: relative; width: 100%; height: 100%; min-height: 240px; background: #101014; }
      .host { position: absolute; inset: 0; overflow: hidden; }
      .host canvas { display: block; width: 100% !important; height: 100% !important; }
      .status { position: absolute; inset: 0; display: grid; place-items: center; padding: 1rem; color: rgba(255,255,255,0.78); font: 500 0.85rem/1.4 system-ui, sans-serif; text-align: center; pointer-events: none; }
    `
    this.hostEl = document.createElement('div')
    this.hostEl.className = 'host'
    this.statusEl = document.createElement('div')
    this.statusEl.className = 'status'
    this.statusEl.textContent = 'Loading splat…'
    root.append(style, this.hostEl, this.statusEl)
  }

  connectedCallback(): void {
    void this.refresh()
  }

  disconnectedCallback(): void {
    this.dispose()
    this.currentSrc = null
    this.currentBytes = null
  }

  attributeChangedCallback(name: string): void {
    if (RECREATE_ATTRS.has(name)) {
      void this.refresh()
      return
    }
    if (this.viewer) {
      this.applyLiveSettings()
    }
  }

  private async refresh(): Promise<void> {
    const token = ++this.inflightToken
    this.dispose()
    const src = this.getAttribute('src')
    if (!src) {
      this.setStatus('No src')
      return
    }

    if (src !== this.currentSrc || !this.currentBytes) {
      this.setStatus('Loading splat…')
      try {
        const res = await fetch(src)
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
        const buffer = await res.arrayBuffer()
        if (token !== this.inflightToken) return
        this.currentBytes = new Uint8Array(buffer)
        this.currentSrc = src
      } catch (err) {
        if (token !== this.inflightToken) return
        this.setStatus(`Failed to fetch: ${(err as Error).message}`)
        return
      }
    }

    const bytes = this.currentBytes
    if (!bytes) return
    const settings = this.resolveSettings(bytes)

    let lib: SplatLib
    try {
      lib = await loadSplatLib()
    } catch (err) {
      if (token !== this.inflightToken) return
      this.setStatus(`Failed to load viewer: ${(err as Error).message}`)
      return
    }
    if (token !== this.inflightToken) return

    const blob = new Blob([bytes as BlobPart], { type: 'application/octet-stream' })
    const blobUrl = URL.createObjectURL(blob)

    const viewerOptions: Record<string, unknown> = {
      rootElement: this.hostEl,
      selfDrivenMode: true,
      useBuiltInControls: true,
      sharedMemoryForWorkers: false,
      ignoreDevicePixelRatio: false,
      maxScreenSpaceSplatSize: settings.maxScreenSize,
    }
    if (settings.cameraPosition) viewerOptions.initialCameraPosition = settings.cameraPosition
    if (settings.cameraTarget) viewerOptions.initialCameraLookAt = settings.cameraTarget
    if (settings.cameraUp) viewerOptions.cameraUp = settings.cameraUp

    let viewer: ViewerLike
    try {
      viewer = new lib.Viewer(viewerOptions)
      this.viewer = viewer
      viewer.start()
      await viewer.addSplatScene(blobUrl, {
        format: lib.SceneFormat.Ply,
        showLoadingUI: false,
        splatAlphaRemovalThreshold: 1,
      })
      if (token !== this.inflightToken) {
        void viewer.dispose().catch(() => undefined)
        return
      }
      this.applyLiveSettings()
      this.setStatus(null)
    } catch (err) {
      if (token === this.inflightToken) {
        this.setStatus(`Failed to load splat: ${(err as Error).message}`)
      }
    } finally {
      URL.revokeObjectURL(blobUrl)
    }
  }

  private applyLiveSettings(): void {
    const viewer = this.viewer
    const bytes = this.currentBytes
    if (!viewer || !bytes) return
    const settings = this.resolveSettings(bytes)
    if (viewer.camera && typeof viewer.camera.fov === 'number') {
      viewer.camera.fov = settings.fov
      viewer.camera.updateProjectionMatrix?.()
    }
    viewer.renderer?.setClearColor?.(settings.bgColor)
    if (viewer.controls) {
      viewer.controls.autoRotate = settings.autoRotate
      if (settings.autoRotate) viewer.controls.autoRotateSpeed = 1.0
    }
  }

  private resolveSettings(bytes: Uint8Array): ResolvedSettings {
    const fileMeta = readSharpViewerMeta(bytes) ?? {}
    const attrMeta = this.readAttributeMeta()
    const merged: SharpViewerMeta = { ...fileMeta, ...attrMeta }
    return {
      cameraPosition: merged.cameraPosition,
      cameraTarget: merged.cameraTarget,
      cameraUp: merged.cameraUp,
      bgColor: merged.bgColor ?? DEFAULTS.bgColor,
      fov: merged.fov ?? DEFAULTS.fov,
      maxScreenSize: merged.maxScreenSize ?? DEFAULTS.maxScreenSize,
      autoRotate: merged.autoRotate ?? DEFAULTS.autoRotate,
    }
  }

  private readAttributeMeta(): SharpViewerMeta {
    const out: SharpViewerMeta = {}
    const cp = parseTriple(this.getAttribute('camera-position'))
    if (cp) out.cameraPosition = cp
    const ct = parseTriple(this.getAttribute('camera-target'))
    if (ct) out.cameraTarget = ct
    const cu = parseTriple(this.getAttribute('camera-up'))
    if (cu) out.cameraUp = cu
    const bg = this.getAttribute('bg-color')
    if (bg && /^#[0-9a-fA-F]{6}$/.test(bg)) out.bgColor = bg.toLowerCase()
    const fovAttr = this.getAttribute('fov')
    if (fovAttr) {
      const n = Number(fovAttr)
      if (Number.isFinite(n) && n > 0 && n < 180) out.fov = n
    }
    const ms = this.getAttribute('max-screen-size')
    if (ms) {
      const n = Number(ms)
      if (Number.isFinite(n) && n > 0) out.maxScreenSize = Math.round(n)
    }
    if (this.hasAttribute('auto-rotate')) {
      const v = this.getAttribute('auto-rotate')
      out.autoRotate = v == null || v === '' || v === 'true' || v === '1'
    }
    return out
  }

  private setStatus(message: string | null): void {
    if (message == null) {
      this.statusEl.style.display = 'none'
      this.statusEl.textContent = ''
    } else {
      this.statusEl.style.display = ''
      this.statusEl.textContent = message
    }
  }

  private dispose(): void {
    if (this.viewer) {
      const v = this.viewer
      this.viewer = null
      void v.dispose().catch(() => undefined)
    }
    this.hostEl.textContent = ''
  }
}

function parseTriple(value: string | null): [number, number, number] | undefined {
  if (!value) return undefined
  const parts = value
    .split(/[\s,]+/)
    .filter((s) => s.length > 0)
    .map(Number)
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return undefined
  return [parts[0], parts[1], parts[2]]
}

if (typeof customElements !== 'undefined' && !customElements.get('sharp-splat')) {
  customElements.define('sharp-splat', SharpSplatElement)
}

export type { SharpViewerMeta }
