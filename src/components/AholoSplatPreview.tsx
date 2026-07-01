import { useEffect, useRef, useState, type ReactNode } from 'react'

import type { PerspectiveCamera, Splat, SplatLoader, Viewer } from '@manycore/aholo-viewer'

import type { CameraSnapshot } from './SplatPreview'

// Aholo ships ~3 MB of runtime, so it's dynamically imported and only pulled in
// when this renderer is actually mounted.
type AholoModule = typeof import('@manycore/aholo-viewer')

interface AholoSplatPreviewProps {
  plyUrl: string | null
  generationKey: number
  initialCameraPosition?: [number, number, number]
  initialCameraTarget?: [number, number, number]
  initialCameraUp?: [number, number, number]
  bgColor: string
  fov: number
  autoRotate: boolean
  splatPosition: [number, number, number]
  splatRotation: [number, number, number]
  splatFlip: [boolean, boolean, boolean]
  onCameraChange?: (snap: CameraSnapshot) => void
  children?: ReactNode
}

// SHARP splats follow the OpenCV-style 3DGS convention, which Aholo renders
// with a -Y up vector (see aholojs.dev getting-started guide).
const DEFAULT_UP: V3 = [0, -1, 0]
const ORBIT_SPEED = 0.005
const ZOOM_SPEED = 0.0015
const AUTO_ROTATE_SPEED = 0.5 // radians per second
const MIN_POLAR = 0.05
const MAX_POLAR = Math.PI - 0.05

export function AholoSplatPreview({
  plyUrl,
  generationKey,
  initialCameraPosition,
  initialCameraTarget,
  initialCameraUp,
  bgColor,
  fov,
  autoRotate,
  splatPosition,
  splatRotation,
  splatFlip,
  onCameraChange,
  children,
}: AholoSplatPreviewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const aholoRef = useRef<AholoModule | null>(null)
  const viewerRef = useRef<Viewer | null>(null)
  const splatRef = useRef<Splat | null>(null)
  const [viewerStatus, setViewerStatus] = useState<string>('Waiting for a generated splat…')

  // Latest-value refs so the render loop and live effects read fresh props
  // without tearing down the viewer.
  const onCameraChangeRef = useRef(onCameraChange)
  onCameraChangeRef.current = onCameraChange
  const fovRef = useRef(fov)
  fovRef.current = fov
  const autoRotateRef = useRef(autoRotate)
  autoRotateRef.current = autoRotate
  const bgColorRef = useRef(bgColor)
  bgColorRef.current = bgColor

  const splatPositionRef = useRef(splatPosition)
  const splatRotationRef = useRef(splatRotation)
  const splatFlipRef = useRef(splatFlip)
  splatPositionRef.current = splatPosition
  splatRotationRef.current = splatRotation
  splatFlipRef.current = splatFlip

  // Orbit state, shared between pointer handlers and the render loop.
  const cameraRef = useRef<PerspectiveCamera | null>(null)
  const targetRef = useRef<V3>([0, 0, 0])
  const offsetRef = useRef<V3>([0, 0, 3])
  const upRef = useRef<V3>([...DEFAULT_UP])
  const cameraDirtyRef = useRef(true)

  useEffect(() => {
    let cancelled = false
    let viewer: Viewer | null = null
    let splat: Splat | null = null
    let rafId = 0
    const cleanups: Array<() => void> = []

    const mount = async () => {
      const container = containerRef.current
      if (!plyUrl || !container) {
        setViewerStatus('Waiting for a generated splat…')
        return
      }

      setViewerStatus('Loading splat preview…')

      let aholo: AholoModule
      let bytes: Uint8Array
      try {
        const [mod, response] = await Promise.all([import('@manycore/aholo-viewer'), fetch(plyUrl)])
        aholo = mod
        aholoRef.current = mod
        bytes = new Uint8Array(await response.arrayBuffer())
      } catch (error) {
        if (!cancelled) setViewerStatus(`Preview failed: ${errMessage(error)}`)
        return
      }
      if (cancelled || !containerRef.current) return

      try {
        viewer = aholo.createViewer('aholo-preview', container, { antialiasing: true, alpha: true })
        viewerRef.current = viewer

        const data = await aholo.SplatLoader.parseSplatData(
          aholo.SplatLoader.SplatFileType.PLY,
          bytes,
          aholo.SplatLoader.SplatPackType.Compressed,
        )
        if (cancelled) return

        // Frame the camera from the splat's point cloud before it's uploaded.
        const bounds = boundsFromSplatData(data)

        splat = await aholo.SplatUtils.createSplat(data)
        if (cancelled) {
          splat.freeGPU()
          return
        }
        splatRef.current = splat

        const camera = new aholo.PerspectiveCamera(fovRef.current, aspectOf(container), 0.05, 5000)
        cameraRef.current = camera

        initOrbitState(bounds, {
          position: initialCameraPosition,
          target: initialCameraTarget,
          up: initialCameraUp,
          fov: fovRef.current,
        })

        applySplatTransform(splat, {
          position: splatPositionRef.current,
          rotation: splatRotationRef.current,
          flip: splatFlipRef.current,
        })

        viewer.getScene().add(splat)
        viewer.setCamera(camera)
        const [r, g, b] = hexToRgb01(bgColorRef.current)
        aholo.setViewerConfig(viewer, {
          pipeline: {
            Background: {
              background: {
                active: aholo.BackgroundMode.BasicBackground,
                basic: { color: new aholo.Color(r, g, b) },
              },
              ground: { enabled: false },
            },
            Splatting: { enabled: true },
            TAA: { enabled: true },
          },
        })

        // Continuous render loop drives orbit, auto-rotate and splat sorting.
        let lastTs = performance.now()
        const frame = (ts: number) => {
          const dt = Math.min((ts - lastTs) / 1000, 0.1)
          lastTs = ts
          if (autoRotateRef.current) {
            offsetRef.current = rotateAroundAxis(offsetRef.current, normalize(upRef.current), AUTO_ROTATE_SPEED * dt)
            cameraDirtyRef.current = true
          }
          if (cameraDirtyRef.current) {
            applyCamera(camera)
            emitCamera()
            cameraDirtyRef.current = false
          }
          viewer?.render()
          rafId = requestAnimationFrame(frame)
        }
        rafId = requestAnimationFrame(frame)

        // Keep the canvas and projection in sync with the container size.
        const resize = () => {
          const el = containerRef.current
          if (!el || !viewer) return
          viewer.resize()
          camera.aspect = aspectOf(el)
          camera.updateProjectionMatrix()
          cameraDirtyRef.current = true
        }
        const ro = new ResizeObserver(resize)
        ro.observe(container)
        cleanups.push(() => ro.disconnect())

        attachOrbitControls(container, cleanups)

        setViewerStatus('Preview ready. Drag to orbit, scroll to zoom.')
      } catch (error) {
        if (!cancelled) setViewerStatus(`Preview failed: ${errMessage(error)}`)
      }
    }

    const emitCamera = () => {
      const cb = onCameraChangeRef.current
      if (!cb) return
      const target = targetRef.current
      const offset = offsetRef.current
      cb({
        position: [target[0] + offset[0], target[1] + offset[1], target[2] + offset[2]],
        target: [...target],
        up: [...upRef.current],
      })
    }

    const applyCamera = (camera: PerspectiveCamera) => {
      const aholo = aholoRef.current
      if (!aholo) return
      const target = targetRef.current
      const offset = offsetRef.current
      const up = upRef.current
      camera.up.set(up[0], up[1], up[2])
      camera.position.set(target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])
      camera.lookAt(new aholo.Vector3(target[0], target[1], target[2]))
    }

    const initOrbitState = (
      bounds: Bounds,
      init: {
        position?: [number, number, number]
        target?: [number, number, number]
        up?: [number, number, number]
        fov: number
      },
    ) => {
      if (init.position) {
        const target = init.target ?? [0, 0, 0]
        targetRef.current = [...target]
        offsetRef.current = [
          init.position[0] - target[0],
          init.position[1] - target[1],
          init.position[2] - target[2],
        ]
        upRef.current = init.up ? [...init.up] : [...DEFAULT_UP]
      } else {
        // Auto-frame: centre on the cloud and pull back to fit it in view.
        targetRef.current = [...bounds.center]
        upRef.current = [...DEFAULT_UP]
        const distance = Math.max(bounds.radius / Math.tan((init.fov * Math.PI) / 360), 0.1) * 1.4
        const dir = normalize([0.35, -0.35, -1])
        offsetRef.current = [dir[0] * distance, dir[1] * distance, dir[2] * distance]
      }
      cameraDirtyRef.current = true
    }

    const attachOrbitControls = (el: HTMLElement, sinks: Array<() => void>) => {
      let dragging = false
      let lastX = 0
      let lastY = 0

      const onPointerDown = (event: PointerEvent) => {
        dragging = true
        lastX = event.clientX
        lastY = event.clientY
        el.setPointerCapture(event.pointerId)
      }
      const onPointerMove = (event: PointerEvent) => {
        if (!dragging) return
        const dx = event.clientX - lastX
        const dy = event.clientY - lastY
        lastX = event.clientX
        lastY = event.clientY
        orbit(dx, dy)
      }
      const endDrag = (event: PointerEvent) => {
        if (!dragging) return
        dragging = false
        try {
          el.releasePointerCapture(event.pointerId)
        } catch {
          // pointer already released
        }
      }
      const onWheel = (event: WheelEvent) => {
        event.preventDefault()
        const scale = Math.exp(event.deltaY * ZOOM_SPEED)
        const offset = offsetRef.current
        const len = length(offset)
        const nextLen = Math.min(Math.max(len * scale, bounds01(len)), len * 100)
        const k = nextLen / (len || 1)
        offsetRef.current = [offset[0] * k, offset[1] * k, offset[2] * k]
        cameraDirtyRef.current = true
      }

      el.addEventListener('pointerdown', onPointerDown)
      el.addEventListener('pointermove', onPointerMove)
      el.addEventListener('pointerup', endDrag)
      el.addEventListener('pointercancel', endDrag)
      el.addEventListener('wheel', onWheel, { passive: false })
      sinks.push(() => {
        el.removeEventListener('pointerdown', onPointerDown)
        el.removeEventListener('pointermove', onPointerMove)
        el.removeEventListener('pointerup', endDrag)
        el.removeEventListener('pointercancel', endDrag)
        el.removeEventListener('wheel', onWheel)
      })
    }

    const orbit = (dx: number, dy: number) => {
      const up = normalize(upRef.current)
      let offset = offsetRef.current
      // Horizontal drag spins around the up axis.
      offset = rotateAroundAxis(offset, up, -dx * ORBIT_SPEED)
      // Vertical drag tilts around the camera-right axis, clamped near the poles.
      const viewDir = normalize([-offset[0], -offset[1], -offset[2]])
      const right = normalize(cross(viewDir, up))
      const tilted = rotateAroundAxis(offset, right, dy * ORBIT_SPEED)
      const polar = Math.acos(clamp(dot(normalize(tilted), up), -1, 1))
      offsetRef.current = polar > MIN_POLAR && polar < MAX_POLAR ? tilted : offset
      cameraDirtyRef.current = true
    }

    void mount()

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      for (const dispose of cleanups) dispose()
      cameraRef.current = null
      viewerRef.current = null
      splatRef.current = null
      aholoRef.current = null
      if (splat) {
        try {
          viewer?.getScene().remove(splat)
        } catch {
          // scene already torn down
        }
        splat.freeGPU()
      }
      viewer?.destroy()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plyUrl, generationKey])

  useEffect(() => {
    const camera = cameraRef.current
    if (!camera) return
    camera.fov = fov
    camera.updateProjectionMatrix()
    cameraDirtyRef.current = true
  }, [fov])

  useEffect(() => {
    const viewer = viewerRef.current
    const aholo = aholoRef.current
    if (!viewer || !aholo) return
    applyBackground(aholo, viewer, bgColor)
  }, [bgColor])

  useEffect(() => {
    const splat = splatRef.current
    if (!splat) return
    applySplatTransform(splat, { position: splatPosition, rotation: splatRotation, flip: splatFlip })
    cameraDirtyRef.current = true
  }, [splatPosition, splatRotation, splatFlip])

  return (
    <section className="panel preview-panel">
      <div className="preview-toolbar">
        <span className="preview-title">Gaussian Splat Preview · Aholo</span>
        <span className="preview-status">{viewerStatus}</span>
      </div>
      <div className="splat-canvas-shell">
        <div ref={containerRef} className="splat-canvas-host" />
        {!plyUrl ? <div className="splat-empty">Generate a splat to preview it here.</div> : null}
      </div>
      {children}
    </section>
  )
}

type V3 = [number, number, number]

interface Bounds {
  center: V3
  radius: number
}

function applyBackground(aholo: AholoModule, viewer: Viewer, hex: string): void {
  const [r, g, b] = hexToRgb01(hex)
  aholo.setViewerConfig(viewer, {
    pipeline: {
      Background: {
        background: {
          active: aholo.BackgroundMode.BasicBackground,
          basic: { color: new aholo.Color(r, g, b) },
        },
        ground: { enabled: false },
      },
    },
  })
}

interface SplatTransform {
  position: V3
  rotation: V3
  flip: [boolean, boolean, boolean]
}

function applySplatTransform(splat: Splat, transform: SplatTransform): void {
  splat.position.set(transform.position[0], transform.position[1], transform.position[2])
  splat.rotation.set(
    (transform.rotation[0] * Math.PI) / 180,
    (transform.rotation[1] * Math.PI) / 180,
    (transform.rotation[2] * Math.PI) / 180,
    'XYZ',
  )
  splat.scale.set(transform.flip[0] ? -1 : 1, transform.flip[1] ? -1 : 1, transform.flip[2] ? -1 : 1)
}

function boundsFromSplatData(data: SplatLoader.SplatData): Bounds {
  const count = data.counts
  if (!count) return { center: [0, 0, 0], radius: 1 }
  const centers = new Float32Array(count * 3)
  data.fillCenters(centers)
  let minX = Infinity
  let minY = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let maxZ = -Infinity
  for (let i = 0; i < count; i++) {
    const x = centers[i * 3]
    const y = centers[i * 3 + 1]
    const z = centers[i * 3 + 2]
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (z < minZ) minZ = z
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
    if (z > maxZ) maxZ = z
  }
  if (!Number.isFinite(minX)) return { center: [0, 0, 0], radius: 1 }
  const center: V3 = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2]
  const radius = Math.max(0.5 * length([maxX - minX, maxY - minY, maxZ - minZ]), 0.05)
  return { center, radius }
}

function aspectOf(el: HTMLElement): number {
  const rect = el.getBoundingClientRect()
  return rect.height > 0 ? rect.width / rect.height : 1
}

function hexToRgb01(hex: string): V3 {
  const normalized = hex.replace('#', '')
  const value =
    normalized.length === 3
      ? normalized
          .split('')
          .map((c) => c + c)
          .join('')
      : normalized.padEnd(6, '0')
  const int = parseInt(value.slice(0, 6), 16)
  if (Number.isNaN(int)) return [0, 0, 0]
  return [((int >> 16) & 255) / 255, ((int >> 8) & 255) / 255, (int & 255) / 255]
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

// --- tiny vec3 helpers (plain arrays to stay independent of the engine types) ---

function length(v: V3): number {
  return Math.hypot(v[0], v[1], v[2])
}

function normalize(v: V3): V3 {
  const len = length(v) || 1
  return [v[0] / len, v[1] / len, v[2] / len]
}

function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function cross(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi)
}

function bounds01(len: number): number {
  return Math.max(len * 0.02, 0.02)
}

// Rodrigues' rotation of v about a unit axis by angle radians.
function rotateAroundAxis(v: V3, axis: V3, angle: number): V3 {
  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  const [kx, ky, kz] = axis
  const d = kx * v[0] + ky * v[1] + kz * v[2]
  const crossV: V3 = [ky * v[2] - kz * v[1], kz * v[0] - kx * v[2], kx * v[1] - ky * v[0]]
  return [
    v[0] * cosA + crossV[0] * sinA + kx * d * (1 - cosA),
    v[1] * cosA + crossV[1] * sinA + ky * d * (1 - cosA),
    v[2] * cosA + crossV[2] * sinA + kz * d * (1 - cosA),
  ]
}
