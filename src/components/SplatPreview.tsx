import { useEffect, useRef, useState } from 'react'

import type { Viewer } from '@mkkellogg/gaussian-splats-3d'

interface SplatPreviewProps {
  plyUrl: string | null
  generationKey: number
}

export function SplatPreview({ plyUrl, generationKey }: SplatPreviewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<Viewer | null>(null)
  const [viewerStatus, setViewerStatus] = useState<string>('Waiting for a generated splat…')

  useEffect(() => {
    let cancelled = false
    let localViewer: Viewer | null = null

    const mount = async () => {
      if (!plyUrl || !containerRef.current) {
        setViewerStatus('Waiting for a generated splat…')
        return
      }

      setViewerStatus('Loading splat preview…')

      const module = await import('@mkkellogg/gaussian-splats-3d')
      if (cancelled || !containerRef.current) {
        return
      }

      localViewer = new module.Viewer({
        rootElement: containerRef.current,
        selfDrivenMode: true,
        useBuiltInControls: true,
        sharedMemoryForWorkers: false,
        ignoreDevicePixelRatio: false,
        inMemoryCompressionLevel: 0,
      })
      viewerRef.current = localViewer

      try {
        localViewer.start()
        await localViewer.addSplatScene(plyUrl, {
          showLoadingUI: false,
          splatAlphaRemovalThreshold: 1,
        })
        if (!cancelled) {
          setViewerStatus('Preview ready. Drag to orbit, scroll to zoom.')
        }
      } catch (error) {
        if (!cancelled) {
          setViewerStatus(`Preview failed: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    }

    void mount()

    return () => {
      cancelled = true
      const viewer = localViewer ?? viewerRef.current
      viewerRef.current = null
      if (viewer) {
        void viewer.dispose().catch(() => undefined)
      }
    }
  }, [plyUrl, generationKey])

  return (
    <div className="preview-panel">
      <div className="preview-toolbar">
        <span className="preview-title">Gaussian Splat Preview</span>
        <span className="preview-status">{viewerStatus}</span>
      </div>
      <div className="splat-canvas-shell">
        <div ref={containerRef} className="splat-canvas-host" />
        {!plyUrl ? <div className="splat-empty">Generate a splat to preview it here.</div> : null}
      </div>
    </div>
  )
}
