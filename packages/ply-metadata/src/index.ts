/**
 * Metadata baked into a SHARP PLY's header as standard PLY `comment` lines so
 * that compatible viewers (in particular `<sharp-splat>`) can restore the
 * creator's preferred default view without changing the binary body.
 */

export interface SharpViewerMeta {
  cameraPosition?: [number, number, number]
  cameraTarget?: [number, number, number]
  cameraUp?: [number, number, number]
  fov?: number
  bgColor?: string
  maxScreenSize?: number
  autoRotate?: boolean
}

export interface PlyHeaderInfo {
  headerText: string
  bodyOffset: number
}

const COMMENT_PREFIX = 'comment sharp-viewer/'
const META_VERSION = 1
const HEADER_MAX_BYTES = 1 << 16
const END_HEADER = 'end_header\n'

const END_HEADER_BYTES = new TextEncoder().encode(END_HEADER)

export function parseHeader(buf: Uint8Array): PlyHeaderInfo | null {
  const limit = Math.min(buf.length, HEADER_MAX_BYTES)
  for (let i = 0; i + END_HEADER_BYTES.length <= limit; i++) {
    let match = true
    for (let j = 0; j < END_HEADER_BYTES.length; j++) {
      if (buf[i + j] !== END_HEADER_BYTES[j]) {
        match = false
        break
      }
    }
    if (match) {
      const bodyOffset = i + END_HEADER_BYTES.length
      const headerText = new TextDecoder('latin1').decode(buf.subarray(0, bodyOffset))
      return { headerText, bodyOffset }
    }
  }
  return null
}

export function readSharpViewerMeta(buf: Uint8Array): SharpViewerMeta | null {
  const info = parseHeader(buf)
  if (!info) return null

  const meta: SharpViewerMeta = {}
  let touched = false

  for (const line of info.headerText.split('\n')) {
    if (!line.startsWith(COMMENT_PREFIX)) continue
    const rest = line.slice(COMMENT_PREFIX.length).trim()
    const space = rest.indexOf(' ')
    if (space < 0) continue
    const key = rest.slice(0, space)
    const value = rest.slice(space + 1).trim()

    switch (key) {
      case 'camera-position': {
        const t = parseTriple(value)
        if (t) {
          meta.cameraPosition = t
          touched = true
        }
        break
      }
      case 'camera-target': {
        const t = parseTriple(value)
        if (t) {
          meta.cameraTarget = t
          touched = true
        }
        break
      }
      case 'camera-up': {
        const t = parseTriple(value)
        if (t) {
          meta.cameraUp = t
          touched = true
        }
        break
      }
      case 'fov': {
        const n = Number(value)
        if (Number.isFinite(n) && n > 0 && n < 180) {
          meta.fov = n
          touched = true
        }
        break
      }
      case 'bg-color': {
        if (/^#[0-9a-fA-F]{6}$/.test(value)) {
          meta.bgColor = value.toLowerCase()
          touched = true
        }
        break
      }
      case 'max-screen-size': {
        const n = Number(value)
        if (Number.isFinite(n) && n > 0) {
          meta.maxScreenSize = Math.round(n)
          touched = true
        }
        break
      }
      case 'auto-rotate': {
        meta.autoRotate = value === '1' || value.toLowerCase() === 'true'
        touched = true
        break
      }
    }
  }

  return touched ? meta : null
}

export function writeSharpViewerMeta(buf: Uint8Array, meta: SharpViewerMeta): Uint8Array {
  const info = parseHeader(buf)
  if (!info) {
    throw new Error('Invalid PLY: end_header marker not found in first 64 KB')
  }

  const lines = info.headerText.split('\n')
  const filtered = lines.filter((line) => !line.startsWith(COMMENT_PREFIX))

  const commentLines: string[] = [`${COMMENT_PREFIX}version ${META_VERSION}`]
  if (meta.cameraPosition) commentLines.push(`${COMMENT_PREFIX}camera-position ${formatTriple(meta.cameraPosition)}`)
  if (meta.cameraTarget) commentLines.push(`${COMMENT_PREFIX}camera-target ${formatTriple(meta.cameraTarget)}`)
  if (meta.cameraUp) commentLines.push(`${COMMENT_PREFIX}camera-up ${formatTriple(meta.cameraUp)}`)
  if (meta.fov !== undefined) commentLines.push(`${COMMENT_PREFIX}fov ${formatNumber(meta.fov)}`)
  if (meta.bgColor) commentLines.push(`${COMMENT_PREFIX}bg-color ${meta.bgColor}`)
  if (meta.maxScreenSize !== undefined)
    commentLines.push(`${COMMENT_PREFIX}max-screen-size ${Math.round(meta.maxScreenSize)}`)
  if (meta.autoRotate !== undefined) commentLines.push(`${COMMENT_PREFIX}auto-rotate ${meta.autoRotate ? 1 : 0}`)

  const formatIdx = filtered.findIndex((l) => l.startsWith('format '))
  if (formatIdx < 0) {
    throw new Error('Invalid PLY: format line not found in header')
  }

  const newHeaderLines = [
    ...filtered.slice(0, formatIdx + 1),
    ...commentLines,
    ...filtered.slice(formatIdx + 1),
  ]
  const newHeaderText = newHeaderLines.join('\n')

  const headerBytes = new TextEncoder().encode(newHeaderText)
  const body = buf.subarray(info.bodyOffset)
  const out = new Uint8Array(headerBytes.length + body.length)
  out.set(headerBytes, 0)
  out.set(body, headerBytes.length)
  return out
}

function parseTriple(value: string): [number, number, number] | undefined {
  const parts = value.split(/\s+/).map(Number)
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return undefined
  return [parts[0], parts[1], parts[2]]
}

function formatTriple([x, y, z]: [number, number, number]): string {
  return `${formatNumber(x)} ${formatNumber(y)} ${formatNumber(z)}`
}

function formatNumber(n: number): string {
  return Number.parseFloat(n.toPrecision(6)).toString()
}
