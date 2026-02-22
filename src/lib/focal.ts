import * as exifr from 'exifr'

import { DEFAULT_FOCAL_MM, FILM_35MM_DIAGONAL_MM } from './sharpConstants'

export type FocalSource = 'exif-35mm' | 'exif-mm-approx' | 'default-30mm'

export interface FocalEstimate {
  focalPx: number
  focalMmUsed: number
  source: FocalSource
  exif35mm?: number
  exifFocalMm?: number
}

export function convertFocalMmToPx(width: number, height: number, focalMm: number): number {
  return (focalMm * Math.hypot(width, height)) / FILM_35MM_DIAGONAL_MM
}

function parseFiniteNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  if (value && typeof value === 'object') {
    const maybeRational = value as { numerator?: unknown; denominator?: unknown }
    const numerator = typeof maybeRational.numerator === 'number' ? maybeRational.numerator : undefined
    const denominator = typeof maybeRational.denominator === 'number' ? maybeRational.denominator : undefined
    if (
      numerator !== undefined &&
      denominator !== undefined &&
      Number.isFinite(numerator) &&
      Number.isFinite(denominator) &&
      denominator !== 0
    ) {
      return numerator / denominator
    }
  }
  return undefined
}

export async function estimateFocalLengthFromFile(
  file: File,
  width: number,
  height: number,
): Promise<FocalEstimate> {
  let exifData: Record<string, unknown> | undefined

  try {
    exifData = (await exifr.parse(file, [
      'FocalLengthIn35mmFilm',
      'FocalLenIn35mmFilm',
      'FocalLength',
    ])) as Record<string, unknown> | undefined
  } catch {
    exifData = undefined
  }

  const focal35mm =
    parseFiniteNumber(exifData?.FocalLengthIn35mmFilm) ??
    parseFiniteNumber(exifData?.FocalLenIn35mmFilm)
  const focalMmExif = parseFiniteNumber(exifData?.FocalLength)

  if (focal35mm !== undefined && focal35mm >= 1) {
    return {
      focalPx: convertFocalMmToPx(width, height, focal35mm),
      focalMmUsed: focal35mm,
      source: 'exif-35mm',
      exif35mm: focal35mm,
      exifFocalMm: focalMmExif,
    }
  }

  if (focalMmExif !== undefined && focalMmExif > 0) {
    const normalizedFocalMm = focalMmExif < 10 ? focalMmExif * 8.4 : focalMmExif
    return {
      focalPx: convertFocalMmToPx(width, height, normalizedFocalMm),
      focalMmUsed: normalizedFocalMm,
      source: 'exif-mm-approx',
      exifFocalMm: focalMmExif,
    }
  }

  return {
    focalPx: convertFocalMmToPx(width, height, DEFAULT_FOCAL_MM),
    focalMmUsed: DEFAULT_FOCAL_MM,
    source: 'default-30mm',
  }
}
