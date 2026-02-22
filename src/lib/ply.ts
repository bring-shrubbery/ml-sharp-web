const SH_COEFF_DEG0 = Math.sqrt(1 / (4 * Math.PI))
const INV_SH_COEFF_DEG0 = 1 / SH_COEFF_DEG0
const COLOR_SPACE_SRGB_INDEX = 1

export interface GaussianBuffers {
  count: number
  meanVectors: Float32Array
  singularValues: Float32Array
  quaternions: Float32Array
  colors: Float32Array
  opacities: Float32Array
}

export interface SharpPlyBuildInput extends GaussianBuffers {
  imageWidth: number
  imageHeight: number
  focalPx: number
}

function clamp01(value: number): number {
  if (value <= 0) return 0
  if (value >= 1) return 1
  return value
}

function linearToSrgb(value: number): number {
  const x = clamp01(value)
  if (x <= 0.0031308) {
    return 12.92 * x
  }
  return 1.055 * x ** (1 / 2.4) - 0.055
}

function rgbToSh0(rgb: number): number {
  return (rgb - 0.5) * INV_SH_COEFF_DEG0
}

function safeLog(value: number, epsilon = 1e-8): number {
  return Math.log(Math.max(value, epsilon))
}

function safeLogit(value: number, epsilon = 1e-6): number {
  const x = Math.min(1 - epsilon, Math.max(epsilon, value))
  return Math.log(x / (1 - x))
}

function quantile(values: Float32Array, q: number): number {
  if (values.length === 0) {
    return 0
  }
  const sorted = Array.from(values)
  sorted.sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)))
  return sorted[index] ?? 0
}

function buildHeader(vertexCount: number): string {
  return [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${vertexCount}`,
    'property float x',
    'property float y',
    'property float z',
    'property float f_dc_0',
    'property float f_dc_1',
    'property float f_dc_2',
    'property float opacity',
    'property float scale_0',
    'property float scale_1',
    'property float scale_2',
    'property float rot_0',
    'property float rot_1',
    'property float rot_2',
    'property float rot_3',
    'element extrinsic 16',
    'property float extrinsic',
    'element intrinsic 9',
    'property float intrinsic',
    'element image_size 2',
    'property uint image_size',
    'element frame 2',
    'property int frame',
    'element disparity 2',
    'property float disparity',
    'element color_space 1',
    'property uchar color_space',
    'element version 3',
    'property uchar version',
    'end_header',
    '',
  ].join('\n')
}

export function buildSharpPlyBinary(input: SharpPlyBuildInput): Uint8Array {
  const {
    count,
    meanVectors,
    singularValues,
    quaternions,
    colors,
    opacities,
    focalPx,
    imageWidth,
    imageHeight,
  } = input

  if (meanVectors.length !== count * 3) {
    throw new Error('meanVectors length does not match count * 3')
  }
  if (singularValues.length !== count * 3) {
    throw new Error('singularValues length does not match count * 3')
  }
  if (quaternions.length !== count * 4) {
    throw new Error('quaternions length does not match count * 4')
  }
  if (colors.length !== count * 3) {
    throw new Error('colors length does not match count * 3')
  }
  if (opacities.length !== count) {
    throw new Error('opacities length does not match count')
  }

  const header = buildHeader(count)
  const encoder = new TextEncoder()
  const headerBytes = encoder.encode(header)

  const vertexStrideBytes = 14 * 4
  const vertexBytesLength = count * vertexStrideBytes
  const extrinsicBytesLength = 16 * 4
  const intrinsicBytesLength = 9 * 4
  const imageSizeBytesLength = 2 * 4
  const frameBytesLength = 2 * 4
  const disparityBytesLength = 2 * 4
  const colorSpaceBytesLength = 1
  const versionBytesLength = 3

  const totalBytes =
    headerBytes.byteLength +
    vertexBytesLength +
    extrinsicBytesLength +
    intrinsicBytesLength +
    imageSizeBytesLength +
    frameBytesLength +
    disparityBytesLength +
    colorSpaceBytesLength +
    versionBytesLength

  const output = new Uint8Array(totalBytes)
  output.set(headerBytes, 0)
  const view = new DataView(output.buffer)
  let offset = headerBytes.byteLength

  const disparities = new Float32Array(count)

  for (let i = 0; i < count; i += 1) {
    const v3 = i * 3
    const v4 = i * 4

    const x = meanVectors[v3]
    const y = meanVectors[v3 + 1]
    const z = meanVectors[v3 + 2]

    disparities[i] = 1 / Math.max(z, 1e-6)

    const sr = linearToSrgb(colors[v3])
    const sg = linearToSrgb(colors[v3 + 1])
    const sb = linearToSrgb(colors[v3 + 2])

    const values = [
      x,
      y,
      z,
      rgbToSh0(sr),
      rgbToSh0(sg),
      rgbToSh0(sb),
      safeLogit(opacities[i]),
      safeLog(singularValues[v3]),
      safeLog(singularValues[v3 + 1]),
      safeLog(singularValues[v3 + 2]),
      quaternions[v4],
      quaternions[v4 + 1],
      quaternions[v4 + 2],
      quaternions[v4 + 3],
    ]

    for (const value of values) {
      view.setFloat32(offset, value, true)
      offset += 4
    }
  }

  const extrinsic = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ])
  for (const value of extrinsic) {
    view.setFloat32(offset, value, true)
    offset += 4
  }

  const intrinsic = new Float32Array([
    focalPx,
    0,
    imageWidth * 0.5,
    0,
    focalPx,
    imageHeight * 0.5,
    0,
    0,
    1,
  ])
  for (const value of intrinsic) {
    view.setFloat32(offset, value, true)
    offset += 4
  }

  view.setUint32(offset, imageWidth, true)
  offset += 4
  view.setUint32(offset, imageHeight, true)
  offset += 4

  view.setInt32(offset, 1, true)
  offset += 4
  view.setInt32(offset, count, true)
  offset += 4

  view.setFloat32(offset, quantile(disparities, 0.1), true)
  offset += 4
  view.setFloat32(offset, quantile(disparities, 0.9), true)
  offset += 4

  view.setUint8(offset, COLOR_SPACE_SRGB_INDEX)
  offset += 1

  view.setUint8(offset, 1)
  offset += 1
  view.setUint8(offset, 5)
  offset += 1
  view.setUint8(offset, 0)
  offset += 1

  if (offset !== output.byteLength) {
    throw new Error(`PLY writer size mismatch (${offset} != ${output.byteLength})`)
  }

  return output
}
