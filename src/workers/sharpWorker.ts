/// <reference lib="WebWorker" />

import * as ort from 'onnxruntime-web/webgpu'

import { buildSharpPlyBinary } from '../lib/ply'
import { SHARP_INTERNAL_RESOLUTION } from '../lib/sharpConstants'
import type {
  LoadModelRequestPayload,
  RunInferenceRequestPayload,
  WorkerInferenceResult,
  WorkerMessage,
  WorkerReply,
  WorkerRequest,
  WorkerStatusMessage,
} from './messages'

const workerScope = self as DedicatedWorkerGlobalScope
const sessionCache = new Map<string, Promise<ort.InferenceSession>>()

ort.env.wasm.numThreads = Math.max(1, Math.min(4, self.navigator.hardwareConcurrency || 2))
ort.env.wasm.simd = true
const ortBaseUrl = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.origin).href
ort.env.wasm.wasmPaths = {
  mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', ortBaseUrl).href,
  wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', ortBaseUrl).href,
}

function postMessageSafe(message: WorkerMessage, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) {
    workerScope.postMessage(message, transfer)
    return
  }
  workerScope.postMessage(message)
}

function postStatus(
  stage: WorkerStatusMessage['stage'],
  message: string,
  requestId?: string,
): void {
  postMessageSafe({ type: 'status', stage, message, requestId })
}

function postError(requestId: string, error: unknown): void {
  const text = error instanceof Error ? error.message : String(error)
  const reply: WorkerReply = {
    type: 'reply',
    requestId,
    ok: false,
    error: text,
  }
  postMessageSafe(reply)
}

function getSession(modelUrl: string): Promise<ort.InferenceSession> {
  const cached = sessionCache.get(modelUrl)
  if (cached) {
    return cached
  }

  const sessionPromise = createSession(modelUrl)
  sessionCache.set(modelUrl, sessionPromise)
  return sessionPromise
}

async function createSession(modelUrl: string): Promise<ort.InferenceSession> {
  try {
    return await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
    })
  } catch (webGpuError) {
    return ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).catch((wasmError) => {
      throw new Error(
        `Could not create ONNX Runtime session with WebGPU or WASM. WebGPU error: ${String(webGpuError)}. WASM error: ${String(wasmError)}`,
      )
    })
  }
}

function getTensor(outputs: ort.InferenceSession.ReturnType, key: string): ort.Tensor {
  const tensor = outputs[key]
  if (!tensor) {
    const available = Object.keys(outputs)
    throw new Error(`Missing output tensor '${key}'. Available outputs: ${available.join(', ')}`)
  }
  return tensor
}

function getTensorAny(
  outputs: ort.InferenceSession.ReturnType,
  keys: readonly string[],
): { tensor: ort.Tensor; key: string } {
  for (const key of keys) {
    const tensor = outputs[key]
    if (tensor) {
      return { tensor, key }
    }
  }
  const available = Object.keys(outputs)
  throw new Error(`Missing required output tensor. Tried: ${keys.join(', ')}. Available: ${available.join(', ')}`)
}

function asFloat32(name: string, tensor: ort.Tensor): Float32Array {
  const data = tensor.data
  if (!(data instanceof Float32Array)) {
    throw new Error(
      `Expected '${name}' tensor to be Float32Array, got ${Object.prototype.toString.call(data)}`,
    )
  }
  return data
}

interface PrunedGaussians {
  count: number
  meanVectors: Float32Array
  singularValues: Float32Array
  quaternions: Float32Array
  colors: Float32Array
  opacities: Float32Array
}

function copyTriplets(source: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length * 3)
  let outOffset = 0
  for (const index of indices) {
    const srcOffset = index * 3
    out[outOffset] = source[srcOffset]
    out[outOffset + 1] = source[srcOffset + 1]
    out[outOffset + 2] = source[srcOffset + 2]
    outOffset += 3
  }
  return out
}

function copyQuads(source: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length * 4)
  let outOffset = 0
  for (const index of indices) {
    const srcOffset = index * 4
    out[outOffset] = source[srcOffset]
    out[outOffset + 1] = source[srcOffset + 1]
    out[outOffset + 2] = source[srcOffset + 2]
    out[outOffset + 3] = source[srcOffset + 3]
    outOffset += 4
  }
  return out
}

function copySingles(source: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length)
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = source[indices[i]]
  }
  return out
}

function flattenBatchTensor(
  tensor: ort.Tensor,
  channels: number,
  label: string,
): { data: Float32Array; count: number } {
  const dims = tensor.dims
  const data = asFloat32(label, tensor)

  if (dims.length < 2) {
    throw new Error(`Output '${label}' should have rank >= 2. Got dims=${dims.join('x')}`)
  }

  const count = channels === 1 ? data.length : Math.floor(data.length / channels)
  if (count <= 0) {
    throw new Error(`Output '${label}' has no data.`)
  }
  if (channels > 1 && count * channels !== data.length) {
    throw new Error(`Output '${label}' length (${data.length}) is not divisible by ${channels}.`)
  }

  return { data, count }
}

function pruneGaussians(
  meanVectors: Float32Array,
  singularValues: Float32Array,
  quaternions: Float32Array,
  colors: Float32Array,
  opacities: Float32Array,
  opacityThreshold: number,
  maxGaussians: number,
): { pruned: PrunedGaussians; totalCount: number } {
  const totalCount = opacities.length
  const threshold = Number.isFinite(opacityThreshold) ? opacityThreshold : 0
  const cappedMax = Number.isFinite(maxGaussians) && maxGaussians > 0 ? Math.floor(maxGaussians) : 0

  const selected: number[] = []
  for (let i = 0; i < totalCount; i += 1) {
    if (opacities[i] >= threshold) {
      selected.push(i)
    }
  }

  if (selected.length === 0) {
    for (let i = 0; i < totalCount; i += 1) {
      selected.push(i)
    }
  }

  if (cappedMax > 0 && selected.length > cappedMax) {
    selected.sort((a, b) => opacities[b] - opacities[a])
    selected.length = cappedMax
    selected.sort((a, b) => a - b)
  }

  const pruned: PrunedGaussians = {
    count: selected.length,
    meanVectors: copyTriplets(meanVectors, selected),
    singularValues: copyTriplets(singularValues, selected),
    quaternions: copyQuads(quaternions, selected),
    colors: copyTriplets(colors, selected),
    opacities: copySingles(opacities, selected),
  }

  return { pruned, totalCount }
}

function quaternionToRotationMatrix(
  qw: number,
  qx: number,
  qy: number,
  qz: number,
): [number, number, number, number, number, number, number, number, number] {
  const norm = Math.hypot(qw, qx, qy, qz) || 1
  const w = qw / norm
  const x = qx / norm
  const y = qy / norm
  const z = qz / norm

  const ww = w * w
  const xx = x * x
  const yy = y * y
  const zz = z * z
  const wx = w * x
  const wy = w * y
  const wz = w * z
  const xy = x * y
  const xz = x * z
  const yz = y * z

  return [
    ww + xx - yy - zz,
    2 * (xy - wz),
    2 * (xz + wy),
    2 * (xy + wz),
    ww - xx + yy - zz,
    2 * (yz - wx),
    2 * (xz - wy),
    2 * (yz + wx),
    ww - xx - yy + zz,
  ]
}

function jacobiRotateSymmetric3x3(matrix: Float64Array, vectors: Float64Array, p: number, q: number): void {
  const pp = p * 3 + p
  const qq = q * 3 + q
  const pq = p * 3 + q
  const qp = q * 3 + p

  const app = matrix[pp]
  const aqq = matrix[qq]
  const apq = matrix[pq]
  if (Math.abs(apq) < 1e-18) {
    return
  }

  const tau = (aqq - app) / (2 * apq)
  const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau))
  const c = 1 / Math.sqrt(1 + t * t)
  const s = t * c

  for (let k = 0; k < 3; k += 1) {
    if (k === p || k === q) {
      continue
    }

    const kp = k * 3 + p
    const pk = p * 3 + k
    const kq = k * 3 + q
    const qk = q * 3 + k

    const mkp = matrix[kp]
    const mkq = matrix[kq]

    const newMkp = c * mkp - s * mkq
    const newMkq = s * mkp + c * mkq

    matrix[kp] = newMkp
    matrix[pk] = newMkp
    matrix[kq] = newMkq
    matrix[qk] = newMkq
  }

  matrix[pp] = c * c * app - 2 * s * c * apq + s * s * aqq
  matrix[qq] = s * s * app + 2 * s * c * apq + c * c * aqq
  matrix[pq] = 0
  matrix[qp] = 0

  for (let k = 0; k < 3; k += 1) {
    const kp = k * 3 + p
    const kq = k * 3 + q
    const vkp = vectors[kp]
    const vkq = vectors[kq]
    vectors[kp] = c * vkp - s * vkq
    vectors[kq] = s * vkp + c * vkq
  }
}

function jacobiEigenSymmetric3x3(matrix: Float64Array, vectors: Float64Array): void {
  vectors.fill(0)
  vectors[0] = 1
  vectors[4] = 1
  vectors[8] = 1

  for (let sweep = 0; sweep < 8; sweep += 1) {
    const offDiag = Math.abs(matrix[1]) + Math.abs(matrix[2]) + Math.abs(matrix[5])
    if (offDiag < 1e-14) {
      break
    }
    jacobiRotateSymmetric3x3(matrix, vectors, 0, 1)
    jacobiRotateSymmetric3x3(matrix, vectors, 0, 2)
    jacobiRotateSymmetric3x3(matrix, vectors, 1, 2)
  }
}

function swapEigenColumns(vectors: Float64Array, c0: number, c1: number): void {
  for (let row = 0; row < 3; row += 1) {
    const i0 = row * 3 + c0
    const i1 = row * 3 + c1
    const temp = vectors[i0]
    vectors[i0] = vectors[i1]
    vectors[i1] = temp
  }
}

function sortEigenpairsDescending(eigenvalues: Float64Array, vectors: Float64Array): void {
  if (eigenvalues[0] < eigenvalues[1]) {
    const temp = eigenvalues[0]
    eigenvalues[0] = eigenvalues[1]
    eigenvalues[1] = temp
    swapEigenColumns(vectors, 0, 1)
  }
  if (eigenvalues[1] < eigenvalues[2]) {
    const temp = eigenvalues[1]
    eigenvalues[1] = eigenvalues[2]
    eigenvalues[2] = temp
    swapEigenColumns(vectors, 1, 2)
  }
  if (eigenvalues[0] < eigenvalues[1]) {
    const temp = eigenvalues[0]
    eigenvalues[0] = eigenvalues[1]
    eigenvalues[1] = temp
    swapEigenColumns(vectors, 0, 1)
  }
}

function ensureProperRotation(vectors: Float64Array): void {
  const r00 = vectors[0]
  const r01 = vectors[1]
  const r02 = vectors[2]
  const r10 = vectors[3]
  const r11 = vectors[4]
  const r12 = vectors[5]
  const r20 = vectors[6]
  const r21 = vectors[7]
  const r22 = vectors[8]

  const det =
    r00 * (r11 * r22 - r12 * r21) -
    r01 * (r10 * r22 - r12 * r20) +
    r02 * (r10 * r21 - r11 * r20)

  if (det < 0) {
    vectors[2] *= -1
    vectors[5] *= -1
    vectors[8] *= -1
  }
}

function quaternionFromRotationMatrix(
  r00: number,
  r01: number,
  r02: number,
  r10: number,
  r11: number,
  r12: number,
  r20: number,
  r21: number,
  r22: number,
): [number, number, number, number] {
  const trace = r00 + r11 + r22
  let qw: number
  let qx: number
  let qy: number
  let qz: number

  if (trace > 0) {
    const s = 2 * Math.sqrt(Math.max(1e-12, trace + 1))
    qw = 0.25 * s
    qx = (r21 - r12) / s
    qy = (r02 - r20) / s
    qz = (r10 - r01) / s
  } else if (r00 > r11 && r00 > r22) {
    const s = 2 * Math.sqrt(Math.max(1e-12, 1 + r00 - r11 - r22))
    qw = (r21 - r12) / s
    qx = 0.25 * s
    qy = (r01 + r10) / s
    qz = (r02 + r20) / s
  } else if (r11 > r22) {
    const s = 2 * Math.sqrt(Math.max(1e-12, 1 + r11 - r00 - r22))
    qw = (r02 - r20) / s
    qx = (r01 + r10) / s
    qy = 0.25 * s
    qz = (r12 + r21) / s
  } else {
    const s = 2 * Math.sqrt(Math.max(1e-12, 1 + r22 - r00 - r11))
    qw = (r10 - r01) / s
    qx = (r02 + r20) / s
    qy = (r12 + r21) / s
    qz = 0.25 * s
  }

  const norm = Math.hypot(qw, qx, qy, qz) || 1
  return [qw / norm, qx / norm, qy / norm, qz / norm]
}

function unprojectGaussiansInPlace(
  gaussians: Pick<PrunedGaussians, 'count' | 'meanVectors' | 'singularValues' | 'quaternions'>,
  scaleX: number,
  scaleY: number,
): void {
  const matrix = new Float64Array(9)
  const vectors = new Float64Array(9)
  const eigenvalues = new Float64Array(3)

  for (let i = 0; i < gaussians.count; i += 1) {
    const idx3 = i * 3
    const idx4 = i * 4

    gaussians.meanVectors[idx3] *= scaleX
    gaussians.meanVectors[idx3 + 1] *= scaleY

    const [r00, r01, r02, r10, r11, r12, r20, r21, r22] = quaternionToRotationMatrix(
      gaussians.quaternions[idx4],
      gaussians.quaternions[idx4 + 1],
      gaussians.quaternions[idx4 + 2],
      gaussians.quaternions[idx4 + 3],
    )

    const v0 = gaussians.singularValues[idx3] ** 2
    const v1 = gaussians.singularValues[idx3 + 1] ** 2
    const v2 = gaussians.singularValues[idx3 + 2] ** 2

    const c00 = r00 * r00 * v0 + r01 * r01 * v1 + r02 * r02 * v2
    const c01 = r00 * r10 * v0 + r01 * r11 * v1 + r02 * r12 * v2
    const c02 = r00 * r20 * v0 + r01 * r21 * v1 + r02 * r22 * v2
    const c11 = r10 * r10 * v0 + r11 * r11 * v1 + r12 * r12 * v2
    const c12 = r10 * r20 * v0 + r11 * r21 * v1 + r12 * r22 * v2
    const c22 = r20 * r20 * v0 + r21 * r21 * v1 + r22 * r22 * v2

    // A * C * A^T where A = diag(scaleX, scaleY, 1)
    matrix[0] = c00 * scaleX * scaleX
    matrix[1] = c01 * scaleX * scaleY
    matrix[2] = c02 * scaleX
    matrix[3] = matrix[1]
    matrix[4] = c11 * scaleY * scaleY
    matrix[5] = c12 * scaleY
    matrix[6] = matrix[2]
    matrix[7] = matrix[5]
    matrix[8] = c22

    jacobiEigenSymmetric3x3(matrix, vectors)
    eigenvalues[0] = matrix[0]
    eigenvalues[1] = matrix[4]
    eigenvalues[2] = matrix[8]
    sortEigenpairsDescending(eigenvalues, vectors)
    ensureProperRotation(vectors)

    gaussians.singularValues[idx3] = Math.sqrt(Math.max(eigenvalues[0], 1e-12))
    gaussians.singularValues[idx3 + 1] = Math.sqrt(Math.max(eigenvalues[1], 1e-12))
    gaussians.singularValues[idx3 + 2] = Math.sqrt(Math.max(eigenvalues[2], 1e-12))

    const [qw, qx, qy, qz] = quaternionFromRotationMatrix(
      vectors[0],
      vectors[1],
      vectors[2],
      vectors[3],
      vectors[4],
      vectors[5],
      vectors[6],
      vectors[7],
      vectors[8],
    )
    gaussians.quaternions[idx4] = qw
    gaussians.quaternions[idx4 + 1] = qx
    gaussians.quaternions[idx4 + 2] = qy
    gaussians.quaternions[idx4 + 3] = qz
  }
}

function resolveOutputTensors(outputs: ort.InferenceSession.ReturnType): {
  meanVectors: ort.Tensor
  singularValues: ort.Tensor
  quaternions: ort.Tensor
  colors: ort.Tensor
  opacities: ort.Tensor
  isNdcOutput: boolean
} {
  const mean = getTensorAny(outputs, ['mean_vectors_ndc', 'mean_vectors'])
  const scales = getTensorAny(outputs, ['singular_values_ndc', 'singular_values'])
  const quats = getTensorAny(outputs, ['quaternions_ndc', 'quaternions'])
  const colors = getTensor(outputs, 'colors')
  const opacities = getTensor(outputs, 'opacities')

  const isNdcOutput =
    mean.key === 'mean_vectors_ndc' ||
    scales.key === 'singular_values_ndc' ||
    quats.key === 'quaternions_ndc'

  return {
    meanVectors: mean.tensor,
    singularValues: scales.tensor,
    quaternions: quats.tensor,
    colors,
    opacities,
    isNdcOutput,
  }
}

function validateModelInputs(session: ort.InferenceSession): { supportsWrapperScalars: boolean } {
  if (session.inputNames.length < 2) {
    throw new Error(
      `Unexpected model inputs (${session.inputNames.join(', ')}). Expected at least image + disparity_factor inputs.`,
    )
  }

  if (session.inputNames.length !== 2 && session.inputNames.length < 5) {
    throw new Error(
      `Unsupported model input count ${session.inputNames.length}. Expected 2 (raw predictor export) or 5 (legacy wrapper export).`,
    )
  }

  return { supportsWrapperScalars: session.inputNames.length >= 5 }
}

async function handleLoadModel(requestId: string, payload: LoadModelRequestPayload): Promise<void> {
  postStatus('loading-model', 'Loading ONNX model…', requestId)
  const session = await getSession(payload.modelUrl)
  validateModelInputs(session)

  const reply: WorkerReply = {
    type: 'reply',
    requestId,
    ok: true,
    result: { modelUrl: payload.modelUrl },
  }
  postMessageSafe(reply)
}

async function handleRunInference(
  requestId: string,
  payload: RunInferenceRequestPayload,
): Promise<void> {
  if (payload.imageWidth <= 0 || payload.imageHeight <= 0) {
    throw new Error('Image width/height must be > 0.')
  }
  if (payload.focalPx <= 0 || !Number.isFinite(payload.focalPx)) {
    throw new Error('Focal length must be a positive finite number.')
  }

  const session = await getSession(payload.modelUrl)
  const { supportsWrapperScalars } = validateModelInputs(session)

  const imageTensorData = new Float32Array(payload.imageTensor)
  const expectedImageValues = 3 * SHARP_INTERNAL_RESOLUTION * SHARP_INTERNAL_RESOLUTION
  if (imageTensorData.length !== expectedImageValues) {
    throw new Error(
      `Unexpected image tensor size ${imageTensorData.length}. Expected ${expectedImageValues}.`,
    )
  }

  postStatus('running-inference', 'Running SHARP inference in the browser…', requestId)

  const feeds: Record<string, ort.Tensor> = {
    [session.inputNames[0]]: new ort.Tensor('float32', imageTensorData, [1, 3, SHARP_INTERNAL_RESOLUTION, SHARP_INTERNAL_RESOLUTION]),
    [session.inputNames[1]]: new ort.Tensor('float32', new Float32Array([payload.disparityFactor]), [1]),
  }
  if (supportsWrapperScalars) {
    feeds[session.inputNames[2]] = new ort.Tensor('float32', new Float32Array([payload.focalPx]), [1])
    feeds[session.inputNames[3]] = new ort.Tensor('float32', new Float32Array([payload.imageWidth]), [1])
    feeds[session.inputNames[4]] = new ort.Tensor('float32', new Float32Array([payload.imageHeight]), [1])
  }

  const outputs = await session.run(feeds)
  const resolved = resolveOutputTensors(outputs)

  const { data: meanVectors, count } = flattenBatchTensor(
    resolved.meanVectors,
    3,
    resolved.isNdcOutput ? 'mean_vectors_ndc' : 'mean_vectors',
  )
  const { data: singularValues, count: singularCount } = flattenBatchTensor(
    resolved.singularValues,
    3,
    resolved.isNdcOutput ? 'singular_values_ndc' : 'singular_values',
  )
  const { data: quaternions, count: quaternionCount } = flattenBatchTensor(
    resolved.quaternions,
    4,
    resolved.isNdcOutput ? 'quaternions_ndc' : 'quaternions',
  )
  const { data: colors, count: colorCount } = flattenBatchTensor(resolved.colors, 3, 'colors')
  const { data: opacities, count: opacityCount } = flattenBatchTensor(resolved.opacities, 1, 'opacities')

  if (
    count !== singularCount ||
    count !== quaternionCount ||
    count !== colorCount ||
    count !== opacityCount
  ) {
    throw new Error(
      `Output count mismatch: means=${count}, scales=${singularCount}, quat=${quaternionCount}, colors=${colorCount}, opacities=${opacityCount}`,
    )
  }

  postStatus('filtering', 'Filtering and capping Gaussians for browser preview/export…', requestId)
  const { pruned, totalCount } = pruneGaussians(
    meanVectors,
    singularValues,
    quaternions,
    colors,
    opacities,
    payload.opacityThreshold,
    payload.maxGaussians,
  )

  if (resolved.isNdcOutput) {
    postStatus('filtering', 'Converting NDC Gaussians to metric space in-browser…', requestId)
    const scaleX = payload.imageWidth / (2 * payload.focalPx)
    const scaleY = payload.imageHeight / (2 * payload.focalPx)
    unprojectGaussiansInPlace(pruned, scaleX, scaleY)
  }

  postStatus('building-ply', 'Building binary .ply for preview and download…', requestId)
  const ply = buildSharpPlyBinary({
    ...pruned,
    imageWidth: payload.imageWidth,
    imageHeight: payload.imageHeight,
    focalPx: payload.focalPx,
  })

  const result: WorkerInferenceResult = {
    plyBuffer: ply.buffer.slice(ply.byteOffset, ply.byteOffset + ply.byteLength),
    selectedGaussians: pruned.count,
    totalGaussians: totalCount,
  }

  const reply: WorkerReply = {
    type: 'reply',
    requestId,
    ok: true,
    result,
  }

  postMessageSafe(reply, [result.plyBuffer as ArrayBuffer])
}

workerScope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { data } = event

  try {
    if (data.type === 'load-model') {
      await handleLoadModel(data.requestId, data.payload)
      return
    }

    if (data.type === 'run-inference') {
      await handleRunInference(data.requestId, data.payload)
      return
    }

    throw new Error(
      `Unknown worker request type: ${(data as { type?: string }).type ?? 'undefined'}`,
    )
  } catch (error) {
    postError(data.requestId, error)
  }
}
