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

function asFloat32(name: string, tensor: ort.Tensor): Float32Array {
  const data = tensor.data
  if (!(data instanceof Float32Array)) {
    throw new Error(`Expected '${name}' tensor to be Float32Array, got ${Object.prototype.toString.call(data)}`)
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

async function handleLoadModel(requestId: string, payload: LoadModelRequestPayload): Promise<void> {
  postStatus('loading-model', 'Loading ONNX model…', requestId)
  const session = await getSession(payload.modelUrl)
  if (session.inputNames.length < 5) {
    throw new Error(
      `Unexpected model inputs (${session.inputNames.join(', ')}). Expected wrapper inputs: image, disparity_factor, f_px, orig_width, orig_height.`,
    )
  }
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
    [session.inputNames[2]]: new ort.Tensor('float32', new Float32Array([payload.focalPx]), [1]),
    [session.inputNames[3]]: new ort.Tensor('float32', new Float32Array([payload.imageWidth]), [1]),
    [session.inputNames[4]]: new ort.Tensor('float32', new Float32Array([payload.imageHeight]), [1]),
  }

  const outputs = await session.run(feeds)

  const meanVectorsTensor = getTensor(outputs, 'mean_vectors')
  const singularValuesTensor = getTensor(outputs, 'singular_values')
  const quaternionsTensor = getTensor(outputs, 'quaternions')
  const colorsTensor = getTensor(outputs, 'colors')
  const opacitiesTensor = getTensor(outputs, 'opacities')

  const { data: meanVectors, count } = flattenBatchTensor(meanVectorsTensor, 3, 'mean_vectors')
  const { data: singularValues, count: singularCount } = flattenBatchTensor(
    singularValuesTensor,
    3,
    'singular_values',
  )
  const { data: quaternions, count: quaternionCount } = flattenBatchTensor(
    quaternionsTensor,
    4,
    'quaternions',
  )
  const { data: colors, count: colorCount } = flattenBatchTensor(colorsTensor, 3, 'colors')
  const { data: opacities, count: opacityCount } = flattenBatchTensor(opacitiesTensor, 1, 'opacities')

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

  postMessageSafe(reply, [result.plyBuffer])
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

    throw new Error(`Unknown worker request type: ${(data as { type?: string }).type ?? 'undefined'}`)
  } catch (error) {
    postError(data.requestId, error)
  }
}
