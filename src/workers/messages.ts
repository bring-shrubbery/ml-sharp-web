export interface WorkerStatusMessage {
  type: 'status'
  requestId?: string
  stage: 'idle' | 'loading-model' | 'running-inference' | 'filtering' | 'building-ply'
  message: string
}

export interface LoadModelRequestPayload {
  modelUrl: string
}

export interface RunInferenceRequestPayload {
  modelUrl: string
  imageTensor: ArrayBufferLike
  imageWidth: number
  imageHeight: number
  focalPx: number
  disparityFactor: number
  opacityThreshold: number
  maxGaussians: number
}

export interface WorkerInferenceResult {
  plyBuffer: ArrayBufferLike
  selectedGaussians: number
  totalGaussians: number
  outputName?: string
}

export type WorkerRequest =
  | {
      type: 'load-model'
      requestId: string
      payload: LoadModelRequestPayload
    }
  | {
      type: 'run-inference'
      requestId: string
      payload: RunInferenceRequestPayload
    }

export type WorkerReply =
  | {
      type: 'reply'
      requestId: string
      ok: true
      result: { modelUrl: string }
    }
  | {
      type: 'reply'
      requestId: string
      ok: true
      result: WorkerInferenceResult
    }
  | {
      type: 'reply'
      requestId: string
      ok: false
      error: string
    }

export type WorkerMessage = WorkerStatusMessage | WorkerReply
