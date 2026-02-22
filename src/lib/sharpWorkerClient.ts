import type {
  LoadModelRequestPayload,
  RunInferenceRequestPayload,
  WorkerInferenceResult,
  WorkerMessage,
  WorkerReply,
  WorkerStatusMessage,
} from '../workers/messages'

interface PendingRequest {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

function createRequestId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export class SharpWorkerClient {
  private worker: Worker
  private pending = new Map<string, PendingRequest>()
  private statusHandler?: (message: WorkerStatusMessage) => void

  constructor(statusHandler?: (message: WorkerStatusMessage) => void) {
    this.statusHandler = statusHandler
    this.worker = new Worker(new URL('../workers/sharpWorker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      this.handleMessage(event.data)
    }
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'Worker error')
      this.rejectAll(error)
    }
  }

  setStatusHandler(statusHandler?: (message: WorkerStatusMessage) => void): void {
    this.statusHandler = statusHandler
  }

  dispose(): void {
    this.rejectAll(new Error('Worker disposed'))
    this.worker.terminate()
  }

  async loadModel(payload: LoadModelRequestPayload): Promise<{ modelUrl: string }> {
    return this.request('load-model', payload)
  }

  async runInference(payload: RunInferenceRequestPayload): Promise<WorkerInferenceResult> {
    return this.request('run-inference', payload, [payload.imageTensor])
  }

  private request<T>(
    type: 'load-model' | 'run-inference',
    payload: LoadModelRequestPayload | RunInferenceRequestPayload,
    transfer: Transferable[] = [],
  ): Promise<T> {
    const requestId = createRequestId()
    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (value: unknown) => void, reject })
    })

    this.worker.postMessage({ type, requestId, payload }, transfer)
    return promise
  }

  private handleMessage(message: WorkerMessage): void {
    if (message.type === 'status') {
      this.statusHandler?.(message)
      return
    }

    this.handleReply(message)
  }

  private handleReply(message: WorkerReply): void {
    const pending = this.pending.get(message.requestId)
    if (!pending) {
      return
    }
    this.pending.delete(message.requestId)

    if (!message.ok) {
      pending.reject(new Error(message.error))
      return
    }

    pending.resolve(message.result)
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error)
    }
    this.pending.clear()
  }
}
