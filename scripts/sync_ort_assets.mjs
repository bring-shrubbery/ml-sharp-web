import { mkdir, copyFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = process.cwd()
const srcDir = resolve(root, 'node_modules/onnxruntime-web/dist')
const dstDir = resolve(root, 'public/ort')

const files = [
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]

await mkdir(dstDir, { recursive: true })
for (const file of files) {
  await copyFile(resolve(srcDir, file), resolve(dstDir, file))
}

console.log(`Copied ORT web assets to ${dstDir}`)
