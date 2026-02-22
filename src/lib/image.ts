import { SHARP_INTERNAL_RESOLUTION } from './sharpConstants'

export interface DecodedImageInfo {
  width: number
  height: number
}

export async function decodeImageBitmap(file: File): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file, {
      // `imageOrientation` is not fully typed in all TS DOM libs.
      imageOrientation: 'from-image' as never,
    })
  } catch {
    return createImageBitmap(file)
  }
}

export async function readImageInfo(file: File): Promise<DecodedImageInfo> {
  const bitmap = await decodeImageBitmap(file)
  try {
    return { width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}

export async function imageFileToSharpTensor(file: File): Promise<{
  tensor: Float32Array
  width: number
  height: number
}> {
  const bitmap = await decodeImageBitmap(file)
  try {
    const tensor = imageBitmapToSharpTensor(bitmap, SHARP_INTERNAL_RESOLUTION)
    return {
      tensor,
      width: bitmap.width,
      height: bitmap.height,
    }
  } finally {
    bitmap.close()
  }
}

export function imageBitmapToSharpTensor(bitmap: ImageBitmap, size: number): Float32Array {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Could not create a 2D canvas context for image preprocessing.')
  }

  context.clearRect(0, 0, size, size)
  context.drawImage(bitmap, 0, 0, size, size)
  const imageData = context.getImageData(0, 0, size, size)
  const pixels = imageData.data
  const pixelCount = size * size

  const tensor = new Float32Array(3 * pixelCount)
  let pixelOffset = 0
  for (let i = 0; i < pixelCount; i += 1) {
    tensor[i] = pixels[pixelOffset] / 255
    tensor[pixelCount + i] = pixels[pixelOffset + 1] / 255
    tensor[pixelCount * 2 + i] = pixels[pixelOffset + 2] / 255
    pixelOffset += 4
  }

  return tensor
}
