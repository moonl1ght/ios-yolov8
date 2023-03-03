//
// Created by moonl1ght 03.03.2023.
//

import Foundation
import MetalKit
import MetalPerformanceShaders
import CoreMedia

final class MTLImageScaler {
  enum Error: Swift.Error {
    case failedToRescale
  }

  private let device: MTLDevice
  private let scaler: MPSImageScale
  private let textureCache: CVMetalTextureCache
  private var rescaledPixelBufferPool: CVPixelBufferPool?

  init(
    device: MTLDevice,
    textureCache: CVMetalTextureCache,
    rescaledSize: CGSize
  ) {
    self.device = device
    self.textureCache = textureCache
    scaler = MPSImageBilinearScale(device: device)
    let rescaledDim = CMVideoDimensions(width: Int32(rescaledSize.width), height: Int32(rescaledSize.height))
    rescaledPixelBufferPool = CVPixelBufferPool.allocate(
      for: rescaledDim, pixelFormat: kCVPixelFormatType_32BGRA, bufferSize: 1
    )
  }

  func rescale(
    _ pixelBuffer: CVPixelBuffer,
    commandQueue: MTLCommandQueue
  ) throws -> MTLCVTexture {
    guard
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let rescaledPixelBufferPool,
      let originalTexture = pixelBuffer.makeMTLTexture(
        usingTextureCache: textureCache, pixelFormat: .bgra8Unorm
      ),
      let rescaledTexture: MTLCVTexture = .make(
        usingPixelBufferPool: rescaledPixelBufferPool,
        textureCache: textureCache,
        pixelFormat: .bgra8Unorm
      )
    else {
      throw Error.failedToRescale
    }
    scaler.encode(
      commandBuffer: commandBuffer,
      sourceTexture: originalTexture,
      destinationTexture: rescaledTexture.texture
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return rescaledTexture
  }
}
