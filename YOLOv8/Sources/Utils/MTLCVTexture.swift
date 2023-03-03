//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit

final class MTLCVTexture {
  let texture: MTLTexture
  let pixelBuffer: CVPixelBuffer

  var width: Int {
    texture.width
  }

  var height: Int {
    texture.height
  }

  private init(texture: MTLTexture, pixelBuffer: CVPixelBuffer) {
    self.texture = texture
    self.pixelBuffer = pixelBuffer
  }
}

extension MTLCVTexture {
  static func make(
    usingPixelBufferPool pixelBufferPool: CVPixelBufferPool,
    textureCache: CVMetalTextureCache,
    pixelFormat: MTLPixelFormat
  ) -> MTLCVTexture? {
    var pixelBuffer: CVPixelBuffer?
    let result = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pixelBufferPool,
      &pixelBuffer
    )
    guard
      result == kCVReturnSuccess,
      let pixelBuffer = pixelBuffer,
      let texture = pixelBuffer.makeMTLTexture(usingTextureCache: textureCache, pixelFormat: pixelFormat)
    else {
      return nil
    }
    return .init(texture: texture, pixelBuffer: pixelBuffer)
  }
}
