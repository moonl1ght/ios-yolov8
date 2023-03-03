//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit

extension CVMetalTexture {
  static func createFromCVPixelBuffer(
    _ cvPixelBuffer: CVPixelBuffer,
    usingTextureCache textureCache: CVMetalTextureCache,
    pixelFormat: MTLPixelFormat,
    planeIndex: Int = 0
  ) -> CVMetalTexture? {
    precondition(planeIndex >= 0, "Plane index must be non negative.")
    let size: MTLSize
    if cvPixelBuffer.isPlanar {
      size = cvPixelBuffer.getMTLSize(forPlane: planeIndex)
    } else {
      size = cvPixelBuffer.mtlSize
    }
    var texture: CVMetalTexture?
    let status = CVMetalTextureCacheCreateTextureFromImage(
      nil, textureCache, cvPixelBuffer, nil, pixelFormat, size.width, size.height, planeIndex, &texture
    )
    if status != kCVReturnSuccess {
      texture = nil
      assertionFailure("Failed to create texture from CVPixelBuffer.")
    }
    return texture
  }
}
