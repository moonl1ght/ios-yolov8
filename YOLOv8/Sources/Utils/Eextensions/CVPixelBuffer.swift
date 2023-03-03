//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit

extension CVPixelBuffer {
  var isPlanar: Bool {
    CVPixelBufferIsPlanar(self)
  }

  var mtlSize: MTLSize {
    MTLSize(
      width: CVPixelBufferGetWidth(self),
      height: CVPixelBufferGetHeight(self),
      depth: 1
    )
  }

  var size: CGSize {
    CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
  }

  var bytesPerRow: Int {
    CVPixelBufferGetBytesPerRow(self)
  }

  var bytesCount: Int {
    let height = CVPixelBufferGetHeight(self)
    return height * bytesPerRow
  }

  func getSize(forPlane planeIndex: Int) -> CGSize {
    CGSize(
      width: CVPixelBufferGetWidthOfPlane(self, planeIndex),
      height: CVPixelBufferGetHeightOfPlane(self, planeIndex)
    )
  }

  func getMTLSize(forPlane planeIndex: Int) -> MTLSize {
    MTLSize(
      width: CVPixelBufferGetWidthOfPlane(self, planeIndex),
      height: CVPixelBufferGetHeightOfPlane(self, planeIndex),
      depth: 1
    )
  }

  func getBytesPerRow(forPlane planeIndex: Int) -> Int {
    CVPixelBufferGetBytesPerRowOfPlane(self, planeIndex)
  }

  func getBytesCount(forPlane planeIndex: Int) -> Int {
    let height = CVPixelBufferGetHeightOfPlane(self, planeIndex)
    return height * getBytesPerRow(forPlane: planeIndex)
  }

  func makeMTLTexture(
    usingTextureCache textureCache: CVMetalTextureCache,
    pixelFormat: MTLPixelFormat,
    planeIndex: Int = 0
  ) -> MTLTexture? {
    if let cvMetalTexture = CVMetalTexture.createFromCVPixelBuffer(
      self, usingTextureCache: textureCache, pixelFormat: pixelFormat, planeIndex: planeIndex
    ) {
      return CVMetalTextureGetTexture(cvMetalTexture)
    } else {
      return nil
    }
  }
}
