//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import AVFoundation

typealias PixelFormat = OSType

extension CVPixelBufferPool {
  static func allocate(
    for dimension: CMVideoDimensions,
    pixelFormat: PixelFormat,
    bufferSize: UInt32
  ) -> CVPixelBufferPool? {
    let outputBufferAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
      kCVPixelBufferWidthKey as String: Int(dimension.width),
      kCVPixelBufferHeightKey as String: Int(dimension.height),
      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
    ]
    let poolAttributes = [kCVPixelBufferPoolMinimumBufferCountKey as String: bufferSize]
    var pixelBufferPool: CVPixelBufferPool?
    let result = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      poolAttributes as NSDictionary?,
      outputBufferAttributes as NSDictionary?,
      &pixelBufferPool
    )
    guard result == kCVReturnSuccess, let pixelBufferPool = pixelBufferPool else {
      return nil
    }
    return pixelBufferPool
  }
}
