//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit

extension CVMetalTextureCache {
  static func createUsingDevice(_ device: MTLDevice) -> CVMetalTextureCache {
    var textureCache: CVMetalTextureCache?
    let result = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    if result == kCVReturnSuccess, let textureCache {
      return textureCache
    } else {
      fatalError("Failed to create CVMetalTextureCache")
    }
  }
}
