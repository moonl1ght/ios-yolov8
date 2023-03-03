//
// Created by moonl1ght 27.02.2023.
//

import Foundation

import MetalKit

extension MTLComputePipelineState {
  ///
  /// Threads per threadgroup, or threadgroup size.
  ///
  var threadgroupSize: MTLSize {
    MTLSizeMake(threadExecutionWidth, maxTotalThreadsPerThreadgroup / threadExecutionWidth, 1)
  }

  ///
  /// Calculate number of threads per grid.
  /// - Note: Avaliable only on device which supports non-uniform threadgroup sizes.
  ///
  func calculateThreadsPerGrid(for texture: MTLTexture) -> MTLSize {
    MTLSize(width: texture.width, height: texture.height, depth: 1)
  }
}
