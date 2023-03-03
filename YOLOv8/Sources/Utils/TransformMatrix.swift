//
// Created by moonl1ght 06.03.2023.
//

import Foundation
import simd

enum TransformMatrix {
  @inlinable
  @inline(__always)
  static func scaling(_ scaling: SIMD3<Float>) -> float4x4 {
    .init(
      [scaling.x, 0, 0, 0],
      [0, scaling.y, 0, 0],
      [0, 0, scaling.z, 0],
      [0, 0, 0, 1]
    )
  }
}
