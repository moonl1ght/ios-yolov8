//
// Created by moonl1ght 06.03.2023.
//

import Foundation
import MetalKit

extension CGSize {
  var mtlSize: MTLSize {
    .init(width: Int(width), height: Int(height), depth: 1)
  }

  var length: CGFloat {
    width * height
  }

  var whRatio: CGFloat {
    guard width != 0 else { return 0 }
    return width / height
  }

  mutating func round() {
    height.round(.toNearestOrEven)
    width.round(.toNearestOrEven)
  }
}
