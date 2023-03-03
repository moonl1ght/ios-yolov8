//
// Created by moonl1ght 06.03.2023.
//

import Foundation

extension CGSize {
  var whRatio: CGFloat {
    guard width != 0 else { return 0 }
    return width / height
  }

  mutating func round() {
    height.round(.toNearestOrEven)
    width.round(.toNearestOrEven)
  }
}
