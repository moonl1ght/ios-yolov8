//
// Created by moonl1ght 09.03.2023.
//

import Foundation

extension BBox {
  var rect: CGRect {
    .init(x: CGFloat(x), y: CGFloat(y), width: CGFloat(w), height: CGFloat(h))
  }

  var className: String {
    ObjectDetectionModel.classes[Int(classId)]
  }
}
