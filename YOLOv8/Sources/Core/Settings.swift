//
// Created by moonl1ght 09.03.2023.
//

import Foundation

enum Settings {
  static let maxDetectedBBoxes = 100
  static var iouThreshold: Float = 0.2
  static var confidenceThreshold: Float = 0.45
  static var segmentationMaskConfidence: Float = 0.5
}
