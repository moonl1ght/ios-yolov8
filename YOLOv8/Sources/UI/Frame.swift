//
// Created by moonl1ght 06.03.2023.
//

import Foundation
import MetalKit

final class Frame {
  let pixelBuffer: CVPixelBuffer
  var bboxes: [BBox] = []
  var processingDuration: Benchmark.MeasureResult = .zero
  var predictionDuration: Benchmark.MeasureResult = .zero

  init(
    pixelBuffer: CVPixelBuffer
  ) {
    self.pixelBuffer = pixelBuffer
  }
}
