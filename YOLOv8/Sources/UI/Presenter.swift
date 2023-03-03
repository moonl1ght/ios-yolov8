//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import Combine

struct FrameInfo {
  let predictionDuration: String
  let processingDuration: String
  let fullProcessingDuration: String
  let detectedBBoxes: String
}

@MainActor
final class Presenter: ObservableObject {
  @Published private(set) var frameInfo: FrameInfo?

  let cameraController: CameraController
  let objectDetectionController: ObjectDetectionController

  private(set) var frame: Frame?

  private let frameInfoSubject = PassthroughSubject<FrameInfo, Never>()
  private var cancellables: Set<AnyCancellable> = Set()

  private let queue = DispatchQueue(label: "com.YOLOv8.queue", qos: .userInteractive)

  init() {
    cameraController = CameraController(queue: queue)
    objectDetectionController = ObjectDetectionController(queue: queue)
    objectDetectionController.delegate = self
    cameraController.delegate = objectDetectionController
    bind()
  }

  private func bind() {
    frameInfoSubject
      .throttle(for: 0.5, scheduler: RunLoop.main, latest: true)
      .sink { [weak self] in
        self?.frameInfo = $0
      }
      .store(in: &cancellables)
  }
}

extension Presenter: ObjectDetectionDelegate {
  func didDetectFrame(_ frame: Frame) {
    let predictionDuration = String(format: "%.4f", frame.predictionDuration.seconds)
    let processingDurationDelta = frame.processingDuration.seconds - frame.predictionDuration.seconds
    let processingDuration = String(format: "%.4f", processingDurationDelta)
    let fullProcessingDuration = String(format: "%.4f", frame.processingDuration.seconds)
    let predictionFPS = String(format: "%.2f", frame.predictionDuration.approxFPS())
    let processingFPS = String(
      format: "%.2f", frame.predictionDuration.approxFPS() - frame.processingDuration.approxFPS()
    )
    let fullProcessingFPS = String(format: "%.2f", frame.processingDuration.approxFPS())
    frameInfoSubject.send(
      FrameInfo(
        predictionDuration: "Prediction duration \(predictionDuration) sec - max FPS \(predictionFPS)",
        processingDuration: "Processing duration \(processingDuration) sec - FPS overhead \(processingFPS)",
        fullProcessingDuration: "Full processing duration \(fullProcessingDuration) sec - max FPS \(fullProcessingFPS)",
        detectedBBoxes: "Detected bounding boxes: \(frame.bboxes.count)"
      )
    )
    self.frame = frame
  }
}
