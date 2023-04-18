//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import CoreML

final class ObjectDetectionModel {
  enum Error: Swift.Error {
    case failedToLoadModel
  }
  static let inputSize = CGSize(width: 640, height: 640)
  static let stide: Int = 8400
  static let segmentationMaskLength: Int = 32
  static let segmentationMaskSize = CGSize(width: 160, height: 160)

  static let classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
  ]

  enum ModelType {
    case normal
    case withSegmentation
  }

  enum ModelSize {
    case nano
    case small
    case large
    case xlarge
  }

  final class Output {
    let output: MLMultiArray
    let proto: MLMultiArray?

    init(output: MLMultiArray, proto: MLMultiArray?) {
      self.output = output
      self.proto = proto
    }
  }

  private var modeln: YOLOv8n?
  private var modelns: YOLOv8nseg?
  private var models: YOLOv8s?
  private var modelss: YOLOv8sseg?
  private var modelType: ModelType = .normal
  private var modalSize: ModelSize = .nano

  func load(modelType: ModelType, modelSize: ModelSize) throws {
    self.modelType = modelType
    self.modalSize = modelSize
    switch modelType {
    case .normal:
      switch modelSize {
      case .nano:
        modeln = try YOLOv8n(configuration: .init())
      case .small:
        models = try YOLOv8s(configuration: .init())
      case .large, .xlarge:
        throw Error.failedToLoadModel
      }
    case .withSegmentation:
      switch modelSize {
      case .nano:
        modelns = try YOLOv8nseg(configuration: .init())
      case .small:
        modelss = try YOLOv8sseg(configuration: .init())
      case .large, .xlarge:
        throw Error.failedToLoadModel
      }
    }

  }

  func predict(image: CVPixelBuffer) -> Output? {
    do {
      switch modelType {
      case .normal:
        switch modalSize {
        case .nano:
          guard let result = try modeln?.prediction(image: image) else { return nil }
          return Output(output: result.var_914, proto: nil)
        case .small:
          guard let result = try models?.prediction(image: image) else { return nil }
          return Output(output: result.var_914, proto: nil)
        case .large, .xlarge:
          return nil
        }
      case .withSegmentation:
        switch modalSize {
        case .nano:
          guard let result = try modelns?.prediction(image: image) else { return nil }
          return Output(output: result.var_1053, proto: result.p)
        case .small:
          guard let result = try modelss?.prediction(image: image) else { return nil }
          return Output(output: result.var_1053, proto: result.p)
        case .large, .xlarge:
          return nil
        }
      }
    } catch {
      assertionFailure(error.localizedDescription)
      return nil
    }
  }
}
