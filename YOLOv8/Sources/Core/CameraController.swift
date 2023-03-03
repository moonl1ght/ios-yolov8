//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import AVFoundation

protocol CameraControllerDelegate: AnyObject {
  func cameraController(didOutput pixelBuffer: CVPixelBuffer)
}

final class CameraController: NSObject {
  enum Error: Swift.Error {
    case failedToConfigure
  }

  weak var delegate: CameraControllerDelegate?

  let session = AVCaptureSession()

  private let queue: DispatchQueue
  private var isConfigured = false

  init(queue: DispatchQueue) {
    self.queue = queue
  }

  func configure() {
    queue.sync {
      session.beginConfiguration()
      defer { session.commitConfiguration() }
      do {
        try configureInput()
        try configureOutput()
        isConfigured = true
      } catch {
        assertionFailure(error.localizedDescription)
      }
    }
  }

  func startSession() {
    queue.async { [self] in
      guard isConfigured else { return }
      session.startRunning()
    }
  }

  func stopSession() {
    queue.async { [self] in
      session.stopRunning()
    }
  }

  private func configureInput() throws {
    session.sessionPreset = .high
    guard
      let device = AVCaptureDevice.default(for: .video),
      let format = device.formats.filter({
        $0.videoSupportedFrameRateRanges.contains {
          $0.maxFrameRate >= 59
        }
      }).last
    else {
      throw Error.failedToConfigure
    }
    try device.lockForConfiguration()
    defer { device.unlockForConfiguration() }
    device.activeFormat = format
    let deviceInput = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(deviceInput) else {
      throw Error.failedToConfigure
    }
    session.addInput(deviceInput)
  }

  private func configureOutput() throws {
    let videoOutput = AVCaptureVideoDataOutput()
    guard session.canAddOutput(videoOutput) else {
      throw Error.failedToConfigure
    }
    videoOutput.setSampleBufferDelegate(self, queue: queue)
    let settings: [String : Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA),
    ]
    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    session.addOutput(videoOutput)
    let connection = session.connections.first
    connection?.videoOrientation = .portrait
  }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      delegate?.cameraController(didOutput: imageBuffer)
    }
  }
}
