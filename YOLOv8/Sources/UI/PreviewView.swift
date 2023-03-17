//
// Created by moonl1ght 06.03.2023.
//

import Foundation
import SwiftUI
import MetalKit

struct PreviewView: UIViewControllerRepresentable {
  let presenter: Presenter

  func makeUIViewController(context: Context) -> some UIViewController {
    PreviewViewController(presenter: presenter)
  }

  func updateUIViewController(_ uiViewController: UIViewControllerType, context: Context) { }
}

final class PreviewViewController: UIViewController {
  let presenter: Presenter

  private let renderer: Renderer
  private let mtkView = MTKView()
  private var resizeFailed = true
  private var bboxLayers: [CAShapeLayer] = []
  private var classLabels: [UILabel] = []
  private let colors: [String: UIColor] = {
    var colors = [String: UIColor]()
    ObjectDetectionModel.classes.forEach { className in
      colors[className] = UIColor(
        red: CGFloat.random(in: 0 ... 0.6),
        green: CGFloat.random(in: 0 ... 0.6),
        blue: CGFloat.random(in: 0 ... 0.6),
        alpha: 1
      )
    }
    return colors
  }()

  init(presenter: Presenter) {
    self.presenter = presenter
    renderer = Renderer()
    super.init(nibName: nil, bundle: nil)
  }

  @available(*, unavailable)
  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    presenter.cameraController.configure()
    presenter.cameraController.startSession()
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    mtkView.drawableSize = mtkView.frame.size
    bboxLayers.forEach {
      $0.frame = mtkView.layer.bounds
    }
  }

  override func viewWillDisappear(_ animated: Bool) {
    presenter.cameraController.stopSession()
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    view.addSubview(mtkView)
    mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    mtkView.device = DeviceManager.device
    mtkView.colorPixelFormat = Renderer.colorPixelFormat
    mtkView.preferredFramesPerSecond = 60
    mtkView.clearsContextBeforeDrawing = false
    mtkView.delegate = self

    bboxLayers = (0 ..< Settings.maxDetectedBBoxes).map { _ in
      let layer = CAShapeLayer()
      layer.isHidden = true
      layer.fillColor = UIColor.clear.cgColor
      mtkView.layer.addSublayer(layer)
      return layer
    }

    classLabels = (0 ..< Settings.maxDetectedBBoxes).map { _ in
      let label = UILabel()
      label.numberOfLines = 1
      label.adjustsFontSizeToFitWidth = true
      label.minimumScaleFactor = 0.2
      label.isHidden = true
      label.textColor = .white
      mtkView.addSubview(label)
      return label
    }

    configureLayout()
  }

  private func configureLayout() {
    mtkView.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      mtkView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      mtkView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      mtkView.topAnchor.constraint(equalTo: view.topAnchor),
      mtkView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
    ])
  }

  private func drawBBoxes(bboxes: [BBox]) {
    for i in 0 ..< Settings.maxDetectedBBoxes {
      bboxLayers[i].isHidden = true
      classLabels[i].isHidden = true
    }
    if bboxes.isEmpty { return }
    for (i, bbox) in bboxes.enumerated() {
      let rect = CGRectOffset(bbox.rect, -Renderer.bboxOffset.x, -Renderer.bboxOffset.y)
      let path = UIBezierPath(rect: rect)
      let bboxLayer = bboxLayers[i]
      bboxLayer.path = path.cgPath
      bboxLayer.strokeColor = colors[bbox.className]?.cgColor
      let lineWidth: CGFloat = 2
      bboxLayer.lineWidth = lineWidth
      bboxLayer.isHidden = false

      let label = classLabels[i]
      label.text = bbox.className.capitalized
      let labelHeight: CGFloat = 20
      label.frame = CGRect(
        origin: .init(x: rect.minX - lineWidth / 2, y: rect.minY - labelHeight),
        size: .init(width: rect.width + lineWidth, height: labelHeight)
      )
      label.backgroundColor = colors[bbox.className]
      label.isHidden = false
    }
  }
}

extension PreviewViewController: MTKViewDelegate {
  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    guard let frame = presenter.frame else {
      resizeFailed = true
      return
    }
    renderer.resize(size: size, textureSize: frame.pixelBuffer.size)
    resizeFailed = false
  }

  func draw(in view: MTKView) {
    guard let frame = presenter.frame else { return }
    if resizeFailed {
      renderer.resize(size: view.drawableSize, textureSize: frame.pixelBuffer.size)
      resizeFailed = false
    }
    drawBBoxes(bboxes: frame.bboxes)
    renderer.render(frame, view: view)
  }
}
