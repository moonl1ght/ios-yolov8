//
// Created by moonl1ght 06.03.2023.
//

import Foundation
import MetalKit

final class Renderer {
  enum AspectMode {
    case scaleToFill
    case aspectFit
    case aspectFill
  }

  static let colorPixelFormat: MTLPixelFormat = .bgra8Unorm
  static let aspectMode: AspectMode = .aspectFill
  static var bboxOffset: CGPoint = .zero
  static var colors: [simd_float3] = []

  private let commandQueue: MTLCommandQueue
  private let backgroundPlane: MTLBuffer
  private let pipelineState: MTLRenderPipelineState
  private let segmentationPipelineState: MTLRenderPipelineState
  private let textureCache: CVMetalTextureCache
  private var uniforms = VertexUniforms()

  init() {
    let device = DeviceManager.device
    guard
      let commandQueue = device.makeCommandQueue(),
      let library = try? device.makeDefaultLibrary(bundle: .main),
      let backgroundPlane = Renderer.createPlaneVertexBuffer(for: device),
      let pipelineState = Renderer.createPipelineState(
        device: device,
        fragmentFunctionName: "fragmentBaseRendering",
        library: library
      ),
      let segmentationPipelineState = Renderer.createPipelineState(
        device: device,
        fragmentFunctionName: "fragmentSegmentationRendering",
        library: library
      )
    else {
      fatalError()
    }
    self.commandQueue = commandQueue
    self.backgroundPlane = backgroundPlane
    self.pipelineState = pipelineState
    self.segmentationPipelineState = segmentationPipelineState
    let textureCache = CVMetalTextureCache.createUsingDevice(device)
    self.textureCache = textureCache
  }

  func resize(size: CGSize, textureSize: CGSize) {
    let ratio = Float(textureSize.whRatio)
    var referenceSize = size
    switch Renderer.aspectMode {
    case .scaleToFill:
      uniforms.scaleMatrix = matrix_identity_float4x4
    case .aspectFit:
      let scaleY = Float(size.width) / ratio / Float(size.height)
      uniforms.scaleMatrix = TransformMatrix.scaling([1, scaleY, 1])
      referenceSize.height *= CGFloat(scaleY)
      Self.bboxOffset.y = abs((referenceSize.height - size.height) / 2).rounded(.toNearestOrEven)
    case .aspectFill:
      let scaleX = Float(size.height) * ratio / Float(size.width)
      uniforms.scaleMatrix = TransformMatrix.scaling([scaleX, 1, 1])
      referenceSize.width *= CGFloat(scaleX)
      Self.bboxOffset.x = abs((referenceSize.width - size.width) / 2).rounded(.toNearestOrEven)
    }
    referenceSize.round()
    ObjectDetectionController.referenceSize = referenceSize
  }

  func render(
    _ frame: Frame,
    view: MTKView
  ) {
    guard
      let renderPassDescriptor = view.currentRenderPassDescriptor,
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
      let texture = frame.pixelBuffer.makeMTLTexture(usingTextureCache: textureCache, pixelFormat: .bgra8Unorm)
    else {
      return
    }

    renderPassDescriptor.colorAttachments[0].loadAction = .clear

    if Settings.isSegmentationEnabled && !frame.bboxes.isEmpty {
      renderEncoder.setRenderPipelineState(segmentationPipelineState)
    } else {
      renderEncoder.setRenderPipelineState(pipelineState)
    }
    renderEncoder.setVertexBuffer(backgroundPlane, offset: 0, index: 0)
    renderEncoder.setVertexBytes(&uniforms, length: MemoryLayout<VertexUniforms>.stride, index: 1)

    renderEncoder.setFragmentTexture(texture, index: 0)

    if Settings.isSegmentationEnabled && !frame.bboxes.isEmpty {
      renderEncoder.setFragmentTexture(frame.maskTexture, index: 1)
      var segmentationParams = SegmentationParams(
        confidence: Settings.segmentationMaskConfidence,
        bboxCount: UInt32(frame.bboxes.count)
      )
      renderEncoder.setFragmentBytes(&segmentationParams, length: MemoryLayout<SegmentationParams>.stride, index: 0)
      var bboxes = frame.bboxes
        .map {
        BBox(
          confidence: $0.confidence,
          x: $0.x - Int32(Renderer.bboxOffset.x),
          y: $0.y,
          w: $0.w,
          h: $0.h,
          classId: $0.classId
        )
      }
      renderEncoder.setFragmentBytes(&bboxes, length: MemoryLayout<BBox>.stride * bboxes.count, index: 1)
      if Self.colors.isEmpty {
        var colors = Array<simd_float3>(repeating: .zero, count: ObjectDetectionModel.classes.count)
        renderEncoder.setFragmentBytes(&colors, length: MemoryLayout<simd_float3>.stride * colors.count, index: 2)
      } else {
        renderEncoder.setFragmentBytes(
          &Self.colors, length: MemoryLayout<simd_float3>.stride * Self.colors.count, index: 2
        )
      }
    }

    renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    renderEncoder.endEncoding()

    guard let currentDrawable = view.currentDrawable else { return }
    commandBuffer.present(currentDrawable)
    commandBuffer.commit()
  }
}

private extension Renderer {
  static let kImagePlaneVertexData: [Float] = [
    -1.0, -1.0, 0.0, 1.0,
     1.0, -1.0, 1.0, 1.0,
     -1.0, 1.0, 0.0, 0.0,
     1.0, 1.0, 1.0, 0.0
  ]

  static func createPlaneVertexBuffer(for device: MTLDevice) -> MTLBuffer? {
    let imagePlaneVertexDataCount = MemoryLayout<Float>.stride * kImagePlaneVertexData.count
    let imagePlaneVertexBuffer = device.makeBuffer(
      bytes: kImagePlaneVertexData, length: imagePlaneVertexDataCount, options: []
    )
    imagePlaneVertexBuffer?.label = "ImagePlaneVertexBuffer"
    return imagePlaneVertexBuffer
  }

  static func createPipelineState(
    device: MTLDevice, fragmentFunctionName: String, library: MTLLibrary
  ) -> MTLRenderPipelineState? {
    let vertexFunction = library.makeFunction(name: "vertexBaseRendering")
    let fragmentFunction = library.makeFunction(name: fragmentFunctionName)

    let imagePlaneVertexDescriptor = MTLVertexDescriptor()

    // Positions
    let vertexAttributePosition = Int(kVertexAttributePosition.rawValue)
    imagePlaneVertexDescriptor.attributes[vertexAttributePosition].format = .float2
    imagePlaneVertexDescriptor.attributes[vertexAttributePosition].offset = 0
    imagePlaneVertexDescriptor.attributes[vertexAttributePosition].bufferIndex = 0

    // Texture coordinates
    let vertexAttributeUV = Int(kVertexAttributeUV.rawValue)
    imagePlaneVertexDescriptor.attributes[vertexAttributeUV].format = .float2
    imagePlaneVertexDescriptor.attributes[vertexAttributeUV].offset = MemoryLayout<Float>.stride * 2
    imagePlaneVertexDescriptor.attributes[vertexAttributeUV].bufferIndex = 0

    // Buffer Layout
    imagePlaneVertexDescriptor.layouts[0].stride = MemoryLayout<SIMD4<Float>>.stride
    imagePlaneVertexDescriptor.layouts[0].stepRate = 1
    imagePlaneVertexDescriptor.layouts[0].stepFunction = .perVertex

    let pipelineDescriptor = MTLRenderPipelineDescriptor()
    pipelineDescriptor.label = "RenderingPipeline-\(fragmentFunctionName)"
    pipelineDescriptor.vertexFunction = vertexFunction
    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexDescriptor = imagePlaneVertexDescriptor
    pipelineDescriptor.colorAttachments[0].pixelFormat = Renderer.colorPixelFormat
    do {
      return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    } catch {
      assertionFailure(error.localizedDescription)
      return nil
    }
  }
}
