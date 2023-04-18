//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit
import MetalPerformanceShaders
import CoreML

protocol ObjectDetectorDelegate: AnyObject {
  func didDetectFrame(_ frame: Frame)
}

final class ObjectDetector {
  static var referenceSize = ObjectDetectionModel.inputSize

  enum Error: Swift.Error {
    case failedToEncodeFilterBBoxes
    case failedToEncodeNonMaximumSuppression
    case failedToEncodeCleanUp
    case failedToEncodeComputeThreadgroupsPerGrid
    case predictionProcessingStageFailed
    case segmentationProcessingStageFailed
    case failedToEncodeSegmentation
  }

  weak var delegate: ObjectDetectorDelegate?

  private let device: MTLDevice = DeviceManager.device
  private let queue: DispatchQueue
  private let commandQueue: MTLCommandQueue
  private let imageScaler: MTLImageScaler
  private let segmentationImageScaler: MTLImageScaler
  private let model = ObjectDetectionModel()
  private let modelType: ObjectDetectionModel.ModelType
  private let modelSize: ObjectDetectionModel.ModelSize

  // Pipeline states
  private let filterBBoxPipelineState: MTLComputePipelineState
  private let computeThreadgroupsPerGridPipelineState: MTLComputePipelineState
  private let nmsPipelineState: MTLComputePipelineState

  private let segmentationScaler: MPSImageScale

  // Buffers
  var bboxes: MTLBuffer?
  var keptBBoxMap: MTLBuffer?
  var maskProposals: MTLBuffer?

  init(
    queue: DispatchQueue,
    modelType: ObjectDetectionModel.ModelType,
    modelSize: ObjectDetectionModel.ModelSize
  ) {
    self.queue = queue
    self.modelType = modelType
    self.modelSize = modelSize
    guard let commandQueue = device.makeCommandQueue(), let library = device.makeDefaultLibrary() else {
      fatalError()
    }
    self.commandQueue = commandQueue
    let textureCache = CVMetalTextureCache.createUsingDevice(device)
    imageScaler = MTLImageScaler(
      device: device, textureCache: textureCache, rescaledSize: ObjectDetectionModel.inputSize
    )
    segmentationImageScaler = MTLImageScaler(
      device: device, textureCache: textureCache, rescaledSize: ObjectDetectionModel.inputSize
    )
    filterBBoxPipelineState = MTLPipelineState.createCompute(
      library: library, device: device, functionName: "filterBBoxes", label: "filterBBoxes"
    )
    computeThreadgroupsPerGridPipelineState = MTLPipelineState.createCompute(
      library: library, device: device, functionName: "computeThreadgroupsPerGrid", label: "computeThreadgroupsPerGrid"
    )
    nmsPipelineState = MTLPipelineState.createCompute(
      library: library, device: device, functionName: "nonMaximumSuppression", label: "nonMaximumSuppression"
    )
    segmentationScaler = MPSImageBilinearScale(device: device)
  }

  func load(_ completion: @escaping (Result<Void, Swift.Error>) -> Void) {
    queue.async { [self] in
      createBuffers()
      do {
        try model.load(modelType: modelType, modelSize: modelSize)
        DispatchQueue.main.async {
          completion(.success(()))
        }
      } catch {
        DispatchQueue.main.async {
          completion(.failure(error))
        }
      }
    }
  }

  private func createBuffers() {
    bboxes = device.makeBuffer(length: MemoryLayout<BBox>.stride * ObjectDetectionModel.stide)
    keptBBoxMap = device.makeBuffer(length: MemoryLayout<Int32>.stride * ObjectDetectionModel.stide)
    maskProposals = device.makeBuffer(
      length: MemoryLayout<Float>.stride * ObjectDetectionModel.stide * ObjectDetectionModel.segmentationMaskLength
    )
  }

  private func encodeFilterBBoxes(
    _ commandBuffer: MTLCommandBuffer,
    prediction: ObjectDetectionModel.Output,
    bboxes: MTLBuffer,
    bboxCount: MTLBuffer,
    maskProposals: MTLBuffer
  ) throws {
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
    let predictionBuffer: MTLBuffer? = prediction.output.withUnsafeMutableBytes { ptr, strides in
      guard let fptr = ptr.assumingMemoryBound(to: Float.self).baseAddress else {
        return nil
      }
      return device.makeBuffer(
        bytes: fptr,
        length: MemoryLayout<Float>.stride * prediction.output.count,
        options: [.storageModeShared]
      )
    }
    computeEncoder.setComputePipelineState(filterBBoxPipelineState)
    var uniforms = BBoxFilterParams(
      factor: .init(
        x: Float(Self.referenceSize.width / ObjectDetectionModel.inputSize.width),
        y: Float(Self.referenceSize.height / ObjectDetectionModel.inputSize.height)
      ),
      confidenceThreshold: Settings.confidenceThreshold,
      stride: Int32(ObjectDetectionModel.stide),
      numberOfClasses: UInt32(ObjectDetectionModel.classes.count),
      segmentationMaskLength: UInt32(ObjectDetectionModel.segmentationMaskLength),
      hasSegmentationMask: modelType == .withSegmentation ? 1 : 0
    )
    computeEncoder.setBytes(&uniforms, length: MemoryLayout<BBoxFilterParams>.stride, index: iParams.index)
    computeEncoder.setBuffer(predictionBuffer, offset: 0, index: iPrediction.index)
    computeEncoder.setBuffer(bboxCount, offset: 0, index: iBBoxCount.index)
    computeEncoder.setBuffer(bboxes, offset: 0, index: iBBoxes.index)
    computeEncoder.setBuffer(maskProposals, offset: 0, index: iMaskProposals.index)
    let threadgroupSize = filterBBoxPipelineState.threadgroupSize
    let threadsPerGrid = MTLSize(width: ObjectDetectionModel.stide, height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadgroupSize)
    computeEncoder.endEncoding()
  }

  private func encodeComputeThreadgroupsPerGrid(
    _ commandBuffer: MTLCommandBuffer,
    bboxCount: MTLBuffer,
    indirectBuffer: MTLBuffer
  ) throws {
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw Error.failedToEncodeComputeThreadgroupsPerGrid
    }
    computeEncoder.setComputePipelineState(computeThreadgroupsPerGridPipelineState)
    computeEncoder.setBuffer(bboxCount, offset: 0, index: iBBoxCount.index)

    let threadgroupSize = nmsPipelineState.threadgroupSize
    var threadgroupWH = [Int32(threadgroupSize.width), Int32(threadgroupSize.height)]

    computeEncoder.setBytes(&threadgroupWH, length: MemoryLayout<Int32>.stride * 2, index: iThreadgroupSize.index)
    computeEncoder.setBuffer(indirectBuffer, offset: 0, index: iThreadgroupsPerGrid.index)

    computeEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: threadgroupSize)
    computeEncoder.endEncoding()
  }

  private func encodeNonMaximumSuppression(
    _ commandBuffer: MTLCommandBuffer,
    bboxCount: MTLBuffer,
    bboxes: MTLBuffer,
    keptBBoxMap: MTLBuffer,
    indirectBuffer: MTLBuffer
  ) throws {
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw Error.failedToEncodeNonMaximumSuppression
    }
    computeEncoder.setComputePipelineState(nmsPipelineState)

    var uniforms = NMSParams(iouThreshold: Settings.iouThreshold)
    computeEncoder.setBytes(&uniforms, length: MemoryLayout<NMSParams>.stride, index: iParams.index)
    computeEncoder.setBuffer(bboxes, offset: 0, index: iBBoxes.index)
    computeEncoder.setBuffer(bboxCount, offset: 0, index: iBBoxCount.index)
    computeEncoder.setBuffer(keptBBoxMap, offset: 0, index: iKeptBBoxMap.index)

    let threadgroupSize = nmsPipelineState.threadgroupSize
    computeEncoder.dispatchThreadgroups(
      indirectBuffer: indirectBuffer, indirectBufferOffset: 0, threadsPerThreadgroup: threadgroupSize
    )
    computeEncoder.endEncoding()
  }

  func encodeCleanUp(
    _ commandBuffer: MTLCommandBuffer,
    bboxes: MTLBuffer,
    keptBBoxMap: MTLBuffer
  ) throws {
    guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
      throw Error.failedToEncodeCleanUp
    }
    blitEncoder.fill(buffer: bboxes, range: 0 ..< bboxes.length, value: 0)
    blitEncoder.fill(buffer: keptBBoxMap, range: 0 ..< keptBBoxMap.length, value: 0)
    blitEncoder.endEncoding()
  }

  private func encodeSegmentation(
    _ maskProposalsMatrix: MPSMatrix,
    protosMatrix: MPSMatrix,
    proposalsCount: Int
  ) throws -> MTLTexture {
    let segmentationMaskSizeLinearLenght = Int(ObjectDetectionModel.segmentationMaskSize.length)
    let matrixMultiplier = MPSMatrixMultiplication(
      device: device,
      resultRows: proposalsCount,
      resultColumns: segmentationMaskSizeLinearLenght,
      interiorColumns: ObjectDetectionModel.segmentationMaskLength
    )

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      throw Error.failedToEncodeSegmentation
    }

    let resultMatrxiDescriptor = MPSMatrixDescriptor(
      rows: proposalsCount,
      columns: segmentationMaskSizeLinearLenght,
      rowBytes: MemoryLayout<Float>.stride * segmentationMaskSizeLinearLenght,
      dataType: .float32
    )
    let resultMatrix = MPSMatrix(device: device, descriptor: resultMatrxiDescriptor)
    matrixMultiplier.encode(
      commandBuffer: commandBuffer,
      leftMatrix: maskProposalsMatrix,
      rightMatrix: protosMatrix,
      resultMatrix: resultMatrix
    )

    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.textureType = .type2DArray
    textureDescriptor.arrayLength = proposalsCount
    textureDescriptor.pixelFormat = .r32Float
    textureDescriptor.width = Int(ObjectDetectionModel.segmentationMaskSize.width)
    textureDescriptor.height = Int(ObjectDetectionModel.segmentationMaskSize.height)
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard
      let blitEncoder = commandBuffer.makeBlitCommandEncoder(),
      let texture = device.makeTexture(descriptor: textureDescriptor)
    else {
      throw Error.failedToEncodeSegmentation
    }

    for i in 0 ..< proposalsCount {
      blitEncoder.copy(
        from: resultMatrix.data,
        sourceOffset: MemoryLayout<Float>.stride * segmentationMaskSizeLinearLenght * i,
        sourceBytesPerRow: MemoryLayout<Float>.stride * Int(ObjectDetectionModel.segmentationMaskSize.width),
        sourceBytesPerImage: MemoryLayout<Float>.stride * segmentationMaskSizeLinearLenght,
        sourceSize: ObjectDetectionModel.segmentationMaskSize.mtlSize,
        to: texture,
        destinationSlice: i,
        destinationLevel: 0,
        destinationOrigin: .init(x: 0, y: 0, z: 0)
      )
    }
    blitEncoder.endEncoding()

    let upscaledTextureDescriptor = MTLTextureDescriptor()
    upscaledTextureDescriptor.textureType = .type2DArray
    upscaledTextureDescriptor.arrayLength = proposalsCount
    upscaledTextureDescriptor.pixelFormat = .r32Float
    upscaledTextureDescriptor.width = Int(Self.referenceSize.width)
    upscaledTextureDescriptor.height = Int(Self.referenceSize.height)
    upscaledTextureDescriptor.usage = [.shaderRead, .shaderWrite]

    let upscaledTexture = device.makeTexture(descriptor: upscaledTextureDescriptor)
    if let upscaledTexture {
      segmentationScaler.encode(
        commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: upscaledTexture
      )
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return upscaledTexture ?? texture
  }

  private func predictionProcessingStage(
    prediction: ObjectDetectionModel.Output
  ) throws -> (resultBBoxes: [BBox], keptProposals: [Float]) {
    var inderectArguments = MTLDispatchThreadgroupsIndirectArguments(threadgroupsPerGrid: (0, 0, 0))
    let bboxCount = device.makeBuffer(length: MemoryLayout<Int32>.stride)

    guard
      let bboxes,
      let maskProposals,
      let keptBBoxMap,
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let bboxCount,
      let inderectBuffer = device.makeBuffer(
        bytes: &inderectArguments, length: MemoryLayout<MTLDispatchThreadgroupsIndirectArguments>.stride
      )
    else {
      throw Error.predictionProcessingStageFailed
    }
    try encodeCleanUp(commandBuffer, bboxes: bboxes, keptBBoxMap: keptBBoxMap)
    try encodeFilterBBoxes(
      commandBuffer,
      prediction: prediction,
      bboxes: bboxes,
      bboxCount: bboxCount,
      maskProposals: maskProposals
    )
    try encodeComputeThreadgroupsPerGrid(commandBuffer, bboxCount: bboxCount, indirectBuffer: inderectBuffer)
    try encodeNonMaximumSuppression(
      commandBuffer,
      bboxCount: bboxCount,
      bboxes: bboxes,
      keptBBoxMap: keptBBoxMap,
      indirectBuffer: inderectBuffer
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let bboxCountPtr = bboxCount.contents().assumingMemoryBound(to: Int32.self)
    let keptBBoxMapPtr = keptBBoxMap.contents().assumingMemoryBound(to: Int32.self)
    let bboxesPtr = bboxes.contents().assumingMemoryBound(to: BBox.self)
    let maskPropsPtr = maskProposals.contents().assumingMemoryBound(to: Float.self)
    var keptProposals: [Float] = []

    var resultBBoxes: [BBox] = []
    let count = min(Int(bboxCountPtr[0]), Settings.maxDetectedBBoxes)
    resultBBoxes.reserveCapacity(count)
    for i in 0 ..< count where keptBBoxMapPtr[i] == bboxCountPtr[0] - 1 {
      resultBBoxes.append(bboxesPtr[i])
      for j in 0 ..< ObjectDetectionModel.segmentationMaskLength {
        keptProposals.append(maskPropsPtr[i * ObjectDetectionModel.segmentationMaskLength + j])
      }
    }
    return (resultBBoxes: resultBBoxes, keptProposals: keptProposals)
  }

  private func segmentationProcessingStage(
    proposalsCount: Int,
    keptProposalsBuffer: MTLBuffer,
    proto: MLMultiArray
  ) throws -> MTLTexture {
    let maskProposalsBufferDescr = MPSMatrixDescriptor(
      rows: proposalsCount,
      columns: ObjectDetectionModel.segmentationMaskLength,
      rowBytes: MemoryLayout<Float>.stride * ObjectDetectionModel.segmentationMaskLength,
      dataType: .float32
    )
    let maskProposalsMatrix = MPSMatrix(buffer: keptProposalsBuffer, descriptor: maskProposalsBufferDescr)
    let columns = Int(ObjectDetectionModel.segmentationMaskSize.width * ObjectDetectionModel.segmentationMaskSize.height)
    let protosBufferDescr = MPSMatrixDescriptor(
      rows: ObjectDetectionModel.segmentationMaskLength,
      columns: columns,
      rowBytes: MemoryLayout<Float>.stride * columns,
      dataType: .float32
    )
    return try proto.withUnsafeMutableBytes { ptr, strides in
      guard
        let baseAddress = ptr.assumingMemoryBound(to: Float.self).baseAddress,
        let protosBuffer = device.makeBuffer(
          bytes: baseAddress,
          length: MemoryLayout<Float>.stride * proto.count,
          options: [.storageModeShared]
        )
      else {
        throw Error.segmentationProcessingStageFailed
      }
      let protosMatrix = MPSMatrix(buffer: protosBuffer, descriptor: protosBufferDescr)
      return try encodeSegmentation(maskProposalsMatrix, protosMatrix: protosMatrix, proposalsCount: proposalsCount)
    }
  }
}

extension ObjectDetector: CameraControllerDelegate {
  func cameraController(didOutput pixelBuffer: CVPixelBuffer) {
    do {
      let frame = Frame(pixelBuffer: pixelBuffer)
      let processingDuration = try Benchmark.measure {
        let texture = try imageScaler.rescale(pixelBuffer, commandQueue: commandQueue)
        var prediction: ObjectDetectionModel.Output?
        let predictionDuration = Benchmark.measure {
          prediction = model.predict(image: texture.pixelBuffer)
        }
        frame.predictionDuration = predictionDuration ?? .zero
        guard let prediction else {
          assertionFailure()
          return
        }
        var (resultBBoxes, keptProposals) = try predictionProcessingStage(prediction: prediction)
        if
          modelType == .withSegmentation,
          !keptProposals.isEmpty,
          let keptProposalsBuffer = device.makeBuffer(
            bytes: &keptProposals, length: MemoryLayout<Float>.stride * keptProposals.count
          ),
          let proto = prediction.proto
        {
          let maskTexture = try segmentationProcessingStage(
            proposalsCount: resultBBoxes.count,
            keptProposalsBuffer: keptProposalsBuffer,
            proto: proto
          )
          frame.maskTexture = maskTexture
        }
        frame.bboxes = resultBBoxes
      }
      frame.processingDuration = processingDuration ?? .zero
      DispatchQueue.main.async { [self] in
        delegate?.didDetectFrame(frame)
      }
    } catch {
      assertionFailure(error.localizedDescription)
    }
  }
}
