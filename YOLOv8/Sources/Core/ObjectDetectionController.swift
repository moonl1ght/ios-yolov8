//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit
import MetalPerformanceShaders

protocol ObjectDetectionDelegate: AnyObject {
  func didDetectFrame(_ frame: Frame)
}

final class ObjectDetectionController {
  static let modelType: ObjectDetectionModel.ModelType = .withSegmentation
  static let modelSize: ObjectDetectionModel.ModelSize = .small

  static var referenceSize = ObjectDetectionModel.inputSize

  enum Error: Swift.Error {
    case failedToProcess
  }

  weak var delegate: ObjectDetectionDelegate?

  private let device: MTLDevice = DeviceManager.device
  private let queue: DispatchQueue
  private let commandQueue: MTLCommandQueue
  private let imageScaler: MTLImageScaler
  private let segmentationImageScaler: MTLImageScaler
  private let model = ObjectDetectionModel()

  // Pipeline states
  private let filterBBoxPipelineState: MTLComputePipelineState
  private let computeThreadgroupsPerGridPipelineState: MTLComputePipelineState
  private let nmsPipelineState: MTLComputePipelineState
  private let cleanUpPipelineState: MTLComputePipelineState

  private let segmentationScaler: MPSImageScale

  // Buffers
  var bboxes: MTLBuffer?
  var keptBBoxMap: MTLBuffer?
  var maskProposals: MTLBuffer?

  init(queue: DispatchQueue) {
    self.queue = queue
    guard let commandQueue = device.makeCommandQueue(), let library = device.makeDefaultLibrary() else {
      fatalError()
    }
    self.commandQueue = commandQueue
    print(commandQueue)
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
      library: library, device: device, functionName: "NMS", label: "NMS"
    )
    cleanUpPipelineState = MTLPipelineState.createCompute(
      library: library, device: device, functionName: "cleanUpBuffers", label: "cleanUpBuffers"
    )
    segmentationScaler = MPSImageBilinearScale(device: device)
  }

  func load(_ completion: @escaping (Result<Void, Swift.Error>) -> Void) {
    queue.async { [self] in
      createBuffers()
      do {
        try model.load(modelType: Self.modelType, modelSize: Self.modelSize)
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

  private func encodeComputeThreadgroupsPerGrid(
    _ commandBuffer: MTLCommandBuffer,
    bboxCount: MTLBuffer,
    indirectBuffer: MTLBuffer
  ) throws {
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw Error.failedToProcess
    }
    computeEncoder.setComputePipelineState(computeThreadgroupsPerGridPipelineState)
    computeEncoder.setBuffer(bboxCount, offset: 0, index: 0)

    let threadgroupSize = nmsPipelineState.threadgroupSize
    var threadgroupWH = [Int32(threadgroupSize.width), Int32(threadgroupSize.height)]

    computeEncoder.setBytes(&threadgroupWH, length: MemoryLayout<Int32>.stride * 2, index: 1)

    computeEncoder.setBuffer(indirectBuffer, offset: 0, index: 2)

    computeEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: threadgroupSize)
    computeEncoder.endEncoding()
  }

  private func encodeNMS(
    _ commandBuffer: MTLCommandBuffer,
    bboxCount: MTLBuffer,
    bboxes: MTLBuffer,
    keptBBoxMap: MTLBuffer,
    indirectBuffer: MTLBuffer
  ) throws {
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw Error.failedToProcess
    }
    computeEncoder.setComputePipelineState(nmsPipelineState)

    var uniforms = NMSParams(iouThreshold: 0.2)
    computeEncoder.setBytes(&uniforms, length: MemoryLayout<NMSParams>.stride, index: 0)
    computeEncoder.setBuffer(bboxes, offset: 0, index: 1)
    computeEncoder.setBuffer(bboxCount, offset: 0, index: 2)
    computeEncoder.setBuffer(keptBBoxMap, offset: 0, index: 3)

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
      throw Error.failedToProcess
    }
    blitEncoder.fill(buffer: bboxes, range: 0 ..< bboxes.length, value: 0)
    blitEncoder.fill(buffer: keptBBoxMap, range: 0 ..< keptBBoxMap.length, value: 0)
    blitEncoder.endEncoding()
  }

  private func segmentation(
    _ maskProposalsMatrix: MPSMatrix,
    protosMatrix: MPSMatrix,
    proposalsCount: Int
  ) -> MTLTexture? {
    let segmentationMaskSizeLinearLenght = Int(ObjectDetectionModel.segmentationMaskSize.length)
    let matrixMultiplier = MPSMatrixMultiplication(
      device: device,
      resultRows: proposalsCount,
      resultColumns: segmentationMaskSizeLinearLenght,
      interiorColumns: ObjectDetectionModel.segmentationMaskLength
    )

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      assertionFailure()
      return nil
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

    guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
      assertionFailure()
      return nil
    }

    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.textureType = .type2DArray
    textureDescriptor.arrayLength = proposalsCount
    textureDescriptor.pixelFormat = .r32Float
    textureDescriptor.width = Int(ObjectDetectionModel.segmentationMaskSize.width)
    textureDescriptor.height = Int(ObjectDetectionModel.segmentationMaskSize.height)
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
      assertionFailure()
      return nil
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
}

extension ObjectDetectionController: CameraControllerDelegate {
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

        //        let bboxes = device.makeBuffer(length: MemoryLayout<BBox>.stride * ObjectDetectionModel.stide)
        //        let keptBBoxMap = device.makeBuffer(length: MemoryLayout<Int32>.stride * ObjectDetectionModel.stide)
        let maskProposals = device.makeBuffer(
          length: MemoryLayout<Float>.stride * ObjectDetectionModel.stide * ObjectDetectionModel.segmentationMaskLength
        )

        guard
          let bboxes,
          let maskProposals,
          let keptBBoxMap,
          let commandBuffer = commandQueue.makeCommandBuffer()
        else {
          assertionFailure()
          return
        }
        try encodeCleanUp(commandBuffer, bboxes: bboxes, keptBBoxMap: keptBBoxMap)
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
          confidenceThreshold: 0.45,
          stride: Int32(ObjectDetectionModel.stide),
          numberOfClasses: UInt32(ObjectDetectionModel.classes.count),
          segmentationMaskLength: UInt32(ObjectDetectionModel.segmentationMaskLength),
          hasSegmentationMask: 1
        )
        computeEncoder.setBytes(&uniforms, length: MemoryLayout<BBoxFilterParams>.stride, index: 0)
        computeEncoder.setBuffer(predictionBuffer, offset: 0, index: 1)
        let bboxCount = device.makeBuffer(length: MemoryLayout<Int32>.stride)
        computeEncoder.setBuffer(bboxCount, offset: 0, index: 2)
        computeEncoder.setBuffer(bboxes, offset: 0, index: 3)
        computeEncoder.setBuffer(maskProposals, offset: 0, index: 4)
        let threadgroupSize = filterBBoxPipelineState.threadgroupSize
        let threadsPerGrid = MTLSize(width: ObjectDetectionModel.stide, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        var inderectArguments = MTLDispatchThreadgroupsIndirectArguments(threadgroupsPerGrid: (0, 0, 0))
        guard
          let bboxCount,
          let inderectBuffer = device.makeBuffer(
            bytes: &inderectArguments, length: MemoryLayout<MTLDispatchThreadgroupsIndirectArguments>.stride
          )
        else {
          assertionFailure()
          return
        }
        try encodeComputeThreadgroupsPerGrid(commandBuffer, bboxCount: bboxCount, indirectBuffer: inderectBuffer)

        try encodeNMS(
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
        let maskPropsPTR = maskProposals.contents().assumingMemoryBound(to: Float.self)
        var keptProposals: [Float] = []

        var resultBBoxes: [BBox] = []
        let count = min(Int(bboxCountPtr[0]), Settings.maxDetectedBBoxes)
        resultBBoxes.reserveCapacity(count)
        for i in 0 ..< count where keptBBoxMapPtr[i] == bboxCountPtr[0] - 1 {
          resultBBoxes.append(bboxesPtr[i])
          for j in 0 ..< 32 {
            keptProposals.append(maskPropsPTR[i * 32 + j])
          }
        }

        if keptProposals.count == 64 {
          print(keptProposals[0 ..< 32])
          print(keptProposals[32 ..< keptProposals.count])
          print(keptProposals.count / 32)

        }
//        print(resultBBoxes.count)
//        print(keptProposals.count)
//        if keptProposals.count == 64 {
//          print(keptProposals)
//          print()
        if !keptProposals.isEmpty {
          let proposalsCount = resultBBoxes.count
          let keptProposalsBuffer = device.makeBuffer(bytes: &keptProposals, length: MemoryLayout<Float>.stride * keptProposals.count)
          let maskProposalsBufferDescr = MPSMatrixDescriptor(
            rows: proposalsCount, columns: 32, rowBytes: MemoryLayout<Float>.stride * 32, dataType: .float32
          )
          let maskProposalsMatrix = MPSMatrix(buffer: keptProposalsBuffer!, descriptor: maskProposalsBufferDescr)

          let protosBufferDescr = MPSMatrixDescriptor(
            rows: 32, columns: 160 * 160, rowBytes: MemoryLayout<Float>.stride * 160 * 160, dataType: .float32
          )

          let maskTexture = prediction.proto!.withUnsafeMutableBytes { ptr, strides in

            let fp = ptr.assumingMemoryBound(to: Float.self)

            let protosBuffer = device.makeBuffer(
              bytes: fp.baseAddress!,
              length: MemoryLayout<Float>.stride * prediction.proto!.count,
              options: [.storageModeShared]
            )

            let protosMatrix = MPSMatrix(buffer: protosBuffer!, descriptor: protosBufferDescr)
            return segmentation(maskProposalsMatrix, protosMatrix: protosMatrix, proposalsCount: proposalsCount)
          }
          frame.maskTexture = maskTexture
//          frame.bboxesBuffer = bboxes
        }
//        }

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
