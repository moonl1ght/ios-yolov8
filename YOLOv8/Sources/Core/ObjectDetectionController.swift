//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit

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
  private let model = ObjectDetectionModel()

  // Pipeline states
  private let filterBBoxPipelineState: MTLComputePipelineState
  private let computeThreadgroupsPerGridPipelineState: MTLComputePipelineState
  private let nmsPipelineState: MTLComputePipelineState
  private let cleanUpPipelineState: MTLComputePipelineState

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
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw Error.failedToProcess
    }
    computeEncoder.setComputePipelineState(cleanUpPipelineState)
    computeEncoder.setBuffer(bboxes, offset: 0, index: 0)
    computeEncoder.setBuffer(keptBBoxMap, offset: 0, index: 1)
    let threadgroupSize = cleanUpPipelineState.threadgroupSize
    let threadsPerGrid = MTLSize(width: ObjectDetectionModel.stide, height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadgroupSize)
    computeEncoder.endEncoding()
  }

  private func segmentation(
    _ maskProposalsMatrix: MPSMatrix,
    protosMatrix: MPSMatrix,
    bbox: inout NMSBBox
  ) -> MTLCVTexture? {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      return nil
    }
    let resdescr = MPSMatrixDescriptor(rows: 1, columns: 160 * 160, rowBytes: MemoryLayout<Float>.stride * 160 * 160, dataType: .float32)
    let resultMatrix = MPSMatrix(device: device, descriptor: resdescr)
    matrixMultiplier.encode(commandBuffer: commandBuffer, leftMatrix: maskProposalsMatrix, rightMatrix: protosMatrix, resultMatrix: resultMatrix)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let pptr = resultMatrix.data.contents().assumingMemoryBound(to: Float.self)

    //    let bpttr = UnsafeMutableBufferPointer(start: pptr, count: 160 * 160)
    //    let imgb = bpttr[0 ..< 160 * 160]

    var pixelBuffer: CVPixelBuffer?

    let outputBufferAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_OneComponent32Float,
      kCVPixelBufferWidthKey as String: 160,
      kCVPixelBufferHeightKey as String: 160,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
    ]
    //    guard let baseAddress = bpttr.baseAddress else { return }
    CVPixelBufferCreateWithBytes(
      kCFAllocatorDefault, 160, 160,
      kCVPixelFormatType_OneComponent32Float,
      pptr,
      MemoryLayout<Float>.stride * 160, nil, nil, outputBufferAttributes as NSDictionary,
      &pixelBuffer
    )

    let depthImage1 = CIImage(cvPixelBuffer: pixelBuffer!)
    let img1 = UIImage(ciImage: depthImage1)

    guard let commandBuffer1 = commandQueue.makeCommandBuffer() else {
      return nil
    }

    //    let texture = pixelBuffer?.makeMTLTexture(usingTextureCache: textureCache, pixelFormat: .depth32Float)
    //    let texture = try? segmentationImageScaler.rescale(pixelBuffer!, commandBuffer: commandBuffer1, pixelFormat: .r32Float).texture

    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.textureType = .type2D
    //    textureDescriptor.arrayLength = 2
    textureDescriptor.pixelFormat = .r32Float
    textureDescriptor.width = 160
    textureDescriptor.height = 160
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    let oldtexture = resultMatrix.data.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: MemoryLayout<Float>.stride * 160)

    let texture = try? segmentationImageScaler.rescale(oldtexture!, commandBuffer: commandBuffer1, pixelFormat: .r32Float)
    //
    ////    TextureLoader
    //
    //    let texture = resultMatrix.
    //
    //    guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
    //    blitEncoder.copy(
    //      from: resultMatrix.data,
    //      sourceOffset: 0,
    //      sourceBytesPerRow: MemoryLayout<Float>.stride * 160,
    //      sourceBytesPerImage: MemoryLayout<Float>.stride * 160 * 160,
    //      sourceSize: .init(width: 160, height: 160, depth: 1),
    //      to: texture!,
    //      destinationSlice: 0, destinationLevel: 0, destinationOrigin: .init(x: 0, y: 0, z: 0))

    guard let computeEncoder = commandBuffer1.makeComputeCommandEncoder() else { return nil }
    computeEncoder.setComputePipelineState(sigmoidPipelineState)
    computeEncoder.setTexture(texture?.texture, index: 0)
    computeEncoder.setBytes(&bbox, length: MemoryLayout<NMSBBox>.stride, index: 0)
    //    let sigmoidOutput = device.makeBuffer(length: MemoryLayout<Float>.stride * 2 * 160 * 160)
    //    computeEncoder.setBuffer(sigmoidOutput, offset: 0, index: 1)
    let threadsPerGrid = sigmoidPipelineState.calculateThreadsPerGrid(for: texture!.texture)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: sigmoidPipelineState.threadgroupSize)
    computeEncoder.endEncoding()

    commandBuffer1.commit()
    commandBuffer1.waitUntilCompleted()

    //
    //    let pptr = sigmoidOutput?.contents().assumingMemoryBound(to: Float.self)
    //
    //    let bpttr = UnsafeMutableBufferPointer(start: pptr, count: 160 * 160)
    //    let imgb = bpttr[0 ..< 160 * 160]
    //
    //    var pixelBuffer1: CVPixelBuffer?
    //
    //    let outputBufferAttributes: [String: Any] = [
    //      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_DepthFloat32,
    //      kCVPixelBufferWidthKey as String: 160,
    //      kCVPixelBufferHeightKey as String: 160,
    //      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
    //    ]
    //    guard let baseAddress = UnsafeMutableBufferPointer(rebasing: imgb).baseAddress else { return }
    //    CVPixelBufferCreateWithBytes(
    //      kCFAllocatorDefault, 160, 160,
    //      kCVPixelFormatType_DepthFloat32,
    //      baseAddress,
    //      MemoryLayout<Float>.stride * 160, nil, nil, outputBufferAttributes as NSDictionary,
    //      &pixelBuffer1
    //    )
    //    let depthImage = CIImage(cvPixelBuffer: texture!.pixelBuffer)
    //    let img = UIImage(ciImage: depthImage)
    //    try body(pixelBuffer)


    return texture
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
//        let maskProposals = device.makeBuffer(
//          length: MemoryLayout<Float>.stride * ObjectDetectionModel.stide * ObjectDetectionModel.segmentationMaskLength
//        )

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
          hasSegmentationMask: 0
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

        var resultBBoxes: [BBox] = []
        let count = max(Int(bboxCountPtr[0]), Settings.maxDetectedBBoxes)
        resultBBoxes.reserveCapacity(count)
        for i in 0 ..< count where keptBBoxMapPtr[i] == bboxCountPtr[0] - 1 {
          resultBBoxes.append(bboxesPtr[i])
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
