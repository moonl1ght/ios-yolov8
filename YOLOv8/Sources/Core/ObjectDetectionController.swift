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
//  private let matrixMultiplier: MPSMatrixMultiplication
//  private let sigmoidPipelineState: MTLComputePipelineState

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
//    sigmoidPipelineState = MTLPipelineState.createCompute(
//      library: library, device: device, functionName: "sigmoid", label: "sigmoid"
//    )
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
    protosMatrix: MPSMatrix
  ) -> [MTLTexture] {
    let matrixMultiplier = MPSMatrixMultiplication(
      device: device, resultRows: 2, resultColumns: 160 * 160, interiorColumns: 32
    )
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      assertionFailure()
      return []
    }
    let resdescr = MPSMatrixDescriptor(rows: 2, columns: 160 * 160, rowBytes: MemoryLayout<Float>.stride * 160 * 160, dataType: .float32)
    let resultMatrix = MPSMatrix(device: device, descriptor: resdescr)
    matrixMultiplier.encode(commandBuffer: commandBuffer, leftMatrix: maskProposalsMatrix, rightMatrix: protosMatrix, resultMatrix: resultMatrix)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let pptr = resultMatrix.data.contents().assumingMemoryBound(to: Float.self)

//    var zeros1 = 0
//    for i in 0 ..< 160 * 160 {
//      if pptr[i] == 0 {
//        zeros1 += 1
//      }
//      print(pptr[i])
//    }
//    print("ZEROS: \(zeros1)")

//    var zeros = 0
//    for i in 160 * 160 ..< 2 * 160 * 160 {
//      if pptr[i] == 0 {
//        zeros += 1
//      }
////      print(pptr[i])
//    }
//    print("ZEROS2: \(zeros)")
//    print()

    //    let bpttr = UnsafeMutableBufferPointer(start: pptr, count: 160 * 160)
    //    let imgb = bpttr[0 ..< 160 * 160]

//    var pixelBuffer: CVPixelBuffer?
//
//    let outputBufferAttributes: [String: Any] = [
//      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_OneComponent32Float,
//      kCVPixelBufferWidthKey as String: 160,
//      kCVPixelBufferHeightKey as String: 160,
//      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
//    ]
    //    guard let baseAddress = bpttr.baseAddress else { return }
//    CVPixelBufferCreateWithBytes(
//      kCFAllocatorDefault, 160, 160,
//      kCVPixelFormatType_OneComponent32Float,
//      pptr,
//      MemoryLayout<Float>.stride * 160, nil, nil, outputBufferAttributes as NSDictionary,
//      &pixelBuffer
//    )
//
//    let depthImage1 = CIImage(cvPixelBuffer: pixelBuffer!)
//    let img1 = UIImage(ciImage: depthImage1)
//
    guard let commandBuffer1 = commandQueue.makeCommandBuffer() else {
      return []
    }

    //    let texture = pixelBuffer?.makeMTLTexture(usingTextureCache: textureCache, pixelFormat: .depth32Float)
    //    let texture = try? segmentationImageScaler.rescale(pixelBuffer!, commandBuffer: commandBuffer1, pixelFormat: .r32Float).texture

    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.textureType = .type2DArray
    textureDescriptor.arrayLength = 2
    textureDescriptor.pixelFormat = .r32Float
//    textureDescriptor.depth = 1
    textureDescriptor.width = 160
    textureDescriptor.height = 160

    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    var textures: [MTLTexture] = []

    let blitEncoder = commandBuffer1.makeBlitCommandEncoder()!

    let tex = device.makeTexture(descriptor: textureDescriptor)

    blitEncoder.copy(
      from: resultMatrix.data,
      sourceOffset: 0,
      sourceBytesPerRow: MemoryLayout<Float>.stride * 160,
      sourceBytesPerImage: MemoryLayout<Float>.stride * 160 * 160,
      sourceSize: .init(width: 160, height: 160, depth: 1),
      to: tex!,
      destinationSlice: 1,
      destinationLevel: 0,
      destinationOrigin: .init(x: 0, y: 0, z: 0)
    )
//    blitEncoder.copy(
//      from: resultMatrix.data,
//      sourceOffset: MemoryLayout<Float>.stride * 160 * 160,
//      sourceBytesPerRow: MemoryLayout<Float>.stride * 160,
//      sourceBytesPerImage: MemoryLayout<Float>.stride * 160 * 160,
//      sourceSize: .init(width: 160, height: 160, depth: 1),
//      to: tex!,
//      destinationSlice: 0,
//      destinationLevel: 0,
//      destinationOrigin: .init(x: 0, y: 0, z: 0)
//    )
    blitEncoder.endEncoding()

//    for i in 0..<2 {
//      let oldtexture = resultMatrix.data.makeTexture(
//        descriptor: textureDescriptor, offset: i * MemoryLayout<Float>.stride * 160 * 160, bytesPerRow: MemoryLayout<Float>.stride * 160
//      )!
//      textures.append(oldtexture)
//    }

    commandBuffer1.commit()
    commandBuffer1.waitUntilCompleted()

    textures.append(tex!)

//    let oldtexture = resultMatrix.data.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: MemoryLayout<Float>.stride * 160)

//    let texture = try? segmentationImageScaler.rescale(commandBuffer1, texture: oldtexture!, pixelFormat: .r32Float)

    return textures
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
          for j in 0 ..< 32 {
            print(maskPropsPTR[i * 32 + j])
          }
          resultBBoxes.append(bboxesPtr[i])
          if keptProposals.count < 64 {
            for j in 0 ..< 32 {
              keptProposals.append(maskPropsPTR[i * 32 + j])
            }
          }
        }
        print(resultBBoxes.count)
        print(keptProposals.count)
        if keptProposals.count == 64 {
          print(keptProposals)
          print()
          let keptProposalsBuffer = device.makeBuffer(bytes: &keptProposals, length: MemoryLayout<Float>.stride * keptProposals.count)
          let maskProposalsBufferDescr = MPSMatrixDescriptor(
            rows: 2, columns: 32, rowBytes: MemoryLayout<Float>.stride * 32, dataType: .float32
          )
          let maskProposalsMatrix = MPSMatrix(buffer: keptProposalsBuffer!, descriptor: maskProposalsBufferDescr)

          let protosBufferDescr = MPSMatrixDescriptor(
            rows: 32, columns: 160 * 160, rowBytes: MemoryLayout<Float>.stride * 160 * 160, dataType: .float32
          )

          let textures = prediction.proto!.withUnsafeMutableBytes { ptr, strides in

            let fp = ptr.assumingMemoryBound(to: Float.self)

            let protosBuffer = device.makeBuffer(
              bytes: fp.baseAddress!,
              length: MemoryLayout<Float>.stride * prediction.proto!.count,
              options: [.storageModeShared]
            )

            let protosMatrix = MPSMatrix(buffer: protosBuffer!, descriptor: protosBufferDescr)
            return segmentation(maskProposalsMatrix, protosMatrix: protosMatrix)
          }
          frame.maskTextures = textures
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
