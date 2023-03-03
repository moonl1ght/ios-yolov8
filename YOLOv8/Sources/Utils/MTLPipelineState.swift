//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import MetalKit

enum MTLPipelineState {
  static func createCompute(
    library: MTLLibrary,
    device: MTLDevice,
    functionName: String,
    label: String? = nil
  ) -> MTLComputePipelineState {
    do {
      let computePipelineDescriptor = MTLComputePipelineDescriptor()
      let converterFunction = library.makeFunction(name: functionName)
      computePipelineDescriptor.computeFunction = converterFunction
      if let label {
        computePipelineDescriptor.label = label
      }
      computePipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
      return try device.makeComputePipelineState(
        descriptor: computePipelineDescriptor, options: [], reflection: nil
      )
    } catch {
      fatalError(error.localizedDescription)
    }
  }
}

