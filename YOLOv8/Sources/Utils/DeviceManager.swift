//
// Created by moonl1ght 27.02.2023.
//

import MetalKit

enum DeviceManager {
  static let device: MTLDevice = {
    guard let device = MTLCreateSystemDefaultDevice() else {
      fatalError("No GPU device")
    } 
    return device
  }()
}
