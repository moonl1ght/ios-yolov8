//
// Created by moonl1ght 27.02.2023.
//

import Foundation

enum Benchmark {
  struct MeasureResult {
    static var zero: MeasureResult { .init(nanoSeconds: 0) }

    let nanoSeconds: UInt64

    var seconds: TimeInterval {
      TimeInterval(nanoSeconds) / TimeInterval(NSEC_PER_SEC)
    }

    func approxFPS() -> TimeInterval {
      1 / seconds
    }
  }

  @inlinable
  @inline(__always)
  @discardableResult
  static func measure(_ work: () throws -> Void) rethrows -> MeasureResult? {
    var info = mach_timebase_info()
    guard mach_timebase_info(&info) == KERN_SUCCESS else { return nil }

    let start = mach_absolute_time()
    try work()
    let end = mach_absolute_time()

    let elapsed = end - start

    let nanos = elapsed * UInt64(info.numer) / UInt64(info.denom)
    return .init(nanoSeconds: nanos)
  }

  @inlinable
  @inline(__always)
  public static var time: TimeInterval {
    ProcessInfo.processInfo.systemUptime
  }
}
