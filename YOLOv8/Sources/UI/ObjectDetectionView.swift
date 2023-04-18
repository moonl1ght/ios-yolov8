//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import AVFoundation
import SwiftUI

struct ObjectDetectionView: View {
  @Environment(\.dismiss) var dismiss
  @StateObject private var presenter: Presenter
  @State private var isLoaded = false
  @State private var loadingFailed = false

  init(modelType: ObjectDetectionModel.ModelType) {
    _presenter = StateObject(wrappedValue: Presenter(modelType: modelType))
  }

  var body: some View {
    ZStack {
      if isLoaded {
        PreviewView(presenter: presenter)
          .ignoresSafeArea()
      } else if loadingFailed {
        Text("Model loading failed")
      } else {
        Text("Loading model...")
      }
      VStack {
        HStack {
          button(systemName: "xmark", action: { dismiss() })
          Spacer()
        }
        .padding()
        Spacer()
        frameInfoView()
      }
    }
    .onAppear {
      presenter.objectDetector.load { result in
        switch result {
        case .success:
          isLoaded = true
        case .failure:
          loadingFailed = true
        }
      }
    }
  }

  @ViewBuilder
  private func frameInfoView() -> some View {
    if let frameInfo = presenter.frameInfo {
      VStack {
        Text(frameInfo.predictionDuration)
        Text(frameInfo.processingDuration)
        Text(frameInfo.fullProcessingDuration)
        Text(frameInfo.detectedBBoxes)
      }
      .foregroundColor(.white)
      .font(.caption)
      .padding(10)
      .background(
        RoundedRectangle(cornerRadius: 20, style: .continuous)
          .fill(Color.black.opacity(0.6))
      )
    } else {
      EmptyView()
    }
  }

  @ViewBuilder
  private func button(
    systemName: String,
    fillColor: Color = .black.opacity(0.6),
    action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      Image(systemName: systemName)
        .frame(width: 45, height: 45)
        .foregroundColor(.white)
        .background(
          RoundedRectangle(cornerRadius: 20, style: .continuous)
            .fill(Color.black.opacity(0.6))
        )
    }
  }
}
