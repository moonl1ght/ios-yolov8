//
// Created by moonl1ght 27.02.2023.
//

import SwiftUI

struct MainView: View {
  @State private var showingObjectDetection = false
  @State private var showingObjectSegmentation = false

  var body: some View {
    VStack(spacing: 20) {
      button(title: "Object detection") {
        showingObjectDetection.toggle()
      }
      button(title: "Object segmentation") {
        showingObjectSegmentation.toggle()
      }
    }
    .fullScreenCover(isPresented: $showingObjectDetection) {
      ObjectDetectionView(modelType: .normal)
    }
    .fullScreenCover(isPresented: $showingObjectSegmentation) {
      ObjectDetectionView(modelType: .withSegmentation)
    }
    .preferredColorScheme(.light)
  }

  @ViewBuilder
  private func button(title: String, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      HStack {
        Text(title).font(.system(size: 20)).bold()
      }
      .frame(width: 250, height: 62)
      .foregroundColor(.white)
      .background(
        RoundedRectangle(cornerRadius: 15, style: .continuous)
          .fill(Color.blue)
      )
    }
  }
}
