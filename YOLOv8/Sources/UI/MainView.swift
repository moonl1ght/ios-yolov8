//
// Created by moonl1ght 27.02.2023.
//

import SwiftUI

struct MainView: View {
  @State private var showingObjectDetection = false

  var body: some View {
    VStack {
      button(title: "Object detection") {
        showingObjectDetection.toggle()
      }
    }
    .fullScreenCover(isPresented: $showingObjectDetection) {
      ObjectDetectionView()
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
