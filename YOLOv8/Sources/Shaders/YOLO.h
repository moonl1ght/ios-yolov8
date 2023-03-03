//
// Created by moonl1ght 27.02.2023.
//

#ifndef YOLO_h
#define YOLO_h

#import <simd/simd.h>

typedef struct NMSParams {
  float iouThreshold;
} NMSParams;

typedef struct BBoxFilterParams {
  simd_float2 factor;
  float confidenceThreshold;
  int stride;
  uint numberOfClasses; // For YOLOv8 = 80
  uint segmentationMaskLength; // For YOLOv8 = 32
  uint8_t hasSegmentationMask;
} BBoxFilterParams;

typedef struct BBox {
  float confidence;
  int x, y, w, h;
  int classId;
} BBox;

#endif /* YOLO_h */
