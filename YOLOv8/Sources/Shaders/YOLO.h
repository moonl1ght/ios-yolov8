//
// Created by moonl1ght 27.02.2023.
//

#ifndef YOLO_h
#define YOLO_h

#import <simd/simd.h>

typedef enum YOLOBufferIndices {
  iParams = 0,
  iBBoxes = 1,
  iBBoxCount = 2,
  iKeptBBoxMap = 3,
  iThreadgroupSize = 4,
  iThreadgroupsPerGrid = 5,
  iPrediction = 6,
  iMaskProposals = 7
} YOLOBufferIndices;

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

typedef struct SegmentationParams {
  float confidence;
  uint bboxCount;
} SegmentationParams;

#endif /* YOLO_h */
