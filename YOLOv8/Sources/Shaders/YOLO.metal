//
// Created by moonl1ght 27.02.2023.
//

#include <metal_stdlib>
#import "YOLO.h"

#define NUM_OF_COORDINATES 4

using namespace metal;

// Intersection over union.
float IOU(BBox box1, BBox box2) {
  float x1 = max(box1.x, box2.x);
  float y1 = max(box1.y, box2.y);
  float x2 = min(box1.x + box1.w, box2.x + box2.w);
  float y2 = min(box1.y + box1.h, box2.y + box2.h);
  float w = max(y2 - y1, 0.0);
  float h = max(x2 - x1, 0.0);
  float intersection = w * h;
  float a1 = box1.w * box1.h;
  float a2 = box2.w * box2.h;
  return intersection / (a1 + a2 - intersection);
}

kernel void nonMaximumSuppression(constant NMSParams& params [[ buffer(iParams) ]],
                                  constant BBox* bboxes [[ buffer(iBBoxes) ]],
                                  constant int& bboxCount [[ buffer(iBBoxCount) ]],
                                  device atomic_int* keptBBoxMap [[ buffer(iKeptBBoxMap) ]],
                                  uint2 gid [[thread_position_in_grid]])
{
  // Skip all the entries below the main diagonal and out of matrix bounds.
  if (gid.x >= uint(bboxCount) || gid.y >= uint(bboxCount) || gid.y >= gid.x) {
    return;
  }
  BBox box1 = bboxes[gid.x];
  BBox box2 = bboxes[gid.y];
  if (box1.classId == box2.classId) {
    float iou = IOU(bboxes[gid.x], bboxes[gid.y]);
    if (iou > params.iouThreshold) {
      if (box1.confidence > box2.confidence) {
        atomic_fetch_add_explicit(&keptBBoxMap[gid.x], 1, memory_order_relaxed);
      } else {
        atomic_fetch_add_explicit(&keptBBoxMap[gid.y], 1, memory_order_relaxed);
      }
    } else {
      atomic_fetch_add_explicit(&keptBBoxMap[gid.x], 1, memory_order_relaxed);
      atomic_fetch_add_explicit(&keptBBoxMap[gid.y], 1, memory_order_relaxed);
    }
  } else {
    atomic_fetch_add_explicit(&keptBBoxMap[gid.x], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&keptBBoxMap[gid.y], 1, memory_order_relaxed);
  }
}

kernel void computeThreadgroupsPerGrid(constant int& bboxCount [[ buffer(iBBoxCount) ]],
                                       constant int* threadgroupSize [[ buffer(iThreadgroupSize) ]],
                                       device int* threadgroupsPerGrid [[ buffer(iThreadgroupsPerGrid) ]])
{
  threadgroupsPerGrid[0] = (bboxCount + threadgroupSize[0] - 1) / threadgroupSize[0];
  threadgroupsPerGrid[1] = (bboxCount + threadgroupSize[1] - 1) / threadgroupSize[1];
  threadgroupsPerGrid[2] = 1;
}

// Filter bounding boxes with high confidence.
kernel void filterBBoxes(constant BBoxFilterParams& params [[ buffer(iParams) ]],
                         constant float* prediction [[ buffer(iPrediction) ]],
                         device atomic_int* bboxCount [[ buffer(iBBoxCount) ]],
                         device BBox* bboxes [[ buffer(iBBoxes) ]],
                         device float* maskProposals [[ buffer(iMaskProposals) ]],
                         uint2 gid [[thread_position_in_grid]])
{
  int stride = params.stride;
  float x = prediction[gid.x];
  float y = prediction[gid.x + stride];
  float w = prediction[gid.x + stride * 2];
  float h = prediction[gid.x + stride * 3];
  BBox box {
    .x = int((x - 0.5 * w) * params.factor.x),
    .y = int((y - 0.5 * h) * params.factor.y),
    .w = int(w * params.factor.x),
    .h = int(h * params.factor.y)
  };
  float maxConfidence = 0;
  int classId;
  uint length = NUM_OF_COORDINATES + params.numberOfClasses;
  for (uint i = NUM_OF_COORDINATES; i < length; ++i) {
    float confidence = prediction[gid.x + stride * i];
    if (confidence > maxConfidence) {
      maxConfidence = confidence;
      classId = i - NUM_OF_COORDINATES;
    }
  }
  if (maxConfidence > params.confidenceThreshold) {
    box.confidence = maxConfidence;
    box.classId = classId;
    int i = atomic_fetch_add_explicit(bboxCount, 1, memory_order_relaxed);
    if (params.hasSegmentationMask) {
      for (uint j = length; j < length + params.segmentationMaskLength; ++j) {
        float prop = prediction[gid.x + stride * j];
        maskProposals[i * params.segmentationMaskLength + j - length] = prop;
      }
    }
    bboxes[i] = box;
  }
}
